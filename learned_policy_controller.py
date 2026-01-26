#!/usr/bin/env python3
"""
Learned Policy Controller for DROID Robot (CleanDIFT Diffusion Policy).

This controller is aligned with the official robomimic policy API:
  - uses model.get_action() (action_queue / action_horizon)
  - uses ObsUtils.process_obs for image preprocessing
  - optionally unnormalizes actions using action_normalization_stats
"""

import os
import json
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as ActionUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory


class LearnedPolicyController:
    """
    Controller that mirrors the official rollout path for diffusion_policy.

    Inputs:
        obs_dict["image"][<camera_name>]: HxWx3 uint8 or float in [0,1]
        obs_dict["robot_state"]: optional low-dim state dict

    Output:
        action (np.ndarray): action vector (unnormalized if enabled)
    """

    def __init__(
        self,
        checkpoint_path: str,
        prompt: Optional[str] = None,
        device: str = "cuda",
        unnormalize_actions: bool = True,
        camera_name_mapping: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        use_ema: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.prompt = prompt
        self.device = device
        self.unnormalize_actions = unnormalize_actions
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print("Loading LearnedPolicyController")
            print("=" * 70)

        self.model, self.config, self.shape_meta, self.obs_stats, self.action_stats = self._load_model(use_ema=use_ema)
        self.use_neg_one_one_norm = self._use_neg_one_one_norm()

        # Horizons from config
        self.obs_horizon = int(self.config.algo.horizon.observation_horizon)
        self.action_horizon = int(self.config.algo.horizon.action_horizon)

        # Buffers
        self.obs_buffer = deque(maxlen=self.obs_horizon)

        # Modalities from config
        self.camera_keys = list(self.config.observation.modalities.obs.rgb)
        self.low_dim_keys = list(self.config.observation.modalities.obs.low_dim)
        self.image_dim = tuple(self.config.observation.image_dim)

        # Camera mapping: model camera key -> env camera name
        if camera_name_mapping is None:
            # Default mapping using DROID camera IDs
            try:
                from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
                self.camera_name_mapping = {
                    "wrist_image_left": f"{hand_camera_id}_left",
                    "varied_camera_1_left_image": f"{varied_camera_1_id}_left",
                    "varied_camera_2_left_image": f"{varied_camera_2_id}_left",
                }
            except Exception:
                self.camera_name_mapping = {}
        else:
            self.camera_name_mapping = camera_name_mapping

        if self.verbose:
            print(f"✓ Model loaded: {os.path.basename(checkpoint_path)}")
            print(f"  Device: {self.device}")
            if self.prompt:
                print(f"  Prompt: '{self.prompt}'")
            else:
                print("  Prompt: (none)")
            action_type = getattr(self.config.train, "action_type", None)
            if action_type is None:
                try:
                    action_type = self.config.train.get("action_type", None)
                except Exception:
                    action_type = None
            if action_type:
                print(f"  Action type: {action_type}")
            print(f"  Obs horizon: {self.obs_horizon}")
            print(f"  Action horizon: {self.action_horizon}")
            print(f"  Cameras: {len(self.camera_keys)}")
            print(f"  Low-dim: {len(self.low_dim_keys)}")
            print(f"  Image norm: {'[-1,1]' if self.use_neg_one_one_norm else '[0,1]'}")
            print("=" * 70)

    def _load_model(self, use_ema: bool = True):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        cfg_payload = checkpoint.get("config")
        if isinstance(cfg_payload, str):
            config_dict = json.loads(cfg_payload)
        else:
            config_dict = cfg_payload

        algo_name = checkpoint.get("algo_name") or (config_dict.get("algo_name") if isinstance(config_dict, dict) else None)
        if algo_name is None:
            raise ValueError("Checkpoint missing algo_name")

        base_config = config_factory(algo_name)
        base_config.update(config_dict)

        # Initialize obs utils
        ObsUtils.initialize_obs_utils_with_config(base_config)

        model = algo_factory(
            algo_name=algo_name,
            config=base_config,
            obs_key_shapes=checkpoint["shape_metadata"]["all_shapes"],
            ac_dim=checkpoint["shape_metadata"]["ac_dim"],
            device=self.device,
        )
        self._maybe_strip_teacher_to_match_checkpoint(model, checkpoint)
        model.deserialize(checkpoint["model"])
        if not use_ema:
            model.ema = None
        model.set_eval()
        model.reset()

        # Observation normalization stats (optional)
        obs_stats = checkpoint.get("obs_normalization_stats")
        if obs_stats is not None:
            for key in obs_stats:
                for stat_key in obs_stats[key]:
                    obs_stats[key][stat_key] = np.array(obs_stats[key][stat_key])

        # Action normalization stats (optional)
        action_stats = checkpoint.get("action_normalization_stats")
        if action_stats is not None:
            for key in action_stats:
                for stat_key in action_stats[key]:
                    action_stats[key][stat_key] = np.array(action_stats[key][stat_key])

        return model, base_config, checkpoint["shape_metadata"], obs_stats, action_stats

    @staticmethod
    def _checkpoint_has_teacher(model_state) -> bool:
        nets_state = model_state.get("nets", {}) if isinstance(model_state, dict) else {}
        return any("unet_feature_extractor_base" in k for k in nets_state.keys())

    def _maybe_strip_teacher_to_match_checkpoint(self, model, checkpoint) -> bool:
        model_state = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
        if self._checkpoint_has_teacher(model_state):
            return False
        nets = getattr(model, "nets", None)
        if nets is None or not hasattr(nets, "modules"):
            return False
        stripped = False
        for module in nets.modules():
            if hasattr(module, "strip_teacher"):
                try:
                    did = module.strip_teacher(strip_text_encoder=False, verbose=self.verbose)
                except TypeError:
                    did = module.strip_teacher()
                stripped = stripped or bool(did)
        if stripped and self.verbose:
            print("✓ Stripped CleanDIFT teacher to match checkpoint")
        return stripped

    def _use_neg_one_one_norm(self) -> bool:
        try:
            enc = self.config.observation.encoder.rgb
            core_kwargs = getattr(enc, "core_kwargs", {}) or {}
            core_class = getattr(enc, "core_class", "") or ""
            backbone_class = core_kwargs.get("backbone_class", "") or ""
            normalize_mode = core_kwargs.get("normalize_mode")
            if normalize_mode in ("neg_one_one", "-1_1", "minus_one_one"):
                return True
            if "CleanDIFT" in core_class or "DIFT" in core_class:
                return True
            if "CleanDIFT" in backbone_class or "DIFT" in backbone_class:
                return True
        except Exception:
            pass
        return False

    def _unnormalize_actions(self, action_vec: np.ndarray) -> np.ndarray:
        if self.action_stats is None:
            return action_vec
        action_keys = list(self.config.train.action_keys)
        action_shapes = {
            k: tuple(self.action_stats[k]["offset"].shape[1:]) for k in action_keys
        }
        ac_dict = ActionUtils.vector_to_action_dict(action_vec, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_stats)

        # Optional rot_6d conversion (if configured)
        action_config = self.config.train.action_config
        for key, value in ac_dict.items():
            this_format = action_config[key].get("format", None)
            if this_format == "rot_6d":
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
                if conversion_format == "rot_axis_angle":
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                elif conversion_format == "rot_euler":
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").squeeze().numpy()
                else:
                    raise ValueError(f"Unknown rot_6d conversion format: {conversion_format}")
                ac_dict[key] = rot

        return ActionUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)

    def _process_camera_obs(self, obs_dict):
        processed = {}
        for camera_key in self.camera_keys:
            cam_name = camera_key.split("/")[-1]
            env_cam_name = self.camera_name_mapping.get(cam_name, cam_name)

            if env_cam_name not in obs_dict.get("image", {}):
                raise KeyError(f"Missing camera '{env_cam_name}' in obs_dict['image'] (mapped from '{cam_name}')")

            raw = obs_dict["image"][env_cam_name]
            raw_min = float(raw.min()) if not torch.is_tensor(raw) else float(raw.min().item())

            if self.use_neg_one_one_norm and raw_min < 0.0:
                # already in [-1, 1] (likely), just ensure CHW and float
                img = torch.as_tensor(raw).float()
                if img.ndim == 3 and img.shape[-1] in (3, 4):
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    img = img.permute(2, 0, 1)
                elif img.ndim == 3 and img.shape[0] in (3, 4):
                    if img.shape[0] == 4:
                        img = img[:3, :, :]
            else:
                # Process with ObsUtils (HWC -> CHW, /255)
                img_proc = ObsUtils.process_obs(raw, obs_key=camera_key)
                img = torch.from_numpy(img_proc) if not torch.is_tensor(img_proc) else img_proc
                if self.use_neg_one_one_norm:
                    img = img * 2.0 - 1.0

            # resize to config image_dim if needed
            target_h, target_w = self.image_dim
            if img.shape[-2:] != (target_h, target_w):
                img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)

            processed[camera_key] = img

        return processed

    def _process_low_dim_obs(self, obs_dict):
        processed = {}
        if not self.low_dim_keys:
            return processed
        robot_state = obs_dict.get("robot_state", {})

        for key in self.low_dim_keys:
            if key == "robot_state/joint_position":
                val = robot_state.get("joint_positions", None)
            elif key == "robot_state/cartesian_position":
                val = robot_state.get("cartesian_position", None)
            elif key == "robot_state/gripper_position":
                val = robot_state.get("gripper_position", None)
            else:
                val = None

            if val is None:
                raise KeyError(f"Missing low_dim key '{key}' from robot_state.")

            val = torch.as_tensor(val).float()
            if val.ndim == 0:
                val = val.unsqueeze(0)
            processed[key] = val

        return processed

    def _stack_observations(self, current_obs):
        # Apply obs normalization if stats exist (only for keys present in stats)
        if self.obs_stats is not None:
            to_norm = {k: v for k, v in current_obs.items() if k in self.obs_stats}
            if to_norm:
                ObsUtils.normalize_dict(to_norm, normalization_stats=self.obs_stats)
                current_obs.update(to_norm)

        self.obs_buffer.append(current_obs)
        while len(self.obs_buffer) < self.obs_horizon:
            self.obs_buffer.append({k: v.clone() if torch.is_tensor(v) else v for k, v in current_obs.items()})

        stacked = {}
        for key in current_obs.keys():
            obs_stack = torch.stack([obs[key] for obs in self.obs_buffer], dim=0)
            stacked[key] = obs_stack.unsqueeze(0)  # [1, T, ...]

        # add prompt as raw_language (list of strings), keep as-is
        if self.prompt:
            stacked["raw_language"] = [self.prompt]

        # move tensors to device
        for key, value in stacked.items():
            if torch.is_tensor(value):
                stacked[key] = value.to(self.device)

        return stacked

    def forward(self, obs_dict):
        cam_obs = self._process_camera_obs(obs_dict)
        low_obs = self._process_low_dim_obs(obs_dict)
        current_obs = {**cam_obs, **low_obs}
        stacked_obs = self._stack_observations(current_obs)

        with torch.no_grad():
            action = self.model.get_action(stacked_obs, eval_mode=False)  # [1, Da]

        action = action.squeeze(0).detach().cpu().numpy()

        if self.unnormalize_actions:
            action = self._unnormalize_actions(action)

        return action

    def reset(self):
        self.obs_buffer.clear()
        self.model.reset()

    def update_prompt(self, new_prompt: Optional[str]):
        self.prompt = new_prompt

    def get_info(self):
        return {"controller_on": True, "movement_enabled": True}
