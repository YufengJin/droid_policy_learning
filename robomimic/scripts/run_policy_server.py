#!/usr/bin/env python3
"""
run_policy_server.py — Trained-policy WebSocket server.

Loads a robomimic checkpoint (supporting DDP-saved checkpoints), converts to
eval mode with optional bfloat16 precision, and serves actions over WebSocket
via policy_websocket.

Internally handles:
  * Action chunking — Diffusion Policy predicts an action_horizon-length chunk;
    the model caches the chunk and pops one action per infer call.
  * Frame stacking — maintains a temporal buffer of the last ``frame_stack``
    observations so the model receives the correct observation horizon.
  * Language conditioning — encodes the task description into a DistilBERT
    768-d embedding and attaches it to every observation.

Client obs format (RoboCasa-compatible):
  primary_image, secondary_image, wrist_image (H,W,3 uint8),
  proprio (9-d: gripper(2)+eef_pos(3)+eef_quat(4)),
  task_description (str)

Usage:
    python -m robomimic.scripts.run_policy_server --ckpt /path/to/model.pth --port 8000

    python -m robomimic.scripts.run_policy_server \\
        --ckpt /path/to/model.pth --port 8000 --device cuda:1 --bf16
"""

import argparse
import logging
import os
import sys
from collections import deque
from typing import Dict

import cv2
import numpy as np
import torch

from policy_websocket import BasePolicy, WebsocketPolicyServer


import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
from robomimic.algo import algo_factory

logger = logging.getLogger(__name__)

_ROBOCASA_PROPRIO_SLICES = {
    "robot0_gripper_qpos": (0, 2),
    "robot0_eef_pos": (2, 5),
    "robot0_eef_quat": (5, 9),
}
_ROBOCASA_PROPRIO_DIM = 9

_CLIENT_TO_OBS = {
    "primary_image": "robot0_agentview_left_image",
    "secondary_image": "robot0_agentview_right_image",
    "wrist_image": "robot0_eye_in_hand_image",
}


def _strip_ddp_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def _infer_obs_mapping(rgb_keys: set) -> dict:
    """Build client_field -> model_obs_key mapping from rgb_keys."""
    mapping = {}
    for k in sorted(rgb_keys):
        kl = k.lower()
        if "eye_in_hand" in kl or "wrist" in kl:
            mapping.setdefault("wrist_image", k)
        elif "agentview_right" in kl:
            mapping.setdefault("secondary_image", k)
        elif "agentview_left" in kl:
            mapping.setdefault("primary_image", k)
    for client_field, default_key in _CLIENT_TO_OBS.items():
        if client_field not in mapping and default_key in rgb_keys:
            mapping[client_field] = default_key
    unmapped = [k for k in rgb_keys if k not in mapping.values()]
    for field in ("primary_image", "secondary_image", "wrist_image"):
        if field not in mapping and unmapped:
            mapping[field] = unmapped.pop(0)
    return mapping


class TrainedPolicyAdapter(BasePolicy):
    """WebSocket BasePolicy backed by a trained robomimic checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        use_bf16: bool = False,
        image_size: tuple | None = None,
        obs_mapping: dict | None = None,
    ):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path)
        algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
        config, _ = FileUtils.config_from_checkpoint(
            algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=True,
        )
        ObsUtils.initialize_obs_utils_with_config(config)

        shape_meta = ckpt_dict["shape_metadata"]
        self.obs_normalization_stats = self._load_norm_stats(
            ckpt_dict, "obs_normalization_stats",
        )
        self.action_normalization_stats = self._load_norm_stats(
            ckpt_dict, "action_normalization_stats",
        )

        model = algo_factory(
            algo_name,
            config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
        model_state = ckpt_dict["model"]
        try:
            model.deserialize(model_state)
        except RuntimeError:
            print("Stripping DDP 'module.' prefix …")
            if "nets" in model_state:
                model_state["nets"] = _strip_ddp_prefix(model_state["nets"])
            if model_state.get("ema") is not None:
                model_state["ema"] = _strip_ddp_prefix(model_state["ema"])
            model.deserialize(model_state)

        model.set_eval()
        model.reset()

        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        if use_bf16:
            model.nets = model.nets.to(dtype=torch.bfloat16)
            if getattr(model, "ema", None) is not None:
                model.ema.averaged_model = model.ema.averaged_model.to(dtype=torch.bfloat16)
            print("Model converted to bfloat16")

        self.model = model
        self.config = config
        self.device = device
        self.frame_stack = getattr(config.train, "frame_stack", 1)
        self.frame_buffer: deque = deque(maxlen=self.frame_stack)

        if image_size is not None:
            self.image_size = image_size
        elif hasattr(config.observation, "image_dim") and config.observation.image_dim:
            dim = config.observation.image_dim
            self.image_size = (dim[0], dim[1] if len(dim) > 1 else dim[0]) if isinstance(dim, (list, tuple)) else (128, 128)
        else:
            self.image_size = (128, 128)  # Unified 128 for RoboCasa

        self.rgb_keys = set()
        self.low_dim_keys = set()
        self.lang_keys = set()
        for modality in ("low_dim", "rgb", "depth", "scan", "lang"):
            keys = getattr(config.observation.modalities.obs, modality, None) or []
            if modality == "rgb":
                self.rgb_keys.update(keys)
            elif modality == "low_dim":
                self.low_dim_keys.update(keys)
            elif modality == "lang":
                self.lang_keys.update(keys)

        self.obs_mapping = obs_mapping or _infer_obs_mapping(self.rgb_keys)
        self.low_dim_shapes = {
            k: shape_meta["all_shapes"][k]
            for k in self.low_dim_keys
            if k in shape_meta.get("all_shapes", {})
        }
        self.use_lang = bool(self.lang_keys)
        self.lang_embedding = None
        self._lang_tokenizer = None
        self._lang_model = None
        self.action_keys = config.train.action_keys
        self.action_config = config.train.action_config
        self.step_count = 0

        print(f"Image size: {self.image_size}, Frame stack: {self.frame_stack}")
        print(f"Obs mapping: {self.obs_mapping}, ac_dim: {shape_meta['ac_dim']}\n")

    @staticmethod
    def _load_norm_stats(ckpt_dict: dict, key: str):
        stats = ckpt_dict.get(key)
        if stats is None:
            return None
        for m in stats:
            for k in stats[m]:
                stats[m][k] = np.array(stats[m][k])
        return stats

    def _encode_language(self, text: str) -> np.ndarray:
        from transformers import AutoTokenizer, AutoModel
        if self._lang_tokenizer is None:
            self._lang_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self._lang_model = AutoModel.from_pretrained(
                "distilbert-base-uncased", torch_dtype=torch.float32
            ).to(self.device)
            self._lang_model.eval()
        with torch.no_grad():
            enc = self._lang_tokenizer(text, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._lang_model(**enc)
            return out.last_hidden_state.sum(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    def _build_single_frame(self, obs: Dict) -> dict:
        """Build model obs from client obs dict (primary_image, secondary_image, wrist_image, proprio)."""
        out = {}
        for client_field, obs_key in self.obs_mapping.items():
            if obs_key not in self.rgb_keys:
                continue
            img = obs.get(client_field)
            if img is None:
                continue
            img = np.asarray(img)
            if img.dtype != np.uint8 and img.dtype.kind == "f":
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 3:
                pass
            elif img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            if self.image_size and (img.shape[0], img.shape[1]) != self.image_size:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            out[obs_key] = ObsUtils.process_obs(img, obs_key=obs_key)

        proprio = obs.get("proprio")
        if proprio is not None and self.low_dim_keys:
            proprio = np.asarray(proprio, dtype=np.float32).ravel()
            if len(proprio) != _ROBOCASA_PROPRIO_DIM:
                raise ValueError(f"proprio length {len(proprio)} != expected {_ROBOCASA_PROPRIO_DIM}")
            for k in self.low_dim_keys:
                if k in _ROBOCASA_PROPRIO_SLICES:
                    lo, hi = _ROBOCASA_PROPRIO_SLICES[k]
                    out[k] = proprio[lo:hi].astype(np.float32)
                elif k in self.low_dim_shapes:
                    dim = int(np.prod(self.low_dim_shapes[k]))
                    out[k] = np.zeros(dim, dtype=np.float32)
                else:
                    out[k] = proprio.copy()

        if self.use_lang and self.lang_embedding is not None:
            for k in self.lang_keys:
                if "raw" in k:
                    continue
                out[k] = self.lang_embedding.copy()

        return out

    def _stack_and_tensorize(self, single_frame: dict) -> dict:
        self.frame_buffer.append(single_frame)
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft({k: v.copy() for k, v in single_frame.items()})
        stacked = {k: np.stack([f[k] for f in self.frame_buffer], axis=0) for k in single_frame}
        ob = TensorUtils.to_tensor(stacked)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.device)
        ob = TensorUtils.to_float(ob)
        if self.dtype == torch.bfloat16:
            ob = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in ob.items()}
        return ob

    def _denormalize_action(self, ac: np.ndarray) -> np.ndarray:
        if self.action_normalization_stats is None:
            return ac
        action_shapes = {k: self.action_normalization_stats[k]["offset"].shape[1:] for k in self.action_normalization_stats}
        ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=self.action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
        for key, value in ac_dict.items():
            fmt = self.action_config[key].get("format")
            if fmt == "rot_6d":
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                conv = self.action_config[key].get("convert_at_runtime", "rot_axis_angle")
                if conv == "rot_axis_angle":
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d)
                elif conv == "rot_euler":
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ")
                else:
                    raise ValueError(f"Unknown rot conversion: {conv}")
                ac_dict[key] = rot.squeeze().numpy()
        return AcUtils.action_dict_to_vector(ac_dict, action_keys=self.action_keys)

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs and obs.get("primary_image") is None and obs.get("secondary_image") is None and obs.get("wrist_image") is None:
            ac_dim = int(obs.get("action_dim", 7))
            return {"actions": np.zeros(ac_dim, dtype=np.float64)}

        task_desc = obs.get("task_description") or obs.get("task_desc", "")
        if self.use_lang and task_desc:
            self.lang_embedding = self._encode_language(str(task_desc))

        single = self._build_single_frame(obs)
        if not single:
            ac_dim = int(obs.get("action_dim", 7))
            return {"actions": np.zeros(ac_dim, dtype=np.float64)}

        obs_tensor = self._stack_and_tensorize(single)
        with torch.no_grad():
            action = self.model.get_action(obs_dict=obs_tensor)
        ac = TensorUtils.to_numpy(action[0]).astype(np.float32)
        ac = self._denormalize_action(ac)
        self.step_count += 1
        if self.step_count % 50 == 0:
            logger.info("  step=%d  action[:4]=%s", self.step_count, ac[:4].tolist())
        return {"actions": ac.astype(np.float64)}

    def reset(self) -> None:
        self.model.reset()
        self.frame_buffer.clear()
        self.step_count = 0


class ResetOnInitPolicy(BasePolicy):
    """Reset inner policy when receiving episode-init obs (action_dim, no images)."""

    def __init__(self, policy: BasePolicy):
        self._policy = policy

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs and "primary_image" not in obs:
            self._policy.reset()
        if obs.get("reset") is True:
            self._policy.reset()
        return self._policy.infer(obs)

    def reset(self) -> None:
        self._policy.reset()


def main():
    parser = argparse.ArgumentParser(
        description="Trained-policy WebSocket server (robomimic checkpoint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket listen port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 to save GPU memory")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    inner = TrainedPolicyAdapter(
        ckpt_path=args.ckpt,
        device=device,
        use_bf16=args.bf16,
        image_size=tuple(args.image_size) if args.image_size else None,
    )
    policy = ResetOnInitPolicy(inner)
    metadata = {"policy_name": "TrainedPolicy", "action_dim": 7}

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(f"Trained-policy server on ws://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        if e.errno == 98:
            print(f"\nERROR: Port {args.port} in use. Try: lsof -ti :{args.port} | xargs kill -9")
            sys.exit(1)
        raise
    print("Server stopped.")


if __name__ == "__main__":
    main()
