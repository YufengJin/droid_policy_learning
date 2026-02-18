#!/usr/bin/env python3
"""
run_policy_server.py — Trained-policy gRPC server.

Loads a robomimic checkpoint (supporting DDP-saved checkpoints), converts to
eval mode with optional bfloat16 precision, and serves actions over gRPC.

Internally handles:
  * Action chunking — Diffusion Policy predicts an action_horizon-length chunk;
    the server caches the chunk and pops one action per GetAction call.
  * Frame stacking — maintains a temporal buffer of the last ``frame_stack``
    observations so the model receives the correct observation horizon.
  * Language conditioning — encodes the task description from Reset into a
    DistilBERT 768-d embedding and attaches it to every observation.

Usage:
    # basic
    python -m robomimic.scripts.run_policy_server --ckpt /path/to/model.pth

    # bfloat16 on specific GPU
    python -m robomimic.scripts.run_policy_server \\
        --ckpt /path/to/model.pth --device cuda:1 --bf16

    # custom observation key mapping
    python -m robomimic.scripts.run_policy_server \\
        --ckpt /path/to/model.pth \\
        --obs_mapping secondary_image:robot0_agentview_right_image \\
                       wrist_image:robot0_eye_in_hand_image
"""

import argparse
import atexit
import json
import os
import signal
import sys
from collections import deque
from concurrent import futures

import cv2
import grpc
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOMIMIC_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _ROBOMIMIC_ROOT not in sys.path:
    sys.path.insert(0, _ROBOMIMIC_ROOT)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
from robomimic.algo import algo_factory

from policy_bridge.grpc.robocasa import policy_service_pb2, policy_service_pb2_grpc

# Global server reference for signal handlers
_server = None


# ---------------------------------------------------------------------------
# RoboCasa gRPC proprio convention (must match robocasa prepare_observation)
# proprio = concat(robot0_gripper_qpos(2), robot0_eef_pos(3), robot0_eef_quat(4))
# ---------------------------------------------------------------------------
_ROBOCASA_PROPRIO_SLICES = {
    "robot0_gripper_qpos": (0, 2),
    "robot0_eef_pos": (2, 5),
    "robot0_eef_quat": (5, 9),
}
_ROBOCASA_PROPRIO_DIM = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(buf: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB uint8 HWC numpy array."""
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _strip_ddp_prefix(state_dict: dict) -> dict:
    """Strip ``module.`` prefix that DDP wrapping adds to state-dict keys."""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def _infer_obs_mapping(rgb_keys: set) -> dict:
    """Build a default gRPC-field → model-obs-key mapping from the config's
    RGB keys, following RoboCasa naming conventions."""
    mapping: dict[str, str] = {}
    for k in rgb_keys:
        kl = k.lower()
        if "eye_in_hand" in kl or "wrist" in kl:
            mapping.setdefault("wrist_image", k)
        elif "agentview_right" in kl:
            mapping.setdefault("secondary_image", k)
        elif "agentview_left" in kl:
            mapping.setdefault("primary_image", k)

    # TODO strict mapping between rgb keys and grpc fields
    unmapped = [k for k in rgb_keys if k not in mapping.values()]
    for field in ("primary_image", "secondary_image", "wrist_image"):
        if field not in mapping and unmapped:
            mapping[field] = unmapped.pop(0)
    return mapping


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------

class TrainedPolicyServicer(policy_service_pb2_grpc.PolicyServiceServicer):
    """gRPC PolicyService backed by a trained robomimic checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        use_bf16: bool = False,
        image_size: tuple | None = None,
        obs_mapping: dict | None = None,
    ):
        print(f"Loading checkpoint: {ckpt_path}")

        # -- checkpoint & config ------------------------------------------
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path)
        algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
        config, _ = FileUtils.config_from_checkpoint(
            algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=True,
        )
        ObsUtils.initialize_obs_utils_with_config(config)

        shape_meta = ckpt_dict["shape_metadata"]

        # -- normalization stats ------------------------------------------
        self.obs_normalization_stats = self._load_norm_stats(
            ckpt_dict, "obs_normalization_stats",
        )
        self.action_normalization_stats = self._load_norm_stats(
            ckpt_dict, "action_normalization_stats",
        )

        # -- model --------------------------------------------------------
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
            print("Direct deserialize failed; stripping DDP 'module.' prefix …")
            if "nets" in model_state:
                model_state["nets"] = _strip_ddp_prefix(model_state["nets"])
            if model_state.get("ema") is not None:
                model_state["ema"] = _strip_ddp_prefix(model_state["ema"])
            model.deserialize(model_state)

        model.set_eval()
        model.reset()

        # -- optional bfloat16 --------------------------------------------
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        if use_bf16:
            model.nets = model.nets.to(dtype=torch.bfloat16)
            if getattr(model, "ema", None) is not None:
                model.ema.averaged_model = model.ema.averaged_model.to(
                    dtype=torch.bfloat16,
                )
            print("Model converted to bfloat16")

        self.model = model
        self.config = config
        self.device = device

        # -- observation metadata -----------------------------------------
        self.frame_stack = getattr(config.train, "frame_stack", 1)
        self.frame_buffer: deque[dict] = deque(maxlen=self.frame_stack)

        if image_size is not None:
            self.image_size = image_size
        elif (
            hasattr(config.observation, "image_dim")
            and config.observation.image_dim is not None
        ):
            dim = config.observation.image_dim
            if isinstance(dim, (list, tuple)) and len(dim) >= 1:
                self.image_size = (dim[0], dim[1] if len(dim) > 1 else dim[0])
            else:
                self.image_size = None
        else:
            self.image_size = None

        self.rgb_keys: set[str] = set()
        self.low_dim_keys: set[str] = set()
        self.lang_keys: set[str] = set()
        all_obs_keys: list[str] = []
        for modality in ("low_dim", "rgb", "depth", "scan", "lang"):
            keys = getattr(config.observation.modalities.obs, modality, None)
            if not keys:
                continue
            all_obs_keys.extend(keys)
            if modality == "rgb":
                self.rgb_keys.update(keys)
            elif modality == "low_dim":
                self.low_dim_keys.update(keys)
            elif modality == "lang":
                self.lang_keys.update(keys)
        self.all_obs_keys = all_obs_keys

        self.obs_mapping = obs_mapping or _infer_obs_mapping(self.rgb_keys)
        self.low_dim_shapes = {
            k: shape_meta["all_shapes"][k]
            for k in self.low_dim_keys
            if k in shape_meta.get("all_shapes", {})
        }
        self.use_lang = bool(self.lang_keys)
        self.lang_embedding: np.ndarray | None = None
        self._lang_tokenizer = None
        self._lang_model = None

        self.action_keys = config.train.action_keys
        self.action_config = config.train.action_config

        self.step_count = 0
        self.episode_count = 0

        print(f"Image size:  {self.image_size}")
        print(f"Frame stack: {self.frame_stack}")
        print(f"Obs mapping: {self.obs_mapping}")
        print(f"RGB keys:    {sorted(self.rgb_keys)}")
        print(f"Lang keys:   {sorted(self.lang_keys)}")
        print(f"ac_dim:      {shape_meta['ac_dim']}")
        print("Policy server ready.\n")

    # -- internal helpers -------------------------------------------------

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
        """Encode *text* with DistilBERT → 768-d float32 vector."""
        from transformers import AutoTokenizer, AutoModel

        if self._lang_tokenizer is None:
            self._lang_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased",
            )
            self._lang_model = AutoModel.from_pretrained(
                "distilbert-base-uncased", torch_dtype=torch.float32,
            ).to(self.device)
            self._lang_model.eval()

        with torch.no_grad():
            enc = self._lang_tokenizer(text, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._lang_model(**enc)
            embedding = out.last_hidden_state.sum(dim=1).squeeze(0)

        return embedding.cpu().numpy().astype(np.float32)

    def _process_image(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG, resize to model's expected resolution, return uint8 HWC."""
        img = _decode_jpeg(jpeg_bytes)
        if self.image_size is not None:
            h, w = self.image_size
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def _build_single_frame(self, request) -> dict:
        """Parse a gRPC ObservationRequest into a *processed* single-frame obs
        dict (numpy arrays).  Images are float32 [0,1] CHW after
        ``ObsUtils.process_obs``."""
        obs: dict[str, np.ndarray] = {}

        for grpc_field, obs_key in self.obs_mapping.items():
            if obs_key not in self.rgb_keys:
                continue
            jpeg_bytes = getattr(request, grpc_field, b"")
            if not jpeg_bytes:
                continue
            img = self._process_image(jpeg_bytes)
            obs[obs_key] = ObsUtils.process_obs(img, obs_key=obs_key)

        if request.proprio and self.low_dim_keys:
            proprio = np.array(request.proprio, dtype=np.float32)
            if len(proprio) != _ROBOCASA_PROPRIO_DIM:
                raise ValueError(
                    f"proprio length {len(proprio)} != expected {_ROBOCASA_PROPRIO_DIM} "
                    "(gripper(2)+eef_pos(3)+eef_quat(4))"
                )
            for k in self.low_dim_keys:
                if k in _ROBOCASA_PROPRIO_SLICES:
                    lo, hi = _ROBOCASA_PROPRIO_SLICES[k]
                    obs[k] = proprio[lo:hi].astype(np.float32)
                elif k in self.low_dim_shapes:
                    shape = self.low_dim_shapes[k]
                    dim = int(np.prod(shape))
                    obs[k] = np.zeros(dim, dtype=np.float32)
                else:
                    obs[k] = proprio.copy()

        if self.use_lang and self.lang_embedding is not None:
            for k in self.lang_keys:
                if "raw" in k:
                    continue
                obs[k] = self.lang_embedding.copy()

        return obs

    def _stack_and_tensorize(self, single_frame: dict) -> dict:
        """Append *single_frame* to the temporal buffer, stack along dim 0,
        and return a batched tensor dict on ``self.device``."""
        self.frame_buffer.append(single_frame)

        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft(
                {k: v.copy() for k, v in single_frame.items()},
            )

        stacked = {
            k: np.stack([f[k] for f in self.frame_buffer], axis=0)
            for k in single_frame
        }

        ob = TensorUtils.to_tensor(stacked)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.device)
        ob = TensorUtils.to_float(ob)

        if self.dtype == torch.bfloat16:
            ob = {
                k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
                for k, v in ob.items()
            }
        return ob

    def _denormalize_action(self, ac: np.ndarray) -> np.ndarray:
        """Reverse action normalisation and optional rot-6d conversion."""
        if self.action_normalization_stats is None:
            return ac

        action_shapes = {
            k: self.action_normalization_stats[k]["offset"].shape[1:]
            for k in self.action_normalization_stats
        }
        ac_dict = AcUtils.vector_to_action_dict(
            ac, action_shapes=action_shapes, action_keys=self.action_keys,
        )
        ac_dict = ObsUtils.unnormalize_dict(
            ac_dict, normalization_stats=self.action_normalization_stats,
        )
        for key, value in ac_dict.items():
            fmt = self.action_config[key].get("format", None)
            if fmt == "rot_6d":
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                conv = self.action_config[key].get(
                    "convert_at_runtime", "rot_axis_angle",
                )
                if conv == "rot_axis_angle":
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d)
                elif conv == "rot_euler":
                    rot = TorchUtils.rot_6d_to_euler_angles(
                        rot_6d=rot_6d, convention="XYZ",
                    )
                else:
                    raise ValueError(f"Unknown rot conversion format: {conv}")
                ac_dict[key] = rot.squeeze().numpy()
        return AcUtils.action_dict_to_vector(
            ac_dict, action_keys=self.action_keys,
        )

    # -- gRPC endpoints ---------------------------------------------------

    def Reset(self, request, context):
        self.model.reset()
        self.frame_buffer.clear()
        self.step_count = 0
        self.episode_count += 1

        if self.use_lang and request.task_description:
            self.lang_embedding = self._encode_language(request.task_description)

        print(
            f"[Reset] episode={self.episode_count}  "
            f"task={request.task_name!r}  "
            f"desc={request.task_description!r}"
        )
        return policy_service_pb2.ResetResponse(success=True)

    def GetAction(self, request, context):
        if self.model.action_queue is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Reset has not been called yet.")
            return policy_service_pb2.ActionResponse()

        single_obs = self._build_single_frame(request)
        obs_tensor = self._stack_and_tensorize(single_obs)

        with torch.no_grad():
            action = self.model.get_action(obs_dict=obs_tensor)

        ac = TensorUtils.to_numpy(action[0]).astype(np.float32)
        ac = self._denormalize_action(ac)

        self.step_count += 1
        if self.step_count % 50 == 0:
            print(
                f"  [GetAction] step={self.step_count}  "
                f"action[:4]={ac[:4].tolist()}"
            )

        return policy_service_pb2.ActionResponse(action=ac.tolist())


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _graceful_shutdown(signum=None, frame=None):
    global _server
    if _server is not None:
        print(f"\nReceived signal {signum}, shutting down …", flush=True)
        stop_event = _server.stop(grace=0)
        stop_event.wait(timeout=5)
        _server = None
        print("Server stopped, port released.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def serve(args):
    global _server

    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    obs_mapping = None
    if args.obs_mapping:
        obs_mapping = {}
        for pair in args.obs_mapping:
            grpc_field, obs_key = pair.split(":")
            obs_mapping[grpc_field] = obs_key

    servicer = TrainedPolicyServicer(
        ckpt_path=args.ckpt,
        device=device,
        use_bf16=args.bf16,
        image_size=tuple(args.image_size) if args.image_size else None,
        obs_mapping=obs_mapping,
    )

    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ("grpc.so_reuseport", 1),
    ]
    _server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=options,
    )
    policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(servicer, _server)

    bind_addr = f"{args.host}:{args.port}"
    _server.add_insecure_port(bind_addr)
    _server.start()
    print(f"Trained-policy server listening on {bind_addr}")
    print("Press Ctrl+C to stop.\n")

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    atexit.register(_graceful_shutdown)

    _server.wait_for_termination()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trained-policy gRPC server (robomimic checkpoint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--port", type=int, default=50051,
        help="gRPC listen port",
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Bind host (localhost = only local access)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device, e.g. cuda:0 (default: auto-detect)",
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Convert model weights to bfloat16 to save GPU memory",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Override image resize resolution (default: from checkpoint config)",
    )
    parser.add_argument(
        "--obs_mapping", type=str, nargs="+", default=None,
        metavar="FIELD:KEY",
        help=(
            "gRPC-field:obs-key pairs, e.g. "
            "secondary_image:robot0_agentview_right_image "
            "wrist_image:robot0_eye_in_hand_image"
        ),
    )
    parser.add_argument(
        "--max_workers", type=int, default=4,
        help="Max gRPC thread-pool workers",
    )
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
