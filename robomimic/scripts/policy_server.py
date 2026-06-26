#!/usr/bin/env python3
"""
policy_server.py — Diffusion-Policy WebSocket server (LIBERO + RoboCasa, benchmark 自动识别).

给一个 robomimic diffusion policy 的 ckpt 即可起服务；按 run_eval 客户端发来的
``obs["__meta__"]["benchmark"]`` 主动识别 benchmark（libero / robocasa），把**原始 robosuite obs**
转换成 ``TrainedPolicyAdapter`` 需要的 prepared obs（primary/secondary/wrist + proprio + task_description），
再交给已训练模型推理。设计参考 openvla-oft 的 vla-scripts/policy_server.py（envelope-first dispatch）。

本文件自包含：模型加载/推理核心 ``TrainedPolicyAdapter``（原 run_policy_server.py，已合并进来）
+ raw→prepared 适配 + benchmark 分发 ``BenchmarkDispatchPolicy``。

obs 约定（对应训练 config）：
  * LIBERO 模型：2 路相机(agentview/eye_in_hand) + robot_states(9D=[gripper_qpos(2),eef_pos(3),eef_quat(4)])，无语言。
  * RoboCasa 模型：3 路相机(agentview_left/right/eye_in_hand) + DistilBERT 语言，无 low_dim（proprio 不使用）。

动作：diffusion policy 训练于同仿真 HDF5，输出已是 env 动作空间（7D，gripper 已是 -1/+1），
不做 openvla 式 gripper normalize/invert；run_eval 端 ``pad_action_for_env`` 负责 7→env_dim 填充。

用法：
    python -m robomimic.scripts.policy_server --ckpt /path/to/model.pth --port 8000
客户端（run_eval，--arm_controller cartesian_pose，7D）：
    # robocasa env: python scripts/run_eval.py --task_name PnPCounterToCab --policy_server_addr localhost:8000
    # libero env:   python scripts/run_eval.py --task_suite_name libero_spatial --policy_server_addr localhost:8000
"""

import argparse
import logging
import sys
from collections import deque
from typing import Any, Dict

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


# ===========================================================================
# TrainedPolicyAdapter —— 载 robomimic ckpt + 推理（含 action chunking / frame
# stacking / 语言编码 / 动作反归一化）。客户端发 prepared obs（primary_image /
# secondary_image / wrist_image / proprio / task_description）。
# ===========================================================================

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
        # locked robomimic Config 上 getattr(默认值) 会抛 RuntimeError（缺失键时），
        # 用 `in` 成员判断安全取值（如 LIBERO 配置无 lang/depth/scan 模态）。
        obs_mod = config.observation.modalities.obs
        for modality in ("low_dim", "rgb", "depth", "scan", "lang"):
            keys = (obs_mod[modality] or []) if modality in obs_mod else []
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
                    # 单一 low_dim 键且维度与 proprio 等长（如 LIBERO 的 robot_states=9D）→ 直接用整段 proprio；
                    # 否则该 low_dim 不在 proprio 内，置零占位。
                    out[k] = proprio.astype(np.float32).copy() if dim == len(proprio) else np.zeros(dim, dtype=np.float32)
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
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d)
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


# ===========================================================================
# raw robosuite obs  ->  TrainedPolicyAdapter 的 prepared obs
# ===========================================================================

RAW_LIBERO_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
RAW_ROBOCASA_KEYS = ("robot0_agentview_left_image", "robot0_eye_in_hand_image")


def _validate(obs: Dict[str, Any], required: tuple, name: str) -> None:
    missing = [k for k in required if obs.get(k) is None]
    if missing:
        raise ValueError(
            "obs 格式 '{}' 缺少必需键 {}；收到键(前20): {}".format(
                name, missing, list(obs.keys())[:20]
            )
        )


def _proprio_libero(raw: Dict[str, Any]) -> np.ndarray:
    """LIBERO robot_states 顺序：[gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9D（与 RoboCasa 同序）。
    经逐列核对训练 HDF5：robot_states[:,0:2]==gripper_states、[:,2:5]==ee_pos、[:,5:9] 为单位四元数
    (== robosuite robot0_eef_quat)。训练把整段 robot_states 作单一 low_dim 键喂模型，故 eval 必须同序拼接。"""
    if raw.get("robot0_eef_pos") is None:
        return np.zeros(9, dtype=np.float32)
    return np.concatenate([
        np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).ravel(),  # 2
        np.asarray(raw["robot0_eef_pos"], dtype=np.float32).ravel(),       # 3
        np.asarray(raw["robot0_eef_quat"], dtype=np.float32).ravel(),      # 4
    ]).astype(np.float32)


def _proprio_robocasa(raw: Dict[str, Any]) -> np.ndarray:
    """RoboCasa proprio 顺序：[gripper_qpos(2), eef_pos(3), eef_quat(4)] = 9D（见 _ROBOCASA_PROPRIO_SLICES）。"""
    if raw.get("robot0_eef_pos") is None:
        return np.zeros(9, dtype=np.float32)
    return np.concatenate([
        np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).ravel(),  # 2
        np.asarray(raw["robot0_eef_pos"], dtype=np.float32).ravel(),       # 3
        np.asarray(raw["robot0_eef_quat"], dtype=np.float32).ravel(),      # 4
    ]).astype(np.float32)


def prepare_obs_from_libero(raw: Dict[str, Any]) -> Dict[str, Any]:
    _validate(raw, RAW_LIBERO_KEYS, "libero")
    return {
        "primary_image": np.asarray(raw["agentview_image"]),
        "wrist_image": np.asarray(raw["robot0_eye_in_hand_image"]),
        "proprio": _proprio_libero(raw),
        "task_description": raw.get("task_description", ""),
    }


def prepare_obs_from_robocasa(raw: Dict[str, Any]) -> Dict[str, Any]:
    _validate(raw, RAW_ROBOCASA_KEYS, "robocasa")
    out = {
        "primary_image": np.asarray(raw["robot0_agentview_left_image"]),
        "wrist_image": np.asarray(raw["robot0_eye_in_hand_image"]),
        "proprio": _proprio_robocasa(raw),
        "task_description": raw.get("task_description", ""),
    }
    if raw.get("robot0_agentview_right_image") is not None:
        out["secondary_image"] = np.asarray(raw["robot0_agentview_right_image"])
    return out


OBS_ADAPTERS = {
    "libero": prepare_obs_from_libero,
    "robocasa": prepare_obs_from_robocasa,
}


def _sniff_benchmark(obs: Dict[str, Any]) -> str:
    """旧客户端无 __meta__ 信封时，按图像键回退识别。"""
    if obs.get("agentview_image") is not None:
        return "libero"
    if obs.get("robot0_agentview_left_image") is not None:
        return "robocasa"
    raise ValueError(
        "无法识别 benchmark：缺少 __meta__ 信封且无已知图像键。"
        "期望 obs['__meta__']['benchmark'] in {} 或键 agentview_image / "
        "robot0_agentview_left_image。收到键(前25): {}".format(
            sorted(OBS_ADAPTERS), list(obs.keys())[:25]
        )
    )


# ---------------------------------------------------------------------------
# 分发策略：raw obs + __meta__  ->  prepared obs  ->  TrainedPolicyAdapter
# ---------------------------------------------------------------------------

class BenchmarkDispatchPolicy(BasePolicy):
    """读 __meta__ 选 benchmark adapter，把 raw obs 转 prepared 再交给 TrainedPolicyAdapter。"""

    def __init__(self, inner: TrainedPolicyAdapter):
        self._inner = inner

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        meta = obs.pop("__meta__", {}) if isinstance(obs, dict) else {}

        # init / handshake：reset 并回零动作（modern: meta.phase=="init"；legacy: 有 action_dim 且无图像）
        phase = meta.get("phase")
        has_images = any(
            obs.get(k) is not None
            for k in ("agentview_image", "robot0_agentview_left_image",
                      "primary_image", "robot0_eye_in_hand_image")
        )
        if phase == "init" or (phase is None and "action_dim" in obs and not has_images):
            self._inner.reset()
            return {"actions": np.zeros(int(obs.get("action_dim", 7)), dtype=np.float64)}

        benchmark = meta.get("benchmark", "") or _sniff_benchmark(obs)
        adapter = OBS_ADAPTERS.get(benchmark)
        if adapter is None:
            raise ValueError("未知 benchmark {!r}；已注册: {}".format(benchmark, sorted(OBS_ADAPTERS)))
        prepared = adapter(obs)
        if meta.get("task_description"):
            prepared["task_description"] = meta["task_description"]
        return self._inner.infer(prepared)

    def reset(self) -> None:
        self._inner.reset()


def main():
    parser = argparse.ArgumentParser(
        description="Diffusion-policy WebSocket server (LIBERO+RoboCasa, benchmark 自动识别)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="robomimic diffusion policy .pth")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 省显存")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    inner = TrainedPolicyAdapter(
        ckpt_path=args.ckpt,
        device=device,
        use_bf16=args.bf16,
        image_size=tuple(args.image_size) if args.image_size else None,
    )
    policy = BenchmarkDispatchPolicy(inner)
    metadata = {"policy_name": "DiffusionPolicy", "action_dim": 7}

    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    print("Diffusion policy server on ws://{}:{} (benchmark auto-detect)".format(args.host, args.port))
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as e:
        if e.errno == 98:
            print("\nERROR: 端口 {} 被占用。试: lsof -ti :{} | xargs kill -9".format(args.port, args.port))
            sys.exit(1)
        raise
    print("Server stopped.")


if __name__ == "__main__":
    main()
