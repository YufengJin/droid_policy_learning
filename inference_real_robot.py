#!/usr/bin/env python3
"""
单卡推理脚本 - 用于验证训练好的模型
加载checkpoint并生成轨迹预测
"""
import argparse
import os
import json
import itertools

import numpy as np
import torch
import tensorflow as tf

from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.rlds_utils import (
    get_droid_standardize_fn,
    robomimic_transform,
    DROID_TO_RLDS_OBS_KEY_MAP,
    DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP,
)
from octo.data.dataset import (
    make_dataset_from_rlds,
    apply_trajectory_transforms,
    apply_frame_transforms,
)

tf.config.set_visible_devices([], "GPU")


def _looks_like_zip(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def _validate_checkpoint(path):
    if not os.path.exists(path):
        return False
    if not _looks_like_zip(path):
        return True
    try:
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            return bad is None
    except Exception:
        return False


def resolve_checkpoint(checkpoint_path, run_dir):
    if checkpoint_path:
        if not _validate_checkpoint(checkpoint_path):
            raise RuntimeError(
                f"Checkpoint损坏或不完整: {checkpoint_path}\n"
                "请使用训练目录里的其它checkpoint，或重新保存/训练。"
            )
        return checkpoint_path
    if not run_dir:
        raise ValueError("必须指定 --checkpoint 或 --run_dir")
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"未找到models目录: {models_dir}")
    candidates = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not candidates:
        raise FileNotFoundError(f"未找到checkpoint文件: {models_dir}/*.pth")
    candidates.sort(key=lambda p: os.path.getmtime(os.path.join(models_dir, p)))
    for fname in reversed(candidates):
        path = os.path.join(models_dir, fname)
        if _validate_checkpoint(path):
            return path
    raise RuntimeError(f"models目录下没有可用的checkpoint: {models_dir}")


def _load_action_normalization_stats(ckpt):
    action_stats = ckpt.get("action_normalization_stats")
    if action_stats is None:
        return None
    for key in action_stats:
        for stat_key in action_stats[key]:
            action_stats[key][stat_key] = np.array(action_stats[key][stat_key])
    return action_stats


def load_model(checkpoint_path, device='cuda:0'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 重建config
    algo_name = ckpt.get("algo_name")
    if algo_name is None:
        algo_name = ckpt.get("config", {}).get("algo_name")
    if algo_name is None:
        raise ValueError("checkpoint中缺少algo_name")
    config = config_factory(algo_name)
    cfg_payload = ckpt.get("config")
    if isinstance(cfg_payload, str):
        cfg_payload = json.loads(cfg_payload)
    config.update(cfg_payload)
    # Initialize obs utils mapping before building networks.
    ObsUtils.initialize_obs_utils_with_config(config)
    # prevent noisy "not found" logs for these optional keys
    if hasattr(ObsUtils, "OBS_KEYS_TO_MODALITIES"):
        ObsUtils.OBS_KEYS_TO_MODALITIES.setdefault("raw_language", "low_dim")
        ObsUtils.OBS_KEYS_TO_MODALITIES.setdefault("pad_mask", "low_dim")
    
    # 创建模型
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=ckpt['shape_metadata']['all_shapes'],
        ac_dim=ckpt['shape_metadata']['ac_dim'],
        device=device,
    )
    
    # 加载权重
    model.deserialize(ckpt['model'])
    model.set_eval()
    
    print(f"✅ 模型加载成功")
    action_stats = _load_action_normalization_stats(ckpt)
    return model, config, action_stats


def _infer_action_dim_from_stats(action_stats):
    if not action_stats:
        return None
    total = 0
    for key in action_stats:
        stats = action_stats[key]
        offset = stats.get("offset")
        if offset is None:
            continue
        total += int(np.prod(np.array(offset).shape))
    return total if total > 0 else None


def load_dataset(config, action_stats=None, action_type=None):
    """加载验证数据集"""
    print("加载数据集...")
    
    # 从config中提取数据集配置
    data_path = config.train.data_path
    dataset_names = config.train.dataset_names
    
    # 相机配置
    obs_modalities = config.observation.modalities.obs.rgb
    image_obs_keys = {
        "primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]], 
        "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]]
    }
    
    # 状态配置
    state_obs_keys = []
    for obs_key in config.observation.modalities.obs.low_dim:
        mapped = DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP.get(obs_key)
        if mapped:
            state_obs_keys.append(mapped)
    
    # action配置
    ac_dim = _infer_action_dim_from_stats(action_stats)
    if ac_dim is None:
        ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
    is_abs_action = [True] * ac_dim

    use_neg_one_one_norm = False
    if hasattr(config.observation.encoder, "rgb"):
        core_class = getattr(config.observation.encoder.rgb, "core_class", "") or ""
        core_kwargs = getattr(config.observation.encoder.rgb, "core_kwargs", {}) or {}
        backbone_class = core_kwargs.get("backbone_class", "") or ""
        normalize_mode = core_kwargs.get("normalize_mode")
        if normalize_mode in ("neg_one_one", "-1_1", "minus_one_one"):
            use_neg_one_one_norm = True
        elif normalize_mode in (None, "zero_one", "imagenet"):
            use_neg_one_one_norm = False
        else:
            if any(tag in core_class for tag in ("CleanDIFT", "DIFT")):
                use_neg_one_one_norm = True
            if any(tag in backbone_class for tag in ("CleanDIFT", "DIFT")):
                use_neg_one_one_norm = True
    
    if not action_type:
        action_type = _safe_cfg_get(config.train, "action_type", "cartesian_abs")
    standardize_fn = get_droid_standardize_fn(action_type)

    BASE_DATASET_KWARGS = {
        "data_dir": data_path,
        "image_obs_keys": image_obs_keys,
        "state_obs_keys": state_obs_keys,
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": is_abs_action,
        "action_normalization_mask": is_abs_action,
        "standardize_fn": standardize_fn,
    }
    
    # 创建数据集
    dataset, _ = make_dataset_from_rlds(
        name=dataset_names[0],
        **BASE_DATASET_KWARGS,
        train=False,  # 使用验证集或测试集
        shuffle=False,
    )

    dataset = apply_trajectory_transforms(
        dataset,
        train=False,
        window_size=config.algo.horizon.observation_horizon,
        future_action_window_size=config.algo.horizon.prediction_horizon - 1,
        subsample_length=config.train.subsample_length,
        skip_unlabeled=True,
    ).flatten()

    dataset = apply_frame_transforms(
        dataset,
        train=False,
        resize_size=dict(
            primary=config.observation.image_dim,
            secondary=config.observation.image_dim,
        ),
        num_parallel_calls=4,
    )
    
    # 应用transform
    has_proprio = len(config.observation.modalities.obs.low_dim) > 0
    dataset = dataset.map(
        lambda x: robomimic_transform(
            x,
            normalize_to_neg_one_one=use_neg_one_one_norm,
            camera_keys=obs_modalities,
            include_proprio=has_proprio,
        ),
        num_parallel_calls=4,
    )
    
    print(f"✅ 数据集加载成功")
    return dataset


def _prepare_obs(batch_obs, device, obs_horizon, language_prompt=None):
    obs_dict = {}
    if language_prompt is not None and str(language_prompt).strip() != "":
        obs_dict["raw_language"] = [str(language_prompt)]
    else:
        raw_language = batch_obs.get("raw_language")
        if raw_language is not None:
            if isinstance(raw_language, np.ndarray):
                entry = raw_language[0] if raw_language.size > 0 else raw_language
            else:
                entry = raw_language
            obs_dict["raw_language"] = [entry]

    for key, value in batch_obs.items():
        if key == "raw_language":
            continue
        arr = value
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            arr = np.array(value, copy=True)
        tensor = torch.as_tensor(arr).float()
        if tensor.ndim >= 1:
            tensor = tensor[:obs_horizon]
        tensor = tensor.unsqueeze(0).to(device)
        obs_dict[key] = tensor

    obs_dict = ObsUtils.process_obs_dict(obs_dict)
    for key, value in obs_dict.items():
        if torch.is_tensor(value):
            obs_dict[key] = value.to(device)
    return obs_dict


def _safe_cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg[key] if key in cfg else default
    except Exception:
        return default


def _unnormalize_actions(action_vec, action_stats, action_keys, action_shapes):
    if action_stats is None:
        return action_vec
    out = np.array(action_vec, copy=True)
    start = 0
    for key, shape in zip(action_keys, action_shapes):
        dim = int(np.prod(shape))
        offset = np.array(action_stats[key]["offset"]).reshape(-1)
        scale = np.array(action_stats[key]["scale"]).reshape(-1)
        end = start + dim
        out[..., start:end] = out[..., start:end] * scale + offset
        start = end
    return out


def _quick_batch_check(batch, config, action_type=None):
    obs = batch.get("obs", {})
    actions = batch.get("actions")
    obs_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)
    pred_horizon = int(config.algo.horizon.prediction_horizon)
    expected_ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
    print("Input/Output check:")
    print(f"  obs_horizon={obs_horizon} action_horizon={action_horizon} pred_horizon={pred_horizon}")
    print(f"  action_dim={expected_ac_dim} actions.shape={getattr(actions, 'shape', None)}")
    print(f"  rgb_keys={list(config.observation.modalities.obs.rgb)}")
    print(f"  low_dim_keys={list(config.observation.modalities.obs.low_dim)}")
    print(f"  obs_keys={sorted(list(obs.keys()))}")
    if action_type:
        print(f"  action_type={action_type}")


def run_inference(
    model,
    config,
    dataset,
    num_episodes=5,
    unnormalize_actions=False,
    action_stats=None,
    action_type=None,
    language_prompt=None,
):
    """
    运行推理并保存结果
    
    Args:
        model: 训练好的模型
        dataset: 验证数据集
        num_episodes: 推理的episode数量
    """
    print(f"\n开始推理 {num_episodes} 个episodes...")

    traj_l2_means = []
    traj_mae_means = []
    
    obs_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)
    action_keys = list(config.train.action_keys)
    action_shapes = list(config.train.action_shapes)

    iterator = dataset.as_numpy_iterator()
    try:
        first_batch = next(iterator)
    except StopIteration:
        print("❌ 数据集为空")
        return []
    _quick_batch_check(first_batch, config, action_type=action_type)
    batches = itertools.chain([first_batch], iterator)

    for ep_idx, batch in enumerate(batches):
        if ep_idx >= num_episodes:
            break
        # 准备输入
        obs_dict = _prepare_obs(
            batch["obs"],
            device=model.device,
            obs_horizon=obs_horizon,
            language_prompt=language_prompt,
        )
        
        # Ground truth actions
        gt_actions = batch["actions"]
        
        # 模型推理 (diffusion policy is stochastic; allow fixed seeds / multi-sample averaging)
        lang_prompts = None
        if language_prompt is not None and str(language_prompt).strip() != "":
            any_tensor = None
            for v in obs_dict.values():
                if torch.is_tensor(v):
                    any_tensor = v
                    break
            batch_size = int(any_tensor.shape[0]) if any_tensor is not None else 1
            lang_prompts = [str(language_prompt)] * batch_size
        with torch.no_grad():
            pred_actions = model._get_action_trajectory(obs_dict, lang_prompts=lang_prompts)
        
        # 转换为numpy
        pred_actions = pred_actions[0].cpu().numpy()
        pred_actions = pred_actions[:action_horizon]
        gt_actions = gt_actions[:pred_actions.shape[0]]

        gt_actions_norm = gt_actions
        pred_actions_norm = pred_actions

        gt_actions_vis = gt_actions
        pred_actions_vis = pred_actions
        if action_stats is not None:
            gt_actions_vis = _unnormalize_actions(gt_actions_vis, action_stats, action_keys, action_shapes)
            pred_actions_vis = _unnormalize_actions(pred_actions_vis, action_stats, action_keys, action_shapes)

        if unnormalize_actions:
            gt_actions = gt_actions_vis
            pred_actions = pred_actions_vis
        
        # 保存结果
        if ep_idx == 0:
            min_len = min(gt_actions.shape[0], pred_actions.shape[0])
            if min_len > 0:
                err = np.linalg.norm(gt_actions[:min_len] - pred_actions[:min_len], axis=-1)
                print(f"  first-episode mean L2 error: {err.mean():.6f}")
        min_len = min(gt_actions.shape[0], pred_actions.shape[0])
        if min_len > 0:
            diff = gt_actions[:min_len] - pred_actions[:min_len]
            traj_l2_means.append(np.linalg.norm(diff, axis=-1).mean())
            traj_mae_means.append(np.abs(diff).mean())
        
        # 可视化已移除，保持推理逻辑纯净
    
    if traj_l2_means:
        l2 = np.array(traj_l2_means)
        mae = np.array(traj_mae_means)
        print(
            f"Trajectory error (per-episode mean): "
            f"L2 mean={l2.mean():.6f} ± {l2.std():.6f}, "
            f"MAE mean={mae.mean():.6f} ± {mae.std():.6f}"
        )
    return None


def main():
    parser = argparse.ArgumentParser(description='单卡推理脚本')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='训练好的checkpoint路径 (model_epoch_*.pth)')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='训练输出目录 (包含models/子目录)')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='推理的episode数量')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备')
    parser.add_argument('--unnormalize_actions', action='store_true', default=False,
                        help='将动作从[-1,1]还原到原始单位后再评估')
    parser.add_argument('--language_prompt', type=str, default=None,
                        help='覆盖数据集语言指令（默认None表示使用数据集/无语言）')
    
    args = parser.parse_args()
    
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.run_dir)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
    
    # 加载模型
    model, config, action_stats = load_model(checkpoint_path, device=args.device)
    
    # 加载数据集
    action_type = _safe_cfg_get(config.train, "action_type", None)
    dataset = load_dataset(config, action_stats=action_stats, action_type=action_type)
    
    # 运行推理
    run_inference(
        model,
        config,
        dataset,
        args.num_episodes,
        unnormalize_actions=args.unnormalize_actions,
        action_stats=action_stats,
        action_type=action_type,
        language_prompt=args.language_prompt,
    )


if __name__ == '__main__':
    main()
