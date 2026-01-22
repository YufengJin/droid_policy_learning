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
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import action_utils as ActionUtils
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
from octo.utils.spec import ModuleSpec

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
    dataset = dataset.map(
        lambda x: robomimic_transform(
            x, 
            normalize_to_neg_one_one=use_neg_one_one_norm,
            camera_keys=obs_modalities,
            include_proprio=True
        ),
        num_parallel_calls=4
    )
    
    print(f"✅ 数据集加载成功")
    return dataset


def _prepare_obs(batch_obs, device, obs_horizon):
    obs_dict = {}
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


def _save_obs_images(batch_obs, camera_keys, output_dir, prefix, max_images=2):
    if not isinstance(batch_obs, dict):
        return
    os.makedirs(output_dir, exist_ok=True)
    for cam_key in camera_keys:
        if cam_key not in batch_obs:
            continue
        frames = batch_obs[cam_key]
        if frames is None or len(frames) == 0:
            continue
        num = min(max_images, frames.shape[0])
        for idx in range(num):
            frame = frames[idx]
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.min() < 0:
                frame = (frame + 1.0) / 2.0
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
            name = "{}_{}_t{}.png".format(prefix, cam_key.replace("/", "_"), idx)
            imageio.imwrite(os.path.join(output_dir, name), frame)


def _actions_to_path(actions_xy, action_type):
    if action_type and "abs" in action_type:
        path = actions_xy - actions_xy[0]
    else:
        path = np.cumsum(actions_xy, axis=0)
    return path


def _safe_cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg[key] if key in cfg else default
    except Exception:
        return default


def _extract_gt_positions(batch_obs, horizon):
    if not isinstance(batch_obs, dict):
        return None
    key = "robot_state/cartesian_position"
    if key not in batch_obs:
        return None
    pos = batch_obs[key]
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    pos = np.array(pos)
    pos = pos[:horizon]
    if pos.ndim >= 2 and pos.shape[-1] >= 3:
        return pos[..., :3]
    return None


def _compute_pred_positions(pred_actions, gt_pos, action_type):
    if pred_actions is None or pred_actions.shape[-1] < 3:
        return None
    deltas = pred_actions[:, :3]
    if action_type and "abs" in action_type:
        return deltas
    if gt_pos is None or gt_pos.shape[0] == 0:
        return np.cumsum(deltas, axis=0)
    start = gt_pos[0]
    return start + np.cumsum(deltas, axis=0)


def _save_overlay_images(batch_obs, camera_keys, output_dir, prefix, gt_pos, pred_pos):
    if not isinstance(batch_obs, dict):
        return
    os.makedirs(output_dir, exist_ok=True)
    for cam_key in camera_keys:
        if cam_key not in batch_obs:
            continue
        frames = batch_obs[cam_key]
        if frames is None or len(frames) == 0:
            continue
        frame = frames[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.min() < 0:
            frame = (frame + 1.0) / 2.0
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(frame)
        ax.axis("off")

        if gt_pos is None or pred_pos is None:
            continue
        gt_xy = gt_pos[:, :2]
        pred_xy = pred_pos[:, :2]
        gt_path = _actions_to_path(gt_xy, "abs")
        pred_path = _actions_to_path(pred_xy, "abs")

        # scale trajectories to image size for visualization
        h, w = frame.shape[:2]
        all_xy = np.concatenate([gt_path, pred_path], axis=0)
        extent = np.max(np.abs(all_xy)) if all_xy.size else 0.0
        if extent > 0:
            scale = 0.4 * min(h, w) / extent
            gt_draw = gt_path * scale
            pred_draw = pred_path * scale
            gt_x = w / 2.0 + gt_draw[:, 0]
            gt_y = h / 2.0 - gt_draw[:, 1]
            pred_x = w / 2.0 + pred_draw[:, 0]
            pred_y = h / 2.0 - pred_draw[:, 1]
            ax.plot(gt_x, gt_y, color="blue", linewidth=3.5, alpha=0.95, label="GT")
            ax.plot(pred_x, pred_y, color="red", linewidth=3.5, alpha=0.95, label="Pred")
            ax.legend(loc="upper right", fontsize=6, framealpha=0.6)

        name = "{}_{}.png".format(prefix, cam_key.replace("/", "_"))
        fig.savefig(os.path.join(output_dir, name), bbox_inches="tight")
        plt.close(fig)


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


def _log_batch_consistency(batch, config):
    print("\n" + "=" * 80)
    print("一致性检查")
    print("=" * 80)
    obs_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)
    pred_horizon = int(config.algo.horizon.prediction_horizon)
    expected_ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
    print(f"配置: obs_horizon={obs_horizon}, action_horizon={action_horizon}, prediction_horizon={pred_horizon}")
    print(f"配置: action_dim={expected_ac_dim}, action_shapes={list(config.train.action_shapes)}")

    obs = batch.get("obs", {})
    actions = batch.get("actions")
    if actions is not None:
        print(f"数据: actions.shape={getattr(actions, 'shape', None)}")
        if hasattr(actions, "shape") and actions.shape[-1] != expected_ac_dim:
            print(f"⚠️ action_dim不匹配: expected={expected_ac_dim}, actual={actions.shape[-1]}")
        if hasattr(actions, "shape") and actions.shape[0] < action_horizon:
            print(f"⚠️ action_horizon过长: horizon={action_horizon}, available={actions.shape[0]}")

    rgb_keys = list(config.observation.modalities.obs.rgb)
    low_dim_keys = list(config.observation.modalities.obs.low_dim)
    print(f"配置: rgb_keys={rgb_keys}")
    print(f"配置: low_dim_keys={low_dim_keys}")
    print(f"数据: obs_keys={sorted(list(obs.keys()))}")
    for key in rgb_keys:
        if key not in obs:
            print(f"⚠️ 缺少RGB输入: {key}")
        else:
            print(f"数据: {key}.shape={getattr(obs[key], 'shape', None)}")
    for key in low_dim_keys:
        if key not in obs:
            print(f"⚠️ 缺少低维输入: {key}")
        else:
            print(f"数据: {key}.shape={getattr(obs[key], 'shape', None)}")
    print("=" * 80 + "\n")


def plot_joint_trajectory(gt_actions, pred_actions, ep_idx, save_path):
    num_dims = min(gt_actions.shape[-1], pred_actions.shape[-1])
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]
    for i in range(num_dims):
        ax = axes[i]
        ax.plot(gt_actions[:, i], color="blue", linewidth=3.0, label="GT")
        ax.plot(pred_actions[:, i], color="red", linewidth=3.0, linestyle="--", label="Pred")
        ax.set_ylabel(f"J{i}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f"Episode {ep_idx}: Joint Trajectory (GT vs Pred)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_inference(model, config, dataset, output_dir, num_episodes=5, unnormalize_actions=False, action_stats=None, action_type=None):
    """
    运行推理并保存结果
    
    Args:
        model: 训练好的模型
        dataset: 验证数据集
        output_dir: 输出目录
        num_episodes: 推理的episode数量
    """
    print(f"\n开始推理 {num_episodes} 个episodes...")
    
    os.makedirs(output_dir, exist_ok=True)
    preview_dir = output_dir
    
    results = []
    
    obs_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)
    action_keys = list(config.train.action_keys)
    action_shapes = list(config.train.action_shapes)

    camera_keys = list(config.observation.modalities.obs.rgb)
    iterator = dataset.as_numpy_iterator()
    try:
        first_batch = next(iterator)
    except StopIteration:
        print("❌ 数据集为空")
        return []
    _log_batch_consistency(first_batch, config)
    batches = itertools.chain([first_batch], iterator)

    for ep_idx, batch in enumerate(tqdm(batches, total=num_episodes)):
        if ep_idx >= num_episodes:
            break
        if ep_idx < 3:
            _save_obs_images(batch.get("obs", {}), camera_keys, preview_dir, f"episode_{ep_idx}")
        
        # 准备输入
        obs_dict = _prepare_obs(batch["obs"], device=model.device, obs_horizon=obs_horizon)
        
        # Ground truth actions
        gt_actions = batch["actions"]
        
        # 模型推理
        with torch.no_grad():
            pred_actions = model._get_action_trajectory(obs_dict)
        
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
        result = {
            'episode_idx': ep_idx,
            'gt_actions_norm': gt_actions_norm,
            'pred_actions_norm': pred_actions_norm,
            'language': batch['obs']['raw_language'][0] if 'raw_language' in batch['obs'] else None,
        }
        if unnormalize_actions:
            result['gt_actions'] = gt_actions
            result['pred_actions'] = pred_actions
        results.append(result)
        
        # 可视化对比
        if ep_idx < 3:  # 只可视化前3个episodes
            plot_error_curve(
                gt_actions,
                pred_actions,
                ep_idx,
                save_path=os.path.join(preview_dir, f'episode_{ep_idx}_error.png')
            )
            if action_type and action_type.startswith("joint"):
                plot_joint_trajectory(
                    gt_actions_vis,
                    pred_actions_vis,
                    ep_idx,
                    save_path=os.path.join(preview_dir, f'episode_{ep_idx}_joint.png')
                )
            else:
                plot_trajectory_comparison(
                    gt_actions, 
                    pred_actions, 
                    ep_idx,
                    save_path=os.path.join(preview_dir, f'episode_{ep_idx}_comparison.png')
                )
                gt_pos = _extract_gt_positions(batch.get("obs", {}), gt_actions_vis.shape[0])
                pred_pos = _compute_pred_positions(pred_actions_vis, gt_pos, action_type)
                _save_overlay_images(
                    batch.get("obs", {}),
                    camera_keys,
                    preview_dir,
                    f"episode_{ep_idx}_overlay",
                    gt_pos,
                    pred_pos,
                )
    
    # 保存所有结果
    save_path = os.path.join(output_dir, 'inference_results.npz')
    np.savez(save_path, results=results)
    print(f"\n✅ 推理完成，结果保存至: {save_path}")
    
    # 计算统计指标
    compute_metrics(results, output_dir, use_normalized=True, suffix="_normalized")
    if unnormalize_actions:
        compute_metrics(results, output_dir, use_normalized=False, suffix="_unnormalized")
    
    return results


def plot_trajectory_comparison(gt_actions, pred_actions, ep_idx, save_path):
    """可视化ground truth和预测的轨迹对比"""
    num_dims = min(gt_actions.shape[-1], 6)  # 最多显示6个维度
    
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2*num_dims))
    if num_dims == 1:
        axes = [axes]
    
    for i in range(num_dims):
        ax = axes[i]
        
        # Ground truth
        ax.plot(gt_actions[:, i], 'b-', label='Ground Truth', linewidth=2)
        
        # Prediction
        if pred_actions.ndim == 2:
            ax.plot(pred_actions[:, i], 'r--', label='Prediction', linewidth=2)
        else:
            # 如果pred_actions是3D的，取第一个
            ax.plot(pred_actions[0, :, i], 'r--', label='Prediction', linewidth=2)
        
        ax.set_ylabel(f'Dim {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    fig.suptitle(f'Episode {ep_idx}: Action Trajectory Comparison')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存可视化: {save_path}")


def plot_error_curve(gt_actions, pred_actions, ep_idx, save_path):
    """保存每步动作误差曲线"""
    min_len = min(gt_actions.shape[0], pred_actions.shape[0])
    if min_len == 0:
        return
    gt = gt_actions[:min_len]
    pred = pred_actions[:min_len]
    err = np.linalg.norm(gt - pred, axis=-1)
    plt.figure(figsize=(10, 3))
    plt.plot(err, 'm-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('L2 Error')
    plt.title(f'Episode {ep_idx}: Action Error (per-step)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存误差曲线: {save_path}")


def compute_metrics(results, output_dir, use_normalized=True, suffix=""):
    """计算推理指标"""
    print("\n" + "=" * 80)
    space_name = "normalized" if use_normalized else "unnormalized"
    print(f"推理指标统计 ({space_name})")
    print("=" * 80)
    
    mse_list = []
    mae_list = []
    
    for result in results:
        if use_normalized:
            gt = result.get('gt_actions_norm')
            pred = result.get('pred_actions_norm')
        else:
            gt = result.get('gt_actions')
            pred = result.get('pred_actions')
        if gt is None or pred is None:
            continue
        
        # 确保维度匹配
        if pred.ndim == 3:
            pred = pred[0]  # 取第一个batch
        
        min_len = min(gt.shape[0], pred.shape[0])
        gt = gt[:min_len]
        pred = pred[:min_len]
        
        # MSE
        mse = np.mean((gt - pred) ** 2)
        mse_list.append(mse)
        
        # MAE
        mae = np.mean(np.abs(gt - pred))
        mae_list.append(mae)
    
    print(f"\n整体指标:")
    print(f"  平均 MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    print(f"  平均 MAE: {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
    
    # 保存指标
    metrics = {
        'mse_per_episode': [float(x) for x in mse_list],
        'mae_per_episode': [float(x) for x in mae_list],
        'mean_mse': float(np.mean(mse_list)),
        'mean_mae': float(np.mean(mae_list)),
        'std_mse': float(np.std(mse_list)),
        'std_mae': float(np.std(mae_list)),
    }
    
    metrics_path = os.path.join(output_dir, f'metrics{suffix}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ 指标保存至: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='单卡推理脚本')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='训练好的checkpoint路径 (model_epoch_*.pth)')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='训练输出目录 (包含models/子目录)')
    parser.add_argument('--output_dir', type=str, default='previews',
                        help='输出目录（默认previews）')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='推理的episode数量')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备')
    parser.add_argument('--unnormalize_actions', action='store_true', default=False,
                        help='将动作从[-1,1]还原到原始单位后再评估')
    parser.add_argument('--action_type', type=str, default=None,
                        help='动作类型 (e.g. cartesian_abs or cartesian_rel); 用于轨迹可视化')
    
    args = parser.parse_args()
    
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.run_dir)
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: Checkpoint不存在: {checkpoint_path}")
        return
    
    print("=" * 80)
    print("Real Robot 模型推理")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    output_dir = args.output_dir
    if output_dir in (".", "./", ""):
        output_dir = "previews"
    print(f"输出目录: {output_dir}")
    print(f"设备: {args.device}")
    print(f"Episode数: {args.num_episodes}")
    print("=" * 80)
    
    # 加载模型
    model, config, action_stats = load_model(checkpoint_path, device=args.device)
    
    # 加载数据集
    cfg_action_type = _safe_cfg_get(config.train, "action_type", None)
    action_type = args.action_type or cfg_action_type
    dataset = load_dataset(config, action_stats=action_stats, action_type=action_type)
    
    # 运行推理
    results = run_inference(
        model,
        config,
        dataset,
        output_dir,
        args.num_episodes,
        unnormalize_actions=args.unnormalize_actions,
        action_stats=action_stats,
        action_type=action_type,
    )
    
    print("\n" + "=" * 80)
    print("✅ 推理完成!")
    print("=" * 80)
    print(f"\n查看结果:")
    print(f"  可视化: {output_dir}/episode_*_comparison.png")
    print(f"  数据: {output_dir}/inference_results.npz")
    print(f"  指标: {output_dir}/metrics_normalized.json")


if __name__ == '__main__':
    main()
