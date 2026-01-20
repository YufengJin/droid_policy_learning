#!/usr/bin/env python

import os
import argparse
import sys

if "--quiet" in sys.argv:
    os.environ["ROBOMIMIC_QUIET"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.action_utils as AcUtils
from robomimic.utils.rlds_utils import (
    DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP,
    DROID_TO_RLDS_OBS_KEY_MAP,
    TorchRLDSDataset,
    droid_dataset_transform,
    robomimic_transform,
)

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec


def _decode_prompts(algo, raw_entries):
    if raw_entries is None:
        return None
    if hasattr(algo, "_decode_language_prompts"):
        try:
            return algo._decode_language_prompts(raw_entries)
        except Exception:
            return None
    return None


def _use_neg_one_one_norm(config):
    try:
        backbone_class = config.observation.encoder.rgb.core_kwargs.get("backbone_class", "")
    except Exception:
        return False
    return "CleanDIFT" in backbone_class or "DIFT" in backbone_class


def _unnormalize_action_vector(actions, action_normalization_stats, action_keys):
    if action_normalization_stats is None:
        return actions
    orig_shape = actions.shape
    if actions.ndim > 2:
        actions = actions.reshape((-1, orig_shape[-1]))
    action_shapes = {
        key: action_normalization_stats[key]["offset"].shape[1:]
        for key in action_keys
    }
    action_dict = AcUtils.vector_to_action_dict(
        actions, action_shapes=action_shapes, action_keys=action_keys
    )
    ObsUtils.unnormalize_dict(action_dict, action_normalization_stats)
    actions = AcUtils.action_dict_to_vector(action_dict, action_keys=action_keys)
    if len(orig_shape) > 2:
        actions = actions.reshape(orig_shape)
    return actions


def _build_rlds_dataset(
    config,
    data_dir,
    dataset_names,
    sample_weights,
    train,
    filter_success,
    transform_kwargs=None,
):
    obs_modalities = config.observation.modalities.obs.rgb
    if len(obs_modalities) == 2:
        obs_modalities = obs_modalities[::-1]  # swap to match possible training order
    if len(obs_modalities) != 2:
        raise ValueError("This script expects exactly 2 RGB cameras in the config.")
    for cam_key in obs_modalities:
        if cam_key not in DROID_TO_RLDS_OBS_KEY_MAP:
            available_keys = list(DROID_TO_RLDS_OBS_KEY_MAP.keys())
            raise KeyError(
                "Camera key '{}' not found in DROID_TO_RLDS_OBS_KEY_MAP.\n"
                "Available keys: {}\n"
                "Your config has: {}\n"
                "Hint: Use --cameras to specify correct camera names, e.g.:\n"
                "  For DROID: --cameras hand_camera_left varied_camera_1_left\n"
                "  For real robot: --cameras wrist_left varied_camera_1_left".format(
                    cam_key, available_keys, obs_modalities
                )
            )

    image_obs_keys = {
        "primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]],
        "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]],
    }
    state_obs_keys = [
        DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[key]
        for key in config.observation.modalities.obs.low_dim
    ]

    ac_dim = sum([shape[1] for shape in config.train.action_shapes])
    is_abs_action = [True] * ac_dim

    base_kwargs = {
        "data_dir": data_dir,
        "image_obs_keys": image_obs_keys,
        "state_obs_keys": state_obs_keys,
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": is_abs_action,
        "action_normalization_mask": is_abs_action,
        "standardize_fn": droid_dataset_transform,
    }

    filter_functions = []
    for name in dataset_names:
        if filter_success and name == "droid":
            filter_functions.append([ModuleSpec.create("robomimic.utils.rlds_utils:filter_success")])
        else:
            filter_functions.append([])

    dataset_kwargs_list = [
        {"name": name, "filter_functions": filters, **base_kwargs}
        for name, filters in zip(dataset_names, filter_functions)
    ]

    combined_stats = combine_dataset_statistics(
        [make_dataset_from_rlds(**kwargs, train=train)[1] for kwargs in dataset_kwargs_list]
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=config.train.shuffle_buffer_size,
        batch_size=None,
        balance_weights=False,
        dataset_statistics=combined_stats,
        traj_transform_kwargs=dict(
            window_size=config.algo.horizon.observation_horizon,
            future_action_window_size=config.algo.horizon.prediction_horizon - 1,
            subsample_length=config.train.subsample_length,
            skip_unlabeled=True,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs=dict(),
            resize_size=dict(
                primary=config.observation.image_dim,
                secondary=config.observation.image_dim,
            ),
            num_parallel_calls=config.train.num_parallel_calls,
        ),
        traj_transform_threads=config.train.traj_transform_threads,
        traj_read_threads=config.train.traj_read_threads,
    )

    if transform_kwargs is None:
        transform_kwargs = {}
    dataset = dataset.map(
        lambda traj: robomimic_transform(traj, **transform_kwargs),
        num_parallel_calls=config.train.traj_transform_threads,
    )
    return TorchRLDSDataset(
        dataset,
        train=train,
        shuffle_buffer_size=config.train.shuffle_buffer_size,
    )


def _plot_trajectories(gt, pred, out_path, xy_dims, title):
    num_samples = gt.shape[0]
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        gt_xy = gt[idx][:, xy_dims]
        pred_xy = pred[idx][:, xy_dims]
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-o", label="gt")
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-o", label="pred")
        ax.set_xlabel(f"action[{xy_dims[0]}]")
        ax.set_ylabel(f"action[{xy_dims[1]}]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Offline RLDS inference and trajectory plot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_names", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="trajectory_compare.png")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--xy_dims", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--filter_success", action="store_true", default=False)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--split", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--quiet", action="store_true", default=False)
    args = parser.parse_args()

    if args.quiet:
        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
    tf.config.set_visible_devices([], "GPU")

    device = TorchUtils.get_torch_device(try_to_use_cuda=not args.no_cuda)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint,
        device=device,
        verbose=False,
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    sample_weights = list(config.train.sample_weights)
    if len(sample_weights) != len(args.dataset_names):
        sample_weights = [1.0] * len(args.dataset_names)

    use_neg_one_one_norm = _use_neg_one_one_norm(config)
    has_proprio = len(config.observation.modalities.obs.low_dim) > 0
    transform_kwargs = dict(
        normalize_to_neg_one_one=use_neg_one_one_norm,
        include_proprio=has_proprio,
        view_dropout_prob=0.0,
        camera_keys=config.observation.modalities.obs.rgb,
    )

    dataset = _build_rlds_dataset(
        config,
        data_dir=args.data_dir,
        dataset_names=args.dataset_names,
        sample_weights=sample_weights,
        train=args.split == "train",
        filter_success=args.filter_success,
        transform_kwargs=transform_kwargs,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    algo = policy.policy
    algo.set_eval()
    action_norm_stats = ckpt_dict.get("action_normalization_stats", None)
    print(f"Action norm stats: {action_norm_stats is not None}")
    if action_norm_stats:
        print(f"Stats keys: {list(action_norm_stats.keys())}")
    action_keys = config.train.action_keys
    To = config.algo.horizon.observation_horizon
    Ta = config.algo.horizon.action_horizon

    all_gt = []
    all_pred = []
    torch.set_grad_enabled(False)
    for idx, batch in enumerate(loader):
        if idx >= args.num_batches:
            break
        batch = TensorUtils.to_device(TensorUtils.to_float(batch), device)
        obs = batch["obs"]
        raw_lang = obs.get("raw_language")
        lang_prompts = _decode_prompts(algo, raw_lang)
        pred = algo._get_action_trajectory(obs_dict=obs, lang_prompts=lang_prompts)
        gt = batch["actions"]
        start = To - 1
        end = start + Ta
        gt = gt[:, start:end]

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        if action_norm_stats is not None:
            # pred is already unnormalized by algo._get_action_trajectory
            gt_np = _unnormalize_action_vector(gt_np, action_norm_stats, action_keys)

        print(f"Batch {idx}: Pred shape {pred_np.shape}, GT shape {gt_np.shape}")
        print(f"Pred sample: {pred_np[0, 0]}")
        print(f"GT sample: {gt_np[0, 0]}")
        all_pred.append(pred_np)
        all_gt.append(gt_np)

    if not all_gt:
        raise RuntimeError("No data collected from dataset; check dataset path or filters.")

    gt_concat = np.concatenate(all_gt, axis=0)
    pred_concat = np.concatenate(all_pred, axis=0)
    title = os.path.basename(args.checkpoint)
    _plot_trajectories(gt_concat, pred_concat, args.output, args.xy_dims, title)
    print(f"Saved trajectory plot to {args.output}")


if __name__ == "__main__":
    main()
