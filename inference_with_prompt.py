#!/usr/bin/env python3
"""
Inference script for CleanDIFT Diffusion Policy with optional language prompt.

Usage:
    # Without prompt (vision-only)
    python inference_with_prompt.py

    # With custom prompt
    python inference_with_prompt.py --prompt "Pick up the red object"
    python inference_with_prompt.py --prompt "Insert the object into the yellow hole"
"""

import os
import json
import argparse
import time
import torch
import numpy as np
import tensorflow as tf
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
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


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        return "cpu"
    return device


def _load_action_normalization_stats(ckpt):
    action_stats = ckpt.get("action_normalization_stats")
    if action_stats is None:
        return None
    for key in action_stats:
        for stat_key in action_stats[key]:
            action_stats[key][stat_key] = np.array(action_stats[key][stat_key])
    return action_stats


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


def _infer_use_neg_one_one_norm(config):
    use_neg_one_one_norm = False
    if hasattr(config.observation, "encoder") and hasattr(config.observation.encoder, "rgb"):
        core_class = getattr(config.observation.encoder.rgb, "core_class", "") or ""
        core_kwargs = getattr(config.observation.encoder.rgb, "core_kwargs", {}) or {}
        backbone_class = core_kwargs.get("backbone_class", "") or ""
        normalize_mode = core_kwargs.get("normalize_mode")
        if normalize_mode in ("neg_one_one", "-1_1", "minus_one_one"):
            use_neg_one_one_norm = True
        elif normalize_mode in (None, "zero_one", "imagenet"):
            use_neg_one_one_norm = False
        else:
            if "CleanDIFT" in core_class or "DIFT" in core_class:
                use_neg_one_one_norm = True
            if "CleanDIFT" in backbone_class or "DIFT" in backbone_class:
                use_neg_one_one_norm = True
    return use_neg_one_one_norm


def load_model(checkpoint_path, device="cpu", verbose=True):
    """Load trained model from checkpoint."""
    device = _resolve_device(device)
    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if verbose:
        file_size = os.path.getsize(checkpoint_path) / (1024**3)
        print(f"   ✓ Checkpoint loaded ({file_size:.2f} GB)")

    # Extract configuration
    cfg_payload = checkpoint.get("config")
    if isinstance(cfg_payload, str):
        config_dict = json.loads(cfg_payload)
    else:
        config_dict = cfg_payload
    shape_meta = checkpoint["shape_metadata"]

    # Create config object
    algo_name = checkpoint.get("algo_name") or (config_dict.get("algo_name") if isinstance(config_dict, dict) else None)
    if algo_name is None:
        raise ValueError("Checkpoint missing algo_name")
    base_config = config_factory(algo_name)
    base_config.update(config_dict)

    # Initialize observation utilities
    ObsUtils.initialize_obs_utils_with_config(base_config)

    if verbose:
        obs_keys = list(base_config.observation.modalities.obs.rgb)
        print(f"   ✓ Registered {len(obs_keys)} observation keys")

    # Create model
    model = algo_factory(
        algo_name=algo_name,
        config=base_config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # Load weights and initialize
    model.deserialize(checkpoint["model"])
    model.set_eval()
    model.reset()  # Initialize inference queues

    if verbose:
        print("   ✓ Model weights loaded and initialized")

    action_stats = _load_action_normalization_stats(checkpoint)
    return model, base_config, shape_meta, action_stats


def load_dataset(config, action_stats=None):
    """Load validation dataset following training preprocessing."""
    data_path = config.train.data_path
    dataset_names = list(config.train.dataset_names)
    if not dataset_names:
        raise ValueError("config.train.dataset_names is empty")

    obs_modalities = list(config.observation.modalities.obs.rgb)
    if len(obs_modalities) != 2:
        raise ValueError(f"Expected 2 RGB cameras, got {len(obs_modalities)}: {obs_modalities}")

    image_obs_keys = {
        "primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]],
        "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]],
    }

    state_obs_keys = []
    for obs_key in list(config.observation.modalities.obs.low_dim):
        mapped = DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP.get(obs_key)
        if mapped:
            state_obs_keys.append(mapped)

    ac_dim = _infer_action_dim_from_stats(action_stats)
    if ac_dim is None:
        ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
    is_abs_action = [True] * ac_dim

    use_neg_one_one_norm = _infer_use_neg_one_one_norm(config)
    standardize_fn = get_droid_standardize_fn(config.train.action_type)

    dataset, _ = make_dataset_from_rlds(
        name=dataset_names[0],
        data_dir=data_path,
        image_obs_keys=image_obs_keys,
        state_obs_keys=state_obs_keys,
        language_key="language_instruction",
        norm_skip_keys=["proprio"],
        action_proprio_normalization_type="bounds",
        absolute_action_mask=is_abs_action,
        action_normalization_mask=is_abs_action,
        standardize_fn=standardize_fn,
        train=False,
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

    has_proprio = len(config.observation.modalities.obs.low_dim) > 0
    view_dropout_prob = getattr(config.train, "view_dropout_prob", 0.0)
    language_prompt = None
    try:
        language_prompt = config.train.get("language_prompt", None)
    except Exception:
        language_prompt = getattr(config.train, "language_prompt", None)
    dataset = dataset.map(
        lambda x: robomimic_transform(
            x,
            normalize_to_neg_one_one=use_neg_one_one_norm,
            camera_keys=obs_modalities,
            include_proprio=has_proprio,
            view_dropout_prob=view_dropout_prob,
            language_prompt=language_prompt,
        ),
        num_parallel_calls=4,
    )

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


def _infer_action_trajectory(model, obs_dict, prompt=None):
    lang_prompts = [str(prompt)] if prompt is not None else None
    with torch.no_grad():
        action_seq = model._get_action_trajectory(obs_dict, lang_prompts=lang_prompts)
    return action_seq


def main():
    parser = argparse.ArgumentParser(
        description="CleanDIFT Diffusion Policy Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/atuin/g108ea/g108ea11/real_robot/CleanDIFTConv_pick_and_up.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Language instruction (optional, if not provided uses vision-only mode)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of episodes to evaluate (<=0 means all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("CleanDIFT Diffusion Policy - Inference")
        print("=" * 70)
        print()

    # Load model
    model, config, shape_meta, action_stats = load_model(args.checkpoint, args.device, verbose)

    # Display configuration
    if verbose:
        print("\n📋 Configuration:")
        if args.prompt:
            print("   Mode: Language-conditioned")
            print(f"   Prompt: '{args.prompt}'")
        else:
            print("   Mode: Vision-only (no language prompt)")
        action_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
        print(f"   Action dim: {action_dim}")
        print(f"   Obs horizon: {config.algo.horizon.observation_horizon}")
        print(f"   Pred horizon: {config.algo.horizon.prediction_horizon}")
        print(f"   Action horizon: {config.algo.horizon.action_horizon}")

    # Evaluate dataset (same pipeline as training)
    dataset = load_dataset(config, action_stats=action_stats)
    iterator = dataset.as_numpy_iterator()

    obs_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)

    traj_l2_means = []
    traj_mae_means = []
    evaluated = 0
    total_infer_time = 0.0
    total_pred_actions = 0
    timed_episodes = 0

    max_episodes = args.num_runs if args.num_runs > 0 else None
    for ep_idx, batch in enumerate(iterator):
        if max_episodes is not None and ep_idx >= max_episodes:
            break

        obs_dict = _prepare_obs(
            batch["obs"],
            device=model.device,
            obs_horizon=obs_horizon,
            language_prompt=args.prompt,
        )

        if verbose and ep_idx == 0:
            print("\n📷 Observation:")
            for key, val in obs_dict.items():
                if torch.is_tensor(val):
                    camera_name = key.split("/")[-1]
                    print(f"   {camera_name}: {val.shape}")

        start_t = time.perf_counter()
        pred_actions = _infer_action_trajectory(model, obs_dict, prompt=args.prompt)
        elapsed = time.perf_counter() - start_t
        pred_actions = pred_actions[0].cpu().numpy()
        pred_actions = pred_actions[:action_horizon]
        if ep_idx > 0:
            total_infer_time += elapsed
            total_pred_actions += int(pred_actions.shape[0])
            timed_episodes += 1

        gt_actions = batch["actions"]
        gt_actions = gt_actions[:pred_actions.shape[0]]

        min_len = min(gt_actions.shape[0], pred_actions.shape[0])
        if min_len > 0:
            diff = gt_actions[:min_len] - pred_actions[:min_len]
            traj_l2_means.append(np.linalg.norm(diff, axis=-1).mean())
            traj_mae_means.append(np.abs(diff).mean())
            if verbose and ep_idx == 0:
                print(f"\n  first-episode mean L2 error: {traj_l2_means[-1]:.6f}")
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("Dataset is empty")

    if verbose:
        l2 = np.array(traj_l2_means)
        mae = np.array(traj_mae_means)
        print(f"\n📊 GT error over {evaluated} episodes (normalized action space):")
        print(f"   L2 mean:  {l2.mean():.6f} ± {l2.std():.6f}")
        print(f"   MAE mean: {mae.mean():.6f} ± {mae.std():.6f}")
        if total_infer_time > 0 and timed_episodes > 0:
            eps_per_sec = timed_episodes / total_infer_time
            act_per_sec = total_pred_actions / total_infer_time
            avg_ep_time = total_infer_time / timed_episodes
            avg_action_time = total_infer_time / total_pred_actions if total_pred_actions > 0 else 0.0
            print("\n⏱️  Inference throughput:")
            print(f"   Episodes/sec: {eps_per_sec:.3f} (warmup skipped)")
            print(f"   Actions/sec:  {act_per_sec:.3f} (warmup skipped)")
            print(f"   Avg episode time: {avg_ep_time:.4f}s")
            if avg_action_time > 0:
                print(f"   Avg action time:  {avg_action_time:.6f}s")

    if verbose:
        print("\n✓ Done!")
        print("=" * 70)

    return traj_l2_means, traj_mae_means


if __name__ == "__main__":
    main()
