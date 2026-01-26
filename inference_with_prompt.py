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
import robomimic.utils.action_utils as ActionUtils
from collections import defaultdict
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
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


def _enable_ddim_inference(model, config, num_inference_timesteps=10):
    """Switch to DDIM sampler while keeping training noise schedule."""
    ddpm_cfg = config.algo.ddpm
    ddim_cfg = config.algo.ddim
    with config.unlocked():
        config.algo.ddpm.enabled = False
        config.algo.ddim.enabled = True
        # Keep training noise schedule consistent with DDPM config.
        config.algo.ddim.num_train_timesteps = ddpm_cfg.num_train_timesteps
        config.algo.ddim.beta_schedule = ddpm_cfg.beta_schedule
        config.algo.ddim.clip_sample = ddpm_cfg.clip_sample
        config.algo.ddim.prediction_type = ddpm_cfg.prediction_type
        config.algo.ddim.num_inference_timesteps = int(num_inference_timesteps)

    model.noise_scheduler = DDIMScheduler(
        num_train_timesteps=ddim_cfg.num_train_timesteps,
        beta_schedule=ddim_cfg.beta_schedule,
        clip_sample=ddim_cfg.clip_sample,
        set_alpha_to_one=ddim_cfg.set_alpha_to_one,
        steps_offset=ddim_cfg.steps_offset,
        prediction_type=ddim_cfg.prediction_type,
    )


def _enable_ddpm_inference(model, config, num_inference_timesteps=None):
    """Switch to DDPM sampler (match training by default)."""
    ddpm_cfg = config.algo.ddpm
    with config.unlocked():
        config.algo.ddim.enabled = False
        config.algo.ddpm.enabled = True
        if num_inference_timesteps is not None:
            config.algo.ddpm.num_inference_timesteps = int(num_inference_timesteps)
    model.noise_scheduler = DDPMScheduler(
        num_train_timesteps=ddpm_cfg.num_train_timesteps,
        beta_schedule=ddpm_cfg.beta_schedule,
        clip_sample=ddpm_cfg.clip_sample,
        prediction_type=ddpm_cfg.prediction_type,
    )


def _checkpoint_has_teacher(model_state) -> bool:
    nets_state = model_state.get("nets", {}) if isinstance(model_state, dict) else {}
    return any("unet_feature_extractor_base" in k for k in nets_state.keys())


def _maybe_strip_teacher_to_match_checkpoint(model, checkpoint, verbose=False) -> bool:
    model_state = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
    if _checkpoint_has_teacher(model_state):
        return False
    nets = getattr(model, "nets", None)
    if nets is None or not hasattr(nets, "modules"):
        return False
    stripped = False
    for module in nets.modules():
        if hasattr(module, "strip_teacher"):
            try:
                did = module.strip_teacher(strip_text_encoder=False, verbose=verbose)
            except TypeError:
                did = module.strip_teacher()
            stripped = stripped or bool(did)
    if stripped and verbose:
        print("   ✓ Stripped CleanDIFT teacher to match checkpoint")
    return stripped


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
    if isinstance(config_dict, dict) and "language_prompt" in config_dict and "language_prompt" not in base_config:
        config_dict = dict(config_dict)
        config_dict.pop("language_prompt", None)
    with base_config.unlocked():
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

    # If checkpoint was saved with teacher stripped, strip before loading to avoid key mismatch.
    _maybe_strip_teacher_to_match_checkpoint(model, checkpoint, verbose=verbose)

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


def _prepare_obs(batch_obs, device, obs_horizon, language_prompt=None, skip_process_obs=False):
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

    if not skip_process_obs:
        obs_dict = ObsUtils.process_obs_dict(obs_dict)
        for key, value in obs_dict.items():
            if torch.is_tensor(value):
                obs_dict[key] = value.to(device)
    return obs_dict


def _get_action_sequence(model, obs_dict, action_horizon):
    """Use official get_action API and its action_queue to build a full sequence."""
    model.reset()
    with torch.no_grad():
        first_action = model.get_action(obs_dict, eval_mode=False)  # [1, Da]
    actions = [first_action.squeeze(0)]
    while len(actions) < action_horizon and model.action_queue is not None and len(model.action_queue) > 0:
        actions.append(model.action_queue.popleft())
    if len(actions) == 0:
        return torch.empty((0, model.ac_dim), device=model.device)
    return torch.stack(actions, dim=0)


def _unnormalize_actions(actions, config, action_norm_stats):
    if action_norm_stats is None:
        return None


def _shift_mae(pred, gt, shift=0):
    if pred.shape[0] <= abs(shift):
        return float("nan")
    if shift == 0:
        return np.abs(pred - gt).mean()
    if shift > 0:
        return np.abs(pred[shift:] - gt[:-shift]).mean()
    s = -shift
    return np.abs(pred[:-s] - gt[s:]).mean()


def _per_key_mae(pred_vec, gt_vec, action_keys, action_shapes):
    """Return per-key, per-dim MAE arrays."""
    pred_dict = ActionUtils.vector_to_action_dict(pred_vec, action_shapes=action_shapes, action_keys=action_keys)
    gt_dict = ActionUtils.vector_to_action_dict(gt_vec, action_shapes=action_shapes, action_keys=action_keys)
    out = {}
    for key in action_keys:
        p = np.asarray(pred_dict[key])
        g = np.asarray(gt_dict[key])
        # mean over time dimension(s), keep last dim
        err = np.abs(p - g)
        while err.ndim > 1:
            err = err.mean(axis=0)
        out[key] = err
    return out
    try:
        action_keys = list(config.train.action_keys)
        action_shapes = {
            k: tuple(action_norm_stats[k]["offset"].shape[1:]) for k in action_keys
        }
        ac_dict = ActionUtils.vector_to_action_dict(actions, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=action_norm_stats)
        return ActionUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
    except Exception:
        return None


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
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="Inference sampler to use (ddim or ddpm).",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps (faster than DDPM).",
    )
    parser.add_argument(
        "--ddpm-steps",
        type=int,
        default=None,
        help="Number of DDPM inference steps (default: config).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cpu or cuda)")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Report inference time per episode (get_action path).",
    )
    parser.add_argument(
        "--report-unnormalized",
        action="store_true",
        help="Also report error in unnormalized action units if stats are available.",
    )
    parser.add_argument(
        "--debug-eval",
        action="store_true",
        help="Print debug stats (action ranges, zero baseline, shift MAE).",
    )
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
    if args.sampler == "ddim":
        _enable_ddim_inference(model, config, num_inference_timesteps=args.ddim_steps)
    else:
        _enable_ddpm_inference(model, config, num_inference_timesteps=args.ddpm_steps)

    # Display configuration
    if verbose:
        print("\n📋 Configuration:")
        if args.prompt:
            print("   Mode: Language-conditioned")
            print(f"   Prompt: '{args.prompt}'")
        else:
            print("   Mode: Vision-only (no language prompt)")
        action_type = getattr(config.train, "action_type", None)
        if action_type is None:
            try:
                action_type = config.train.get("action_type", None)
            except Exception:
                action_type = None
        if action_type:
            print(f"   Action type: {action_type}")
        if args.sampler == "ddim":
            print(f"   Sampler: DDIM ({args.ddim_steps} steps)")
        else:
            steps = args.ddpm_steps if args.ddpm_steps is not None else config.algo.ddpm.num_inference_timesteps
            print(f"   Sampler: DDPM ({steps} steps)")
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
    use_neg_one_one_norm = _infer_use_neg_one_one_norm(config)

    traj_l2_means = []
    traj_mae_means = []
    traj_l2_unnorm = []
    traj_mae_unnorm = []
    dbg_shift_mae = { -1: [], 0: [], 1: [] }
    dbg_zero_mae = []
    dbg_shift_mae_un = { -1: [], 0: [], 1: [] }
    dbg_zero_mae_un = []
    dbg_key_mae = defaultdict(list)
    dbg_key_mae_un = defaultdict(list)
    infer_times = []
    evaluated = 0

    max_episodes = args.num_runs if args.num_runs > 0 else None
    for ep_idx, batch in enumerate(iterator):
        if max_episodes is not None and ep_idx >= max_episodes:
            break

        obs_dict = _prepare_obs(
            batch["obs"],
            device=model.device,
            obs_horizon=obs_horizon,
            language_prompt=args.prompt,
            skip_process_obs=use_neg_one_one_norm,
        )

        if verbose and ep_idx == 0:
            print("\n📷 Observation:")
            for key, val in obs_dict.items():
                if torch.is_tensor(val):
                    camera_name = key.split("/")[-1]
                    print(f"   {camera_name}: {val.shape}")

        if args.profile:
            start_t = time.perf_counter()
        pred_actions = _get_action_sequence(model, obs_dict, action_horizon=action_horizon)
        if args.profile:
            infer_times.append(time.perf_counter() - start_t)
        pred_actions = pred_actions.cpu().numpy()

        gt_actions = batch["actions"]
        gt_actions = gt_actions[:pred_actions.shape[0]]

        min_len = min(gt_actions.shape[0], pred_actions.shape[0])
        if min_len > 0:
            diff = gt_actions[:min_len] - pred_actions[:min_len]
            traj_l2_means.append(np.linalg.norm(diff, axis=-1).mean())
            traj_mae_means.append(np.abs(diff).mean())
            if args.report_unnormalized:
                pred_un = _unnormalize_actions(pred_actions[:min_len], config, action_stats)
                gt_un = _unnormalize_actions(gt_actions[:min_len], config, action_stats)
                if pred_un is not None and gt_un is not None:
                    diff_un = gt_un - pred_un
                    traj_l2_unnorm.append(np.linalg.norm(diff_un, axis=-1).mean())
                    traj_mae_unnorm.append(np.abs(diff_un).mean())
            if verbose and ep_idx == 0:
                print(f"\n  first-episode mean L2 error: {traj_l2_means[-1]:.6f}")
            if args.debug_eval:
                pred_slice = pred_actions[:min_len]
                gt_slice = gt_actions[:min_len]
                action_keys = list(config.train.action_keys)
                action_shapes = {
                    k: tuple(action_stats[k]["offset"].shape[1:]) if (action_stats and k in action_stats) else tuple(config.train.action_shapes[action_keys.index(k)])
                    for k in action_keys
                }
                per_key = _per_key_mae(pred_slice, gt_slice, action_keys, action_shapes)
                for k, v in per_key.items():
                    dbg_key_mae[k].append(v)
                for s in (-1, 0, 1):
                    dbg_shift_mae[s].append(_shift_mae(pred_slice, gt_slice, shift=s))
                dbg_zero_mae.append(np.abs(gt_slice).mean())
                if args.report_unnormalized:
                    pred_un = _unnormalize_actions(pred_slice, config, action_stats)
                    gt_un = _unnormalize_actions(gt_slice, config, action_stats)
                    if pred_un is not None and gt_un is not None:
                        per_key_un = _per_key_mae(pred_un, gt_un, action_keys, action_shapes)
                        for k, v in per_key_un.items():
                            dbg_key_mae_un[k].append(v)
                        for s in (-1, 0, 1):
                            dbg_shift_mae_un[s].append(_shift_mae(pred_un, gt_un, shift=s))
                        dbg_zero_mae_un.append(np.abs(gt_un).mean())
                if verbose and ep_idx == 0:
                    print("\n🔎 Debug (normalized action space):")
                    print(f"   pred min/max: {pred_slice.min():.3f} / {pred_slice.max():.3f}")
                    print(f"   gt   min/max: {gt_slice.min():.3f} / {gt_slice.max():.3f}")
                    print(f"   zero-action MAE: {dbg_zero_mae[-1]:.6f}")
                    print(f"   shift MAE (pred vs gt+1): {dbg_shift_mae[-1][-1]:.6f}")
                    print(f"   shift MAE (aligned):      {dbg_shift_mae[0][-1]:.6f}")
                    print(f"   shift MAE (pred vs gt-1): {dbg_shift_mae[1][-1]:.6f}")
                    if args.report_unnormalized and len(dbg_zero_mae_un) > 0:
                        print("\n🔎 Debug (unnormalized action units):")
                        print(f"   pred min/max: {pred_un.min():.3f} / {pred_un.max():.3f}")
                        print(f"   gt   min/max: {gt_un.min():.3f} / {gt_un.max():.3f}")
                        print(f"   zero-action MAE: {dbg_zero_mae_un[-1]:.6f}")
                        print(f"   shift MAE (pred vs gt+1): {dbg_shift_mae_un[-1][-1]:.6f}")
                        print(f"   shift MAE (aligned):      {dbg_shift_mae_un[0][-1]:.6f}")
                        print(f"   shift MAE (pred vs gt-1): {dbg_shift_mae_un[1][-1]:.6f}")
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("Dataset is empty")

    if verbose:
        l2 = np.array(traj_l2_means)
        mae = np.array(traj_mae_means)
        print(f"\n📊 GT error over {evaluated} episodes (normalized action space):")
        print(f"   L2 mean:  {l2.mean():.6f} ± {l2.std():.6f}")
        print(f"   MAE mean: {mae.mean():.6f} ± {mae.std():.6f}")
        if args.report_unnormalized and len(traj_l2_unnorm) > 0:
            l2u = np.array(traj_l2_unnorm)
            maeu = np.array(traj_mae_unnorm)
            print(f"\n📊 GT error over {evaluated} episodes (unnormalized action units):")
            print(f"   L2 mean:  {l2u.mean():.6f} ± {l2u.std():.6f}")
            print(f"   MAE mean: {maeu.mean():.6f} ± {maeu.std():.6f}")
        if args.profile and len(infer_times) > 0:
            times = np.array(infer_times)
            print(f"\n⏱️  Inference time (get_action path):")
            for i, t in enumerate(times):
                print(f"   Episode {i+1}: {t:.4f}s")
            if len(times) > 1:
                print(f"   Avg: {times.mean():.4f}s ± {times.std():.4f}s")

            # Report ms/action and FPS
            ms_per_action = (times / action_horizon) * 1000.0
            fps = action_horizon / np.maximum(times, 1e-8)
            print(f"\n⏱️  Speed:")
            for i, (ms, f) in enumerate(zip(ms_per_action, fps)):
                print(f"   Episode {i+1}: {ms:.2f} ms/action | {f:.3f} actions/sec")
            if len(times) > 1:
                print(f"   Avg: {ms_per_action.mean():.2f} ms/action ± {ms_per_action.std():.2f}")
                print(f"   Avg: {fps.mean():.3f} actions/sec ± {fps.std():.3f}")

        if args.debug_eval:
            print("\n🔎 Debug summary (normalized action space):")
            for s in (-1, 0, 1):
                vals = np.array(dbg_shift_mae[s], dtype=float)
                if vals.size > 0:
                    print(f"   shift {s:+d} MAE: {np.nanmean(vals):.6f} ± {np.nanstd(vals):.6f}")
            if len(dbg_zero_mae) > 0:
                z = np.array(dbg_zero_mae, dtype=float)
                print(f"   zero-action MAE: {np.nanmean(z):.6f} ± {np.nanstd(z):.6f}")
            if len(dbg_key_mae) > 0:
                print("\n🔎 Per-key MAE (normalized):")
                for key, arrs in dbg_key_mae.items():
                    stacked = np.stack(arrs, axis=0)
                    mean = stacked.mean(axis=0)
                    if key.endswith("joint_position"):
                        labels = [f"j{i}" for i in range(mean.shape[0])]
                    elif key.endswith("gripper_position"):
                        labels = ["gripper"]
                    else:
                        labels = [f"d{i}" for i in range(mean.shape[0])]
                    items = ", ".join([f"{l}:{v:.4f}" for l, v in zip(labels, mean)])
                    print(f"   {key}: {items}")
            if args.report_unnormalized and len(dbg_zero_mae_un) > 0:
                print("\n🔎 Debug summary (unnormalized action units):")
                for s in (-1, 0, 1):
                    vals = np.array(dbg_shift_mae_un[s], dtype=float)
                    if vals.size > 0:
                        print(f"   shift {s:+d} MAE: {np.nanmean(vals):.6f} ± {np.nanstd(vals):.6f}")
                z = np.array(dbg_zero_mae_un, dtype=float)
                print(f"   zero-action MAE: {np.nanmean(z):.6f} ± {np.nanstd(z):.6f}")
                if len(dbg_key_mae_un) > 0:
                    print("\n🔎 Per-key MAE (unnormalized):")
                    for key, arrs in dbg_key_mae_un.items():
                        stacked = np.stack(arrs, axis=0)
                        mean = stacked.mean(axis=0)
                        if key.endswith("joint_position"):
                            labels = [f"j{i}" for i in range(mean.shape[0])]
                        elif key.endswith("gripper_position"):
                            labels = ["gripper"]
                        else:
                            labels = [f"d{i}" for i in range(mean.shape[0])]
                        items = ", ".join([f"{l}:{v:.4f}" for l, v in zip(labels, mean)])
                        print(f"   {key}: {items}")

    if verbose:
        print("\n✓ Done!")
        print("=" * 70)

    return traj_l2_means, traj_mae_means


if __name__ == "__main__":
    main()
