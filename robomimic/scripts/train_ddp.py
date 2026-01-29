"""
DDP (Distributed Data Parallel) training entry point.

与 train_rlds.py 一样通过 Hydra 从 train_configs 读取配置；启动方式使用 torchrun。

Usage:
  # 多卡（与 train_rlds 相同：Hydra + train_configs/train_rlds.yaml）:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_ddp

  # 带覆盖:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_ddp \
      train.data_path=/workspace/dataset train.dataset_names=[insert_pin] experiment.name=my_exp

  # 可选 load_from（与 train_rlds 一致，会替换为 JSON 配置）:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_ddp load_from=/path/to/generated.json

  # 单卡（不设 RANK 时退化为单进程，仍用 Hydra + train_configs）:
  python -m robomimic.scripts.train_ddp

Config 始终通过 Hydra 从 robomimic/scripts/train_configs 读取（与 train_rlds 一致）；运行用 torchrun。
"""

import json
import os
import sys
import traceback
import datetime
import psutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOMIMIC_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _ROBOMIMIC_ROOT not in sys.path:
    sys.path.insert(0, _ROBOMIMIC_ROOT)

import numpy as np
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tensorflow as tf

from collections import OrderedDict

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.action_utils as ActionUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.eval_utils as EvalUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.rlds_utils import (
    droid_dataset_transform,
    robomimic_transform,
    DROID_TO_RLDS_OBS_KEY_MAP,
    DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP,
    TorchRLDSDataset,
)

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec


def setup_ddp():
    """Initialize DDP if RANK is set (torchrun). Returns (rank, world_size, local_rank, use_ddp)."""
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank >= 0 and world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_config_from_args():
    """
    与 train_rlds.py 一致：通过 Hydra 从 train_configs 读取配置。
    Compose from train_configs/train_rlds.yaml + sys.argv overrides；可选 load_from 替换为 JSON。
    """
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = os.path.join(_SCRIPT_DIR, "train_configs")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError("Config directory not found: {}".format(config_dir))
    overrides = list(sys.argv[1:])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
        cfg = compose(config_name="train_rlds", overrides=overrides)
    load_from = OmegaConf.select(cfg, "load_from")
    if load_from is not None and str(load_from).strip() != "":
        with open(os.path.expanduser(load_from), "r") as f:
            cfg = OmegaConf.create(json.load(f))
    debug = OmegaConf.select(cfg, "debug")
    if debug is None:
        debug = False
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Resolved config is not a dict")
    cfg_dict.pop("load_from", None)
    cfg_dict.pop("debug", None)
    cfg_dict.pop("ddp", None)  # Hydra-only, 不写入 robomimic config
    config = config_factory(cfg_dict["algo_name"])
    with config.values_unlocked():
        config.update(cfg_dict)
    exp_name = config.experiment.name
    if exp_name is None or exp_name == "null" or (isinstance(exp_name, str) and exp_name.strip() == ""):
        algo_name = config.algo_name
        dataset_names = config.train.dataset_names
        if isinstance(dataset_names, list) and len(dataset_names) > 0:
            dataset_str = dataset_names[0] if len(dataset_names) == 1 else "_".join(dataset_names[:2])
            if len(dataset_names) > 2:
                dataset_str += f"_plus{len(dataset_names)-2}"
        else:
            dataset_str = "unknown_dataset"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment.name = f"{algo_name}_{dataset_str}_{timestamp}"
    if debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 200
        config.experiment.mse.every_n_epochs = 2
        config.experiment.save.every_n_epochs = 1
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = "/tmp/tmp_trained_models"
        config.experiment.logging.terminal_output_to_txt = True
        config.experiment.logging.log_tb = False
        config.experiment.logging.log_wandb = False
    config.lock()
    return config, debug


def train_ddp(config, device, rank, world_size, local_rank, use_ddp, debug=False):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(1)

    is_main = rank == 0
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if is_main and config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    if is_main:
        print("\n============= DDP Training Run =============")
        print("world_size={} rank={} local_rank={}".format(world_size, rank, local_rank))
        print(config)
        print("")

    ObsUtils.initialize_obs_utils_with_config(config)
    ds_format = config.train.data_format

    if ds_format == "droid_rlds":
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=None, ds_format=ds_format)
        obs_normalization_stats = None
        tf.config.set_visible_devices([], "GPU")
        obs_modalities = config.observation.modalities.obs.rgb
        assert len(obs_modalities) == 2
        ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
        action_config = config.train.action_config
        is_abs_action = [True] * ac_dim
        BASE_DATASET_KWARGS = {
            "data_dir": config.train.data_path,
            "image_obs_keys": {
                "primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]],
                "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]],
            },
            "state_obs_keys": [DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[obs_key] for obs_key in config.observation.modalities.obs.low_dim],
            "language_key": "language_instruction",
            "norm_skip_keys": ["proprio"],
            "action_proprio_normalization_type": "bounds",
            "absolute_action_mask": is_abs_action,
            "action_normalization_mask": is_abs_action,
            "standardize_fn": droid_dataset_transform,
        }
        dataset_names = config.train.dataset_names
        filter_functions = [
            [ModuleSpec.create("robomimic.utils.rlds_utils:filter_success")] if d_name == "droid" else []
            for d_name in dataset_names
        ]
        dataset_kwargs_list = [
            {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS}
            for d_name, f_functions in zip(dataset_names, filter_functions)
        ]
        combined_dataset_statistics = combine_dataset_statistics(
            [make_dataset_from_rlds(**dk, train=True)[1] for dk in dataset_kwargs_list]
        )
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            config.train.sample_weights,
            train=True,
            shuffle_buffer_size=config.train.shuffle_buffer_size,
            batch_size=None,
            balance_weights=False,
            dataset_statistics=combined_dataset_statistics,
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
        rlds_dataset_stats = (
            dataset.dataset_statistics[0]
            if isinstance(dataset.dataset_statistics, list)
            else dataset.dataset_statistics
        )
        action_stats = ActionUtils.get_action_stats_dict(
            rlds_dataset_stats["action"], config.train.action_keys, config.train.action_shapes
        )
        action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
        dataset = dataset.map(robomimic_transform, num_parallel_calls=config.train.traj_transform_threads)
        pytorch_dataset = TorchRLDSDataset(
            dataset, train=True, rank=rank, world_size=world_size
        )
        train_loader = DataLoader(
            pytorch_dataset,
            batch_size=config.train.batch_size,
            num_workers=0,
        )
        data_loader_iter = iter(train_loader)
        rlds_batch = next(data_loader_iter)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=None,
            batch=rlds_batch,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=is_main,
            config=config,
        )
        validset = None
        valid_loader = None
    else:
        rlds_batch = None
        data_loader_iter = None
        eval_dataset_cfg = config.train.data[0]
        dataset_path = os.path.expanduser(eval_dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            batch=None,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=is_main,
            config=config,
        )
        trainset, validset = TrainUtils.load_data_for_training(
            config, obs_keys=shape_meta["all_obs_keys"]
        )
        obs_normalization_stats = None
        if config.train.hdf5_normalize_obs:
            obs_normalization_stats = trainset.get_obs_normalization_stats()
        action_normalization_stats = trainset.get_action_normalization_stats()
        train_sampler = trainset.get_dataset_sampler()
        if use_ddp:
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(train_sampler is None),
            )
        train_loader = DataLoader(
            dataset=trainset,
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
        )
        if config.experiment.validate and validset is not None:
            valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank) if use_ddp else validset.get_dataset_sampler()
            valid_loader = DataLoader(
                dataset=validset,
                sampler=valid_sampler,
                batch_size=config.train.batch_size,
                shuffle=(valid_sampler is None),
                num_workers=min(config.train.num_data_workers, 1),
                drop_last=True,
            )
        else:
            valid_loader = None

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env

    envs = OrderedDict()
    if config.experiment.rollout.enabled and is_main:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            env_names.extend(config.experiment.additional_envs)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            env = EnvUtils.wrap_env_from_config(env, config=config)
            envs[env.name] = env

    data_logger = None
    if is_main:
        data_logger = DataLogger(
            log_dir,
            config,
            log_tb=config.experiment.logging.log_tb,
            log_wandb=config.experiment.logging.log_wandb,
        )
        with open(os.path.join(log_dir, "..", "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None:
        if is_main:
            print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        print("\n============= Model Summary =============")
        print(model.module if use_ddp else model)
        print("")
        flush_warnings()

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    for epoch in range(1, config.train.num_epochs + 1):
        if use_ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        step_log, data_loader_iter = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
            data_loader_iter=data_loader_iter,
        )
        unwrapped = model.module if use_ddp else model
        unwrapped.on_epoch_end(epoch)

        epoch_ckpt_name = "model_epoch_{}".format(epoch)
        should_save_ckpt = False
        ckpt_reason = None
        if config.experiment.save.enabled:
            time_check = (
                config.experiment.save.every_n_seconds is not None
                and (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            )
            epoch_check = (
                config.experiment.save.every_n_epochs is not None
                and epoch > 0
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

        if is_main:
            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)

        if config.experiment.validate and valid_loader is not None:
            with torch.no_grad():
                step_log_v, _ = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            if is_main:
                for k, v in step_log_v.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                    else:
                        data_logger.record("Valid/{}".format(k), v, epoch)
                if "Loss" in step_log_v and (best_valid_loss is None or step_log_v["Loss"] <= best_valid_loss):
                    best_valid_loss = step_log_v["Loss"]
                    if config.experiment.save.on_best_validation:
                        epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        video_paths = None
        rollout_check = (
            config.experiment.rollout.enabled
            and is_main
            and (epoch > config.experiment.rollout.warmstart)
            and ((epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time"))
        )
        if rollout_check and envs:
            rollout_model = RolloutPolicy(
                unwrapped,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
            if is_main:
                for env_name in all_rollout_logs:
                    rollout_logs = all_rollout_logs[env_name]
                    for k, v in rollout_logs.items():
                        if k.startswith("Time_"):
                            data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                        else:
                            data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)
                    print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                    print("Env: {}".format(env_name))
                    print(json.dumps(rollout_logs, sort_keys=True, indent=4))
                updated_stats = TrainUtils.should_save_from_rollout_logs(
                    all_rollout_logs=all_rollout_logs,
                    best_return=best_return,
                    best_success_rate=best_success_rate,
                    epoch_ckpt_name=epoch_ckpt_name,
                    save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                    save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
                )
                best_return = updated_stats["best_return"]
                best_success_rate = updated_stats["best_success_rate"]
                epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
                if updated_stats["should_save_ckpt"]:
                    should_save_ckpt = True
                if updated_stats["ckpt_reason"] is not None:
                    ckpt_reason = updated_stats["ckpt_reason"]

        if ds_format == "droid_rlds" and rlds_batch is not None and config.experiment.mse.enabled and is_main:
            should_save_mse = (
                (config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0)
                or (config.experiment.mse.on_save_ckpt and should_save_ckpt)
            )
            if should_save_mse:
                unwrapped.set_eval()
                unwrapped.reset()
                input_batch = unwrapped.process_batch_for_training(rlds_batch)
                input_batch = unwrapped.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
                with torch.no_grad():
                    predicted_actions = unwrapped.get_action(input_batch["obs"])
                    actual_actions = input_batch["actions"]
                predicted_actions_np = TensorUtils.to_numpy(predicted_actions)
                actual_actions_np = TensorUtils.to_numpy(actual_actions)
                predicted_pos = predicted_actions_np[..., :3]
                actual_pos = actual_actions_np[..., :3]
                predicted_rot = predicted_actions_np[..., 3:9]
                actual_rot = actual_actions_np[..., 3:9]
                pos_err_stats = EvalUtils.compute_pos_err(predicted_pos, actual_pos)
                rot_err_stats = EvalUtils.compute_rot_err(predicted_rot, actual_rot, rot_format="6d")
                mse_log = {
                    "evaluate/cartesian_position_error/mean": pos_err_stats["mean"],
                    "evaluate/cartesian_position_error/max": pos_err_stats["max"],
                    "evaluate/cartesian_position_error/min": pos_err_stats["min"],
                    "evaluate/cartesian_position_error/std": pos_err_stats["std"],
                    "evaluate/cartesian_position_error/mse": pos_err_stats["mse"],
                    "evaluate/rotation_error/mean": rot_err_stats["mean"],
                    "evaluate/rotation_error/max": rot_err_stats["max"],
                    "evaluate/rotation_error/min": rot_err_stats["min"],
                    "evaluate/rotation_error/std": rot_err_stats["std"],
                    "evaluate/rotation_error/mse": rot_err_stats.get("mse", 0.0),
                }
                for k, v in mse_log.items():
                    data_logger.record("{}".format(k), v, epoch)
                unwrapped.set_train()

        if is_main and video_paths and not (
            (should_save_ckpt and ckpt_reason != "valid") or config.experiment.keep_all_videos
        ):
            for env_name in video_paths:
                try:
                    os.remove(video_paths[env_name])
                except Exception:
                    pass

        if should_save_ckpt and is_main:
            save_model = unwrapped
            TrainUtils.save_model(
                model=save_model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        if is_main:
            mem_usage = int(psutil.Process().memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    if is_main and data_logger is not None:
        data_logger.close()


def main():
    rank, world_size, local_rank, use_ddp = setup_ddp()
    try:
        config, debug = get_config_from_args()
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        train_ddp(
            config,
            device=device,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            use_ddp=use_ddp,
            debug=debug,
        )
        res_str = "finished run successfully!"
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    finally:
        cleanup_ddp()
    if rank == 0:
        print(res_str)


if __name__ == "__main__":
    main()
