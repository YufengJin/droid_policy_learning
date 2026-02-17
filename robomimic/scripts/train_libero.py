"""
LIBERO DDP (Distributed Data Parallel) training entry point.

使用 LIBERODataset 读取 LIBERO HDF5 数据，支持 JPEG 压缩。

Usage:
  # 多卡训练:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_libero

  # 带覆盖:
  torchrun --nproc_per_node=8 -m robomimic.scripts.train_libero \
      'train.data=[{path: /workspace/datasets/libero/}]' \
      train.batch_size=32 experiment.name=libero_diffusion

  # 单卡:
  python -m robomimic.scripts.train_libero

Config 通过 Hydra 从 robomimic/scripts/train_configs/train_libero.yaml 读取。
"""

import json
import os
import signal
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

from collections import OrderedDict

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.libero_dataset import LIBERODataset


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
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


def _register_signal_handlers():
    def _handler(signum, frame):
        cleanup_ddp()
        sys.exit(128 + (signum if signum is not None else 0))
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config_from_args():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = os.path.join(_SCRIPT_DIR, "train_configs")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError("Config directory not found: {}".format(config_dir))
    overrides = list(sys.argv[1:])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
        cfg = compose(config_name="train_libero", overrides=overrides)

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

    config = config_factory(cfg_dict["algo_name"])
    with config.values_unlocked():
        config.update(cfg_dict)

    exp_name = config.experiment.name
    if exp_name is None or exp_name == "null" or (isinstance(exp_name, str) and exp_name.strip() == ""):
        algo_name = config.algo_name
        data_cfg = config.train.data
        if isinstance(data_cfg, list) and len(data_cfg) > 0:
            first_path = data_cfg[0].get("path", "unknown") if isinstance(data_cfg[0], dict) else str(data_cfg[0])
            dataset_str = os.path.splitext(os.path.basename(first_path))[0]
            if len(data_cfg) > 1:
                dataset_str += "_plus{}".format(len(data_cfg) - 1)
        else:
            dataset_str = "libero"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment.name = "{}_{}_{}_libero".format(algo_name, dataset_str, timestamp)

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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_libero(config, device, rank, world_size, local_rank, use_ddp, debug=False):
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
        print("\n============= LIBERO DDP Training =============")
        print("world_size={} rank={} local_rank={}".format(world_size, rank, local_rank))
        print("effective batch_size = {} * {} = {}".format(
            config.train.batch_size, world_size, config.train.batch_size * world_size))
        print(config)
        print("")

    ObsUtils.initialize_obs_utils_with_config(config)
    ds_format = config.train.data_format

    # -----------------------------------------------------------------------
    # Load LIBERO dataset
    # -----------------------------------------------------------------------
    eval_dataset_cfg = config.train.data[0]
    data_dir = os.path.expanduser(eval_dataset_cfg["path"])
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Dataset at provided path {} not found!".format(data_dir))

    all_obs_keys = []
    if hasattr(config.observation.modalities.obs, "low_dim"):
        all_obs_keys.extend(config.observation.modalities.obs.low_dim)
    if hasattr(config.observation.modalities.obs, "rgb"):
        all_obs_keys.extend(config.observation.modalities.obs.rgb)

    image_size = None
    if hasattr(config.observation, "image_dim") and config.observation.image_dim is not None:
        img_dim = config.observation.image_dim
        if isinstance(img_dim, (list, tuple)) and len(img_dim) >= 1:
            image_size = (img_dim[0], img_dim[1] if len(img_dim) > 1 else img_dim[0])

    trainset = LIBERODataset(
        data_dir=data_dir,
        obs_keys=all_obs_keys,
        action_keys=config.train.action_keys,
        seq_length=config.train.seq_length,
        frame_stack=config.train.frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        action_config=config.train.action_config,
        filter_key=config.train.hdf5_filter_key,
        image_size=image_size,
        normalize_actions=True,
    )

    validset = None
    if config.experiment.validate:
        hdf5_validation_filter_key = getattr(config.train, "hdf5_validation_filter_key", "valid")
        try:
            validset = LIBERODataset(
                data_dir=data_dir,
                obs_keys=all_obs_keys,
                action_keys=config.train.action_keys,
                seq_length=config.train.seq_length,
                frame_stack=config.train.frame_stack,
                pad_seq_length=config.train.pad_seq_length,
                pad_frame_stack=config.train.pad_frame_stack,
                action_config=config.train.action_config,
                filter_key=hdf5_validation_filter_key,
                image_size=image_size,
                normalize_actions=True,
            )
        except Exception as e:
            if is_main:
                print("Warning: could not create validation dataset: {}".format(e))
            validset = None

    obs_normalization_stats = None
    action_normalization_stats = trainset.get_action_normalization_stats()

    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=data_dir, ds_format=ds_format)
    except Exception:
        env_meta = {"env_name": "LIBERO", "type": 1, "env_kwargs": {}}
    if hasattr(config.experiment, "env_meta_update_dict") and config.experiment.env_meta_update_dict:
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)

    sample = trainset[0]
    all_shapes = OrderedDict()
    for k, v in sample["obs"].items():
        all_shapes[k] = list(v.shape[1:]) if len(v.shape) > 1 else list(v.shape)
    ac_dim = sample["actions"].shape[-1]
    shape_meta = {
        "all_shapes": all_shapes,
        "all_obs_keys": all_obs_keys,
        "ac_dim": ac_dim,
        "use_images": any(k in str(all_obs_keys) for k in ["image", "rgb"]),
    }

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_sampler = trainset.get_dataset_sampler()
    if use_ddp:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.train.seed)

    train_loader = DataLoader(
        dataset=trainset, sampler=train_sampler, batch_size=config.train.batch_size,
        shuffle=(train_sampler is None), num_workers=config.train.num_data_workers,
        drop_last=True, pin_memory=True,
    )

    valid_loader = None
    if config.experiment.validate and validset is not None:
        valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset, sampler=valid_sampler, batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None), num_workers=min(config.train.num_data_workers, 1),
            drop_last=True, pin_memory=True,
        )

    # -----------------------------------------------------------------------
    # Environment & Logger
    # -----------------------------------------------------------------------
    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env

    envs = OrderedDict()
    if config.experiment.rollout.enabled and is_main:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            env_names.extend(config.experiment.additional_envs)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(env_meta=env_meta, env_name=env_name, render=False, render_offscreen=config.experiment.render_video, use_image_obs=shape_meta["use_images"])
            env = EnvUtils.wrap_env_from_config(env, config=config)
            envs[env.name] = env

    data_logger = None
    if is_main:
        data_logger = DataLogger(log_dir, config, log_tb=config.experiment.logging.log_tb, log_wandb=config.experiment.logging.log_wandb)
        with open(os.path.join(log_dir, "..", "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = algo_factory(algo_name=config.algo_name, config=config, obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=device)

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

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()
    data_loader_iter = None

    for epoch in range(1, config.train.num_epochs + 1):
        if use_ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        step_log, data_loader_iter = TrainUtils.run_epoch(
            model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats, data_loader_iter=data_loader_iter,
        )
        unwrapped = model.module if use_ddp else model
        unwrapped.on_epoch_end(epoch)

        epoch_ckpt_name = "model_epoch_{}".format(epoch)
        should_save_ckpt = False
        ckpt_reason = None
        if config.experiment.save.enabled:
            time_check = config.experiment.save.every_n_seconds is not None and (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = config.experiment.save.every_n_epochs is not None and epoch > 0 and (epoch % config.experiment.save.every_n_epochs == 0)
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
                step_log_v, _ = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps, obs_normalization_stats=obs_normalization_stats)
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
        rollout_check = config.experiment.rollout.enabled and is_main and (epoch > config.experiment.rollout.warmstart) and ((epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time"))
        if rollout_check and envs:
            rollout_model = RolloutPolicy(unwrapped, obs_normalization_stats=obs_normalization_stats, action_normalization_stats=action_normalization_stats)
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model, envs=envs, horizon=config.experiment.rollout.horizon, use_goals=config.use_goals,
                num_episodes=config.experiment.rollout.n, render=False,
                video_dir=video_dir if config.experiment.render_video else None, epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5), terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
            if is_main:
                for env_name in all_rollout_logs:
                    rollout_logs = all_rollout_logs[env_name]
                    for k, v in rollout_logs.items():
                        if k.startswith("Time_"):
                            data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                        else:
                            data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)
                updated_stats = TrainUtils.should_save_from_rollout_logs(all_rollout_logs=all_rollout_logs, best_return=best_return, best_success_rate=best_success_rate, epoch_ckpt_name=epoch_ckpt_name, save_on_best_rollout_return=config.experiment.save.on_best_rollout_return, save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate)
                best_return = updated_stats["best_return"]
                best_success_rate = updated_stats["best_success_rate"]
                epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
                if updated_stats["should_save_ckpt"]:
                    should_save_ckpt = True
                if updated_stats["ckpt_reason"] is not None:
                    ckpt_reason = updated_stats["ckpt_reason"]

        # MSE evaluation
        if config.experiment.mse.enabled and is_main:
            should_save_mse = (config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0) or (config.experiment.mse.on_save_ckpt and should_save_ckpt)
            if should_save_mse:
                unwrapped.set_eval()
                unwrapped.reset()
                try:
                    mse_batch = next(iter(train_loader))
                except StopIteration:
                    mse_batch = None
                if mse_batch is not None:
                    input_batch = unwrapped.process_batch_for_training(mse_batch)
                    input_batch = unwrapped.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
                    with torch.no_grad():
                        predicted_actions = unwrapped.get_action(input_batch["obs"])
                        actual_actions = input_batch["actions"]
                    predicted_np = TensorUtils.to_numpy(predicted_actions)
                    actual_np = TensorUtils.to_numpy(actual_actions)
                    mse_all = float(np.mean((predicted_np - actual_np) ** 2))
                    mae_all = float(np.mean(np.abs(predicted_np - actual_np)))
                    for k, v in {"evaluate/action_mse": mse_all, "evaluate/action_mae": mae_all}.items():
                        data_logger.record(k, v, epoch)
                unwrapped.set_train()

        if is_main and video_paths and not ((should_save_ckpt and ckpt_reason != "valid") or config.experiment.keep_all_videos):
            for env_name in video_paths:
                try:
                    os.remove(video_paths[env_name])
                except Exception:
                    pass

        if should_save_ckpt and is_main:
            TrainUtils.save_model(model=unwrapped, config=config, env_meta=env_meta, shape_meta=shape_meta, ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"), obs_normalization_stats=obs_normalization_stats, action_normalization_stats=action_normalization_stats)

        if is_main:
            mem_usage = int(psutil.Process().memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    if is_main and data_logger is not None:
        data_logger.close()


def main():
    rank, world_size, local_rank, use_ddp = setup_ddp()
    _register_signal_handlers()
    try:
        config, debug = get_config_from_args()
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        train_libero(config, device=device, rank=rank, world_size=world_size, local_rank=local_rank, use_ddp=use_ddp, debug=debug)
        res_str = "finished run successfully!"
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    finally:
        cleanup_ddp()
    if rank == 0:
        print(res_str)


if __name__ == "__main__":
    main()
