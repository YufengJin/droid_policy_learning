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

Config 通过 Hydra 从 robomimic/scripts/train_configs/ 读取；默认 train_libero.yaml。
Drift Policy：使用 train_libero_drift.yaml 时设置环境变量
LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift，或保留默认配置并传 algo_name=drift_policy。
详见 docs/drift_libero_training_report.md。

Debug（debug=true）时为避免整库载入导致 OOM（进程被 Killed）：
  - 默认在配置路径下存在子目录 ``libero_10`` 时只使用该子目录；
  - 或通过环境变量 ``LIBERO_DEBUG_DATA_DIR`` 指定小数据目录；
  - ``dataset_statistics.json`` 写到 ``/tmp/robomimic_dataset_cache/``（只读数据盘也可跑）；
  - 默认最多载入 ``LIBERO_DEBUG_MAX_DEMOS``（默认 16）个 demo，可加大或设为更大整数。

非 debug、数据目录只读：统计会自动写入 ``/tmp/robomimic_dataset_cache/`` 并发出 ``UserWarning``（无需设置环境变量）。可选 ``LIBERO_DATASET_STATISTICS_PATH`` 强制指定统计文件路径。

减少终端噪音：默认不再打印完整 config / 整网结构；需要时设 ``LIBERO_VERBOSE_CONFIG=1``、``LIBERO_VERBOSE_MODEL=1``。
"""

import hashlib
import json
import os
import signal
import sys
import tempfile
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
import robomimic.utils.eval_utils as EvalUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.libero_dataset import LIBERODataset
from robomimic.utils.action_space_utils import ACTION_SPACE_DIMS, get_rot_slice, get_rot_format_for_eval


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
    # e.g. LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift for Drift Policy (see docs/drift_libero_training_report.md)
    hydra_config_name = os.environ.get("LIBERO_TRAIN_HYDRA_CONFIG", "train_libero").strip() or "train_libero"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
        cfg = compose(config_name=hydra_config_name, overrides=overrides)

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
        config.train.num_epochs = 20
        config.experiment.mse.every_n_epochs = 2
        config.experiment.save.every_n_epochs = 10
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.train.output_dir = "/tmp/tmp_trained_models"
        config.experiment.logging.terminal_output_to_txt = True
        config.experiment.logging.log_tb = False
        config.experiment.logging.log_wandb = False
        # Avoid loading full LIBERO tree + duplicate valid set (OOM / Killed on large roots).
        config.experiment.validate = False
        config.experiment.mse.enabled = False
        config.train.num_data_workers = 0
        if config.train.batch_size > 8:
            config.train.batch_size = 8

    config.lock()
    return config, debug


def _libero_debug_dataset_kwargs(debug, config_data_path):
    """
    When debug=True: prefer LIBERO_DEBUG_DATA_DIR, else libero_10 under config path if present;
    write dataset_statistics.json under /tmp (read-only dataset mounts); cap demos via max_demos.
    Returns dict with keys used by LIBERODataset (optional kwargs only when debug).
    """
    if not debug:
        return {}
    env_p = os.environ.get("LIBERO_DEBUG_DATA_DIR", "").strip()
    if env_p:
        data_dir = os.path.expanduser(env_p)
    else:
        data_dir = os.path.expanduser(config_data_path)
        if os.path.isdir(data_dir):
            sub = os.path.join(data_dir, "libero_10")
            if os.path.isdir(sub):
                data_dir = sub
    cache_root = os.path.join(tempfile.gettempdir(), "robomimic_dataset_cache")
    os.makedirs(cache_root, exist_ok=True)
    sig = hashlib.md5(os.path.abspath(data_dir).encode("utf-8")).hexdigest()[:16]
    stats_path = os.path.join(cache_root, "dataset_statistics_{}.json".format(sig))
    try:
        max_demos = max(1, int(os.environ.get("LIBERO_DEBUG_MAX_DEMOS", "16")))
    except ValueError:
        max_demos = 16
    return {
        "_resolved_data_dir": data_dir,
        "dataset_statistics_path": stats_path,
        "max_demos": max_demos,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_libero(config, device, rank, world_size, local_rank, use_ddp, debug=False):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(1)

    is_main = rank == 0
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config, prompt_on_exists=False)

    if is_main and config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    if is_main:
        print("\n============= LIBERO DDP Training =============")
        print("world_size={} rank={} local_rank={}".format(world_size, rank, local_rank))
        print("effective batch_size = {} * {} = {}".format(
            config.train.batch_size, world_size, config.train.batch_size * world_size))
        if os.environ.get("LIBERO_VERBOSE_CONFIG", "").strip() in ("1", "true", "yes"):
            print(config)
            print("")
        else:
            print("algo_name={}  train.batch_size={}  train.num_epochs={}  experiment.validate={}".format(
                config.algo_name, config.train.batch_size, config.train.num_epochs, config.experiment.validate))
            print("")

    ObsUtils.initialize_obs_utils_with_config(config)
    ds_format = config.train.data_format

    # -----------------------------------------------------------------------
    # Load LIBERO dataset
    # -----------------------------------------------------------------------
    eval_dataset_cfg = config.train.data[0]
    config_data_path = eval_dataset_cfg["path"]
    if debug:
        dbg_ds_kw = _libero_debug_dataset_kwargs(True, config_data_path)
        data_dir = dbg_ds_kw.pop("_resolved_data_dir")
    else:
        data_dir = os.path.expanduser(config_data_path)
        dbg_ds_kw = {}

    _stats_env = os.environ.get("LIBERO_DATASET_STATISTICS_PATH", "").strip()
    if _stats_env:
        dbg_ds_kw = dict(dbg_ds_kw)
        dbg_ds_kw["dataset_statistics_path"] = os.path.expanduser(_stats_env)

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

    # Action space: pos_euler (7D, default) | pos_rot6d (10D) | pos_axisangle (7D)
    action_space = getattr(config.train, "action_space", "pos_euler") or "pos_euler"
    ac_dim = ACTION_SPACE_DIMS[action_space]
    # Override action_shapes in config to match the actual ac_dim
    with config.values_unlocked():
        config.train.action_shapes = [[1, ac_dim]]

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
        action_space=action_space,
        **dbg_ds_kw,
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
                action_space=action_space,
                **dbg_ds_kw,
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
        modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        tail_shape = tuple(v.shape[1:])
        all_shapes[k] = ObsUtils.get_processed_shape(obs_modality=modality, input_shape=tail_shape)
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
    # Logger
    # 评测改为 client/server（见 robomimic.scripts.policy_server + benchmarks/*/run_eval.py），
    # 训练循环不再建仿真环境、不做 in-process rollout。
    # -----------------------------------------------------------------------
    # ckpt 预加载（仅一次）：resume 时先取出 wandb_run_id，供 DataLogger 续接同一 run
    ckpt_path = config.experiment.ckpt_path
    resume = getattr(config.experiment, "resume", False)
    ckpt_dict = None
    resume_wandb_run_id = None
    if ckpt_path is not None:
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        if resume:
            resume_wandb_run_id = ckpt_dict.get("wandb_run_id", None)

    data_logger = None
    if is_main:
        data_logger = DataLogger(log_dir, config, log_tb=config.experiment.logging.log_tb, log_wandb=config.experiment.logging.log_wandb, wandb_run_id=resume_wandb_run_id)
        with open(os.path.join(log_dir, "..", "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = algo_factory(algo_name=config.algo_name, config=config, obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=device)

    start_epoch = 1
    best_valid_loss = None
    if ckpt_dict is not None:
        if is_main:
            print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        model.deserialize(ckpt_dict["model"])
        # 完整 resume：恢复 optimizer/scheduler/epoch（需 ckpt 由本流程保存，含这些字段）
        if resume and "epoch" in ckpt_dict and "optimizer" in ckpt_dict:
            if is_main:
                print("RESUMING: loading optimizer, scheduler, epoch from checkpoint")
            TrainUtils._load_optimizer_state(model.optimizers, ckpt_dict["optimizer"])
            TrainUtils._load_scheduler_state(model.lr_schedulers, ckpt_dict["lr_scheduler"])
            start_epoch = ckpt_dict["epoch"] + 1
            if "best_valid_loss" in ckpt_dict:
                best_valid_loss = ckpt_dict["best_valid_loss"]
            if is_main:
                print("RESUME: starting from epoch {}".format(start_epoch))

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        if os.environ.get("LIBERO_VERBOSE_MODEL", "").strip() in ("1", "true", "yes"):
            print("\n============= Model Summary =============")
            print(model.module if use_ddp else model)
            print("")
        flush_warnings()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    last_ckpt_time = time.time()
    data_loader_iter = None

    for epoch in range(start_epoch, config.train.num_epochs + 1):
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
        save_best = False   # 触发“最优模型”单文件覆盖保存（model_best.pth）
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
                        save_best = True   # 仅保留单个 model_best.pth（覆盖），不按 epoch 累积

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
                    # Per-component pos/rot metrics (respects action_space)
                    pos_err_stats = EvalUtils.compute_pos_err(predicted_np[..., :3], actual_np[..., :3])
                    rot_start, rot_end = get_rot_slice(action_space)
                    rot_fmt = get_rot_format_for_eval(action_space)
                    pred_rot = predicted_np[..., rot_start:rot_end]
                    actual_rot = actual_np[..., rot_start:rot_end]
                    if rot_fmt == "euler":
                        pred_rot_in = TorchUtils.euler_angles_to_matrix(
                            torch.tensor(pred_rot, dtype=torch.float32), "XYZ"
                        )
                        actual_rot_in = TorchUtils.euler_angles_to_matrix(
                            torch.tensor(actual_rot, dtype=torch.float32), "XYZ"
                        )
                        rot_err_stats = EvalUtils.compute_rot_err(pred_rot_in, actual_rot_in, rot_format="matrix")
                    else:
                        rot_err_stats = EvalUtils.compute_rot_err(
                            torch.tensor(pred_rot, dtype=torch.float32),
                            torch.tensor(actual_rot, dtype=torch.float32),
                            rot_format=rot_fmt,
                        )
                    mse_log = {
                        "evaluate/action_mse": mse_all,
                        "evaluate/action_mae": mae_all,
                        "evaluate/cartesian_position_error/mean": pos_err_stats["mean"],
                        "evaluate/rotation_error/mean": rot_err_stats["mean"],
                    }
                    for k, v in mse_log.items():
                        data_logger.record(k, v, epoch)
                    print("MSE Log Epoch {}".format(epoch))
                    print(json.dumps(mse_log, sort_keys=True, indent=4))
                unwrapped.set_train()

        if is_main and (should_save_ckpt or save_best):
            # 周期 ckpt 按 epoch 累积（每 ~1h 一个）；best 只保留一个可覆盖的 model_best.pth
            save_fnames = []
            if should_save_ckpt:
                save_fnames.append("model_epoch_{}.pth".format(epoch))
            if save_best:
                save_fnames.append("model_best.pth")
            for _fname in save_fnames:
                TrainUtils.save_model(model=unwrapped, config=config, env_meta=env_meta, shape_meta=shape_meta, ckpt_path=os.path.join(ckpt_dir, _fname), obs_normalization_stats=obs_normalization_stats, action_normalization_stats=action_normalization_stats, epoch=epoch, best_valid_loss=best_valid_loss, wandb_run_id=(data_logger.wandb_run_id if data_logger is not None else None))

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
