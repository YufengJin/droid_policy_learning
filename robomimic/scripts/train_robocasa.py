"""
RoboCasa DDP (Distributed Data Parallel) training entry point.

使用 RoboCasaDataset 读取 RoboCasa HDF5 数据（robosuite 标准格式），支持 JPEG 压缩、动作截断等。
path 可为单个 .h5 或目录（扫描目录下所有 HDF5）；train/val 由 HDF5 mask 与 train.hdf5_filter_key 控制。

Usage:
  # 多卡训练:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_robocasa

  # 带覆盖:
  torchrun --nproc_per_node=8 -m robomimic.scripts.train_robocasa \\
      'train.data=[{path: /workspace/datasets/robocasa/}]' \\
      train.batch_size=32 experiment.name=robocasa_diffusion

  # 单卡:
  python -m robomimic.scripts.train_robocasa

  # Debug 模式:
  python -m robomimic.scripts.train_robocasa debug=true

  # 从 JSON 配置启动:
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_robocasa load_from=/path/to/config.json

  # 在 docker 中运行
  docker exec -it -w /workspace/droid_policy_learning droid-dev-headless micromamba run -n droid_env python -m robomimic.scripts.train_robocasa

Config 通过 Hydra 从 robomimic/scripts/train_configs/train_robocasa.yaml 读取。
"""

import json
import os
import signal
import sys
import traceback
import datetime
import psutil

# Paths: allow running as module from repo root
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
from robomimic.utils.robocasa_dataset import RoboCasaDataset


def _robocasa_collate_fn(batch):
    """
    Custom collate for RoboCasaDataset.
    Stacks arrays and converts to torch.Tensor so the model receives tensors.
    """
    elem = batch[0]
    if isinstance(elem, dict):
        out = {}
        for k in elem:
            if k == "obs":
                obs_batch = {}
                for obs_k in elem["obs"]:
                    vals = [b["obs"][obs_k] for b in batch]
                    if isinstance(vals[0], (np.ndarray, torch.Tensor)):
                        stacked = np.stack(vals, axis=0) if isinstance(vals[0], np.ndarray) else torch.stack(vals, dim=0)
                        obs_batch[obs_k] = torch.from_numpy(stacked) if isinstance(stacked, np.ndarray) else stacked
                    else:
                        obs_batch[obs_k] = vals
                out["obs"] = obs_batch
            elif k == "actions":
                actions_np = np.stack([b["actions"] for b in batch], axis=0)
                out["actions"] = torch.from_numpy(actions_np)
            else:
                out[k] = [b[k] for b in batch]
        return out
    return torch.utils.data.dataloader.default_collate(batch)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

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
    """Destroy DDP process group so the process exits cleanly and releases MASTER_PORT."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _register_signal_handlers():
    """On SIGINT/SIGTERM, cleanup DDP and exit so the port is released."""

    def _handler(signum, frame):
        cleanup_ddp()
        sys.exit(128 + (signum if signum is not None else 0))

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config_from_args():
    """
    从 Hydra YAML + 命令行覆盖构建 robomimic 配置。
    支持 load_from=path/to/config.json 完全替代 YAML。
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
        cfg = compose(config_name="train_robocasa", overrides=overrides)

    # 可选：从 JSON 加载完整配置（覆盖上述 YAML）
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

    # Hydra 专用键不传给 robomimic（robomimic config 为 key-locked，未知 key 会导致 update 失败）
    _HYDRA_ONLY_KEYS = ("load_from", "debug", "resume")
    for _k in _HYDRA_ONLY_KEYS:
        cfg_dict.pop(_k, None)

    config = config_factory(cfg_dict["algo_name"])
    with config.values_unlocked():
        config.update(cfg_dict)

    # RoboCasa：若 YAML 中未写某个 modality（如 low_dim 整段注释掉），config.update 不会覆盖
    # base_config 的默认值，会保留 base_config 的 low_dim = [robot0_eef_pos, ..., object]。
    # 这里按 cfg_dict 中“是否出现”该 key 来补写：未出现则置为 []，避免误用默认 low_dim。
    _modality_keys = ("low_dim", "rgb", "depth", "scan", "lang")
    _obs_cfg = cfg_dict.get("observation", {}).get("modalities", {}).get("obs", {})
    for _mk in _modality_keys:
        if _mk not in _obs_cfg:
            config.observation.modalities.obs[_mk] = []
    _goal_cfg = cfg_dict.get("observation", {}).get("modalities", {}).get("goal", {})
    for _mk in _modality_keys:
        if _mk not in _goal_cfg:
            config.observation.modalities.goal[_mk] = []

    # 未指定实验名时：{algo}_{dataset_basename}_{timestamp}_robocasa
    exp_name = config.experiment.name
    if exp_name is None or exp_name == "null" or (isinstance(exp_name, str) and exp_name.strip() == ""):
        algo_name = config.algo_name
        # Derive name from dataset path(s)
        data_cfg = config.train.data
        if isinstance(data_cfg, list) and len(data_cfg) > 0:
            first_path = data_cfg[0].get("path", "unknown") if isinstance(data_cfg[0], dict) else str(data_cfg[0])
            dataset_str = os.path.splitext(os.path.basename(first_path))[0]
            if len(data_cfg) > 1:
                dataset_str += "_plus{}".format(len(data_cfg) - 1)
        else:
            dataset_str = "robocasa"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment.name = "{}_{}_{}_robocasa".format(algo_name, dataset_str, timestamp)

    if debug:
        config.unlock()
        config.lock_keys()
        # 缩短 epoch / 减少 rollout / 关日志，便于快速调试
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

def train_robocasa(config, device, rank, world_size, local_rank, use_ddp, debug=False):
    """
    RoboCasa HDF5 DDP training loop.

    Args:
        config: robomimic config object
        device: torch device for this process
        rank: global rank
        world_size: total number of processes
        local_rank: local rank on this node
        use_ddp: whether DDP is active
        debug: whether in debug mode
    """
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(1)

    is_main = rank == 0
    # prompt_on_exists=False: never block on input() when exp dir exists (slurm/container-safe)
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config, prompt_on_exists=False)

    if is_main and config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    if is_main:
        print("\n============= RoboCasa DDP Training =============")
        print("world_size={} rank={} local_rank={}".format(world_size, rank, local_rank))
        print("effective batch_size = {} * {} = {}".format(
            config.train.batch_size, world_size, config.train.batch_size * world_size
        ))
        print(config)
        print("")

    # -----------------------------------------------------------------------
    # 观测与数据格式
    # -----------------------------------------------------------------------
    ObsUtils.initialize_obs_utils_with_config(config)
    ds_format = config.train.data_format

    # -----------------------------------------------------------------------
    # 数据集：仅使用 train.data[0]，支持目录或单个 .h5 文件
    # -----------------------------------------------------------------------
    eval_dataset_cfg = config.train.data[0]
    data_dir = os.path.expanduser(eval_dataset_cfg["path"])
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Dataset at provided path {} not found!".format(data_dir))

    # 观测键：从 config 的 modalities.obs 中收集所有启用的 key。
    # 使用 getattr + 过滤 None / 空列表，确保注释掉某个 modality 时不会误加 key。
    all_obs_keys = []
    for modality_name in ("low_dim", "rgb", "depth", "scan", "lang"):
        keys = getattr(config.observation.modalities.obs, modality_name, None)
        if keys:  # skip None, empty list, MISSING
            all_obs_keys.extend(keys)

    # 图像尺寸：若配置了 image_dim 则传给 RoboCasaDataset 做 resize
    image_size = None
    if hasattr(config.observation, "image_dim") and config.observation.image_dim is not None:
        img_dim = config.observation.image_dim
        if isinstance(img_dim, (list, tuple)) and len(img_dim) >= 1:
            image_size = (img_dim[0], img_dim[1] if len(img_dim) > 1 else img_dim[0])

    # 训练集
    trainset = RoboCasaDataset(
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
        verbose=debug,
    )

    # 验证集：同一 data_dir，用 mask/valid 的 demo（需 HDF5 内有 valid mask）
    validset = None
    if config.experiment.validate:
        hdf5_validation_filter_key = getattr(config.train, "hdf5_validation_filter_key", "valid")
        try:
            validset = RoboCasaDataset(
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
                verbose=debug,
            )
        except Exception as e:
            if is_main:
                print("Warning: could not create validation dataset: {}".format(e))
            validset = None

    obs_normalization_stats = None
    action_normalization_stats = trainset.get_action_normalization_stats()

    # -----------------------------------------------------------------------
    # env_meta / shape_meta：供 rollout、保存 checkpoint 等使用
    # -----------------------------------------------------------------------
    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=data_dir, ds_format=ds_format)
    except Exception:
        env_meta = {"env_name": "RoboCasa", "type": 1, "env_kwargs": {}}
    if hasattr(config.experiment, "env_meta_update_dict") and config.experiment.env_meta_update_dict:
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)

    # shape_meta：从 dataset 取一条样本推断各 obs 的 shape。
    # 注意：只取 all_obs_keys 中的 key，忽略 sample 中可能存在的多余 key。
    # 图像：Encoder/CropRandomizer 期望 (C,H,W)，dataset 返回 (seq,H,W,C) -> 转为 (C,H,W)
    # 语言：lang_fixed/language_distilbert 为预计算 DistilBERT 768-d embedding
    rgb_keys = set(getattr(config.observation.modalities.obs, "rgb", None) or [])
    sample = trainset[0]
    all_shapes = OrderedDict()
    _lang_shape_added = False
    for k in all_obs_keys:
        if k in ("raw_language", "lang_fixed/language_distilbert"):
            if not _lang_shape_added:
                all_shapes["lang_fixed/language_distilbert"] = [768]  # DistilBERT output
                _lang_shape_added = True
            continue
        if k not in sample["obs"]:
            raise KeyError("Obs key '{}' not found in dataset sample. "
                           "Available keys: {}".format(k, list(sample["obs"].keys())))
        v = sample["obs"][k]
        shape = list(v.shape[1:]) if len(v.shape) > 1 else list(v.shape)
        if k in rgb_keys and len(shape) == 3:
            shape = [shape[2], shape[0], shape[1]]  # (H,W,C) -> (C,H,W)
        all_shapes[k] = shape
    ac_dim = sample["actions"].shape[-1]
    shape_meta = {
        "all_shapes": all_shapes,
        "all_obs_keys": all_obs_keys,
        "ac_dim": ac_dim,
        "use_images": any(k in str(all_obs_keys) for k in ["image", "rgb"]),
    }
    if is_main:
        print("all_obs_keys = {}".format(all_obs_keys))
        print("shape_meta['all_shapes'] = {}".format(dict(all_shapes)))

    # -----------------------------------------------------------------------
    # DataLoader：DDP 时用 DistributedSampler 保证各 rank 数据不重叠
    # -----------------------------------------------------------------------
    train_sampler = trainset.get_dataset_sampler()
    if use_ddp:
        train_sampler = DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.train.seed,
        )

    _num_workers = config.train.num_data_workers or 0
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=_num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_num_workers > 0),
        collate_fn=_robocasa_collate_fn,
    )

    valid_loader = None
    if config.experiment.validate and validset is not None:
        valid_sampler = (
            DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)
            if use_ddp
            else validset.get_dataset_sampler()
        )
        _valid_workers = min(_num_workers, 1)
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=_valid_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(_valid_workers > 0),
            collate_fn=_robocasa_collate_fn,
        )

    # -----------------------------------------------------------------------
    # 环境：仅 rank 0 创建，用于 rollout 评估与录视频
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 日志：TensorBoard / WandB，仅 rank 0；并写 config.json
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 模型：由 algo_name 创建（如 diffusion_policy），obs_key_shapes 决定 encoder 输入
    # -----------------------------------------------------------------------
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()
    data_loader_iter = None
    start_epoch = 1

    ckpt_path = config.experiment.ckpt_path
    resume = getattr(config.experiment, "resume", False)
    if ckpt_path is not None:
        if is_main:
            print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        # Load into unwrapped algo; DDP wrap happens below
        model.deserialize(ckpt_dict["model"])
        if resume and "epoch" in ckpt_dict and "optimizer" in ckpt_dict:
            if is_main:
                print("RESUMING: loading optimizer, scheduler, epoch from checkpoint")
            TrainUtils._load_optimizer_state(model.optimizers, ckpt_dict["optimizer"])
            TrainUtils._load_scheduler_state(model.lr_schedulers, ckpt_dict["lr_scheduler"])
            start_epoch = ckpt_dict["epoch"] + 1
            if "best_valid_loss" in ckpt_dict:
                best_valid_loss = ckpt_dict["best_valid_loss"]
            if "best_return" in ckpt_dict:
                best_return = ckpt_dict["best_return"]
            if "best_success_rate" in ckpt_dict:
                best_success_rate = ckpt_dict["best_success_rate"]
            if is_main:
                print("RESUME: starting from epoch {}".format(start_epoch))

    # DDP 包装，多卡时梯度同步
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        print("\n============= Model Summary =============")
        print(model.module if use_ddp else model)
        print("")
        flush_warnings()

    # -----------------------------------------------------------------------
    # 训练循环：每 epoch 跑 train / valid / 可选 rollout / MSE / 保存
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, config.train.num_epochs + 1):
        # DDP：每 epoch 重置 sampler 的 epoch，保证各 rank 打乱顺序不同且可复现
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

        # -------------------------------------------------------------------
        # 是否本 epoch 保存：按间隔 / 时间 / 指定 epoch 列表
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # 训练指标写入 DataLogger（rank 0）
        # -------------------------------------------------------------------
        if is_main:
            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)

        # -------------------------------------------------------------------
        # 验证集前向与 loss，可选触发 best_validation 保存
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # Rollout：rank 0 在环境中跑 policy，录视频，可选按 return/success 保存
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # MSE：从 train_loader 取一批，算预测动作与真实动作的 MSE/MAE（rank 0）
        # -------------------------------------------------------------------
        if config.experiment.mse.enabled and is_main:
            should_save_mse = (
                (config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0)
                or (config.experiment.mse.on_save_ckpt and should_save_ckpt)
            )
            if should_save_mse:
                unwrapped.set_eval()
                unwrapped.reset()
                # Sample a batch for MSE evaluation
                try:
                    mse_batch = next(iter(train_loader))
                except StopIteration:
                    mse_batch = None
                if mse_batch is not None:
                    input_batch = unwrapped.process_batch_for_training(mse_batch)
                    input_batch = unwrapped.postprocess_batch_for_training(
                        input_batch, obs_normalization_stats=obs_normalization_stats
                    )
                    with torch.no_grad():
                        predicted_actions = unwrapped.get_action(input_batch["obs"])
                        actual_actions = input_batch["actions"]
                    predicted_np = TensorUtils.to_numpy(predicted_actions)
                    actual_np = TensorUtils.to_numpy(actual_actions)
                    # Compute position and rotation errors
                    mse_all = float(np.mean((predicted_np - actual_np) ** 2))
                    mae_all = float(np.mean(np.abs(predicted_np - actual_np)))
                    mse_log = {
                        "evaluate/action_mse": mse_all,
                        "evaluate/action_mae": mae_all,
                    }
                    for k, v in mse_log.items():
                        data_logger.record(k, v, epoch)
                unwrapped.set_train()

        # -------------------------------------------------------------------
        # 非保留时删除本 epoch 的 rollout 视频以省磁盘
        # -------------------------------------------------------------------
        if is_main and video_paths and not (
            (should_save_ckpt and ckpt_reason != "valid") or config.experiment.keep_all_videos
        ):
            for env_name in video_paths:
                try:
                    os.remove(video_paths[env_name])
                except Exception:
                    pass

        # -------------------------------------------------------------------
        # 写 checkpoint（rank 0），含 optimizer/scheduler/epoch 供 resume
        # -------------------------------------------------------------------
        if should_save_ckpt and is_main:
            model_to_save = unwrapped
            TrainUtils.save_model(
                model=model_to_save,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                best_return=best_return,
                best_success_rate=best_success_rate,
            )

        # -------------------------------------------------------------------
        # 内存占用记录（rank 0）
        # -------------------------------------------------------------------
        if is_main:
            mem_usage = int(psutil.Process().memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # -----------------------------------------------------------------------
    # 收尾：关闭 logger，DDP 在 main() 的 finally 里 cleanup
    # -----------------------------------------------------------------------
    if is_main and data_logger is not None:
        data_logger.close()


# ---------------------------------------------------------------------------
# 入口：解析配置、setup DDP、跑 train_robocasa、cleanup
# ---------------------------------------------------------------------------

def main():
    rank, world_size, local_rank, use_ddp = setup_ddp()
    _register_signal_handlers()
    try:
        config, debug = get_config_from_args()
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        train_robocasa(
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
