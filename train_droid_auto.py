#!/usr/bin/env python3
import argparse
import datetime
import math
import os
import sys
import time
import torch

from robomimic.config import config_factory
from robomimic.scripts.train import train

os.environ.setdefault("TORCH_FX_DISABLE_SYMBOLIC_SHAPES", "1")
os.environ.setdefault("ROBOMIMIC_STAGE_LOG", "0")


def _expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _timestamp() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime("%m-%d-%H%M%S")


def _validate_sample_weights(dataset_names, sample_weights):
    if sample_weights is None:
        return [1] * len(dataset_names)
    if len(sample_weights) != len(dataset_names):
        raise ValueError("sample_weights数量必须和dataset_names一致")
    return list(sample_weights)


def _normalize_camera_key(key: str) -> str:
    k = key.strip()
    if k.startswith("camera/"):
        return k
    aliases = {
        "wrist_image_left": "camera/image/wrist_image_left",
        "wrist_left": "camera/image/wrist_image_left",
        "exterior_image_1_left": "camera/image/varied_camera_1_left_image",
        "exterior_1": "camera/image/varied_camera_1_left_image",
        "exterior_image_2_left": "camera/image/varied_camera_2_left_image",
        "exterior_2": "camera/image/varied_camera_2_left_image",
        "varied_camera_1_left_image": "camera/image/varied_camera_1_left_image",
        "varied_camera_2_left_image": "camera/image/varied_camera_2_left_image",
    }
    return aliases.get(k, k)


def _visible_gpu_count():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        env = env.strip()
        if not env:
            return 0
        return len([x for x in env.split(",") if x.strip() != ""])
    return torch.cuda.device_count()


def _get_dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return True, rank, world, local_rank
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if rank_env is not None:
        try:
            rank = int(rank_env)
            world = int(world_env) if world_env is not None else 1
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return True, rank, world, local_rank
        except ValueError:
            pass
    return False, 0, 1, 0


def _init_distributed(args):
    if not args.use_ddp:
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed不可用，无法启用DDP")
    if torch.distributed.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")


def _estimate_dataset_len(args, config, sample_weights, action_type):
    try:
        from octo.data.dataset import make_dataset_from_rlds
        from robomimic.utils.rlds_utils import (
            get_droid_standardize_fn,
            DROID_TO_RLDS_OBS_KEY_MAP,
            DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP,
        )
    except Exception:
        return None

    obs_modalities = list(config.observation.modalities.obs.rgb)
    if len(obs_modalities) != 2:
        return None
    state_obs_keys = [
        DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[k]
        for k in config.observation.modalities.obs.low_dim
        if k in DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP
    ]
    ac_dim = sum([ac_comp[1] for ac_comp in config.train.action_shapes])
    is_abs_action = [True] * ac_dim
    standardize_fn = get_droid_standardize_fn(action_type)

    base_kwargs = {
        "data_dir": config.train.data_path,
        "image_obs_keys": {
            "primary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[0]],
            "secondary": DROID_TO_RLDS_OBS_KEY_MAP[obs_modalities[1]],
        },
        "state_obs_keys": state_obs_keys,
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": is_abs_action,
        "action_normalization_mask": is_abs_action,
        "standardize_fn": standardize_fn,
    }

    stats_list = []
    for name in args.dataset_names:
        try:
            _, stats = make_dataset_from_rlds(name=name, **base_kwargs, train=True)
        except Exception:
            return None
        stats_list.append(stats)

    total = 0.0
    for stats, weight in zip(stats_list, sample_weights or [1] * len(stats_list)):
        if not isinstance(stats, dict):
            continue
        val = None
        for key in ("num_transitions", "num_samples", "total_transitions", "total_steps", "steps"):
            if key in stats:
                val = stats[key]
                break
        if val is None:
            continue
        total += float(val) * float(weight)
    if total <= 0:
        return None
    return int(total)


def build_config(args):
    config = config_factory("diffusion_policy")

    exp_name = f"{_timestamp()}-{args.name}" if args.append_timestamp else args.name
    log_root = _expand_path(args.log_dir)
    data_path = _expand_path(args.data_path)

    use_cleandift = args.visual_encoder == "CleanDIFTConv"
    use_resnet50 = args.visual_encoder == "ResNet50Conv"

    sample_weights = _validate_sample_weights(args.dataset_names, args.sample_weights)

    camera_keys = args.cameras or [
        "camera/image/varied_camera_1_left_image",
        "camera/image/varied_camera_2_left_image",
    ]
    camera_keys = [_normalize_camera_key(key) for key in camera_keys]
    if len(camera_keys) != 2:
        raise ValueError(f"需要恰好2个相机,当前有{len(camera_keys)}个")

    with config.values_unlocked():
        if args.action_type == "cartesian_abs":
            action_keys = [
                "action/abs_pos",
                "action/abs_rot_6d",
                "action/gripper_position",
            ]
            action_shapes = [(1, 3), (1, 6), (1, 1)]
            action_config = {
                "action/abs_pos": {"normalization": "min_max"},
                "action/abs_rot_6d": {
                    "normalization": "min_max",
                    "format": "rot_6d",
                    "convert_at_runtime": "rot_euler",
                },
                "action/gripper_position": {"normalization": "min_max"},
            }
        elif args.action_type == "joint_velocity":
            action_keys = [
                "action/joint_velocity",
                "action/gripper_position",
            ]
            action_shapes = [(1, 7), (1, 1)]
            action_config = {
                "action/joint_velocity": {"normalization": "min_max"},
                "action/gripper_position": {"normalization": "min_max"},
            }
        elif args.action_type == "joint_position":
            action_keys = [
                "action/joint_position",
                "action/gripper_position",
            ]
            action_shapes = [(1, 7), (1, 1)]
            action_config = {
                "action/joint_position": {"normalization": "min_max"},
                "action/gripper_position": {"normalization": "min_max"},
            }
        elif args.action_type == "cartesian_velocity":
            action_keys = [
                "action/cartesian_velocity",
                "action/gripper_position",
            ]
            action_shapes = [(1, 6), (1, 1)]
            action_config = {
                "action/cartesian_velocity": {"normalization": "min_max"},
                "action/gripper_position": {"normalization": "min_max"},
            }
        else:
            raise ValueError(f"Unknown action_type: {args.action_type}")

    config.experiment.name = exp_name
    config.experiment.validate = False
    config.experiment.logging.log_wandb = not args.no_wandb
    config.experiment.logging.log_tb = True
    config.experiment.logging.terminal_output_to_txt = bool(args.verbose)
    if args.wandb_proj_name:
        config.experiment.logging.wandb_proj_name = args.wandb_proj_name
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.save_freq
    config.experiment.ckpt_path = args.resume_from
    config.experiment.rollout.enabled = bool(args.enable_rollout)
    config.experiment.render = False
    config.experiment.render_video = True

    if args.steps_per_epoch is not None:
        config.experiment.epoch_every_n_steps = int(args.steps_per_epoch)
    else:
        config.experiment.epoch_every_n_steps = None
    config.experiment.validation_epoch_every_n_steps = 100

    if args.final_only:
        config.experiment.save.every_n_epochs = None
        config.experiment.save.epochs = [int(args.num_epochs)]
        config.experiment.save.on_best_validation = False
        config.experiment.save.on_best_rollout_success_rate = False
        if args.final_ckpt_name:
            final_name = args.final_ckpt_name
        else:
            dataset_name = args.dataset_names[0] if len(args.dataset_names) > 0 else "dataset"
            final_name = f"{args.visual_encoder}_{dataset_name}"
        final_name = final_name.replace("/", "_")
        config.experiment.save.final_ckpt_name = final_name

    config.train.data_format = "droid_rlds"
    config.train.data_path = data_path
    config.train.dataset_names = list(args.dataset_names)
    config.train.sample_weights = list(sample_weights)
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.num_epochs
    config.train.shuffle_buffer_size = args.shuffle_buffer_size
    config.train.subsample_length = args.subsample_length
    config.train.num_parallel_calls = args.num_parallel_calls
    config.train.traj_transform_threads = args.traj_transform_threads
    config.train.traj_read_threads = args.traj_read_threads
    config.train.seed = args.seed
    config.train.cuda = True
    config.train.use_ddp = args.use_ddp
    config.train.use_amp = args.use_amp
    config.train.amp_dtype = args.amp_dtype
    config.train.action_keys = action_keys
    config.train.action_shapes = action_shapes
    config.train.action_config = action_config
    config.train.action_type = args.action_type
    config.train.output_dir = log_root
    if args.language_prompt is not None:
        with config.unlocked():
            config.train.language_prompt = args.language_prompt
    if args.checkpoint_dir:
        config.train.checkpoint_dir = _expand_path(args.checkpoint_dir)
    if args.disable_ema:
        with config.unlocked():
            config.algo.ema.enabled = False
    if args.strip_teacher:
        with config.unlocked():
            config.train.strip_cleandift_teacher = True

    config.observation.image_dim = [256, 256]

    rgb_only = use_cleandift
    if args.rgb_only:
        rgb_only = True
    if args.use_low_dim:
        rgb_only = False

    if rgb_only:
        config.observation.modalities.obs.low_dim = []
    else:
        if args.action_type.startswith("joint"):
            config.observation.modalities.obs.low_dim = [
                "robot_state/joint_position",
                "robot_state/gripper_position",
            ]
        else:
            config.observation.modalities.obs.low_dim = [
                "robot_state/cartesian_position",
                "robot_state/gripper_position",
            ]

    config.observation.modalities.obs.rgb = camera_keys

    if use_cleandift:
        config.observation.encoder.rgb.core_class = "CleanDIFTRGBCore"
        config.observation.encoder.rgb.core_kwargs = {
            "sd_version": "sd21",
            "feature_key": ["us3", "us6", "us8"],
            "freeze_backbone": bool(args.freeze_backbone),
            "use_text_condition": True,
            "map_out_dim": 512,
            "resize_shape": (256, 256),
            "normalize_mode": "zero_one",
            "use_fp32": False,
            "fpn_dim": 256,
            "fpn_num_queries": 8,
            "fpn_dropout": 0.1,
            "layer_scale_init": 0.1,
        }
        if args.cleandift_checkpoint and str(args.cleandift_checkpoint).lower() not in ("none", "null", ""):
            config.observation.encoder.rgb.core_kwargs["custom_checkpoint"] = _expand_path(args.cleandift_checkpoint)

        config.observation.encoder.rgb.obs_randomizer_class = "ColorRandomizer"
        config.observation.encoder.rgb.obs_randomizer_kwargs = {}

        config.algo.cleandift_alignment_weight = float(args.alignment_weight)
        config.algo.cleandift_alignment_freq = 1.0
        config.algo.cleandift_alignment_freq_min = 0.2
        config.algo.cleandift_alignment_freq_drop_ratio = 0.01
        config.algo.cleandift_alignment_warmdown_frac = 0.5
        config.algo.cleandift_alignment_warmdown_steps = None
        config.algo.cleandift_alignment_min_decay_factor = 0.001
        config.algo.cleandift_alignment_decay_power = 2.0
    elif use_resnet50:
        config.observation.encoder.rgb.core_class = "VisualCore"
        config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet50Conv"
        config.observation.encoder.rgb.core_kwargs.normalize_mode = "imagenet"
        config.observation.encoder.rgb.core_kwargs.backbone_kwargs = {
            "pretrained": True,
            "use_cam": False,
            "downsample": False,
        }
        config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
        config.observation.encoder.rgb.core_kwargs.pool_kwargs = {
            "num_kp": 32,
            "learnable_temperature": False,
            "temperature": 1.0,
            "noise_std": 0.0,
        }
        config.observation.encoder.rgb.core_kwargs.feature_dimension = 512
        config.observation.encoder.rgb.core_kwargs.flatten = True

        config.observation.encoder.rgb.obs_randomizer_class = ["ColorRandomizer", "CropRandomizer"]
        config.observation.encoder.rgb.obs_randomizer_kwargs = [
            {},
            {"crop_height": 224, "crop_width": 224, "num_crops": 1, "pos_enc": False},
        ]

        config.algo.cleandift_alignment_weight = 0.0
    else:
        raise ValueError(f"不支持的visual encoder: {args.visual_encoder}")

    config.observation.encoder.rgb.fuser = None

    config.lock()
    return config


def main():
    parser = argparse.ArgumentParser(description="DROID自动训练脚本")

    parser.add_argument("--name", type=str, required=True, help="实验名称")
    parser.add_argument("--append_timestamp", action="store_true", default=True,
                        help="在实验名称前添加时间戳")
    parser.add_argument("--log_dir", type=str, default="$WORK/logs",
                        help="日志目录")

    parser.add_argument("--data_path", type=str, default="$WORK/datasets",
                        help="数据集根目录")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=["droid"],
                        help="数据集名称列表")
    parser.add_argument("--sample_weights", type=float, nargs="+", default=None,
                        help="数据集采样权重")

    parser.add_argument("--action_type", type=str, default="cartesian_abs",
                        choices=["cartesian_abs", "joint_velocity", "joint_position", "cartesian_velocity"],
                        help="动作空间类型")

    parser.add_argument("--visual_encoder", type=str, default="ResNet50Conv",
                        choices=["ResNet50Conv", "CleanDIFTConv"],
                        help="视觉编码器类型")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="冻结backbone（仅CleanDIFT）")
    parser.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false",
                        help="不冻结backbone（仅CleanDIFT）")
    parser.add_argument("--alignment_weight", type=float, default=0.1,
                        help="对齐损失权重（仅CleanDIFT）")
    parser.add_argument("--cleandift_checkpoint", type=str, default=None,
                        help="CleanDIFT自定义checkpoint目录")
    parser.add_argument("--disable_ema", action="store_true", default=False,
                        help="关闭EMA以节省显存")
    parser.add_argument("--strip_teacher", action="store_true", default=False,
                        help="移除CleanDIFT teacher/base UNet以节省显存（推理/冻结backbone时安全）")
    parser.add_argument("--rgb_only", action="store_true", default=False,
                        help="仅使用RGB输入，禁用low_dim状态")
    parser.add_argument("--use_low_dim", action="store_true", default=False,
                        help="显式启用low_dim状态输入")
    parser.add_argument("--language_prompt", type=str, default=None,
                        help="覆盖数据集语言指令（默认None表示使用数据集/无语言）")

    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="相机列表（需要恰好2个）。")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小（每个GPU）")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练epoch数")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="启用AMP混合精度")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="AMP精度类型")
    parser.add_argument("--use_ddp", action="store_true", default=False,
                        help="启用DDP多GPU训练")
    parser.add_argument("--show_progress", dest="show_progress", action="store_true", default=True,
                        help="显示训练进度条")
    parser.add_argument("--no_progress", dest="show_progress", action="store_false",
                        help="关闭训练进度条")
    parser.add_argument("--shuffle_buffer_size", type=int, default=100000,
                        help="TF shuffle buffer大小")
    parser.add_argument("--subsample_length", type=int, default=100,
                        help="子采样长度")
    parser.add_argument("--num_parallel_calls", type=int, default=200,
                        help="并行调用数")
    parser.add_argument("--traj_transform_threads", type=int, default=48,
                        help="轨迹转换线程数")
    parser.add_argument("--traj_read_threads", type=int, default=48,
                        help="轨迹读取线程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    parser.add_argument("--use_true_epochs", action="store_true", default=False,
                        help="使用完整数据集作为一个epoch")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="每个epoch的步数（覆盖use_true_epochs）")

    parser.add_argument("--save_freq", type=int, default=10,
                        help="保存频率（每N个epoch）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="恢复训练的checkpoint路径")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="checkpoint保存目录（覆盖默认log目录）")
    parser.add_argument("--final_only", action="store_true", default=False,
                        help="只在最后一个epoch保存checkpoint")
    parser.add_argument("--final_ckpt_name", type=str, default=None,
                        help="最后保存的checkpoint文件名（不含后缀）")

    parser.add_argument("--wandb_proj_name", type=str, default="droid_policy",
                        help="Wandb项目名称")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="禁用wandb")

    parser.add_argument("--enable_rollout", action="store_true", default=False,
                        help="启用评估rollout")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="启用详细输出")

    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank = _get_dist_info()
    if args.use_ddp:
        _init_distributed(args)

    rank_log_dir = None
    if rank != 0:
        rank_log_dir = _expand_path(args.log_dir or ".")
        rank_log_dir = os.path.join(rank_log_dir, "rank_logs")
        try:
            os.makedirs(rank_log_dir, exist_ok=True)
            log_path = os.path.join(rank_log_dir, f"train_droid_auto_rank{rank}.log")
        except OSError:
            log_path = f"/tmp/train_droid_auto_rank{rank}.log"
        sys.stdout = open(log_path, "w")
        sys.stderr = sys.stdout

    if args.visual_encoder == "CleanDIFTConv":
        try:
            import omegaconf  # noqa: F401
        except ModuleNotFoundError as exc:
            if rank == 0:
                print(
                    "CleanDIFT requires omegaconf. Install it with:\n"
                    "  pip install omegaconf\n"
                    "or\n"
                    "  conda install -c conda-forge omegaconf",
                    flush=True,
                )
            raise
        try:
            from agents.encoders.cleandift_img_encoder import CleanDIFTImgEncoder  # noqa: F401
        except Exception as exc:
            if rank == 0:
                print(f"CleanDIFT import failed: {exc}", flush=True)
            raise

    device_index = local_rank
    if torch.cuda.is_available() and args.use_ddp:
        visible = _visible_gpu_count()
        if visible == 0:
            raise RuntimeError("No visible CUDA devices found.")
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = build_config(args)
    with config.values_unlocked():
        config.experiment.logging.log_wandb = (rank == 0) and (not args.no_wandb)
        if rank != 0:
            os.environ["WANDB_MODE"] = "disabled"

    if args.steps_per_epoch is None and args.use_true_epochs:
        sample_weights = _validate_sample_weights(args.dataset_names, args.sample_weights)
        dataset_len = _estimate_dataset_len(args, config, sample_weights, args.action_type)
        if dataset_len is not None:
            global_batch = args.batch_size * (world_size if args.use_ddp else 1)
            steps = int(math.ceil(dataset_len / float(global_batch)))
            with config.values_unlocked():
                config.experiment.epoch_every_n_steps = steps
        else:
            pass

    if rank == 0 and args.use_ddp:
        rank_log_root = _expand_path(args.log_dir or ".")
        rank_log_root = os.path.join(rank_log_root, "rank_logs")
        os.makedirs(rank_log_root, exist_ok=True)

    try:
        train(config, device=device)
    except Exception as exc:
        if rank == 0:
            print("=" * 80)
            print(f"✗ 训练失败: {exc}")
            print("=" * 80)
        raise


if __name__ == "__main__":
    main()
