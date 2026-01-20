#!/usr/bin/env python

import argparse
import datetime
import json
import os
import time

import torch

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train


def _expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _timestamp() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime("%m-%d-%H%M%S")


def _validate_sample_weights(dataset_names, sample_weights):
    if sample_weights is None:
        return [1] * len(dataset_names)
    if len(sample_weights) != len(dataset_names):
            raise ValueError(
            f"--sample_weights must match --dataset_names length ({len(dataset_names)}), "
            f"got {len(sample_weights)}"
        )
    return sample_weights


def _normalize_camera_key(key: str) -> str:
    aliases = {
        "hand_camera_left": "camera/image/hand_camera_left_image",
        "hand_left": "camera/image/hand_camera_left_image",
        "wrist_left": "camera/image/wrist_image_left",
        "wrist_image_left": "camera/image/wrist_image_left",
        "varied_camera_1_left": "camera/image/varied_camera_1_left_image",
        "exterior_1": "camera/image/varied_camera_1_left_image",
        "exterior_image_1_left": "camera/image/varied_camera_1_left_image",
        "varied_camera_1_left": "camera/image/varied_camera_1_left_image",
        "varied_camera_2_left": "camera/image/varied_camera_2_left_image",
        "exterior_2": "camera/image/varied_camera_2_left_image",
        "exterior_image_2_left": "camera/image/varied_camera_2_left_image",
    }
    return aliases.get(key, key)


def build_config(args):
    config = config_factory("diffusion_policy")

    exp_name = f"{_timestamp()}-{args.name}" if args.append_timestamp else args.name
    log_root = _expand_path(args.log_dir)
    data_path = _expand_path(args.data_path)

    use_cleandift = args.visual_encoder == "CleanDIFTConv"
    sample_weights = _validate_sample_weights(args.dataset_names, args.sample_weights)
    camera_keys = args.cameras or [
        "camera/image/varied_camera_1_left_image",
        "camera/image/varied_camera_2_left_image",
    ]
    camera_keys = [_normalize_camera_key(key) for key in camera_keys]
    if len(camera_keys) != 2:
            raise ValueError(f"DROID RLDS expects exactly 2 cameras, got {len(camera_keys)}")

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
        config.experiment.logging.terminal_output_to_txt = True
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
        elif args.use_true_epochs:
            config.experiment.epoch_every_n_steps = None
        config.experiment.validation_epoch_every_n_steps = 100

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
