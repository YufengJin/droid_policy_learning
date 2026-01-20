#!/usr/bin/env python

import os

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train


def build_reference_config():
    config = config_factory("diffusion_policy")

    data_path = os.path.expandvars("$WORK/datasets")
    log_root = os.path.expandvars("$WORK/logs")

    with config.values_unlocked():
        config.experiment.name = "droid_reference"
        config.experiment.rollout.enabled = False
        config.experiment.validate = False
        config.experiment.save.every_n_epochs = 50

        config.train.data_format = "droid_rlds"
        config.train.data_path = data_path
        config.train.dataset_names = ["droid"]
        config.train.sample_weights = [1]
        config.train.batch_size = 128
        config.train.num_epochs = 100000
        config.train.output_dir = os.path.join(log_root, "droid", "im", "diffusion_policy")
        config.train.shuffle_buffer_size = 50000
        config.train.subsample_length = 100
        config.train.num_parallel_calls = 200
        config.train.traj_transform_threads = 48
        config.train.traj_read_threads = 48

        config.train.action_keys = [
            "action/abs_pos",
            "action/abs_rot_6d",
            "action/gripper_position",
        ]
        config.train.action_shapes = [(1, 3), (1, 6), (1, 1)]
        config.train.action_config = {
            "action/abs_pos": {"normalization": "min_max"},
            "action/abs_rot_6d": {
                "normalization": "min_max",
                "format": "rot_6d",
                "convert_at_runtime": "rot_euler",
            },
            "action/gripper_position": {"normalization": "min_max"},
        }

        config.algo.ddim.enabled = True
        config.algo.ddpm.enabled = False
        config.algo.noise_samples = 8

        config.observation.image_dim = [128, 128]
        config.observation.modalities.obs.rgb = [
            "camera/image/varied_camera_1_left_image",
            "camera/image/varied_camera_2_left_image",
        ]
        config.observation.modalities.obs.low_dim = [
            "robot_state/cartesian_position",
            "robot_state/gripper_position",
        ]

        config.observation.encoder.rgb.core_class = "VisualCore"
        config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet50Conv"
        config.observation.encoder.rgb.core_kwargs.backbone_kwargs = {
            "pretrained": True,
            "use_cam": False,
            "downsample": False,
        }
        config.observation.encoder.rgb.core_kwargs.feature_dimension = 512
        config.observation.encoder.rgb.core_kwargs.flatten = True
        config.observation.encoder.rgb.obs_randomizer_class = [
            "ColorRandomizer",
            "CropRandomizer",
        ]
        config.observation.encoder.rgb.obs_randomizer_kwargs = [
            {},
            {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False},
        ]

    config.lock()
    return config


if __name__ == "__main__":
    cfg = build_reference_config()
    print(cfg)
    # device = TorchUtils.get_torch_device(try_to_use_cuda=cfg.train.cuda)
    # train(cfg, device=device)
