"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict, Optional, Sequence

import os
import numpy as np
import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg

def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def droid_dataset_transform(trajectory: Dict[str, Any], action_type: str = "cartesian_abs") -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    action_dict = trajectory["action_dict"]
    if action_type == "cartesian_abs":
        T = action_dict["cartesian_position"][:, :3]
        R = mat_to_rot6d(euler_to_rmat(action_dict["cartesian_position"][:, 3:6]))
        trajectory["action"] = tf.concat(
            (
                T,
                R,
                action_dict["gripper_position"],
            ),
            axis=-1,
        )
    elif action_type == "cartesian_velocity":
        trajectory["action"] = tf.concat(
            (
                action_dict["cartesian_velocity"],
                action_dict["gripper_position"],
            ),
            axis=-1,
        )
    elif action_type == "joint_velocity":
        trajectory["action"] = tf.concat(
            (
                action_dict["joint_velocity"],
                action_dict["gripper_position"],
            ),
            axis=-1,
        )
    elif action_type == "joint_position":
        trajectory["action"] = tf.concat(
            (
                action_dict["joint_position"],
                action_dict["gripper_position"],
            ),
            axis=-1,
        )
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    return trajectory


def droid_dataset_transform_cartesian_abs(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return droid_dataset_transform(trajectory, action_type="cartesian_abs")


def droid_dataset_transform_cartesian_velocity(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return droid_dataset_transform(trajectory, action_type="cartesian_velocity")


def droid_dataset_transform_joint_velocity(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return droid_dataset_transform(trajectory, action_type="joint_velocity")


def droid_dataset_transform_joint_position(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return droid_dataset_transform(trajectory, action_type="joint_position")


def get_droid_standardize_fn(action_type: str):
    mapping = {
        "cartesian_abs": droid_dataset_transform_cartesian_abs,
        "cartesian_velocity": droid_dataset_transform_cartesian_velocity,
        "joint_velocity": droid_dataset_transform_joint_velocity,
        "joint_position": droid_dataset_transform_joint_position,
    }
    if action_type not in mapping:
        raise ValueError(f"Unknown action_type: {action_type}")
    return mapping[action_type]


def robomimic_transform(
    trajectory: Dict[str, Any],
    normalize_to_neg_one_one: bool = False,
    include_proprio: bool = True,
    view_dropout_prob: float = 0.0,
    camera_keys: Optional[Sequence[str]] = None,
    language_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    if camera_keys is None:
        camera_keys = [
            "camera/image/varied_camera_1_left_image",
            "camera/image/varied_camera_2_left_image",
        ]
    if len(camera_keys) != 2:
        raise ValueError(f"robomimic_transform expects 2 camera keys, got {len(camera_keys)}")

    img1 = tf.cast(trajectory["observation"]["image_primary"], tf.float32)
    img2 = tf.cast(trajectory["observation"]["image_secondary"], tf.float32)

    if normalize_to_neg_one_one:
        img1 = img1 / 127.5 - 1.0
        img2 = img2 / 127.5 - 1.0

    if view_dropout_prob and view_dropout_prob > 0.0:
        drop = tf.less(tf.random.uniform([], 0.0, 1.0), float(view_dropout_prob))
        drop_value = -1.0 if normalize_to_neg_one_one else 0.0
        fill_img1 = tf.ones_like(img1) * tf.cast(drop_value, img1.dtype)
        fill_img2 = tf.ones_like(img2) * tf.cast(drop_value, img2.dtype)

        def _do_drop():
            drop_first = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)
            img1_d = tf.where(drop_first, fill_img1, img1)
            img2_d = tf.where(drop_first, img2, fill_img2)
            return img1_d, img2_d

        def _no_drop():
            return img1, img2

        img1, img2 = tf.cond(drop, _do_drop, _no_drop)

    # Convert (T, H, W, C) -> (T, C, H, W) for robomimic encoders.
    img1 = tf.transpose(img1, perm=[0, 3, 1, 2])
    img2 = tf.transpose(img2, perm=[0, 3, 1, 2])

    if language_prompt is not None and str(language_prompt).strip() != "":
        try:
            lang_shape = tf.shape(trajectory["task"]["language_instruction"])
        except Exception:
            try:
                lang_shape = tf.shape(trajectory["observation"]["pad_mask"])
            except Exception:
                lang_shape = tf.shape(img1)[:1]
        raw_language = tf.fill(lang_shape, tf.constant(str(language_prompt)))
    else:
        task = trajectory.get("task", {})
        if isinstance(task, dict) and "language_instruction" in task:
            raw_language = tf.fill(tf.shape(task["language_instruction"]), "Pick up the red object from the green tray and insert it into the yellow hole.")
        else:
            raw_language = tf.fill(tf.shape(img1)[:1], "Pick up the red object from the green tray and insert it into the yellow hole.")

    obs_dict = {
        camera_keys[0]: img1,
        camera_keys[1]: img2,
        "raw_language": raw_language,
        "pad_mask": trajectory["observation"]["pad_mask"][..., None],
    }

    if include_proprio and ("proprio" in trajectory.get("observation", {})):
        obs_dict["robot_state/cartesian_position"] = trajectory["observation"]["proprio"][..., :6]
        obs_dict["robot_state/gripper_position"] = trajectory["observation"]["proprio"][..., -1:]

    return {
        "obs": obs_dict,
        "actions": trajectory["action"][1:],
    }


DROID_TO_RLDS_OBS_KEY_MAP = {
    "camera/image/hand_camera_left_image": "wrist_image_left",
    "camera/image/wrist_image_left": "wrist_image_left",
    "camera/image/varied_camera_1_left_image": "exterior_image_1_left",
    "camera/image/varied_camera_2_left_image": "exterior_image_2_left",
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
    "robot_state/joint_position": "joint_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
        shuffle_buffer_size=None,
        dataset_length=None,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._shuffle_buffer_size = shuffle_buffer_size
        self._dataset_length = dataset_length

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        if self._dataset_length is not None:
            return self._dataset_length

        dataset_stats = None
        if hasattr(self._rlds_dataset, "dataset_statistics"):
            dataset_stats = self._rlds_dataset.dataset_statistics
        elif hasattr(self._rlds_dataset, "_dataset_statistics"):
            dataset_stats = self._rlds_dataset._dataset_statistics

        if dataset_stats is None:
            if not os.environ.get("ROBOMIMIC_QUIET"):
                print("Warning: Dataset statistics not available; using shuffle_buffer_size estimate.")
            return self._shuffle_buffer_size if self._shuffle_buffer_size is not None else 100000

        if isinstance(dataset_stats, dict):
            dataset_stats = [dataset_stats]

        lengths = []
        for stats in dataset_stats:
            if "num_transitions" in stats:
                lengths.append(stats["num_transitions"])
            elif "num_samples" in stats:
                lengths.append(stats["num_samples"])
            elif "total_transitions" in stats:
                lengths.append(stats["total_transitions"])
            elif "total_steps" in stats:
                lengths.append(stats["total_steps"])
            elif "steps" in stats:
                lengths.append(stats["steps"])
        if not lengths:
            if not os.environ.get("ROBOMIMIC_QUIET"):
                print("Warning: Dataset statistics missing length keys; using shuffle_buffer_size estimate.")
            return self._shuffle_buffer_size if self._shuffle_buffer_size is not None else 100000
        lengths = np.array(lengths)
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)
