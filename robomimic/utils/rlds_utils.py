"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
import torch
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg

# Default for DROID RLDS when train.action_space is unset (preserves historical 10D pos+rot6d+gripper).
RLDS_DROID_DEFAULT_ACTION_SPACE = "pos_rot6d"


def resolve_rlds_droid_action_space(train_cfg) -> str:
    """Return validated action_space for DROID RLDS; default pos_rot6d if missing."""
    from robomimic.utils.action_space_utils import ACTION_SPACE_DIMS

    raw = getattr(train_cfg, "action_space", None)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return RLDS_DROID_DEFAULT_ACTION_SPACE
    s = str(raw).strip()
    if s not in ACTION_SPACE_DIMS:
        raise ValueError(
            "train.action_space must be one of {}; got {!r}".format(
                list(ACTION_SPACE_DIMS.keys()), s
            )
        )
    return s


def rlds_droid_action_keys_and_shapes(action_space: str) -> Tuple[List[str], List[List[int]]]:
    """action_keys / action_shapes aligned with make_droid_dataset_transform output layout."""
    if action_space == "pos_rot6d":
        return (
            ["action/abs_pos", "action/abs_rot_6d", "action/gripper_position"],
            [[1, 3], [1, 6], [1, 1]],
        )
    if action_space == "pos_euler":
        return (
            ["action/abs_pos", "action/abs_rot_euler", "action/gripper_position"],
            [[1, 3], [1, 3], [1, 1]],
        )
    if action_space == "pos_axisangle":
        return (
            ["action/abs_pos", "action/abs_rot_axis_angle", "action/gripper_position"],
            [[1, 3], [1, 3], [1, 1]],
        )
    raise ValueError("Unknown action_space: {}".format(action_space))


def apply_rlds_droid_action_space_to_config(config) -> str:
    """
    Set config.train.action_space, action_keys, action_shapes for DROID RLDS.
    Fills missing action_config entries needed for normalization / rollout.
    """
    action_space = resolve_rlds_droid_action_space(config.train)
    keys, shapes = rlds_droid_action_keys_and_shapes(action_space)
    defaults = {
        "action/abs_pos": {"normalization": "min_max"},
        "action/abs_rot_6d": {
            "normalization": "min_max",
            "format": "rot_6d",
            "convert_at_runtime": "rot_euler",
        },
        "action/abs_rot_euler": {"normalization": "min_max", "format": "rot_euler"},
        "action/abs_rot_axis_angle": {"normalization": "min_max"},
        "action/gripper_position": {"normalization": "min_max"},
    }
    with config.values_unlocked():
        config.train.action_space = action_space
        config.train.action_keys = keys
        config.train.action_shapes = shapes
        ac = config.train.action_config
        for k in keys:
            if k not in ac:
                if k not in defaults:
                    raise KeyError(
                        "action_config missing key {!r} and no default; add it to YAML".format(k)
                    )
                ac[k] = defaults[k]
    return action_space


def _np_euler_xyz_to_axis_angle(euler: np.ndarray) -> np.ndarray:
    """Batch (..., 3) euler -> axis-angle rotvec; float32 out.

    Uses intrinsic xyz convention (scipy lowercase "xyz") to match
    tensorflow_graphics.rotation_matrix_3d.from_euler used by the other
    DROID RLDS standardize functions (pos_rot6d, pos_euler).
    """
    from scipy.spatial.transform import Rotation as R

    euler = np.asarray(euler, dtype=np.float64)
    shape = euler.shape
    flat = euler.reshape(-1, 3)
    aa = R.from_euler("xyz", flat).as_rotvec().astype(np.float32)
    return aa.reshape(shape)


def _euler_to_axis_angle_tf(euler: tf.Tensor) -> tf.Tensor:
    aa = tf.numpy_function(_np_euler_xyz_to_axis_angle, [euler], tf.float32)
    aa.set_shape(euler.shape)
    return aa


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def _droid_action_pos_euler_grip(trajectory: Dict[str, Any]):
    cart = tf.cast(trajectory["action_dict"]["cartesian_position"], tf.float32)
    T = cart[:, :3]
    euler = cart[:, 3:6]
    g = tf.cast(trajectory["action_dict"]["gripper_position"], tf.float32)
    return T, euler, g


# Separate top-level callables so Octo's get_dataset_statistics (inspect.getsource(standardize_fn))
# produces different cache keys per action dimension; identical inner-function source would reuse 10D stats.
def droid_standardize_pos_rot6d(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    T, euler, g = _droid_action_pos_euler_grip(trajectory)
    R = mat_to_rot6d(euler_to_rmat(euler))
    trajectory["action"] = tf.concat((T, R, g), axis=-1)
    return trajectory


def droid_standardize_pos_euler(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    T, euler, g = _droid_action_pos_euler_grip(trajectory)
    trajectory["action"] = tf.concat((T, euler, g), axis=-1)
    return trajectory


def droid_standardize_pos_axisangle(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    T, euler, g = _droid_action_pos_euler_grip(trajectory)
    aa = _euler_to_axis_angle_tf(euler)
    trajectory["action"] = tf.concat((T, aa, g), axis=-1)
    return trajectory


_DROID_STANDARDIZE_BY_SPACE = {
    "pos_rot6d": droid_standardize_pos_rot6d,
    "pos_euler": droid_standardize_pos_euler,
    "pos_axisangle": droid_standardize_pos_axisangle,
}


def make_droid_dataset_transform(action_space: str):
    """
    RLDS standardize_fn for DROID: trajectory['action'] in the chosen space.

    Source: cartesian_position [xyz, euler_xyz(3)], gripper_position (extrinsic XYZ euler).
    """
    try:
        return _DROID_STANDARDIZE_BY_SPACE[action_space]
    except KeyError:
        raise ValueError("Unknown action_space: {}".format(action_space))


def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible default: pos_rot6d (10D), same as historical RLDS DROID pipeline."""
    return droid_standardize_pos_rot6d(trajectory)


def robomimic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "camera/image/varied_camera_1_left_image": 
                tf.cast(trajectory["observation"]["image_primary"], tf.float32) / 255.0,             # role-ros2/droid dataset is uint8 [0,255]
            "camera/image/varied_camera_2_left_image": 
                tf.cast(trajectory["observation"]["image_secondary"], tf.float32) / 255.0,             # role-ros2/droid dataset is uint8 [0,255]
            "raw_language": trajectory["task"]["language_instruction"],
            "robot_state/cartesian_position": trajectory["observation"]["proprio"][..., :6],
            "robot_state/gripper_position": trajectory["observation"]["proprio"][..., -1:],
            "pad_mask": trajectory["observation"]["pad_mask"][..., None],
        },
        "actions": trajectory["action"][1:],
    }

DROID_TO_RLDS_OBS_KEY_MAP = {
    # "camera/image/varied_camera_1_left_image": "exterior_image_1_left",    
    # "camera/image/varied_camera_2_left_image": "exterior_image_2_left"
    "camera/image/varied_camera_1_left_image": "wrist_image_left",   
    "camera/image/varied_camera_2_left_image": "exterior_image_1_left"
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            rlds_dataset: Octo RLDS dataset.
            train: whether this is training split.
            rank: for DDP, this process's rank (0..world_size-1). Used to shard data.
            world_size: for DDP, total number of processes. If world_size > 1, each rank
                only yields samples where (index % world_size) == rank.
        """
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        for i, sample in enumerate(self._rlds_dataset.as_numpy_iterator()):
            if self._world_size > 1 and (i % self._world_size) != self._rank:
                continue
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            total_len = int(0.95 * total_len)
        else:
            total_len = int(0.05 * total_len)
        if self._world_size > 1:
            total_len = (total_len + self._world_size - 1) // self._world_size
        return total_len

