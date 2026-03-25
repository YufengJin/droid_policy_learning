"""
Action space utilities for RoboCasa / LIBERO training pipelines.

Supports three action space representations:
  pos_euler    (7D):  [pos(3), euler_xyz(3), gripper(1)]  — default
  pos_rot6d   (10D):  [pos(3), rot_6d(6),   gripper(1)]
  pos_axisangle (7D): [pos(3), axis_angle(3), gripper(1)]

Euler convention: extrinsic XYZ (= scipy 'XYZ' uppercase), matching PyTorch3D's
euler_angles_to_matrix("XYZ") which computes R_X @ R_Y @ R_Z.
All conversions use scipy.spatial.transform for a mathematically correct roundtrip.
"""

import numpy as np
import scipy.spatial.transform

ACTION_SPACE_DIMS = {
    "pos_euler": 7,
    "pos_rot6d": 10,
    "pos_axisangle": 7,
}

# Euler convention used throughout RoboCasa/LIBERO pipelines (extrinsic XYZ)
_EULER_CONVENTION = "XYZ"


def _euler_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert (..., 3) euler angles (extrinsic XYZ) to (..., 3, 3) rotation matrices."""
    flat = euler.reshape(-1, 3)
    mats = scipy.spatial.transform.Rotation.from_euler(_EULER_CONVENTION, flat).as_matrix()
    return mats.reshape(euler.shape[:-1] + (3, 3))


def _matrix_to_euler(mats: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 3) euler angles (extrinsic XYZ)."""
    flat = mats.reshape(-1, 3, 3)
    euler = scipy.spatial.transform.Rotation.from_matrix(flat).as_euler(_EULER_CONVENTION)
    return euler.reshape(mats.shape[:-2] + (3,)).astype(np.float32)


def _matrix_to_rot6d(mats: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 6) 6D representation (first two columns)."""
    # 6D = first two columns of rotation matrix, flattened
    return np.concatenate([mats[..., :, 0], mats[..., :, 1]], axis=-1).astype(np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert (..., 6) 6D representation to (..., 3, 3) orthonormal rotation matrices."""
    # Gram-Schmidt orthonormalization
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)  # (..., 3, 3)


def _matrix_to_axis_angle(mats: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 3) axis-angle vectors."""
    flat = mats.reshape(-1, 3, 3)
    aa = scipy.spatial.transform.Rotation.from_matrix(flat).as_rotvec()
    return aa.reshape(mats.shape[:-2] + (3,)).astype(np.float32)


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices."""
    flat = aa.reshape(-1, 3)
    mats = scipy.spatial.transform.Rotation.from_rotvec(flat).as_matrix()
    return mats.reshape(aa.shape[:-1] + (3, 3)).astype(np.float32)


def convert_actions_from_euler(actions_euler: np.ndarray, target_space: str) -> np.ndarray:
    """
    Convert (..., 7) euler actions to the target action space.

    Args:
        actions_euler: (..., 7) array with [pos(3), euler_xyz(3), gripper(1)]
        target_space: one of 'pos_euler', 'pos_rot6d', 'pos_axisangle'

    Returns:
        Converted actions array of shape (..., ACTION_SPACE_DIMS[target_space])
    """
    if target_space == "pos_euler":
        return actions_euler

    pos = actions_euler[..., :3]       # (..., 3)
    euler = actions_euler[..., 3:6]    # (..., 3)
    gripper = actions_euler[..., 6:7]  # (..., 1)

    mats = _euler_to_matrix(euler)     # (..., 3, 3)

    if target_space == "pos_rot6d":
        rot = _matrix_to_rot6d(mats)   # (..., 6)
        return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    elif target_space == "pos_axisangle":
        rot = _matrix_to_axis_angle(mats)  # (..., 3)
        return np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    else:
        raise ValueError("Unknown action_space: {}".format(target_space))


def convert_actions_to_euler(actions: np.ndarray, source_space: str) -> np.ndarray:
    """
    Convert actions back to euler representation.

    Args:
        actions: (..., D) array in source_space format
        source_space: one of 'pos_euler', 'pos_rot6d', 'pos_axisangle'

    Returns:
        Array of shape (..., 7) with [pos(3), euler_xyz(3), gripper(1)]
    """
    if source_space == "pos_euler":
        return actions

    pos = actions[..., :3]       # (..., 3)
    gripper = actions[..., -1:]  # (..., 1)

    if source_space == "pos_rot6d":
        mats = _rot6d_to_matrix(actions[..., 3:9])   # (..., 3, 3)
    elif source_space == "pos_axisangle":
        mats = _axis_angle_to_matrix(actions[..., 3:6])  # (..., 3, 3)
    else:
        raise ValueError("Unknown action_space: {}".format(source_space))

    euler = _matrix_to_euler(mats)  # (..., 3)
    return np.concatenate([pos, euler, gripper], axis=-1).astype(np.float32)


def get_rot_slice(action_space: str) -> tuple:
    """Return (start, end) index of the rotation component in an action vector."""
    if action_space in ("pos_euler", "pos_axisangle"):
        return (3, 6)
    elif action_space == "pos_rot6d":
        return (3, 9)
    else:
        raise ValueError("Unknown action_space: {}".format(action_space))


def get_rot_format_for_eval(action_space: str) -> str:
    """Return the rot_format string expected by eval_utils.compute_rot_err."""
    if action_space == "pos_euler":
        return "euler"       # caller must convert via euler_angles_to_matrix first
    elif action_space == "pos_rot6d":
        return "6d"
    elif action_space == "pos_axisangle":
        return "axis_angle"
    else:
        raise ValueError("Unknown action_space: {}".format(action_space))
