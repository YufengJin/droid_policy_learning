"""
Evaluation utilities for computing errors between predicted and actual values.
Supports position, rotation, and joint space error computation.
"""
import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils


def compute_pos_err(predicted_pos, actual_pos):
    """
    Compute position error between predicted and actual positions.
    
    Args:
        predicted_pos: Predicted positions, shape (*, 3) where * can be any dimensions
        actual_pos: Actual positions, shape (*, 3) where * can be any dimensions
    
    Returns:
        dict: Dictionary containing position error statistics:
            - 'error': Position error vectors (predicted - actual), shape (*, 3)
            - 'magnitude': L2 norm of position error for each sample, shape (*,)
            - 'mse': Mean squared error (scalar)
            - 'mean': Mean error magnitude (scalar)
            - 'max': Maximum error magnitude (scalar)
            - 'min': Minimum error magnitude (scalar)
            - 'std': Standard deviation of error magnitude (scalar)
    """
    # Convert to numpy if needed
    if isinstance(predicted_pos, torch.Tensor):
        predicted_pos = TensorUtils.to_numpy(predicted_pos)
    if isinstance(actual_pos, torch.Tensor):
        actual_pos = TensorUtils.to_numpy(actual_pos)
    
    # Compute position error
    pos_error = predicted_pos - actual_pos  # (*, 3)
    pos_error_squared = pos_error ** 2  # (*, 3)
    
    # Compute MSE for each dimension, then average across dimensions
    mse_per_dim = pos_error_squared.mean(axis=tuple(range(len(pos_error_squared.shape) - 1)))  # (3,)
    mse = mse_per_dim.mean()  # scalar
    
    # Compute statistics across all samples
    pos_error_flat = pos_error.reshape(-1, 3)  # (N, 3) where N = total number of samples
    pos_error_magnitude = np.linalg.norm(pos_error_flat, axis=1)  # (N,) - L2 norm for each sample
    
    return {
        'error': pos_error,
        'magnitude': pos_error_magnitude,
        'mse': float(mse),
        'mean': float(pos_error_magnitude.mean()),
        'max': float(pos_error_magnitude.max()),
        'min': float(pos_error_magnitude.min()),
        'std': float(pos_error_magnitude.std()),
    }


def compute_rot_err(predicted_rot, actual_rot, rot_format='6d'):
    """
    Compute rotation error in radians between predicted and actual rotations.
    
    Args:
        predicted_rot: Predicted rotations, shape (*, D) where D depends on rot_format
        actual_rot: Actual rotations, shape (*, D) where D depends on rot_format
        rot_format: Rotation format, one of:
            - '6d': 6D rotation representation (default), shape (*, 6)
            - 'matrix': Rotation matrices, shape (*, 3, 3)
            - 'axis_angle': Axis-angle representation, shape (*, 3)
    
    Returns:
        dict: Dictionary containing rotation error statistics:
            - 'error_rad': Rotation error in radians for each sample, shape (*,)
            - 'mean': Mean rotation error in radians (scalar)
            - 'max': Maximum rotation error in radians (scalar)
            - 'min': Minimum rotation error in radians (scalar)
            - 'std': Standard deviation of rotation error in radians (scalar)
            - 'mse': Mean squared error of 6D representation (only for '6d' format)
    """
    # Convert to torch tensors if needed
    if isinstance(predicted_rot, np.ndarray):
        predicted_rot = torch.from_numpy(predicted_rot).float()
    elif not isinstance(predicted_rot, torch.Tensor):
        predicted_rot = torch.tensor(predicted_rot).float()
    
    if isinstance(actual_rot, np.ndarray):
        actual_rot = torch.from_numpy(actual_rot).float()
    elif not isinstance(actual_rot, torch.Tensor):
        actual_rot = torch.tensor(actual_rot).float()
    
    # Ensure tensors are on the same device (use CPU by default for evaluation)
    if predicted_rot.device != actual_rot.device:
        actual_rot = actual_rot.to(predicted_rot.device)
    
    # Convert to rotation matrices
    if rot_format == '6d':
        # Convert 6D representation to rotation matrices
        pred_mat = TorchUtils.rotation_6d_to_matrix(predicted_rot)
        actual_mat = TorchUtils.rotation_6d_to_matrix(actual_rot)
    elif rot_format == 'matrix':
        pred_mat = predicted_rot
        actual_mat = actual_rot
    elif rot_format == 'axis_angle':
        # Convert axis-angle to rotation matrices
        pred_mat = TorchUtils.axis_angle_to_matrix(predicted_rot)
        actual_mat = TorchUtils.axis_angle_to_matrix(actual_rot)
    else:
        raise ValueError(f"Unsupported rotation format: {rot_format}. Must be one of: '6d', 'matrix', 'axis_angle'")
    
    # Compute relative rotation: R_rel = R_pred^T * R_actual
    # This gives the rotation needed to go from predicted to actual
    pred_mat_T = pred_mat.transpose(-2, -1)  # (*, 3, 3)
    rel_rot = torch.matmul(pred_mat_T, actual_mat)  # (*, 3, 3)
    
    # Convert relative rotation to axis-angle representation
    # The magnitude of the axis-angle vector is the rotation angle in radians
    rel_axis_angle = TorchUtils.matrix_to_axis_angle(rel_rot)  # (*, 3)
    rot_error_rad = torch.norm(rel_axis_angle, p=2, dim=-1)  # (*,) - angle in radians
    
    # Convert to numpy for statistics
    rot_error_rad_np = rot_error_rad.detach().cpu().numpy()
    if isinstance(rot_error_rad_np, np.ndarray) and rot_error_rad_np.size == 1:
        rot_error_rad_np = rot_error_rad_np.item()
    rot_error_rad_flat = rot_error_rad_np.flatten() if isinstance(rot_error_rad_np, np.ndarray) else np.array([rot_error_rad_np])
    
    result = {
        'error_rad': rot_error_rad_np,
        'mean': float(rot_error_rad_flat.mean()),
        'max': float(rot_error_rad_flat.max()),
        'min': float(rot_error_rad_flat.min()),
        'std': float(rot_error_rad_flat.std()),
    }
    
    # For 6D format, also compute MSE of the 6D representation
    if rot_format == '6d':
        rot_error_6d = predicted_rot - actual_rot  # (*, 6)
        rot_error_squared = rot_error_6d ** 2  # (*, 6)
        mse_per_dim = rot_error_squared.mean(axis=tuple(range(len(rot_error_squared.shape) - 1)))  # (6,)
        mse = mse_per_dim.mean()  # scalar
        result['mse'] = float(mse.detach().cpu().item() if isinstance(mse, torch.Tensor) else mse)
    
    return result


def compute_joint_err(predicted_joint, actual_joint):
    """
    Compute joint space error between predicted and actual joint positions/velocities.
    
    Args:
        predicted_joint: Predicted joint values, shape (*, D) where D is number of joints
        actual_joint: Actual joint values, shape (*, D) where D is number of joints
    
    Returns:
        dict: Dictionary containing joint error statistics:
            - 'error': Joint error vectors (predicted - actual), shape (*, D)
            - 'magnitude': L2 norm of joint error for each sample, shape (*,)
            - 'mse': Mean squared error (scalar)
            - 'mean': Mean error magnitude (scalar)
            - 'max': Maximum error magnitude (scalar)
            - 'min': Minimum error magnitude (scalar)
            - 'std': Standard deviation of error magnitude (scalar)
            - 'per_joint_mse': MSE for each joint, shape (D,)
            - 'per_joint_mean': Mean error for each joint, shape (D,)
    """
    # Convert to numpy if needed
    if isinstance(predicted_joint, torch.Tensor):
        predicted_joint = TensorUtils.to_numpy(predicted_joint)
    if isinstance(actual_joint, torch.Tensor):
        actual_joint = TensorUtils.to_numpy(actual_joint)
    
    # Compute joint error
    joint_error = predicted_joint - actual_joint  # (*, D)
    joint_error_squared = joint_error ** 2  # (*, D)
    
    # Compute MSE for each joint, then average across joints
    mse_per_joint = joint_error_squared.mean(axis=tuple(range(len(joint_error_squared.shape) - 1)))  # (D,)
    mse = mse_per_joint.mean()  # scalar
    
    # Compute statistics across all samples
    joint_error_flat = joint_error.reshape(-1, joint_error.shape[-1])  # (N, D) where N = total number of samples
    joint_error_magnitude = np.linalg.norm(joint_error_flat, axis=1)  # (N,) - L2 norm for each sample
    
    # Per-joint mean error (absolute value)
    per_joint_mean = np.abs(joint_error_flat).mean(axis=0)  # (D,)
    
    return {
        'error': joint_error,
        'magnitude': joint_error_magnitude,
        'mse': float(mse),
        'mean': float(joint_error_magnitude.mean()),
        'max': float(joint_error_magnitude.max()),
        'min': float(joint_error_magnitude.min()),
        'std': float(joint_error_magnitude.std()),
        'per_joint_mse': mse_per_joint.tolist(),
        'per_joint_mean': per_joint_mean.tolist(),
    }
