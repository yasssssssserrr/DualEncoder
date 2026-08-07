"""Geometry and SE(3) transformations utilities for ultrasound probe tracking."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def compute_relative_transforms(transforms: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame relative transforms T_rel[i] = inv(T[i]) @ T[i+1].
    
    Args:
        transforms: Array of shape (N, 4, 4) containing homogeneous transformation matrices.
        
    Returns:
        rel_transforms: Array of shape (N-1, 4, 4) containing relative step transforms.
    """
    N = len(transforms)
    if N < 2:
        return np.zeros((0, 4, 4), dtype=transforms.dtype)
    
    rel_transforms = np.zeros((N - 1, 4, 4), dtype=transforms.dtype)
    for i in range(N - 1):
        t_inv = np.linalg.inv(transforms[i])
        rel_transforms[i] = t_inv @ transforms[i + 1]
    return rel_transforms


def transform_to_pose_vector(transform_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform matrix to 6-DoF vector [tx, ty, tz, rx, ry, rz] (rotvec in radians)."""
    t = transform_matrix[:3, 3]
    r = Rotation.from_matrix(transform_matrix[:3, :3]).as_rotvec()
    return np.concatenate([t, r])


def pose_vector_to_transform(pose_vec: np.ndarray) -> np.ndarray:
    """Convert a 6-DoF vector [tx, ty, tz, rx, ry, rz] to a 4x4 transform matrix."""
    t = pose_vec[:3]
    rot = Rotation.from_rotvec(pose_vec[3:]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def compute_translation_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute Euclidean translation distance between two 4x4 transforms in mm."""
    pos1 = t1[:3, 3] if t1.shape == (4, 4) else t1[:3]
    pos2 = t2[:3, 3] if t2.shape == (4, 4) else t2[:3]
    return float(np.linalg.norm(pos1 - pos2))


def compute_rotation_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Compute geodesic rotation angle between two 3x3 rotation matrices in degrees."""
    if r1.shape == (4, 4):
        r1 = r1[:3, :3]
    if r2.shape == (4, 4):
        r2 = r2[:3, :3]
    r_rel = r1.T @ r2
    rot = Rotation.from_matrix(r_rel)
    angle_rad = rot.magnitude()
    return float(np.degrees(angle_rad))


def integrate_relative_poses(rel_poses: np.ndarray, init_pose: np.ndarray = None) -> np.ndarray:
    """Integrate sequence of relative 4x4 transforms or 6-DoF vectors into global trajectory.
    
    Args:
        rel_poses: (N-1, 4, 4) or (N-1, 6)
        init_pose: (4, 4) starting pose, defaults to identity.
        
    Returns:
        global_poses: (N, 4, 4) global trajectory.
    """
    N = len(rel_poses) + 1
    global_poses = np.zeros((N, 4, 4), dtype=np.float64)
    if init_pose is None:
        init_pose = np.eye(4, dtype=np.float64)
    global_poses[0] = init_pose
    
    for i in range(len(rel_poses)):
        if rel_poses[i].shape == (6,):
            t_step = pose_vector_to_transform(rel_poses[i])
        else:
            t_step = rel_poses[i]
        global_poses[i + 1] = global_poses[i] @ t_step
        
    return global_poses
