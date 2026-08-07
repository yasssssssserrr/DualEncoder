"""Trajectory reconstruction and pose evaluation against robot tracker ground truth."""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from src.utils.geometry import compute_translation_distance, compute_rotation_angle_deg


def pose_vec_to_mat(vec: np.ndarray) -> np.ndarray:
    """Convert DualTrack 6-DoF output [tx, ty, tz, rx_deg, ry_deg, rz_deg] to 4x4 matrix."""
    t = vec[:3]
    euler_deg = vec[3:]
    rot = R.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def mat_to_pose_vec(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to [tx, ty, tz, rx_deg, ry_deg, rz_deg]."""
    t = mat[:3, 3]
    euler_deg = R.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([t, euler_deg])


@dataclass
class TrajectoryMetrics:
    """Dataclass holding trajectory reconstruction metrics."""
    sweep_id: str
    num_frames: int
    mean_gpe_mm: float  # Mean Global Position Error along trajectory (mm)
    max_gpe_mm: float   # Maximum Global Position Error (mm)
    endpoint_error_mm: float  # Error at final frame (mm)
    mean_lpe_mm: float  # Mean Local (step-to-step) Error (mm)
    mean_rot_error_deg: float  # Mean rotation error (degrees)
    drift_percent: float  # (Endpoint Error / Total Trajectory Length) * 100
    pred_trajectory: np.ndarray  # (N, 4, 4)
    gt_trajectory: np.ndarray    # (N, 4, 4)


def evaluate_sweep_trajectory(
    pred_rel_vectors: np.ndarray | torch.Tensor,
    gt_transforms: np.ndarray,
    sweep_id: str = "sweep",
) -> TrajectoryMetrics:
    """Evaluate predicted relative pose vectors against ground-truth robot transforms.
    
    Args:
        pred_rel_vectors: Shape (N-1, 6) or (1, N-1, 6) containing predicted [tx, ty, tz, rx, ry, rz].
        gt_transforms: Shape (N, 4, 4) ground truth robot tool transforms.
        sweep_id: Name/ID of sweep.
        
    Returns:
        metrics: TrajectoryMetrics dataclass.
    """
    if isinstance(pred_rel_vectors, torch.Tensor):
        pred_rel_vectors = pred_rel_vectors.detach().cpu().numpy()
    if pred_rel_vectors.ndim == 3:
        pred_rel_vectors = pred_rel_vectors[0]

    N = len(gt_transforms)
    assert len(pred_rel_vectors) == N - 1, f"Expected {N-1} relative steps, got {len(pred_rel_vectors)}"

    # 1. Compute relative GT transforms
    gt_rel_transforms = np.zeros((N - 1, 4, 4), dtype=np.float64)
    gt_rel_vectors = np.zeros((N - 1, 6), dtype=np.float64)
    for i in range(N - 1):
        rel = np.linalg.inv(gt_transforms[i]) @ gt_transforms[i + 1]
        gt_rel_transforms[i] = rel
        gt_rel_vectors[i] = mat_to_pose_vec(rel)

    # 2. Integrate predicted relative steps into global trajectory (starting at GT pose 0)
    pred_glob = np.zeros((N, 4, 4), dtype=np.float64)
    pred_glob[0] = gt_transforms[0]
    for i in range(N - 1):
        step_mat = pose_vec_to_mat(pred_rel_vectors[i])
        pred_glob[i + 1] = pred_glob[i] @ step_mat

    # 3. Compute Position and Rotation Errors
    gpe_per_frame = np.zeros(N)
    rot_err_per_frame = np.zeros(N)
    for i in range(N):
        gpe_per_frame[i] = compute_translation_distance(pred_glob[i], gt_transforms[i])
        rot_err_per_frame[i] = compute_rotation_angle_deg(pred_glob[i], gt_transforms[i])

    lpe_per_step = np.zeros(N - 1)
    for i in range(N - 1):
        pred_step = pose_vec_to_mat(pred_rel_vectors[i])
        lpe_per_step[i] = compute_translation_distance(pred_step, gt_rel_transforms[i])

    # 4. Total Trajectory Length
    total_length = 0.0
    for i in range(N - 1):
        total_length += compute_translation_distance(gt_transforms[i], gt_transforms[i + 1])
    total_length = max(total_length, 1e-4)

    endpoint_error = float(gpe_per_frame[-1])
    drift_pct = float((endpoint_error / total_length) * 100.0)

    return TrajectoryMetrics(
        sweep_id=sweep_id,
        num_frames=N,
        mean_gpe_mm=float(np.mean(gpe_per_frame)),
        max_gpe_mm=float(np.max(gpe_per_frame)),
        endpoint_error_mm=endpoint_error,
        mean_lpe_mm=float(np.mean(lpe_per_step)),
        mean_rot_error_deg=float(np.mean(rot_err_per_frame)),
        drift_percent=drift_pct,
        pred_trajectory=pred_glob,
        gt_trajectory=gt_transforms,
    )
