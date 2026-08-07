"""Official Dense Displacement Field (DDF) 5-point landmark metrics (LPE, GPE, FDR, Max Drift)."""
from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import numpy as np
import torch

from src.utils.geometry import compute_translation_distance


def make_image_points(
    H: int, W: int, mode: Literal["all-pts", "5pt-landmark", "corners"] = "5pt-landmark"
) -> np.ndarray:
    """Generate pixel coordinates in 4D homogeneous coordinates."""
    if mode == "all-pts":
        ind = np.indices((W, H)).transpose(1, 2, 0).reshape(-1, 2)
    elif mode == "5pt-landmark":
        ind = np.array(
            [
                [0, 0],
                [W, 0],
                [0, H],
                [W, H],
                [W // 2, H // 2],
            ]
        )
    elif mode == "corners":
        ind = np.array([[0, 0], [W, 0], [0, H], [W, H]])
    else:
        raise ValueError(f"Unexpected mode {mode}")

    extra_coords = np.array([[0, 1]]).repeat(ind.shape[0], 0)
    ind = np.concatenate([ind, extra_coords], axis=-1)
    return ind.T  # Shape: (4, num_points)


def cal_global_ddf(
    transformation_global: torch.Tensor,
    tform_calib_scale: torch.Tensor,
    image_points: torch.Tensor,
) -> torch.Tensor:
    """Generate global DDF in mm for image points with respect to first frame.
    
    Args:
        transformation_global: Shape (N-1, 4, 4)
        tform_calib_scale: Shape (4, 4) pixel-to-image calibration matrix
        image_points: Shape (4, P)
        
    Returns:
        global_ddf: Shape (N-1, 3, P) in mm
    """
    global_allpts = torch.matmul(
        transformation_global, torch.matmul(tform_calib_scale, image_points)
    )
    base_pts = torch.matmul(tform_calib_scale, image_points)[0:3, :].expand(
        global_allpts.shape[0], -1, -1
    )
    return global_allpts[:, 0:3, :] - base_pts


def cal_local_ddf(
    transformation_local: torch.Tensor,
    tform_calib_scale: torch.Tensor,
    image_points: torch.Tensor,
) -> torch.Tensor:
    """Generate local DDF in mm for image points with respect to immediately previous frame.
    
    Args:
        transformation_local: Shape (N-1, 4, 4)
        tform_calib_scale: Shape (4, 4)
        image_points: Shape (4, P)
        
    Returns:
        local_ddf: Shape (N-1, 3, P) in mm
    """
    local_allpts = torch.matmul(
        transformation_local, torch.matmul(tform_calib_scale, image_points)
    )
    base_pts = torch.matmul(tform_calib_scale, image_points)[0:3, :].expand(
        local_allpts.shape[0], -1, -1
    )
    return local_allpts[:, 0:3, :] - base_pts


@dataclass
class DDFMetrics:
    """Official DualTrack / TUS-REC DDF Metrics."""
    lpe_mm: float           # Local Point Error (mm)
    lpe_um: float           # Local Point Error in micrometers (µm)
    max_lpe_mm: float       # Maximum single-step Local Point Error (mm)
    gpe_mm: float           # Global Point Error (mm)
    max_gpe_mm: float       # Maximum Global Point Error (mm)
    final_drift_rate_pct: float  # Final Drift Rate (FDR in %)
    total_trajectory_length_mm: float


def compute_official_ddf_metrics(
    pred_transforms_glob: np.ndarray | torch.Tensor,
    gt_transforms_glob: np.ndarray | torch.Tensor,
    pixel_to_mm_spacing: Tuple[float, float] = (0.0786, 0.0786),
    image_shape: Tuple[int, int] = (512, 485),  # (H, W)
    mode: Literal["all-pts", "5pt-landmark", "corners"] = "5pt-landmark",
) -> DDFMetrics:
    """Compute official 5-point landmark DDF metrics.
    
    Args:
        pred_transforms_glob: Shape (N, 4, 4) predicted global tracking
        gt_transforms_glob: Shape (N, 4, 4) ground truth global tracking
        pixel_to_mm_spacing: (spacing_y, spacing_x) in mm/pixel
        image_shape: (H, W) in pixels
        mode: Landmark mode
    """
    if isinstance(pred_transforms_glob, np.ndarray):
        pred_glob_t = torch.tensor(pred_transforms_glob, dtype=torch.float32)
    else:
        pred_glob_t = pred_transforms_glob.float()

    if isinstance(gt_transforms_glob, np.ndarray):
        gt_glob_t = torch.tensor(gt_transforms_glob, dtype=torch.float32)
    else:
        gt_glob_t = gt_transforms_glob.float()

    N = len(gt_glob_t)
    H, W = image_shape

    # Construct calibration scaling matrix (pixel -> mm)
    calib = torch.eye(4, dtype=torch.float32)
    calib[0, 0] = pixel_to_mm_spacing[1]  # x spacing
    calib[1, 1] = pixel_to_mm_spacing[0]  # y spacing

    # Normalize tracking relative to frame 0
    gt_0_inv = torch.linalg.inv(gt_glob_t[0])
    pred_0_inv = torch.linalg.inv(pred_glob_t[0])
    
    gt_glob_norm = torch.matmul(gt_0_inv.unsqueeze(0), gt_glob_t)
    pred_glob_norm = torch.matmul(pred_0_inv.unsqueeze(0), pred_glob_t)

    # Local step transforms
    gt_loc = torch.zeros((N - 1, 4, 4), dtype=torch.float32)
    pred_loc = torch.zeros((N - 1, 4, 4), dtype=torch.float32)
    for i in range(N - 1):
        gt_loc[i] = torch.matmul(torch.linalg.inv(gt_glob_t[i]), gt_glob_t[i + 1])
        pred_loc[i] = torch.matmul(torch.linalg.inv(pred_glob_t[i]), pred_glob_t[i + 1])

    # Image points
    img_pts = torch.tensor(make_image_points(H, W, mode=mode), dtype=torch.float32)

    # Calculate DDFs
    pred_glob_ddf = cal_global_ddf(pred_glob_norm[1:], calib, img_pts)
    gt_glob_ddf = cal_global_ddf(gt_glob_norm[1:], calib, img_pts)

    pred_loc_ddf = cal_local_ddf(pred_loc, calib, img_pts)
    gt_loc_ddf = cal_local_ddf(gt_loc, calib, img_pts)

    # Errors
    # Shape: (N-1, P)
    global_err = torch.sqrt(torch.sum((pred_glob_ddf - gt_glob_ddf) ** 2, dim=1)).mean(dim=-1)
    local_err = torch.sqrt(torch.sum((pred_loc_ddf - gt_loc_ddf) ** 2, dim=1)).mean(dim=-1)

    # Total Trajectory Length
    total_length = 0.0
    gt_glob_np = gt_glob_t.numpy()
    for i in range(N - 1):
        total_length += compute_translation_distance(gt_glob_np[i], gt_glob_np[i + 1])
    total_length = max(total_length, 1e-4)

    endpoint_err = float(global_err[-1].item())
    fdr_pct = (endpoint_err / total_length) * 100.0

    avg_lpe = float(local_err.mean().item())
    avg_gpe = float(global_err.mean().item())

    return DDFMetrics(
        lpe_mm=avg_lpe,
        lpe_um=avg_lpe * 1000.0,
        max_lpe_mm=float(local_err.max().item()),
        gpe_mm=avg_gpe,
        max_gpe_mm=float(global_err.max().item()),
        final_drift_rate_pct=float(fdr_pct),
        total_trajectory_length_mm=float(total_length),
    )
