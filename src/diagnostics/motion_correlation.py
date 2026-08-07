"""Motion sensitivity and displacement correlation analysis for feature representations."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from src.utils.geometry import compute_translation_distance, compute_rotation_angle_deg


@dataclass
class CorrelationMetrics:
    """Dataclass holding correlation statistics between feature distances and physical motion."""
    stage_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    mean_feature_dist: float
    monotonicity_score: float  # Percentage of strictly increasing steps


def compute_feature_distance_matrix(
    features: np.ndarray | torch.Tensor,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute pairwise distance matrix between frames in feature space.
    
    Args:
        features: Shape (N, D) where N is number of frames, D is feature dimension.
        metric: 'cosine', 'euclidean', or 'correlation'.
        
    Returns:
        dist_mat: Shape (N, N) distance matrix.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    elif features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    return cdist(features, features, metric=metric)


def compute_physical_distance_matrix(
    transforms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise translation (mm) and rotation (deg) matrices from 4x4 transforms.
    
    Args:
        transforms: Shape (N, 4, 4) homogeneous transforms.
        
    Returns:
        trans_dist: Shape (N, N) translation distances in mm.
        rot_dist: Shape (N, N) rotation distances in degrees.
    """
    N = len(transforms)
    trans_dist = np.zeros((N, N), dtype=np.float64)
    rot_dist = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            td = compute_translation_distance(transforms[i], transforms[j])
            rd = compute_rotation_angle_deg(transforms[i], transforms[j])
            trans_dist[i, j] = trans_dist[j, i] = td
            rot_dist[i, j] = rot_dist[j, i] = rd

    return trans_dist, rot_dist


def evaluate_motion_correlation(
    features: np.ndarray | torch.Tensor,
    transforms: np.ndarray,
    stage_name: str = "features",
    max_lag: int = 15,
) -> CorrelationMetrics:
    """Evaluate how well feature distance correlates with physical robot displacement.
    
    Args:
        features: (N, D) feature embeddings.
        transforms: (N, 4, 4) robot ground-truth transforms.
        stage_name: Label for this feature level.
        max_lag: Maximum frame step delta |i - j| to evaluate.
        
    Returns:
        metrics: CorrelationMetrics dataclass.
    """
    feat_dist_mat = compute_feature_distance_matrix(features, metric="cosine")
    trans_dist_mat, _ = compute_physical_distance_matrix(transforms)

    N = len(transforms)
    feat_dists = []
    phys_dists = []

    for lag in range(1, min(max_lag + 1, N)):
        for i in range(N - lag):
            j = i + lag
            feat_dists.append(feat_dist_mat[i, j])
            phys_dists.append(trans_dist_mat[i, j])

    feat_dists = np.array(feat_dists)
    phys_dists = np.array(phys_dists)

    # Clean NaNs or constant values
    valid_mask = np.isfinite(feat_dists) & np.isfinite(phys_dists)
    feat_dists = feat_dists[valid_mask]
    phys_dists = phys_dists[valid_mask]

    if len(feat_dists) < 3 or np.std(feat_dists) < 1e-7 or np.std(phys_dists) < 1e-7:
        return CorrelationMetrics(
            stage_name=stage_name,
            pearson_r=0.0,
            pearson_p=1.0,
            spearman_rho=0.0,
            spearman_p=1.0,
            mean_feature_dist=float(np.mean(feat_dists)) if len(feat_dists) > 0 else 0.0,
            monotonicity_score=0.0,
        )

    pr, pp = pearsonr(phys_dists, feat_dists)
    sr, sp = spearmanr(phys_dists, feat_dists)

    # Monotonicity check: For adjacent frame steps from anchor frame 0, does distance increase?
    anchor_feat_dist = feat_dist_mat[0, :min(max_lag, N)]
    anchor_phys_dist = trans_dist_mat[0, :min(max_lag, N)]
    diffs = np.diff(anchor_feat_dist)
    monotonic_ratio = float(np.mean(diffs >= -1e-4)) if len(diffs) > 0 else 0.0

    return CorrelationMetrics(
        stage_name=stage_name,
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        mean_feature_dist=float(np.mean(feat_dists)),
        monotonicity_score=monotonic_ratio,
    )
