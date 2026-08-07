"""Elevational speckle decorrelation curve and beam width analysis."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.geometry import compute_translation_distance


@dataclass
class DecorrelationStats:
    """Dataclass holding speckle decorrelation analysis results."""
    lags_frames: np.ndarray
    displacements_mm: np.ndarray
    cosine_similarities: np.ndarray
    fwhm_mm: float  # Full Width at Half Maximum (mm) where correlation drops to 0.5
    decay_length_mm: float  # Distance where correlation drops to 1/e (~0.368)


def compute_speckle_decorrelation(
    features: np.ndarray | torch.Tensor,
    transforms: np.ndarray,
    max_lag: int = 20,
) -> DecorrelationStats:
    """Compute feature similarity as a function of physical translation distance.
    
    Args:
        features: Shape (N, D) or (1, N, D).
        transforms: Shape (N, 4, 4) ground truth robot poses.
        max_lag: Maximum frame offset.
        
    Returns:
        stats: DecorrelationStats.
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    elif features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    # Normalize feature vectors for cosine similarity
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_feats = features / norms

    N = len(transforms)
    max_lag = min(max_lag, N - 1)

    lags = np.arange(0, max_lag + 1)
    sims = np.zeros(len(lags))
    disps_mm = np.zeros(len(lags))

    sims[0] = 1.0
    disps_mm[0] = 0.0

    for lag in range(1, max_lag + 1):
        lag_sims = []
        lag_disps = []
        for i in range(N - lag):
            j = i + lag
            dot_prod = float(np.dot(norm_feats[i], norm_feats[j]))
            disp = compute_translation_distance(transforms[i], transforms[j])
            lag_sims.append(dot_prod)
            lag_disps.append(disp)
        sims[lag] = float(np.mean(lag_sims))
        disps_mm[lag] = float(np.mean(lag_disps))

    # Calculate FWHM (displacement where similarity reaches 0.5)
    fwhm = float(np.nan)
    for k in range(len(sims) - 1):
        if sims[k] >= 0.5 and sims[k + 1] <= 0.5:
            # Linear interpolation
            frac = (0.5 - sims[k]) / (sims[k + 1] - sims[k] + 1e-8)
            fwhm = float(disps_mm[k] + frac * (disps_mm[k + 1] - disps_mm[k]))
            break
    if np.isnan(fwhm) and sims[-1] > 0.5:
        fwhm = float(disps_mm[-1])

    # Calculate decay length (1/e ~ 0.368)
    decay_len = float(np.nan)
    target = 1.0 / np.e
    for k in range(len(sims) - 1):
        if sims[k] >= target and sims[k + 1] <= target:
            frac = (target - sims[k]) / (sims[k + 1] - sims[k] + 1e-8)
            decay_len = float(disps_mm[k] + frac * (disps_mm[k + 1] - disps_mm[k]))
            break
    if np.isnan(decay_len) and sims[-1] > target:
        decay_len = float(disps_mm[-1])

    return DecorrelationStats(
        lags_frames=lags,
        displacements_mm=disps_mm,
        cosine_similarities=sims,
        fwhm_mm=fwhm,
        decay_length_mm=decay_len,
    )
