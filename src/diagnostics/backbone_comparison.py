"""Comparative Diagnostics: USFM (Vision Transformer) vs. ResNet-18 (CNN)."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import torch

from src.loaders.mhd_loader import RobotSweep
from src.loaders.preprocessor import prepare_sweep_batch
from src.models.usfm_bridge import BackboneComparisonExtractor


@dataclass
class BackboneEvaluationResult:
    backbone_name: str
    feature_dim: int
    pearson_r: float
    spearman_rho: float
    fwhm_mm: float
    crop_stability_cosine: float
    inference_time_ms_per_frame: float
    memory_mb: float


def compute_decorrelation_curve_and_fwhm(
    features: np.ndarray,
    positions: np.ndarray,
    max_dist: float = 40.0,
    bin_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute pairwise feature cosine similarity as a function of elevational distance."""
    norm_feat = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_feat, norm_feat.T)
    dist_matrix = cdist(positions, positions, metric="euclidean")

    bins = np.arange(0, max_dist + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    mean_sims = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        mask = (dist_matrix >= bins[i]) & (dist_matrix < bins[i + 1])
        if np.any(mask):
            mean_sims[i] = np.mean(sim_matrix[mask])
        else:
            mean_sims[i] = np.nan

    # Fill NaNs
    valid = ~np.isnan(mean_sims)
    if np.sum(valid) > 1:
        mean_sims = np.interp(bin_centers, bin_centers[valid], mean_sims[valid])

    # Compute FWHM (distance where similarity drops to 0.5)
    fwhm = max_dist
    for d, s in zip(bin_centers, mean_sims):
        if s <= 0.5:
            fwhm = float(d)
            break

    return bin_centers, mean_sims, float(fwhm)


def evaluate_backbone_comparison(
    extractor: BackboneComparisonExtractor,
    sweeps: List[RobotSweep],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, BackboneEvaluationResult]:
    """Evaluate and compare ResNet-18 vs. USFM across multiple sweeps."""
    rn_r_list, rn_rho_list, rn_fwhm_list, rn_crop_sim_list, rn_lat_list = [], [], [], [], []
    usfm_r_list, usfm_rho_list, usfm_fwhm_list, usfm_crop_sim_list, usfm_lat_list = [], [], [], [], []

    for sweep in sweeps:
        batch = prepare_sweep_batch(sweep.frames, device=device)
        images = batch["global_encoder_images"] # (1, N, 1, 224, 224)
        N = sweep.num_frames
        positions = sweep.transforms[:, :3, 3] # (N, 3)

        # 1. ResNet-18 Evaluation
        t0 = time.perf_counter()
        rn_pooled, _ = extractor.forward_resnet(images)
        t_rn = (time.perf_counter() - t0) * 1000.0 / N
        rn_features = rn_pooled[0].cpu().numpy()
        rn_norm = rn_features / (np.linalg.norm(rn_features, axis=-1, keepdims=True) + 1e-8)

        # Cumulative displacement along sweep
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        cum_dist = np.concatenate([[0.0], np.cumsum(diffs)])

        # Normalized feature displacement relative to frame 0
        rn_feat_dist = np.linalg.norm(rn_norm - rn_norm[0:1], axis=-1)
        r_rn, _ = pearsonr(cum_dist, rn_feat_dist)
        rho_rn, _ = spearmanr(cum_dist, rn_feat_dist)
        _, _, fwhm_rn = compute_decorrelation_curve_and_fwhm(rn_features, positions)

        rn_r_list.append(r_rn)
        rn_rho_list.append(rho_rn)
        rn_fwhm_list.append(fwhm_rn)
        rn_lat_list.append(t_rn)

        # 2. USFM Evaluation
        t0 = time.perf_counter()
        usfm_pooled, _ = extractor.forward_usfm(images)
        t_usfm = (time.perf_counter() - t0) * 1000.0 / N
        usfm_features = usfm_pooled[0].cpu().numpy()
        usfm_norm = usfm_features / (np.linalg.norm(usfm_features, axis=-1, keepdims=True) + 1e-8)

        usfm_feat_dist = np.linalg.norm(usfm_norm - usfm_norm[0:1], axis=-1)
        r_usfm, _ = pearsonr(cum_dist, usfm_feat_dist)
        rho_usfm, _ = spearmanr(cum_dist, usfm_feat_dist)
        _, _, fwhm_usfm = compute_decorrelation_curve_and_fwhm(usfm_features, positions)

        usfm_r_list.append(r_usfm)
        usfm_rho_list.append(rho_usfm)
        usfm_fwhm_list.append(fwhm_usfm)
        usfm_lat_list.append(t_usfm)

    # Crop stability (Start 50 vs Center Crop)
    sample_sweep = sweeps[0]
    from src.diagnostics.crop_sensitivity import apply_center_crop, apply_start50_crop
    std_img = apply_center_crop(sample_sweep.frames, target_size=(224, 224)).to(device)
    crop_img = apply_start50_crop(sample_sweep.frames, target_size=(224, 224)).to(device)

    rn_std, _ = extractor.forward_resnet(std_img)
    rn_crop, _ = extractor.forward_resnet(crop_img)
    rn_sim = float(np.mean(np.sum(
        (rn_std[0].cpu().numpy() / np.linalg.norm(rn_std[0].cpu().numpy(), axis=-1, keepdims=True)) *
        (rn_crop[0].cpu().numpy() / np.linalg.norm(rn_crop[0].cpu().numpy(), axis=-1, keepdims=True)),
        axis=-1,
    )))

    usfm_std, _ = extractor.forward_usfm(std_img)
    usfm_crop, _ = extractor.forward_usfm(crop_img)
    usfm_sim = float(np.mean(np.sum(
        (usfm_std[0].cpu().numpy() / np.linalg.norm(usfm_std[0].cpu().numpy(), axis=-1, keepdims=True)) *
        (usfm_crop[0].cpu().numpy() / np.linalg.norm(usfm_crop[0].cpu().numpy(), axis=-1, keepdims=True)),
        axis=-1,
    )))

    return {
        "ResNet-18 (CNN)": BackboneEvaluationResult(
            backbone_name="ResNet-18 (3D/2D CNN)",
            feature_dim=512,
            pearson_r=float(np.mean(rn_r_list)),
            spearman_rho=float(np.mean(rn_rho_list)),
            fwhm_mm=float(np.mean(rn_fwhm_list)),
            crop_stability_cosine=rn_sim,
            inference_time_ms_per_frame=float(np.mean(rn_lat_list)),
            memory_mb=11.7, # standard ResNet18 parameter footprint
        ),
        "USFM (Vision Transformer)": BackboneEvaluationResult(
            backbone_name="USFM (HVIT Hierarchical ViT)",
            feature_dim=256,
            pearson_r=float(np.mean(usfm_r_list)),
            spearman_rho=float(np.mean(usfm_rho_list)),
            fwhm_mm=float(np.mean(usfm_fwhm_list)),
            crop_stability_cosine=usfm_sim,
            inference_time_ms_per_frame=float(np.mean(usfm_lat_list)),
            memory_mb=86.2, # ViT-Base 12-block Transformer footprint
        ),
    }
