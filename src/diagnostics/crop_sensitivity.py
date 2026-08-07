"""Preprocessing and Crop Sensitivity Analysis in Feature Embedding Space."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.feature_extractors import DualTrackFeatureExtractor


def apply_center_crop(frames: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Standard Center Crop: Crop max square from center and resize."""
    N, H, W = frames.shape
    crop_size = min(H, W)
    y_start = (H - crop_size) // 2
    x_start = (W - crop_size) // 2
    cropped = frames[:, y_start : y_start + crop_size, x_start : x_start + crop_size]
    
    resized = np.zeros((N, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(N):
        resized[i] = cv2.resize(cropped[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(resized / 255.0).unsqueeze(0).unsqueeze(2)  # (1, N, 1, H, W)
    return tensor


def apply_start50_crop(frames: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Start 50 Crop: Remove top 50 probe surface reverberation artifact pixels."""
    N, H, W = frames.shape
    cropped = frames[:, 50:, :]  # (N, H-50, W)
    
    # Square crop from remaining
    H_new = H - 50
    crop_size = min(H_new, W)
    y_start = (H_new - crop_size) // 2
    x_start = (W - crop_size) // 2
    cropped = cropped[:, y_start : y_start + crop_size, x_start : x_start + crop_size]
    
    resized = np.zeros((N, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(N):
        resized[i] = cv2.resize(cropped[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(resized / 255.0).unsqueeze(0).unsqueeze(2)
    return tensor


def apply_direct_resize(frames: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Direct Full-Field Resize without cropping (anamorphic squeeze)."""
    N, H, W = frames.shape
    resized = np.zeros((N, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(N):
        resized[i] = cv2.resize(frames[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(resized / 255.0).unsqueeze(0).unsqueeze(2)
    return tensor


def compute_cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """Compute mean cosine similarity between two feature sets."""
    f1_flat = f1.reshape(f1.shape[0], -1)
    f2_flat = f2.reshape(f2.shape[0], -1)
    
    norm1 = np.linalg.norm(f1_flat, axis=-1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(f2_flat, axis=-1, keepdims=True) + 1e-8
    
    sims = np.sum((f1_flat / norm1) * (f2_flat / norm2), axis=-1)
    return float(np.mean(sims))


@dataclass
class CropSensitivityMetrics:
    crop_name: str
    stage1_cosine_sim: float
    stage2_cosine_sim: float
    stage3_cosine_sim: float
    global_cosine_sim: float


def evaluate_crop_sensitivity(
    extractor: DualTrackFeatureExtractor,
    frames: np.ndarray,
    crop_variations: Dict[str, torch.Tensor],
) -> List[CropSensitivityMetrics]:
    """Compare feature representations from various crops against standard center crop."""
    extractor.eval()
    device = extractor.device

    # Standard reference (Center Crop)
    std_local = apply_center_crop(frames, target_size=(256, 256)).to(device)
    std_global = apply_center_crop(frames, target_size=(224, 224)).to(device)

    with torch.no_grad():
        std_s1 = extractor.extract_stage1_cnn(std_local, pool=True).cpu().numpy()[0]
        std_s2 = extractor.extract_stage2_vit_cls(std_local).cpu().numpy()[0]
        std_s3 = extractor.extract_stage3_temporal(std_local, project_to_decoder_dim=False).cpu().numpy()[0]
        std_glob, _ = extractor.extract_global_context(std_global)
        std_glob = std_glob.cpu().numpy()[0]

    results = []
    for crop_name, (test_loc, test_glob) in crop_variations.items():
        test_loc = test_loc.to(device)
        test_glob = test_glob.to(device)

        with torch.no_grad():
            t_s1 = extractor.extract_stage1_cnn(test_loc, pool=True).cpu().numpy()[0]
            t_s2 = extractor.extract_stage2_vit_cls(test_loc).cpu().numpy()[0]
            t_s3 = extractor.extract_stage3_temporal(test_loc, project_to_decoder_dim=False).cpu().numpy()[0]
            t_glob, _ = extractor.extract_global_context(test_glob)
            t_glob = t_glob.cpu().numpy()[0]

        sim_s1 = compute_cosine_similarity(std_s1, t_s1)
        sim_s2 = compute_cosine_similarity(std_s2, t_s2)
        sim_s3 = compute_cosine_similarity(std_s3, t_s3)
        sim_glob = compute_cosine_similarity(std_glob, t_glob)

        results.append(
            CropSensitivityMetrics(
                crop_name=crop_name,
                stage1_cosine_sim=sim_s1,
                stage2_cosine_sim=sim_s2,
                stage3_cosine_sim=sim_s3,
                global_cosine_sim=sim_glob,
            )
        )

    return results
