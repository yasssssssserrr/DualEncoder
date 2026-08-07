"""Differentiable Calibration Consistency Losses & Bone-Weighted Optimization.

Implements differentiable spatial transformation, bone-cortex attention weighting,
and feature-metric loss functions for ultrasound probe calibration compensation.
"""
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_bone_cortex_attention_mask(
    image: Union[np.ndarray, torch.Tensor],
    intensity_threshold: float = 0.55,
    shadow_suppression: bool = True,
) -> torch.Tensor:
    """Computes a 2D spatial attention weight mask prioritizing hyperechoic bone interfaces.
    
    Args:
        image: (H, W) or (B, 1, H, W) ultrasound B-mode image in [0, 1].
        intensity_threshold: Minimum normalized intensity for cortical bone reflection.
        shadow_suppression: Whether to suppress acoustic shadow regions.
        
    Returns:
        bone_mask: Tensor of same spatial shape in [0, 1].
    """
    if isinstance(image, np.ndarray):
        img_t = torch.from_numpy(np.array(image, copy=True)).float()
    else:
        img_t = image.clone().float()

    if img_t.ndim == 2:
        img_t = img_t.unsqueeze(0).unsqueeze(0)
    elif img_t.ndim == 3:
        img_t = img_t.unsqueeze(1)

    # Normalize
    img_norm = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8)

    # 1. Hyperechoic reflection response (soft sigmoid around threshold)
    reflection_weight = torch.sigmoid((img_norm - intensity_threshold) * 12.0)

    # 2. Vertical gradient (Sobel y) to highlight top-surface cortical reflection interface
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=img_t.device).view(1, 1, 3, 3)
    grad_y = F.conv2d(img_norm, sobel_y, padding=1)
    kortex_edge = torch.clamp(-grad_y, min=0.0)  # Transition from dark to bright (top boundary)

    bone_mask = reflection_weight * (1.0 + 2.0 * kortex_edge)
    bone_mask = bone_mask / (bone_mask.max() + 1e-8)
    return bone_mask


def create_2d_rigid_affine_matrix(
    tx: torch.Tensor,
    ty: torch.Tensor,
    theta_rad: torch.Tensor,
) -> torch.Tensor:
    """Creates a differentiable 2D affine transformation matrix (B, 2, 3) for grid_sample.
    
    Args:
        tx: Horizontal translation in normalized coordinates [-1, 1].
        ty: Vertical translation in normalized coordinates [-1, 1].
        theta_rad: In-plane rotation angle in radians.
        
    Returns:
        affine_mat: (B, 2, 3) tensor.
    """
    B = tx.shape[0] if tx.ndim > 0 else 1
    cos_t = torch.cos(theta_rad).view(B, 1)
    sin_t = torch.sin(theta_rad).view(B, 1)
    tx_col = tx.view(B, 1)
    ty_col = ty.view(B, 1)

    row1 = torch.cat([cos_t, -sin_t, tx_col], dim=1)  # (B, 3)
    row2 = torch.cat([sin_t, cos_t, ty_col], dim=1)   # (B, 3)
    affine_mat = torch.stack([row1, row2], dim=1)     # (B, 2, 3)
    return affine_mat


def differentiable_spatial_warp_2d(
    feature_map: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    theta_rad: torch.Tensor,
) -> torch.Tensor:
    """Warps spatial feature maps using a differentiable 2D rigid transformation.
    
    Args:
        feature_map: (B, C, H, W) spatial features.
        tx, ty: Translations.
        theta_rad: Rotation in radians.
        
    Returns:
        warped_features: (B, C, H, W) transformed feature map with active grad.
    """
    affine_mat = create_2d_rigid_affine_matrix(tx, ty, theta_rad)
    grid = F.affine_grid(affine_mat, feature_map.size(), align_corners=False)
    warped = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped


class BoneWeightedCalibrationLoss(nn.Module):
    """Bone-prioritized spatial feature consistency loss for calibration optimization."""

    def __init__(self, metric: str = "cosine", bone_boost_factor: float = 3.0):
        super().__init__()
        self.metric = metric
        self.bone_boost_factor = bone_boost_factor

    def forward(
        self,
        fmap_a: torch.Tensor,
        fmap_b_warped: torch.Tensor,
        bone_weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes weighted feature consistency loss between target and warped source frames.
        
        Args:
            fmap_a: Target spatial feature map (B, C, H, W).
            fmap_b_warped: Warped source spatial feature map (B, C, H, W).
            bone_weight_mask: Optional (B, 1, H, W) attention mask.
            
        Returns:
            loss: Scalar differentiable loss tensor.
        """
        # Downsample weight mask to match feature map resolution if needed
        if bone_weight_mask is not None and bone_weight_mask.shape[-2:] != fmap_a.shape[-2:]:
            weight = F.interpolate(bone_weight_mask, size=fmap_a.shape[-2:], mode="bilinear", align_corners=False)
        elif bone_weight_mask is not None:
            weight = bone_weight_mask
        else:
            weight = torch.ones((fmap_a.shape[0], 1, fmap_a.shape[2], fmap_a.shape[3]), device=fmap_a.device)

        # Apply bone boost weighting
        weight = 1.0 + self.bone_boost_factor * weight

        if self.metric == "cosine":
            # Cosine similarity along channel dimension C
            cos_sim = F.cosine_similarity(fmap_a, fmap_b_warped, dim=1, eps=1e-8)  # (B, H, W)
            loss_map = 1.0 - cos_sim  # In [0, 2]
            weighted_loss = (loss_map.unsqueeze(1) * weight).sum() / (weight.sum() + 1e-8)
            return weighted_loss

        elif self.metric == "mse":
            diff_sq = (fmap_a - fmap_b_warped) ** 2  # (B, C, H, W)
            weighted_loss = (diff_sq * weight).sum() / (weight.sum() * fmap_a.shape[1] + 1e-8)
            return weighted_loss

        elif self.metric == "ncc":
            # Normalized Cross Correlation
            a_c = fmap_a - fmap_a.mean(dim=(2, 3), keepdim=True)
            b_c = fmap_b_warped - fmap_b_warped.mean(dim=(2, 3), keepdim=True)
            num = (a_c * b_c * weight).sum(dim=(1, 2, 3))
            den = torch.sqrt((a_c**2 * weight).sum(dim=(1, 2, 3)) * (b_c**2 * weight).sum(dim=(1, 2, 3)) + 1e-8)
            return (1.0 - num / den).mean()

        else:
            raise ValueError(f"Unknown metric: {self.metric}")
