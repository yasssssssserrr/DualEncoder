"""Calibration Compensation & Differentiable Feature Alignment Module."""
from src.calibration.spatial_feature_extractor import (
    DualTrackSpatialFeatureExtractor,
    SpatialFeatureOutputs,
    IntermediateLayerHook,
)
from src.calibration.loss_functions import (
    compute_bone_cortex_attention_mask,
    differentiable_spatial_warp_2d,
    create_2d_rigid_affine_matrix,
    BoneWeightedCalibrationLoss,
)

__all__ = [
    "DualTrackSpatialFeatureExtractor",
    "SpatialFeatureOutputs",
    "IntermediateLayerHook",
    "compute_bone_cortex_attention_mask",
    "differentiable_spatial_warp_2d",
    "create_2d_rigid_affine_matrix",
    "BoneWeightedCalibrationLoss",
]
