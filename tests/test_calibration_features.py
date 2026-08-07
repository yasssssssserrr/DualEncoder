"""Unit tests for Differentiable Layer 2 Spatial Feature Extractor and Calibration Loss."""
import numpy as np
import torch
import pytest
from src.calibration.spatial_feature_extractor import DualTrackSpatialFeatureExtractor, IntermediateLayerHook
from src.calibration.loss_functions import (
    compute_bone_cortex_attention_mask,
    prepare_joint_binary_mask,
    create_2d_rigid_affine_matrix,
    differentiable_spatial_warp_2d,
    BoneWeightedCalibrationLoss,
)


@pytest.fixture(scope="module")
def spatial_extractor():
    return DualTrackSpatialFeatureExtractor(device="cpu")


def test_spatial_extractor_initialization(spatial_extractor):
    assert spatial_extractor is not None
    assert spatial_extractor.global_resnet is not None
    for p in spatial_extractor.parameters():
        assert not p.requires_grad


def test_forward_layer2_shape_and_gradient(spatial_extractor):
    # Input tensor with requires_grad
    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    fmap = spatial_extractor.forward_layer2(x)

    # Layer 2 has 128 channels and 56x56 spatial resolution (fine-grained bone detail)
    assert fmap.shape == (2, 128, 56, 56)
    assert fmap.requires_grad

    # Test gradient flow back to input x
    loss = fmap.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 1, 224, 224)


def test_forward_multiscale(spatial_extractor):
    x = torch.randn(1, 1, 224, 224)
    out = spatial_extractor.forward_multiscale(x, layers=("layer1", "layer2", "layer3", "layer4"))

    assert "layer1" in out and out["layer1"].shape == (1, 64, 112, 112)
    assert "layer2" in out and out["layer2"].shape == (1, 128, 56, 56)
    assert "layer3" in out and out["layer3"].shape == (1, 256, 28, 28)
    assert "layer4" in out and out["layer4"].shape == (1, 512, 14, 14)


def test_bone_attention_mask():
    img = torch.zeros(224, 224)
    img[100:110, 50:150] = 0.9  # Bright bone-like strip
    mask = compute_bone_cortex_attention_mask(img)

    assert mask.shape == (1, 1, 224, 224)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    # The bone region should have significantly higher attention than background
    assert mask[0, 0, 105, 100] > mask[0, 0, 10, 10]


def test_differentiable_spatial_warping_and_grad():
    fmap = torch.randn(1, 128, 28, 28)
    tx = torch.nn.Parameter(torch.tensor([0.05]))
    ty = torch.nn.Parameter(torch.tensor([-0.02]))
    theta = torch.nn.Parameter(torch.tensor([0.1]))

    warped = differentiable_spatial_warp_2d(fmap, tx, ty, theta)
    assert warped.shape == (1, 128, 28, 28)
    assert warped.requires_grad

    loss = warped.sum()
    loss.backward()
    assert tx.grad is not None and not torch.isnan(tx.grad)
    assert ty.grad is not None and not torch.isnan(ty.grad)
    assert theta.grad is not None and not torch.isnan(theta.grad)


def test_bone_weighted_calibration_loss():
    fmap_a = torch.randn(1, 128, 28, 28)
    fmap_b = fmap_a.clone() + 0.05 * torch.randn(1, 128, 28, 28)
    bone_mask = torch.rand(1, 1, 224, 224)

    loss_fn = BoneWeightedCalibrationLoss(metric="cosine", bone_boost_factor=3.0)
    loss = loss_fn(fmap_a, fmap_b, bone_weight_mask=bone_mask)

    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_prepare_joint_binary_mask():
    # Simulate a binary mask from JOINT segmentation (e.g. 512x485)
    joint_raw_mask = np.zeros((512, 485), dtype=np.uint8)
    joint_raw_mask[200:240, 100:350] = 1  # Segmented bone cortex

    # Test with target_size, dilation, and smoothing
    processed = prepare_joint_binary_mask(
        joint_raw_mask,
        target_size=(56, 56),
        dilation_radius=2,
        smooth_sigma=1.0,
    )

    assert processed.shape == (1, 1, 56, 56)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
    # Bone region should have positive activation
    assert processed.sum() > 0.0


def test_joint_mask_loss_modes():
    fmap_a = torch.randn(1, 128, 56, 56)
    fmap_b = fmap_a.clone() + 0.1 * torch.randn(1, 128, 56, 56)
    
    # Binary JOINT mask
    joint_mask = torch.zeros(1, 1, 56, 56)
    joint_mask[0, 0, 20:30, 20:40] = 1.0

    # 1. Hard mask mode
    loss_hard_fn = BoneWeightedCalibrationLoss(metric="cosine", mask_mode="hard")
    loss_hard = loss_hard_fn(fmap_a, fmap_b, bone_weight_mask=joint_mask)
    assert torch.isfinite(loss_hard)
    assert loss_hard.item() > 0.0

    # 2. Boost mode
    loss_boost_fn = BoneWeightedCalibrationLoss(metric="cosine", mask_mode="boost", bone_boost_factor=5.0)
    loss_boost = loss_boost_fn(fmap_a, fmap_b, bone_weight_mask=joint_mask)
    assert torch.isfinite(loss_boost)

    # 3. Normalized mode
    loss_norm_fn = BoneWeightedCalibrationLoss(metric="cosine", mask_mode="normalized")
    loss_norm = loss_norm_fn(fmap_a, fmap_b, bone_weight_mask=joint_mask)
    assert torch.isfinite(loss_norm)

