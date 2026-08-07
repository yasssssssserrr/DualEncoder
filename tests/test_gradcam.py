"""Unit tests for UltrasoundGradCAM, EigenCAM, LatentOcclusion, and Faithfulness evaluation."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diagnostics.gradcam import (
    ExplanationMetadata,
    LatentOcclusion,
    TrajectoryDirectionEstimator,
    UltrasoundEigenCAM,
    UltrasoundGradCAM,
    evaluate_cam_faithfulness_deletion,
    resolve_target_layer,
    safe_min_max_normalize,
)


class DummyUltrasoundEncoder(nn.Module):
    """Simple 2D/3D CNN encoder mimicking VideoResNet with 512-D output for fast unit tests."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 5D input (B, N, 1, H, W)
        if x.ndim == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmaps = self.layer4(x)  # (N, 512, H/16, W/16)
        return fmaps


@pytest.fixture
def dummy_encoder():
    model = DummyUltrasoundEncoder(in_channels=1, feature_dim=512)
    model.eval()
    return model


def test_resolve_target_layer(dummy_encoder):
    """Test resolution of named layers and modules."""
    layer4_mod, name = resolve_target_layer(dummy_encoder, "layer4")
    assert layer4_mod is dummy_encoder.layer4
    assert "layer4" in name

    layer1_mod, name = resolve_target_layer(dummy_encoder, dummy_encoder.layer1)
    assert layer1_mod is dummy_encoder.layer1

    with pytest.raises(ValueError):
        resolve_target_layer(dummy_encoder, "non_existent_layer")


def test_safe_min_max_normalize():
    """Test safe normalization under normal and edge conditions (constant, zeros, NaNs)."""
    # 1. Normal map
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = safe_min_max_normalize(tensor)
    assert norm.min().item() == pytest.approx(0.0)
    assert norm.max().item() == pytest.approx(1.0)

    # 2. Constant map
    const_tensor = torch.ones((3, 3)) * 5.0
    const_norm = safe_min_max_normalize(const_tensor)
    assert torch.all(const_norm == 0.0)

    # 3. Signed normalization
    signed_tensor = torch.tensor([[-10.0, 0.0], [5.0, 10.0]])
    signed_norm = safe_min_max_normalize(signed_tensor, signed=True)
    assert signed_norm.min().item() == pytest.approx(-1.0)
    assert signed_norm.max().item() == pytest.approx(1.0)

    # 4. NaN / Inf hazard handling
    hazard_tensor = torch.tensor([[float("nan"), 1.0], [float("inf"), -float("inf")]])
    clean_norm = safe_min_max_normalize(hazard_tensor)
    assert not torch.isnan(clean_norm).any()
    assert not torch.isinf(clean_norm).any()


def test_gradcam_hooks_registration_and_cleanup(dummy_encoder):
    """Test that hooks register and clean up completely without leaving dangling references."""
    cam = UltrasoundGradCAM(dummy_encoder, target_layer="layer4")
    assert cam._forward_hook_handle is not None

    x = torch.randn(1, 1, 64, 64)
    cam_map, meta = cam.explain(x, objective="energy")
    assert cam_map.shape == (1, 64, 64)
    assert meta.method == "gradcam"
    assert meta.objective == "energy"

    # Clean up
    cam.remove_hooks()
    assert cam._forward_hook_handle is None
    assert cam._backward_hook_handle is None


def test_gradcam_context_manager(dummy_encoder):
    """Test context manager cleanup."""
    x = torch.randn(2, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_map, meta = cam.explain(x, objective="latent_dimension", target_dimension=10)
        assert cam_map.shape == (2, 64, 64)
    assert cam._forward_hook_handle is None


def test_gradcam_batch_processing_and_isolation(dummy_encoder):
    """Test that batch size > 1 produces N independent (N, H, W) maps."""
    N = 4
    x = torch.randn(N, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_map, meta = cam.explain(x, objective="latent_dimension", target_dimension=42)
        assert cam_map.shape == (N, 64, 64)
        assert not torch.isnan(cam_map).any()
        assert cam_map.min() >= 0.0 and cam_map.max() <= 1.0


def test_gradcam_objective_latent_dimension_bounds(dummy_encoder):
    """Test bounds checking on target_dimension."""
    x = torch.randn(1, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4", embedding_dim=512) as cam:
        # Valid
        _, meta = cam.explain(x, objective="latent_dimension", target_dimension=0)
        assert meta.target_dimension == 0
        _, meta = cam.explain(x, objective="latent_dimension", target_dimension=511)
        assert meta.target_dimension == 511

        # Out of bounds
        with pytest.raises(ValueError):
            cam.explain(x, objective="latent_dimension", target_dimension=512)
        with pytest.raises(ValueError):
            cam.explain(x, objective="latent_dimension", target_dimension=-1)


def test_gradcam_objective_latent_direction(dummy_encoder):
    """Test latent direction projection objective S = z^T v."""
    x = torch.randn(2, 1, 64, 64)
    direction = torch.randn(512)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_map, meta = cam.explain(
            x, objective="latent_direction", direction=direction, normalize_direction=True
        )
        assert cam_map.shape == (2, 64, 64)
        assert meta.objective == "latent_direction"


def test_gradcam_objective_pairwise_similarity(dummy_encoder):
    """Test pairwise similarity objective with branch separation."""
    x_a = torch.randn(1, 1, 64, 64)
    x_b = torch.randn(1, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_a, meta_a = cam.explain(
            x_a, objective="similarity", target_frame=x_b, explain_branch="a"
        )
        assert cam_a.shape == (1, 64, 64)
        assert meta_a.objective == "similarity"
        assert meta_a.explained_branch == "a"


def test_gradcam_signed_attribution_modes(dummy_encoder):
    """Test signed attribution modes (positive, negative, absolute, signed)."""
    x = torch.randn(1, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_pos, meta = cam.explain(x, objective="energy", relu=False, signed_mode="positive")
        assert cam_pos.shape == (1, 64, 64)
        assert meta.signed_attribution is True

        cam_signed, _ = cam.explain(x, objective="energy", relu=False, signed_mode="signed")
        assert cam_signed.min() >= -1.0 and cam_signed.max() <= 1.0


def test_eigencam(dummy_encoder):
    """Test UltrasoundEigenCAM gradient-free SVD projection."""
    x = torch.randn(2, 1, 64, 64)
    with UltrasoundEigenCAM(dummy_encoder, target_layer="layer4") as ecam:
        cam_map, meta = ecam.explain(x)
        assert cam_map.shape == (2, 64, 64)
        assert meta.method == "eigencam"
        assert not torch.isnan(cam_map).any()


def test_latent_occlusion(dummy_encoder):
    """Test LatentOcclusion sliding-window perturbation sensitivity."""
    x = torch.randn(1, 1, 64, 64)
    occ = LatentOcclusion(dummy_encoder, patch_size=(16, 16), stride=(8, 8), metric="cosine")
    occ_map, meta = occ.explain(x)
    assert occ_map.shape == (1, 64, 64)
    assert meta.method == "latent_occlusion"
    assert occ_map.min() >= 0.0 and occ_map.max() <= 1.0


def test_trajectory_direction_estimator():
    """Test TrajectoryDirectionEstimator PCA and sign orientation."""
    N, D = 30, 512
    # Simulated sweep with strong linear trajectory in dim 0
    t = np.linspace(0, 10, N)
    embeddings = np.zeros((N, D))
    embeddings[:, 0] = -t  # Negative physical orientation
    embeddings += np.random.randn(N, D) * 0.01

    v_traj, r = TrajectoryDirectionEstimator.estimate_trajectory_direction(
        embeddings=embeddings, physical_positions=t
    )
    assert v_traj.shape == (D,)
    assert r > 0.95  # Sign must be flipped to correlate positively with sweep progression


def test_faithfulness_deletion(dummy_encoder):
    """Test deletion faithfulness evaluation test."""
    x = torch.randn(1, 1, 64, 64)
    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_map, _ = cam.explain(x, objective="latent_dimension", target_dimension=5)

    results = evaluate_cam_faithfulness_deletion(
        model=dummy_encoder,
        image=x,
        cam_mask=cam_map,
        objective_type="latent_dimension",
        target_dimension=5,
        mask_fractions=[0.1, 0.3],
        num_random_trials=2,
    )
    assert "audc_top" in results
    assert "audc_random" in results
    assert "audc_least" in results
    assert len(results["top_attribution_drops"]) == 2


def test_trajectory_linear_probe():
    """Test TrajectoryLinearProbe ridge fitting and cross-validation."""
    from src.diagnostics.gradcam import TrajectoryLinearProbe

    N, D = 40, 64
    t = np.linspace(0, 15, N)
    # Synthetic embeddings with strong linear progression along first 4 components
    Z = np.random.randn(N, D) * 0.1
    Z[:, :4] += t[:, None] * 0.5

    probe = TrajectoryLinearProbe(alpha=1.0, cv_folds=5).fit(Z, t)
    assert probe.weights.shape == (D,)
    assert probe.direction_vector.shape == (D,)
    assert np.isclose(np.linalg.norm(probe.direction_vector), 1.0)
    assert probe.pearson_r > 0.90
    assert probe.cv_pearson_r > 0.85
    assert probe.cv_r2_score > 0.70


def test_gradcam_trajectory_probe_objective(dummy_encoder):
    """Test UltrasoundGradCAM with trajectory_probe objective."""
    x = torch.randn(1, 1, 64, 64)
    w_traj = np.random.randn(512)
    w_traj = w_traj / np.linalg.norm(w_traj)

    with UltrasoundGradCAM(dummy_encoder, target_layer="layer4") as cam:
        cam_map, meta = cam.explain(x, objective="trajectory_probe", direction=w_traj)
        assert cam_map.shape == (1, 64, 64)
        assert meta.objective == "trajectory_probe"
        assert meta.target_direction_name == "trajectory_probe_direction"
        assert not torch.isnan(cam_map).any()

