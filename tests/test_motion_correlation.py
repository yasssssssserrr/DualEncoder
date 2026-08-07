"""Unit tests for diagnostic metrics: Motion correlation, decorrelation, and trajectory evaluation."""
import numpy as np
import pytest
import torch

from src.diagnostics.decorrelation_curve import compute_speckle_decorrelation
from src.diagnostics.motion_correlation import (
    compute_feature_distance_matrix,
    compute_physical_distance_matrix,
    evaluate_motion_correlation,
)
from src.diagnostics.trajectory_eval import evaluate_sweep_trajectory


def test_distance_matrix_computations(sample_forearm_sweep):
    transforms = sample_forearm_sweep.transforms[:10]  # (10, 4, 4)
    trans_mat, rot_mat = compute_physical_distance_matrix(transforms)

    assert trans_mat.shape == (10, 10)
    assert rot_mat.shape == (10, 10)
    # Diagonal should be zero
    np.testing.assert_allclose(np.diag(trans_mat), 0.0)
    np.testing.assert_allclose(np.diag(rot_mat), 0.0)
    # Symmetric
    np.testing.assert_allclose(trans_mat, trans_mat.T)


def test_motion_correlation_synthetic(synthetic_ultrasound_sweep):
    sweep = synthetic_ultrasound_sweep
    # Construct mock monotonic directional feature representation
    N = sweep.num_frames
    features = np.zeros((N, 64))
    for i in range(N):
        angle = i * (np.pi / (2 * N))
        features[i, 0] = np.cos(angle)
        features[i, 1] = np.sin(angle)

    metrics = evaluate_motion_correlation(features, sweep.transforms, stage_name="synthetic")
    assert metrics.pearson_r > 0.95
    assert metrics.spearman_rho > 0.95
    assert metrics.monotonicity_score >= 0.9


def test_speckle_decorrelation_computation(sample_forearm_sweep):
    N = 20
    transforms = sample_forearm_sweep.transforms[:N]
    # Create exponentially decaying features
    features = np.random.randn(N, 64)
    for i in range(1, N):
        features[i] = 0.8 * features[i - 1] + 0.2 * np.random.randn(64)

    stats = compute_speckle_decorrelation(features, transforms, max_lag=10)
    assert len(stats.lags_frames) == 11
    assert stats.cosine_similarities[0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(stats.displacements_mm >= 0.0)


def test_trajectory_eval_perfect_identity(sample_forearm_sweep):
    transforms = sample_forearm_sweep.transforms[:10]
    # Compute perfect ground truth relative vectors
    from src.diagnostics.trajectory_eval import mat_to_pose_vec
    pred_rel_vecs = np.zeros((9, 6))
    for i in range(9):
        rel = np.linalg.inv(transforms[i]) @ transforms[i + 1]
        pred_rel_vecs[i] = mat_to_pose_vec(rel)

    metrics = evaluate_sweep_trajectory(pred_rel_vecs, transforms, sweep_id="test_gt")
    assert metrics.mean_gpe_mm < 1e-4
    assert metrics.endpoint_error_mm < 1e-4
    assert metrics.drift_percent < 1e-4
