"""Diagnostic and evaluation modules."""
from .motion_correlation import (
    CorrelationMetrics,
    compute_feature_distance_matrix,
    compute_physical_distance_matrix,
    evaluate_motion_correlation,
)
from .decorrelation_curve import (
    DecorrelationStats,
    compute_speckle_decorrelation,
)
from .domain_gap_analysis import (
    linear_cka,
    compute_pca_embeddings,
)
from .trajectory_eval import (
    TrajectoryMetrics,
    evaluate_sweep_trajectory,
    pose_vec_to_mat,
    mat_to_pose_vec,
)
