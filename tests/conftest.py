"""Pytest configuration and shared fixtures for DualEncoder tests."""
from pathlib import Path
import numpy as np
import pytest
import torch

from src.config import CHECKPOINT_PATH, DEVICE, ROBOT_DATA_DIR, TUSREC_TRAIN_DIR
from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep, RobotSweep
from src.models.feature_extractors import DualTrackFeatureExtractor


@pytest.fixture(scope="session")
def robot_data_path() -> Path:
    return ROBOT_DATA_DIR


@pytest.fixture(scope="session")
def forearm_mhd_files(robot_data_path: Path):
    if not robot_data_path.exists():
        pytest.skip(f"Robot dataset directory not found: {robot_data_path}")
    files = list_forearm_phantom_scans(robot_data_path)
    if not files:
        pytest.skip(f"No *forearm_phantom_scan.mhd found in {robot_data_path}")
    return files


@pytest.fixture(scope="session")
def sample_forearm_sweep(forearm_mhd_files) -> RobotSweep:
    return load_robot_sweep(forearm_mhd_files[0])


@pytest.fixture(scope="session")
def feature_extractor() -> DualTrackFeatureExtractor:
    return DualTrackFeatureExtractor(checkpoint_path=CHECKPOINT_PATH, device="cpu")


@pytest.fixture
def synthetic_ultrasound_sweep():
    """Create a mock 16-frame ultrasound sweep with moving speckle pattern."""
    N, H, W = 16, 512, 485
    frames = np.zeros((N, H, W), dtype=np.uint8)
    transforms = np.zeros((N, 4, 4), dtype=np.float64)

    # Base speckle pattern
    np.random.seed(42)
    base_pattern = (np.random.randn(H, W) * 30 + 128).clip(0, 255).astype(np.uint8)

    for i in range(N):
        # Shift speckle pattern by 2 pixels per frame
        shift_y = (i * 2) % H
        frames[i] = np.roll(base_pattern, shift_y, axis=0)

        # Ground truth transform translating along Z by 0.5 mm per frame
        T = np.eye(4, dtype=np.float64)
        T[2, 3] = i * 0.5  # 0.5 mm translation
        transforms[i] = T

    return RobotSweep(
        sweep_id="synthetic_test",
        mhd_path=Path("synthetic.mhd"),
        raw_path=Path("synthetic.raw"),
        frames=frames,
        transforms=transforms,
        timestamps=np.arange(N, dtype=np.float64) * 0.05,
        spacing_mm=(0.0786, 0.0786, 1.0),
        dimensions=(W, H, N),
    )
