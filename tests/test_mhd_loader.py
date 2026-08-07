"""Unit tests for MetaImage (.mhd/.raw) dataset loader and preprocessors."""
import numpy as np
import pytest
import torch

from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep, parse_mhd_header
from src.loaders.preprocessor import preprocess_frames_for_global_encoder, preprocess_frames_for_local_encoder, prepare_sweep_batch


def test_forearm_files_discovery(robot_data_path):
    files = list_forearm_phantom_scans(robot_data_path)
    assert len(files) == 10, f"Expected 10 forearm phantom scans, found {len(files)}"
    for f in files:
        assert f.name.endswith("forearm_phantom_scan.mhd")


def test_parse_mhd_header(forearm_mhd_files):
    mhd_path = forearm_mhd_files[0]
    tags, transforms, timestamps, spacing, dim_size, raw_filename = parse_mhd_header(mhd_path)

    assert "DimSize" in tags
    assert dim_size == (485, 512, 47)
    assert len(transforms) == 47
    assert len(timestamps) == 47
    assert transforms.shape == (47, 4, 4)
    assert raw_filename.endswith(".raw")


def test_load_robot_sweep(sample_forearm_sweep):
    sweep = sample_forearm_sweep
    assert sweep.num_frames == 47
    assert sweep.frames.shape == (47, 512, 485)
    assert sweep.frames.dtype == np.uint8
    assert sweep.transforms.shape == (47, 4, 4)
    assert len(sweep.timestamps) == 47

    # Check image values are within valid brightness range
    assert sweep.frames.min() >= 0
    assert sweep.frames.max() <= 255


def test_transforms_rigid_body(sample_forearm_sweep):
    transforms = sample_forearm_sweep.transforms
    for i, T in enumerate(transforms):
        # Homogeneous bottom row
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-5)
        # Proper rotation matrix: det(R) == 1.0, R @ R.T == I
        R = T[:3, :3]
        det = np.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-3, err_msg=f"Frame {i} rotation det not 1.0")
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-3, err_msg=f"Frame {i} rotation not orthogonal")


def test_preprocessor_local_and_global(sample_forearm_sweep):
    frames = sample_forearm_sweep.frames  # (47, 512, 485)
    
    local_tensor = preprocess_frames_for_local_encoder(frames, target_size=(256, 256))
    assert local_tensor.shape == (1, 47, 1, 256, 256)
    assert local_tensor.dtype == torch.float32
    assert local_tensor.min() >= 0.0
    assert local_tensor.max() <= 1.0

    global_tensor = preprocess_frames_for_global_encoder(frames, target_size=(224, 224))
    assert global_tensor.shape == (1, 47, 1, 224, 224)
    assert global_tensor.dtype == torch.float32
    assert global_tensor.min() >= 0.0
    assert global_tensor.max() <= 1.0
