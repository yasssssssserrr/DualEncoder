"""Unit tests for Multi-Stage Feature Extractors."""
import numpy as np
import pytest
import torch

from src.models.feature_extractors import DualTrackFeatureExtractor, ExtractedFeatures


def test_feature_extractor_instantiation(feature_extractor):
    assert feature_extractor is not None
    assert feature_extractor.full_model is not None


def test_stage1_cnn_extraction(feature_extractor, synthetic_ultrasound_sweep):
    frames = synthetic_ultrasound_sweep.frames  # (16, 512, 485)
    fmaps = feature_extractor.extract_stage1_cnn(frames, pool=False)
    assert fmaps.shape == (1, 16, 512, 16, 16)

    pooled = feature_extractor.extract_stage1_cnn(frames, pool=True)
    assert pooled.shape == (1, 16, 512)
    assert torch.all(torch.isfinite(pooled))


def test_stage2_vit_cls_extraction(feature_extractor, synthetic_ultrasound_sweep):
    frames = synthetic_ultrasound_sweep.frames
    cls_tokens = feature_extractor.extract_stage2_vit_cls(frames)
    assert cls_tokens.shape == (1, 16, 64)
    assert torch.all(torch.isfinite(cls_tokens))


def test_stage3_temporal_extraction(feature_extractor, synthetic_ultrasound_sweep):
    frames = synthetic_ultrasound_sweep.frames
    tokens_64 = feature_extractor.extract_stage3_temporal(frames, project_to_decoder_dim=False)
    assert tokens_64.shape == (1, 16, 64)

    tokens_512 = feature_extractor.extract_stage3_temporal(frames, project_to_decoder_dim=True)
    assert tokens_512.shape == (1, 16, 512)
    assert torch.all(torch.isfinite(tokens_512))


def test_global_encoder_extraction(feature_extractor, synthetic_ultrasound_sweep):
    frames = synthetic_ultrasound_sweep.frames
    global_feats, indices = feature_extractor.extract_global_context(frames)
    assert global_feats.ndim == 3
    assert global_feats.shape[0] == 1
    assert global_feats.shape[2] == 512
    assert torch.all(torch.isfinite(global_feats))


def test_full_hierarchy_extraction_on_robot_sweep(feature_extractor, sample_forearm_sweep):
    sweep = sample_forearm_sweep  # (47 frames)
    features = feature_extractor.extract_all_hierarchy_levels(sweep.frames[:16], sweep_id=sweep.sweep_id)

    assert isinstance(features, ExtractedFeatures)
    assert features.stage1_fmaps.shape == (1, 16, 512, 16, 16)
    assert features.stage1_pooled.shape == (1, 16, 512)
    assert features.stage2_vit_cls.shape == (1, 16, 64)
    assert features.stage3_temporal.shape == (1, 16, 64)
    assert features.stage3_projected.shape == (1, 16, 512)
    assert features.global_features.shape[2] == 512
    assert features.pred_rel_poses.shape == (1, 15, 6)
