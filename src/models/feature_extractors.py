"""Unified Multi-Stage Feature Extractor for DualTrack."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import einops
import numpy as np
import torch
import torch.nn as nn

from src.config import CHECKPOINT_PATH, DEVICE
from src.loaders.preprocessor import preprocess_frames_for_global_encoder, preprocess_frames_for_local_encoder
from src.models.dualtrack_bridge import build_dualtrack_model


@dataclass
class ExtractedFeatures:
    """Dataclass holding extracted feature representations across all hierarchy levels."""
    sweep_id: str
    num_frames: int
    stage1_fmaps: Optional[torch.Tensor] = None  # (B, N, C=512, H=16, W=16)
    stage1_pooled: Optional[torch.Tensor] = None  # (B, N, C=512)
    stage2_vit_cls: Optional[torch.Tensor] = None  # (B, N, C=64)
    stage3_temporal: Optional[torch.Tensor] = None  # (B, N, C=64)
    stage3_projected: Optional[torch.Tensor] = None  # (B, N, C=512)
    global_features: Optional[torch.Tensor] = None  # (B, N_sparse, C=512)
    sparse_indices: Optional[torch.Tensor] = None  # (B, N_sparse)
    pred_rel_poses: Optional[torch.Tensor] = None  # (B, N-1, 6)


class DualTrackFeatureExtractor(nn.Module):
    """Wrapper around DualTrack model providing direct access to intermediate feature levels."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[Path, str]] = CHECKPOINT_PATH,
        device: str = DEVICE,
    ):
        super().__init__()
        self.device = device
        self.full_model = build_dualtrack_model(checkpoint_path=checkpoint_path, device=device, eval_mode=True)
        self.eval()

    @property
    def local_encoder_spt(self):
        """Access LocalEncoderSPTAttn module."""
        return self.full_model.local_encoder[0]

    @property
    def spatial_vit_module(self):
        """Access FeatureExtractorWithSpatialSelfAttentionV1 module."""
        # Unpack FrozenModuleWrapper
        mod = self.local_encoder_spt.backbone
        if hasattr(mod, "module"):
            mod = mod.module
        return mod

    @property
    def cnn_backbone_module(self):
        """Access VideoResNet 3D CNN backbone."""
        mod = self.spatial_vit_module.backbone
        if hasattr(mod, "module"):
            mod = mod.module
        return mod

    @property
    def temporal_attn_module(self):
        """Access SimpleTemporalAttn module."""
        return self.local_encoder_spt.temporal_encoder

    @property
    def local_proj_linear(self):
        """Access Linear(64, 512) projecting to decoder hidden size."""
        return self.full_model.local_encoder[1]

    @property
    def global_encoder_module(self):
        """Access Global Context Encoder."""
        return self.full_model.global_encoder

    @torch.no_grad()
    def extract_stage1_cnn(
        self,
        frames: Union[np.ndarray, torch.Tensor],
        pool: bool = False,
    ) -> torch.Tensor:
        """Extract Stage 1 (3D CNN VideoResNet) feature maps.
        
        Args:
            frames: (N, H, W) or (B, N, 1, 256, 256).
            pool: If True, return (B, N, 512) spatially pooled features.
                  If False, return (B, N, 512, 16, 16) feature maps.
        """
        x = preprocess_frames_for_local_encoder(frames, device=self.device)
        # 3D CNN forward pass
        fmaps = self.cnn_backbone_module(x)  # (B, N, C=512, H=16, W=16)
        if pool:
            return fmaps.mean(dim=(-1, -2))  # (B, N, 512)
        return fmaps

    @torch.no_grad()
    def extract_stage2_vit_cls(
        self,
        frames: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Extract Stage 2 (Spatial ViT Self-Attention) [CLS] token embeddings.
        
        Args:
            frames: (N, H, W) or (B, N, 1, 256, 256).
            
        Returns:
            cls_tokens: Tensor of shape (B, N, 64).
        """
        x = preprocess_frames_for_local_encoder(frames, device=self.device)
        cls_tokens = self.spatial_vit_module(x)  # (B, N, 64)
        return cls_tokens

    @torch.no_grad()
    def extract_stage3_temporal(
        self,
        frames: Union[np.ndarray, torch.Tensor],
        project_to_decoder_dim: bool = False,
    ) -> torch.Tensor:
        """Extract Stage 3 (Temporal Attention) sequence representations.
        
        Args:
            frames: (N, H, W) or (B, N, 1, 256, 256).
            project_to_decoder_dim: If True, project from 64 to 512.
            
        Returns:
            temp_tokens: Tensor of shape (B, N, 64) or (B, N, 512).
        """
        x = preprocess_frames_for_local_encoder(frames, device=self.device)
        tokens_64 = self.local_encoder_spt(x)  # (B, N, 64)
        if project_to_decoder_dim:
            return self.local_proj_linear(tokens_64)  # (B, N, 512)
        return tokens_64

    @torch.no_grad()
    def extract_global_context(
        self,
        frames: Union[np.ndarray, torch.Tensor],
        sample_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Global Context Encoder representations for sparsely sampled frames.
        
        Args:
            frames: (N, H, W) or (B, N, 1, 224, 224).
            sample_indices: (B, N_sparse) sample indices. If None, uses model's sampler.
            
        Returns:
            global_features: (B, N_sparse, 512)
            sparse_indices: (B, N_sparse)
        """
        x = preprocess_frames_for_global_encoder(frames, device=self.device)
        B, N, C, H, W = x.shape

        if sample_indices is None:
            # Subsample using regular grid sampler
            sample_indices = self.full_model.sampler(x, None)

        subsampled = []
        for i in range(B):
            subsampled.append(x[i][sample_indices[i]].contiguous())
        subsampled_x = torch.stack(subsampled, dim=0)

        global_features = self.global_encoder_module(subsampled_x, sample_indices)
        return global_features, sample_indices

    @torch.no_grad()
    def extract_all_hierarchy_levels(
        self,
        frames: Union[np.ndarray, torch.Tensor],
        sweep_id: str = "sweep",
    ) -> ExtractedFeatures:
        """Extract representations across all encoder levels in a single pass."""
        local_x = preprocess_frames_for_local_encoder(frames, device=self.device)
        global_x = preprocess_frames_for_global_encoder(frames, device=self.device)
        B, N = local_x.shape[0], local_x.shape[1]

        # 1. Stage 1 CNN
        stage1_fmaps = self.cnn_backbone_module(local_x)
        stage1_pooled = stage1_fmaps.mean(dim=(-1, -2))

        # 2. Stage 2 ViT CLS
        # Fold sequence into batch
        fmaps_flat = einops.rearrange(stage1_fmaps, "b n c h w -> (b n) c h w")
        vit_out = self.spatial_vit_module.vit(fmaps_flat).last_hidden_state
        stage2_cls = einops.rearrange(vit_out[:, 0, :], "(b n) c -> b n c", b=B, n=N)

        # 3. Stage 3 Temporal
        stage3_temp = self.temporal_attn_module(stage2_cls)
        stage3_proj = self.local_proj_linear(stage3_temp)

        # 4. Global Context
        global_feats, sparse_indices = self.extract_global_context(global_x)

        # 5. Full Trajectory Prediction
        pred_poses = self.full_model(global_x, local_x)

        return ExtractedFeatures(
            sweep_id=sweep_id,
            num_frames=N,
            stage1_fmaps=stage1_fmaps,
            stage1_pooled=stage1_pooled,
            stage2_vit_cls=stage2_cls,
            stage3_temporal=stage3_temp,
            stage3_projected=stage3_proj,
            global_features=global_feats,
            sparse_indices=sparse_indices,
            pred_rel_poses=pred_poses,
        )
