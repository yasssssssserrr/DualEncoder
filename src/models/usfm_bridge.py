"""Bridge module for building and comparing USFM and ResNet18 backbones."""
import os
from pathlib import Path
import sys
from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange

from src.models.dualtrack_bridge import setup_dualtrack_path

setup_dualtrack_path()

import src.models.misc
import src.models.usfm
from src.models.usfm import get_usfm_backbone, USFMWrapperFor3DFeatureMaps
from src.models.video_resnet import VideoResnetWrapperForFeatureMaps, video_rn18_no_temporal


class SpatialMeanPooling(nn.Module):
    """Pools 2D spatial dimensions (H, W) -> feature vector."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean((-1, -2))


def build_resnet18_backbone(
    weights_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Builds ResNet-18 feature extraction backbone with spatial mean pooling."""
    setup_dualtrack_path()
    rn_model = VideoResnetWrapperForFeatureMaps(video_rn18_no_temporal())
    backbone = nn.Sequential(rn_model, SpatialMeanPooling())
    backbone.num_features = 512

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Extract global_encoder backbone weights if present in full checkpoint
        rn_state = {}
        for k, v in state.items():
            if k.startswith("global_encoder.backbone.0."):
                rn_state[k.replace("global_encoder.backbone.0.", "")] = v
            elif k.startswith("backbone.0."):
                rn_state[k.replace("backbone.0.", "")] = v
            elif not k.startswith("local_encoder.") and not k.startswith("decoder."):
                rn_state[k] = v
        if rn_state:
            rn_model.load_state_dict(rn_state, strict=False)
        else:
            backbone.load_state_dict(state, strict=False)

    backbone.to(device)
    backbone.eval()
    return backbone


def build_usfm_backbone(
    image_size: int = 224,
    projection_dim: int = 256,
    weights_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Builds USFM (Hierarchical ViT Ultrasound Foundation Model) backbone."""
    setup_dualtrack_path()
    usfm_core = get_usfm_backbone(pretrained_path=weights_path if (weights_path and Path(weights_path).exists()) else None, image_size=image_size)
    wrapper = USFMWrapperFor3DFeatureMaps(usfm_core, projection_dim=projection_dim, output_axes="b n c h w")
    backbone = nn.Sequential(wrapper, SpatialMeanPooling())
    backbone.num_features = projection_dim

    backbone.to(device)
    backbone.eval()
    return backbone


class BackboneComparisonExtractor(nn.Module):
    """Wrapper that runs both ResNet-18 and USFM on identical ultrasound sequences."""
    def __init__(
        self,
        resnet_weights: Optional[str] = None,
        usfm_weights: Optional[str] = None,
        image_size: int = 224,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.resnet18 = build_resnet18_backbone(weights_path=resnet_weights, device=device)
        self.usfm = build_usfm_backbone(image_size=image_size, projection_dim=256, weights_path=usfm_weights, device=device)

    @torch.no_grad()
    def forward_resnet(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract ResNet-18 pooled features and 2D spatial feature map.
        
        Input: (B, N, C, H, W)
        Returns: pooled (B, N, 512), spatial_map (B, N, 512, H', W')
        """
        x = x.to(self.device)
        if x.shape[2] != 1:
            x_rn = x[:, :, 0:1, :, :]
        else:
            x_rn = x
        
        feat_map = self.resnet18[0](x_rn) # (B, N, 512, 14, 14)
        pooled = self.resnet18[1](feat_map) # (B, N, 512)
        return pooled, feat_map

    @torch.no_grad()
    def forward_usfm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract USFM pooled features and 2D spatial feature map.
        
        Input: (B, N, C, H, W)
        Returns: pooled (B, N, 256), spatial_map (B, N, 256, H', W')
        """
        x = x.to(self.device)
        if x.shape[2] == 1:
            x_usfm = x.repeat(1, 1, 3, 1, 1)
        else:
            x_usfm = x
        
        feat_map = self.usfm[0](x_usfm) # (B, N, 256, 14, 14)
        pooled = self.usfm[1](feat_map) # (B, N, 256)
        return pooled, feat_map
