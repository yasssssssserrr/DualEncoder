"""Spatial Feature Extractor for Calibration Compensation and 3D Volume Reconstruction.

Extracts intermediate spatial feature maps (Layer 1, Layer 2, Layer 3, Stage 1 CNN)
from the frozen DualTrack feature extractor while preserving full differentiable
gradient flow for downstream calibration matrix optimization (ΔT_calib).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_extractors import DualTrackFeatureExtractor


@dataclass
class SpatialFeatureOutputs:
    """Dataclass holding extracted spatial feature maps with active computation graphs."""
    layer1: Optional[torch.Tensor] = None  # (B, 64, 56, 56)
    layer2: Optional[torch.Tensor] = None  # (B, 128, 28, 28) - Optimal for Bone Cortex
    layer3: Optional[torch.Tensor] = None  # (B, 256, 14, 14)
    layer4: Optional[torch.Tensor] = None  # (B, 512, 7, 7)
    stage1_3d: Optional[torch.Tensor] = None  # (B, N, 512, 16, 16) - Speckle decorrelation


class IntermediateLayerHook:
    """Forward hook to capture layer activations without breaking gradient backprop."""

    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self._hook_fn)
        self.features: Optional[torch.Tensor] = None

    def _hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
        self.features = output

    def remove(self):
        self.hook.remove()


class DualTrackSpatialFeatureExtractor(nn.Module):
    """Differentiable Spatial Feature Extractor for Calibration Compensation.
    
    Extracts high-resolution spatial feature maps (especially Layer 2) from
    the frozen DualTrack backbone. The weights remain frozen, but gradients
    flow backward to input images, spatial warping grids, and calibration matrices.
    """

    def __init__(
        self,
        dualtrack_extractor: Optional[DualTrackFeatureExtractor] = None,
        target_layers: Tuple[str, ...] = ("layer2",),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.target_layers = target_layers

        if dualtrack_extractor is None:
            self.extractor = DualTrackFeatureExtractor(device=device)
        else:
            self.extractor = dualtrack_extractor

        # Freeze all backbone weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.extractor.eval()

        # Resolve the underlying ResNet backbone module inside the Global Context Encoder
        self.global_resnet = self._get_global_resnet()

    def _get_global_resnet(self) -> nn.Module:
        """Unwraps DDP/wrappers to access the 2D/3D ResNet backbone."""
        mod = self.extractor.global_encoder_module
        if hasattr(mod, "backbone"):
            mod = mod.backbone
        if isinstance(mod, (nn.Sequential, list, tuple)):
            mod = mod[0]
        if hasattr(mod, "backbone"):
            mod = mod.backbone
        if hasattr(mod, "module"):
            mod = mod.module
        return mod

    def _ensure_5d(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool, Tuple[int, ...]]:
        """Ensures input tensor is 5D (B, C, 1, H, W) for VideoResNet stem."""
        if x.ndim == 2:  # (H, W)
            return x.unsqueeze(0).unsqueeze(0).unsqueeze(0), False, (1, 1)
        elif x.ndim == 3:  # (N, H, W) -> (1, 1, N, H, W) or (B, H, W)
            return x.unsqueeze(1).unsqueeze(2), False, (x.shape[0], 1)
        elif x.ndim == 4:  # (B, 1, H, W) -> (B, 1, 1, H, W)
            return x.unsqueeze(2), False, (x.shape[0], 1)
        elif x.ndim == 5:  # (B, N, 1, H, W) -> (B*N, 1, 1, H, W)
            B, N, C, H, W = x.shape
            return x.view(B * N, C, 1, H, W), True, (B, N)
        else:
            raise ValueError(f"Unsupported tensor ndim: {x.ndim}")

    def forward_layer2(self, x: torch.Tensor) -> torch.Tensor:
        """Direct forward pass through ResNet up to Layer 2 with active gradient tracking.
        
        Args:
            x: Input ultrasound tensor (B, 1, H=224, W=224) or (B, N, 1, 224, 224).
            
        Returns:
            fmaps: Spatial feature maps from Layer 2 (B, 128, 28, 28) or (B, N, 128, 28, 28).
        """
        x_5d, is_sequence, (B, N) = self._ensure_5d(x)

        # 1. Stem (Conv3d + BatchNorm3d + ReLU)
        h = self.global_resnet.stem(x_5d)    # (B*N, 45 or 64, 1, 112, 112)
        h1 = self.global_resnet.layer1(h)   # (B*N, 64, 1, 56, 56)
        h2 = self.global_resnet.layer2(h1)  # (B*N, 128, 1, 28, 28) - Sharp Bone Cortex

        # Remove singleton temporal/depth dimension (B*N, C, H', W')
        fmap_2d = h2.squeeze(2)

        if is_sequence:
            return fmap_2d.view(B, N, fmap_2d.shape[1], fmap_2d.shape[2], fmap_2d.shape[3])
        return fmap_2d

    def forward_multiscale(
        self,
        x: torch.Tensor,
        layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    ) -> Dict[str, torch.Tensor]:
        """Extracts multi-scale spatial feature maps in a single differentiable forward pass.
        
        Args:
            x: Input tensor (B, 1, H=224, W=224) or (B, N, 1, 224, 224).
            layers: Tuple of layer names to extract.
            
        Returns:
            Dict mapping layer name to feature tensor.
        """
        x_5d, is_sequence, (B, N) = self._ensure_5d(x)
        outputs: Dict[str, torch.Tensor] = {}

        # Stem
        h = self.global_resnet.stem(x_5d)

        # Sequential extraction
        h1 = self.global_resnet.layer1(h)
        if "layer1" in layers:
            out_h1 = h1.squeeze(2)
            outputs["layer1"] = out_h1.view(B, N, *out_h1.shape[1:]) if is_sequence else out_h1

        h2 = self.global_resnet.layer2(h1)
        if "layer2" in layers:
            out_h2 = h2.squeeze(2)
            outputs["layer2"] = out_h2.view(B, N, *out_h2.shape[1:]) if is_sequence else out_h2

        h3 = self.global_resnet.layer3(h2)
        if "layer3" in layers:
            out_h3 = h3.squeeze(2)
            outputs["layer3"] = out_h3.view(B, N, *out_h3.shape[1:]) if is_sequence else out_h3

        h4 = self.global_resnet.layer4(h3)
        if "layer4" in layers:
            out_h4 = h4.squeeze(2)
            outputs["layer4"] = out_h4.view(B, N, *out_h4.shape[1:]) if is_sequence else out_h4

        return outputs

    def extract_with_hooks(
        self,
        x: torch.Tensor,
        layers: Tuple[str, ...] = ("layer2",),
    ) -> Dict[str, torch.Tensor]:
        """Alternative hook-based feature extraction (supports arbitrary nested submodules).
        
        Args:
            x: Input tensor (B, N, 1, 224, 224).
            layers: Layer attribute names on the backbone.
            
        Returns:
            Dict mapping layer name to tensor.
        """
        hooks = {}
        for layer_name in layers:
            submod = getattr(self.global_resnet, layer_name)
            hooks[layer_name] = IntermediateLayerHook(submod)

        try:
            # Dummy forward pass through global encoder module
            if x.ndim == 4:
                x_5d = x.unsqueeze(1)
            else:
                x_5d = x
            idx = torch.zeros((x_5d.shape[0], x_5d.shape[1]), dtype=torch.long, device=x.device)
            _ = self.extractor.global_encoder_module(x_5d, idx)

            results = {}
            for name, hook in hooks.items():
                if hook.features is not None:
                    results[name] = hook.features
            return results
        finally:
            for hook in hooks.values():
                hook.remove()
