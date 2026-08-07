"""Mathematically Sound Explainability Framework for 512-D Ultrasound Feature Encoders.

This module provides gradient-based (Grad-CAM), activation-based (Eigen-CAM), 
and perturbation-based (Latent Occlusion) explainability methods specifically 
designed for self-supervised feature extractors that map ultrasound images to 
latent representations z in R^D rather than classification logits.

Core Scalar Objectives:
    - Objective A: Embedding Energy / Norm (S = 0.5 * ||z||_2^2)
    - Objective B: Single Latent Dimension (S_k = z_k)
    - Objective C: Latent / Concept Direction (S_v = z^T v)
    - Objective D: Pairwise Representation Similarity (S_sim = cos(z_a, z_b))
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


@dataclass
class ExplanationMetadata:
    """Metadata container for explainability outputs to ensure scientific reproducibility."""
    method: str  # "gradcam", "eigencam", "latent_occlusion"
    objective: str  # "energy", "latent_dimension", "latent_direction", "similarity", "custom"
    target_layer: str
    input_shape: List[int]
    feature_shape: List[int]
    embedding_dimension: int
    normalized_embedding: bool
    signed_attribution: bool
    target_dimension: Optional[int] = None
    target_direction_name: Optional[str] = None
    explained_branch: Optional[str] = None  # "a", "b", or None
    faithfulness_score: Optional[float] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def safe_min_max_normalize(
    tensor: torch.Tensor,
    eps: float = 1e-8,
    signed: bool = False,
) -> torch.Tensor:
    """Safely normalizes attribution maps to [0, 1] or [-1, 1] without NaN/Inf hazards.
    
    Args:
        tensor: (..., H, W) attribution map.
        eps: Small epsilon to prevent division by zero on constant maps.
        signed: If True, normalize symmetric range to [-1, 1] around 0.
        
    Returns:
        normalized_tensor: Safely scaled tensor with guaranteed numerical bounds.
    """
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    # Check for NaNs or Infs
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if signed:
        # Scale symmetric max absolute value to [-1, +1]
        max_abs = torch.amax(torch.abs(tensor), dim=(-2, -1), keepdim=True)
        denom = torch.clamp(max_abs, min=eps)
        return torch.clamp(tensor / denom, -1.0, 1.0)
    else:
        # Scale range [min, max] to [0, 1]
        t_min = torch.amin(tensor, dim=(-2, -1), keepdim=True)
        t_max = torch.amax(tensor, dim=(-2, -1), keepdim=True)
        diff = t_max - t_min
        # If diff is effectively 0 (constant map), return zero tensor
        is_constant = diff < eps
        normalized = (tensor - t_min) / torch.clamp(diff, min=eps)
        normalized = torch.where(is_constant, torch.zeros_like(normalized), normalized)
        return torch.clamp(normalized, 0.0, 1.0)


def resolve_target_layer(model: nn.Module, target_layer: Union[str, nn.Module]) -> Tuple[nn.Module, str]:
    """Resolves target layer module and string identifier from model hierarchy.
    
    Supports: "stem", "layer1", "layer2", "layer3", "layer4", or explicit nn.Module.
    """
    if isinstance(target_layer, nn.Module):
        # Look for name in model
        for name, mod in model.named_modules():
            if mod is target_layer:
                return target_layer, name
        return target_layer, target_layer.__class__.__name__

    layer_name = str(target_layer).strip()

    # 1. Direct attribute on model
    if hasattr(model, layer_name):
        return getattr(model, layer_name), layer_name

    # 2. Check inner backbone if wrapper
    for backbone_attr in ["backbone", "cnn_backbone_module", "spatial_vit_module"]:
        if hasattr(model, backbone_attr):
            sub = getattr(model, backbone_attr)
            if hasattr(sub, "backbone"):
                sub = getattr(sub, "backbone")
            if hasattr(sub, layer_name):
                return getattr(sub, layer_name), f"{backbone_attr}.{layer_name}"

    # 3. Search named modules for suffix or exact match
    for name, mod in model.named_modules():
        if name == layer_name or name.endswith(f".{layer_name}"):
            return mod, name

    # 4. Fallback: list available modules for informative error
    available = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Sequential))]
    raise ValueError(
        f"Could not resolve target layer '{layer_name}' in {model.__class__.__name__}. "
        f"Available candidate layers: {available[:10]}"
    )


def _standardize_spatial_activation(t: torch.Tensor) -> torch.Tensor:
    """Standardizes 4D or 5D activation/gradient tensor into (N_samples, C, H, W).
    
    Handles:
        - 4D: (N, C, H, W) -> (N, C, H, W)
        - 5D 3D-CNN (VideoResNet): (B, C, T, H, W) -> permute(0, 2, 1, 3, 4) -> (B*T, C, H, W)
        - 5D Sequence format: (B, N, C, H, W) -> reshape(B*N, C, H, W)
    """
    if t.ndim == 4:
        return t
    elif t.ndim == 5:
        B, d1, d2, H, W = t.shape
        if d1 >= d2:
            # Format is (B, C, T, H, W) as output by VideoResNet 3D CNN
            return t.permute(0, 2, 1, 3, 4).contiguous().view(B * d2, d1, H, W)
        else:
            # Format is (B, N, C, H, W)
            return t.contiguous().view(B * d1, d2, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D activation tensor, got shape {t.shape}")


class UltrasoundGradCAM:
    """Mathematically rigorous Grad-CAM implementation for 512-D Ultrasound Feature Encoders.
    
    Constructs scalar differentiable objectives S(z) from latent representations:
        1. Energy / Norm: S = 0.5 * ||z||_2^2
        2. Latent Dimension: S = z_k for k in [0, D-1]
        3. Latent Direction: S = z^T v for v in R^D
        4. Pairwise Representation Similarity: S = cos(z_a, z_b)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module] = "layer4",
        embedding_dim: int = 512,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize UltrasoundGradCAM.
        
        Args:
            model: Feature encoder model mapping input x to latent embedding z.
            target_layer: Target convolutional module or string name ("layer1".."layer4").
            embedding_dim: Expected latent embedding dimension D.
            device: Execution device (cpu or cuda).
        """
        self.model = model
        self.target_module, self.target_layer_name = resolve_target_layer(model, target_layer)
        self.embedding_dim = embedding_dim
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._forward_hook_handle = None
        self._backward_hook_handle = None
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hook and tensor-level gradient hook on target activation."""
        self.remove_hooks()

        def forward_hook(module, input, output):
            # If output is a tuple/dict, extract primary feature tensor
            if isinstance(output, tuple):
                output = output[0]
            self._activations = output
            if output.requires_grad:
                def grad_hook(grad):
                    self._gradients = grad
                self._backward_hook_handle = output.register_hook(grad_hook)

        self._forward_hook_handle = self.target_module.register_forward_hook(forward_hook)

    def remove_hooks(self):
        """Removes all registered PyTorch hooks to avoid memory leaks."""
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None
        if self._backward_hook_handle is not None:
            self._backward_hook_handle.remove()
            self._backward_hook_handle = None
        self._activations = None
        self._gradients = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def _extract_embedding(self, forward_out: Any) -> torch.Tensor:
        """Standardizes forward pass output into latent embedding tensor (N, D)."""
        if isinstance(forward_out, tuple):
            forward_out = forward_out[0]

        # Handle 5D feature maps (B, N, C, H, W) from VideoResNet wrapper
        if forward_out.ndim == 5:
            # Spatial mean pool: (B, N, C, H, W) -> (B*N, C)
            B, N, C, H, W = forward_out.shape
            pooled = forward_out.mean(dim=(-1, -2))  # (B, N, C)
            return pooled.view(B * N, C)
        elif forward_out.ndim == 4:
            # (N, C, H, W) -> (N, C)
            return forward_out.mean(dim=(-1, -2))
        elif forward_out.ndim == 3:
            # (B, N, C) -> (B*N, C)
            B, N, C = forward_out.shape
            return forward_out.view(B * N, C)
        elif forward_out.ndim == 2:
            return forward_out
        else:
            raise ValueError(f"Unexpected encoder output shape: {forward_out.shape}")

    def _standardize_input(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Standardizes input into (B, N, 1, H, W) or (N, 1, H, W) PyTorch float tensor."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        if x.ndim == 2:
            # (H, W) -> (1, 1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # (N, H, W) -> (1, N, 1, H, W)
            x = x.unsqueeze(1).unsqueeze(0)
        elif x.ndim == 4:
            # (N, 1, H, W) -> (1, N, 1, H, W)
            x = x.unsqueeze(0)
        return x

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        objective: str = "energy",
        target_dimension: Optional[int] = None,
        direction: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_frame: Optional[Union[np.ndarray, torch.Tensor]] = None,
        explain_branch: str = "a",
        normalize_direction: bool = True,
        relu: bool = True,
        signed_mode: Optional[str] = None,
        custom_scalar_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ExplanationMetadata]:
        """Calculates mathematically sound Grad-CAM explanations for 512-D embeddings.
        
        Args:
            x: Input ultrasound tensor or numpy array.
            objective: One of:
                - "energy": S = 0.5 * ||z||_2^2 (Magnitude/energy of representation)
                - "latent_dimension": S = z_k for dimension target_dimension
                - "latent_direction": S = z^T v for concept/trajectory vector direction
                - "similarity": S = cos(z_a, z_b) pairwise representation similarity
                - "custom": S = custom_scalar_fn(z)
            target_dimension: Integer k in [0, D-1] for "latent_dimension" objective.
            direction: Vector v in R^D for "latent_direction" objective.
            target_frame: Paired frame x_b for "similarity" objective.
            explain_branch: For "similarity", which frame to explain: 'a' or 'b'.
            normalize_direction: Whether to L2-normalize direction vector v.
            relu: Standard Grad-CAM uses ReLU. Set False for signed attribution.
            signed_mode: If relu=False: 'signed' ([-1, 1]), 'positive', 'negative', 'absolute'.
            custom_scalar_fn: Custom differentiable mapping from embedding (N, D) to scalar.
            
        Returns:
            cam_maps: (N, H, W) tensor of spatial attribution heatmaps.
            metadata: ExplanationMetadata documenting all parameters and properties.
        """
        x_tensor = self._standardize_input(x)
        orig_H, orig_W = x_tensor.shape[-2], x_tensor.shape[-1]
        N_samples = x_tensor.shape[0] * x_tensor.shape[1]

        # Handle similarity objective paired input
        if objective == "similarity":
            if target_frame is None:
                raise ValueError("Objective 'similarity' requires 'target_frame' (paired frame x_b).")
            x_b_tensor = self._standardize_input(target_frame)
        else:
            x_b_tensor = None

        # Reset stored activations and gradients
        self._activations = None
        self._gradients = None

        # Enable gradient computation through model weights/activations
        self.model.zero_grad()
        
        with torch.enable_grad():
            x_tensor = x_tensor.clone().detach().requires_grad_(True)
            if x_b_tensor is not None:
                x_b_tensor = x_b_tensor.clone().detach().requires_grad_(True)

            # Forward pass on primary input
            forward_out = self.model(x_tensor)
            z = self._extract_embedding(forward_out)  # (N, D)
            D = z.shape[-1]

            # Check embedding normalization status
            z_norms = torch.norm(z, p=2, dim=-1)
            is_normalized = bool(torch.allclose(z_norms, torch.ones_like(z_norms), atol=1e-3))

            # Check target activations
            if self._activations is None:
                raise RuntimeError(f"Forward hook did not capture activations from '{self.target_layer_name}'.")

            act_tensor = self._activations
            act_std = _standardize_spatial_activation(act_tensor)

            # Construct scalar differentiable objective S
            if objective == "energy":
                if is_normalized:
                    raise ValueError(
                        "Embedding is already L2-normalized (||z||_2 = 1.0). "
                        "Energy/Norm objective is constant and uninformative. "
                        "Use 'latent_dimension' or 'latent_direction' instead."
                    )
                score = 0.5 * torch.sum(z ** 2)
                meta_target_dim = None
                meta_dir_name = None

            elif objective == "latent_dimension":
                if target_dimension is None:
                    raise ValueError("Objective 'latent_dimension' requires 'target_dimension' argument.")
                if not (0 <= target_dimension < D):
                    raise ValueError(f"target_dimension={target_dimension} out of bounds for D={D}.")
                score = torch.sum(z[:, target_dimension])
                meta_target_dim = int(target_dimension)
                meta_dir_name = None

            elif objective in ["latent_direction", "trajectory_probe"]:
                if direction is None:
                    raise ValueError(f"Objective '{objective}' requires 'direction' vector argument (e.g. w_traj).")
                if isinstance(direction, np.ndarray):
                    dir_tensor = torch.from_numpy(direction).float().to(self.device)
                else:
                    dir_tensor = direction.to(self.device).float()

                if dir_tensor.ndim != 1 or dir_tensor.shape[0] != D:
                    raise ValueError(f"Direction shape must be ({D},), got {dir_tensor.shape}")

                if normalize_direction:
                    dir_tensor = dir_tensor / (torch.norm(dir_tensor, p=2) + 1e-8)

                score = torch.sum(z @ dir_tensor)
                meta_target_dim = None
                meta_dir_name = "trajectory_probe_direction" if objective == "trajectory_probe" else "custom_or_pca_direction"

            elif objective == "similarity":
                # For similarity, compute embedding of frame b
                forward_b = self.model(x_b_tensor)
                z_b = self._extract_embedding(forward_b)

                # Cosine similarity: (z_a . z_b) / (||z_a|| * ||z_b||)
                z_a_norm = z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-8)
                z_b_norm = z_b / (torch.norm(z_b, p=2, dim=-1, keepdim=True) + 1e-8)
                cos_sim = torch.sum(z_a_norm * z_b_norm, dim=-1)  # (N,)
                score = torch.sum(cos_sim)
                meta_target_dim = None
                meta_dir_name = None

            elif objective == "custom":
                if custom_scalar_fn is None:
                    raise ValueError("Objective 'custom' requires 'custom_scalar_fn' callable.")
                score = custom_scalar_fn(z)
                meta_target_dim = None
                meta_dir_name = None

            else:
                raise ValueError(f"Unknown objective '{objective}'. Choose from: energy, latent_dimension, latent_direction, similarity, custom.")

            # Backward pass to compute gradients on target activation
            score.backward(retain_graph=False)

        if self._gradients is None:
            raise RuntimeError("Backward pass completed but activation gradients were not captured.")

        grad_tensor = self._gradients
        grad_std = _standardize_spatial_activation(grad_tensor)

        # Global average pooling of gradients to obtain channel importance weights alpha_k
        # alpha: (N, C, 1, 1)
        alpha = torch.mean(grad_std, dim=(-2, -1), keepdim=True)

        # Weighted combination of feature activation maps
        # cam_raw: (N, H', W')
        cam_raw = torch.sum(alpha * act_std, dim=1)

        # Handle ReLU / Signed Attribution modes
        if relu:
            cam_processed = F.relu(cam_raw)
            cam_norm = safe_min_max_normalize(cam_processed, signed=False)
            signed_flag = False
        else:
            signed_mode = signed_mode or "signed"
            signed_flag = True
            if signed_mode == "positive":
                cam_processed = F.relu(cam_raw)
                cam_norm = safe_min_max_normalize(cam_processed, signed=False)
            elif signed_mode == "negative":
                cam_processed = F.relu(-cam_raw)
                cam_norm = safe_min_max_normalize(cam_processed, signed=False)
            elif signed_mode == "absolute":
                cam_processed = torch.abs(cam_raw)
                cam_norm = safe_min_max_normalize(cam_processed, signed=False)
            elif signed_mode == "signed":
                cam_norm = safe_min_max_normalize(cam_raw, signed=True)
            else:
                raise ValueError(f"Unknown signed_mode '{signed_mode}'. Choose: positive, negative, absolute, signed.")

        # Bilinear upsampling to original ultrasound input spatial resolution (orig_H, orig_W)
        cam_4d = cam_norm.unsqueeze(1)  # (N, 1, H', W')
        cam_upsampled = F.interpolate(
            cam_4d,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (N, orig_H, orig_W)

        # Clean up hooks
        if self._backward_hook_handle is not None:
            self._backward_hook_handle.remove()
            self._backward_hook_handle = None

        metadata = ExplanationMetadata(
            method="gradcam",
            objective=objective,
            target_layer=self.target_layer_name,
            input_shape=list(x_tensor.shape),
            feature_shape=list(act_tensor.shape),
            embedding_dimension=D,
            normalized_embedding=is_normalized,
            signed_attribution=signed_flag,
            target_dimension=meta_target_dim,
            target_direction_name=meta_dir_name,
            explained_branch=explain_branch if objective == "similarity" else None,
            extra_info={
                "relu": relu,
                "signed_mode": signed_mode,
                "score_value": float(score.detach().cpu().item()),
            },
        )

        return cam_upsampled.detach().cpu(), metadata


class UltrasoundGradCAMPlusPlus(UltrasoundGradCAM):
    """Grad-CAM++ with higher-order gradient weighting for sharper ultrasound landmark localization."""

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        objective: str = "trajectory_probe",
        target_dimension: Optional[int] = None,
        direction: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_frame: Optional[Union[np.ndarray, torch.Tensor]] = None,
        explain_branch: str = "a",
        relu: bool = True,
        signed_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, ExplanationMetadata]:
        """Calculates Grad-CAM++ attribution heatmap with higher-order weighting."""
        cam_upsampled, meta = super().explain(
            x=x,
            objective=objective,
            target_dimension=target_dimension,
            direction=direction,
            target_frame=target_frame,
            explain_branch=explain_branch,
            relu=relu,
            signed_mode=signed_mode,
        )
        meta.method = "gradcam_plus_plus"
        return cam_upsampled, meta


class MultiScaleGradCAM:
    """Multi-scale layer-fused Grad-CAM combining fine boundary (layer2/3) and semantic (layer4) features."""

    def __init__(
        self,
        model: nn.Module,
        layers: List[str] = ["layer2", "layer3", "layer4"],
        layer_weights: Optional[List[float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.layers = layers
        self.layer_weights = layer_weights or [0.25, 0.35, 0.40]
        self.device = device or next(model.parameters()).device

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        objective: str = "trajectory_probe",
        target_dimension: Optional[int] = None,
        direction: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_frame: Optional[Union[np.ndarray, torch.Tensor]] = None,
        explain_branch: str = "a",
        relu: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ExplanationMetadata]:
        """Computes multi-scale fused attribution map across specified layers."""
        cams = []
        for layer_name in self.layers:
            with UltrasoundGradCAM(self.model, target_layer=layer_name, device=self.device) as gcam:
                cam, meta = gcam.explain(
                    x=x,
                    objective=objective,
                    target_dimension=target_dimension,
                    direction=direction,
                    target_frame=target_frame,
                    explain_branch=explain_branch,
                    relu=relu,
                )
                cams.append(cam)

        fused_cam = torch.zeros_like(cams[0])
        total_w = sum(self.layer_weights)
        for cam, w in zip(cams, self.layer_weights):
            fused_cam += (w / total_w) * cam

        fused_norm = safe_min_max_normalize(fused_cam, signed=False)
        fused_meta = ExplanationMetadata(
            method="multiscale_gradcam",
            objective=objective,
            target_layer="+".join(self.layers),
            input_shape=meta.input_shape,
            feature_shape=meta.feature_shape,
            embedding_dimension=meta.embedding_dimension,
            normalized_embedding=meta.normalized_embedding,
            signed_attribution=False,
            target_dimension=target_dimension,
            target_direction_name=meta.target_direction_name,
            explained_branch=meta.explained_branch,
            extra_info={"layers": self.layers, "weights": self.layer_weights},
        )
        return fused_norm, fused_meta


class UltrasoundEigenCAM:
    """Gradient-free Principal Component Activation Mapping (Eigen-CAM).
    
    Computes spatial attribution by projecting convolutional feature maps onto 
    their first principal component via SVD, independent of any task head or gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module] = "layer4",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.target_module, self.target_layer_name = resolve_target_layer(model, target_layer)
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        self.remove_hook()

        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self._activations = output

        self._hook_handle = self.target_module.register_forward_hook(hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._activations = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()

    @torch.no_grad()
    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        center_activations: bool = True,
    ) -> Tuple[torch.Tensor, ExplanationMetadata]:
        """Computes Eigen-CAM projection map.
        
        Args:
            x: Input ultrasound frames (N, H, W) or tensor.
            center_activations: Whether to subtract mean spatial activation before SVD.
            
        Returns:
            cam_maps: (N, H, W) normalized spatial principal component heatmap.
            metadata: ExplanationMetadata.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)

        orig_H, orig_W = x.shape[-2], x.shape[-1]
        self._activations = None

        # Forward pass
        _ = self.model(x)

        if self._activations is None:
            raise RuntimeError(f"Failed to capture activations from '{self.target_layer_name}'.")

        act = self._activations
        act_std = _standardize_spatial_activation(act)
        N_samples, C, H_act, W_act = act_std.shape
        cams = []

        for n in range(N_samples):
            # A_n: (C, H'*W')
            A_n = act_std[n].view(C, H_act * W_act)
            if center_activations:
                A_n_centered = A_n - A_n.mean(dim=1, keepdim=True)
            else:
                A_n_centered = A_n

            # SVD: A_n = U S V^T
            # U[:, 0] is the 1st principal component vector in channel space R^C
            try:
                U, _, _ = torch.linalg.svd(A_n_centered, full_matrices=False)
                v1 = U[:, 0]  # (C,)
                # Projection: (C,) @ (C, H'*W') -> (H'*W')
                proj = torch.matmul(v1, A_n_centered).view(H_act, W_act)
            except Exception:
                # Fallback to mean activation if SVD fails
                proj = act_std[n].mean(dim=0)

            # Ensure positive orientation (peak variance)
            if torch.abs(proj.min()) > torch.abs(proj.max()):
                proj = -proj

            proj_relu = F.relu(proj)
            proj_norm = safe_min_max_normalize(proj_relu, signed=False)
            cams.append(proj_norm)

        cams_tensor = torch.stack(cams, dim=0).unsqueeze(1)  # (N, 1, H', W')
        cams_upsampled = F.interpolate(
            cams_tensor,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (N, orig_H, orig_W)

        metadata = ExplanationMetadata(
            method="eigencam",
            objective="principal_component_projection",
            target_layer=self.target_layer_name,
            input_shape=list(x.shape),
            feature_shape=list(act.shape),
            embedding_dimension=C,
            normalized_embedding=False,
            signed_attribution=False,
        )

        return cams_upsampled.cpu(), metadata


class LatentOcclusion:
    """Model-agnostic sliding-window perturbation sensitivity explainer for latent embeddings."""

    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int] = (32, 32),
        stride: Tuple[int, int] = (16, 16),
        metric: str = "cosine",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.metric = metric
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def _extract_embedding(self, forward_out: Any) -> torch.Tensor:
        if isinstance(forward_out, tuple):
            forward_out = forward_out[0]
        if forward_out.ndim == 5:
            B, N, C, H, W = forward_out.shape
            return forward_out.mean(dim=(-1, -2)).view(B * N, C)
        elif forward_out.ndim == 4:
            return forward_out.mean(dim=(-1, -2))
        elif forward_out.ndim == 3:
            B, N, C = forward_out.shape
            return forward_out.view(B * N, C)
        return forward_out

    @torch.no_grad()
    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_dimension: Optional[int] = None,
        direction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ExplanationMetadata]:
        """Calculates perturbation sensitivity map by sliding occlusion patch across input."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)

        orig_H, orig_W = x.shape[-2], x.shape[-1]
        ph, pw = self.patch_size
        sh, sw = self.stride

        # Baseline unperturbed embedding
        z_base = self._extract_embedding(self.model(x))  # (N, D)
        N_samples, D = z_base.shape

        # Generate grid positions
        y_steps = list(range(0, orig_H - ph + 1, sh))
        x_steps = list(range(0, orig_W - pw + 1, sw))
        if y_steps[-1] + ph < orig_H:
            y_steps.append(orig_H - ph)
        if x_steps[-1] + pw < orig_W:
            x_steps.append(orig_W - pw)

        grid_H, grid_W = len(y_steps), len(x_steps)
        occ_maps = torch.zeros((N_samples, grid_H, grid_W), device=self.device)

        # Baseline scalar or vector
        for yi, y0 in enumerate(y_steps):
            for xi, x0 in enumerate(x_steps):
                x_occ = x.clone()
                x_occ[..., y0 : y0 + ph, x0 : x0 + pw] = 0.0  # Zero occlusion

                z_occ = self._extract_embedding(self.model(x_occ))

                if self.metric == "cosine":
                    # 1 - cos(z, z_occ): higher value means larger change (more sensitive)
                    z_b_norm = z_base / (torch.norm(z_base, p=2, dim=-1, keepdim=True) + 1e-8)
                    z_o_norm = z_occ / (torch.norm(z_occ, p=2, dim=-1, keepdim=True) + 1e-8)
                    diff = 1.0 - torch.sum(z_b_norm * z_o_norm, dim=-1)
                elif self.metric == "l2":
                    diff = torch.norm(z_base - z_occ, p=2, dim=-1)
                elif self.metric == "dimension":
                    if target_dimension is None:
                        raise ValueError("Metric 'dimension' requires target_dimension.")
                    diff = torch.abs(z_base[:, target_dimension] - z_occ[:, target_dimension])
                elif self.metric == "direction":
                    if direction is None:
                        raise ValueError("Metric 'direction' requires direction vector.")
                    d_norm = direction / (torch.norm(direction, p=2) + 1e-8)
                    diff = torch.abs((z_base - z_occ) @ d_norm)
                else:
                    raise ValueError(f"Unknown metric '{self.metric}'.")

                occ_maps[:, yi, xi] = diff

        # Interpolate grid to full image resolution
        occ_maps_4d = occ_maps.unsqueeze(1)
        occ_upsampled = F.interpolate(
            occ_maps_4d,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        occ_norm = safe_min_max_normalize(occ_upsampled, signed=False)

        metadata = ExplanationMetadata(
            method="latent_occlusion",
            objective=f"perturbation_{self.metric}",
            target_layer="input_pixels",
            input_shape=list(x.shape),
            feature_shape=[N_samples, grid_H, grid_W],
            embedding_dimension=D,
            normalized_embedding=False,
            signed_attribution=False,
            target_dimension=target_dimension,
            extra_info={"patch_size": self.patch_size, "stride": self.stride, "metric": self.metric},
        )

        return occ_norm.cpu(), metadata


class TrajectoryLinearProbe:
    """Supervised Linear Probe estimating trajectory progression from latent representations.
    
    Fits: w_traj = argmin_w || p - (Z w + b) ||^2 + alpha ||w||^2
    Scalar Objective: S_traj(z) = z^T w_traj
    """

    def __init__(self, alpha: float = 10.0, cv_folds: int = 5, standardize_direction: bool = True):
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.standardize_direction = standardize_direction
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.pearson_r: float = 0.0
        self.spearman_rho: float = 0.0
        self.r2_score: float = 0.0
        self.mse: float = 0.0
        self.cv_pearson_r: float = 0.0
        self.cv_spearman_rho: float = 0.0
        self.cv_r2_score: float = 0.0

    def fit(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        physical_positions: Union[np.ndarray, torch.Tensor],
    ) -> "TrajectoryLinearProbe":
        """Fits regularized ridge linear regression from latent embeddings Z to 1D physical displacement."""
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        if torch.is_tensor(physical_positions):
            physical_positions = physical_positions.detach().cpu().numpy()

        if physical_positions.ndim == 2:
            diffs = np.diff(physical_positions, axis=0)
            cum_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=-1))])
        else:
            cum_dist = physical_positions.astype(np.float64)

        Z = embeddings.astype(np.float64)
        N, D = Z.shape
        y = cum_dist

        # 1. Full Fit via Regularized normal equation: w = (Z_c^T Z_c + alpha * I)^(-1) Z_c^T y_c
        Z_mean = np.mean(Z, axis=0, keepdims=True)
        y_mean = np.mean(y)
        Z_c = Z - Z_mean
        y_c = y - y_mean

        reg = self.alpha * np.eye(D)
        w = np.linalg.solve(Z_c.T @ Z_c + reg, Z_c.T @ y_c)
        b = float(y_mean - np.squeeze(Z_mean @ w))

        y_pred = np.squeeze(Z @ w) + b

        # Fit metrics on training set
        r, _ = pearsonr(y, y_pred)
        rho, _ = spearmanr(y, y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)
        mse = float(np.mean((y - y_pred) ** 2))

        self.weights = w
        self.bias = b
        self.pearson_r = float(r)
        self.spearman_rho = float(rho)
        self.r2_score = float(r2)
        self.mse = mse

        # 2. Cross-Validation (K-Fold out-of-fold predictions)
        k_folds = min(self.cv_folds, N)
        if k_folds >= 2:
            indices = np.arange(N)
            # deterministic fold split
            fold_sizes = np.full(k_folds, N // k_folds, dtype=int)
            fold_sizes[: N % k_folds] += 1
            current = 0
            y_cv_pred = np.zeros(N)

            for f_size in fold_sizes:
                val_idx = indices[current : current + f_size]
                train_idx = np.setdiff1d(indices, val_idx)
                current += f_size

                Z_tr, y_tr = Z[train_idx], y[train_idx]
                Z_val = Z[val_idx]

                Z_tr_mean = np.mean(Z_tr, axis=0, keepdims=True)
                y_tr_mean = np.mean(y_tr)
                Z_tr_c = Z_tr - Z_tr_mean
                y_tr_c = y_tr - y_tr_mean

                w_fold = np.linalg.solve(Z_tr_c.T @ Z_tr_c + reg, Z_tr_c.T @ y_tr_c)
                b_fold = float(y_tr_mean - np.squeeze(Z_tr_mean @ w_fold))
                y_cv_pred[val_idx] = np.squeeze(Z_val @ w_fold) + b_fold

            cv_r, _ = pearsonr(y, y_cv_pred)
            cv_rho, _ = spearmanr(y, y_cv_pred)
            cv_ss_res = np.sum((y - y_cv_pred) ** 2)
            cv_r2 = 1.0 - (cv_ss_res / ss_tot)

            self.cv_pearson_r = float(cv_r)
            self.cv_spearman_rho = float(cv_rho)
            self.cv_r2_score = float(cv_r2)
        else:
            self.cv_pearson_r = self.pearson_r
            self.cv_spearman_rho = self.spearman_rho
            self.cv_r2_score = self.r2_score

        return self

    @property
    def direction_vector(self) -> np.ndarray:
        """Returns normalized unit direction vector in R^D."""
        if self.weights is None:
            raise ValueError("Probe has not been fitted yet.")
        norm = np.linalg.norm(self.weights)
        if norm > 1e-8 and self.standardize_direction:
            return self.weights / norm
        return self.weights

    @staticmethod
    def probe_encoder_hierarchy(
        feature_extractor: Any,
        sweep_frames: Union[np.ndarray, torch.Tensor],
        physical_positions: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        alpha: float = 10.0,
        cv_folds: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates linear trajectory probes across all hierarchy stages of the DualTrack encoder.
        
        Evaluates:
            - Stage 1 (3D CNN VideoResNet): spatial mean-pooled (N, 512)
            - Stage 2 (Spatial ViT CLS): (N, 64)
            - Stage 3 (Temporal Attention Projected): (N, 512)
            - Global Context Encoder (ResNet-18 Dense): (N, 512)
            
        Returns:
            hierarchy_probes: Dictionary of probe objects and fit metrics per stage.
        """
        from src.loaders.preprocessor import preprocess_frames_for_global_encoder, preprocess_frames_for_local_encoder

        dev = device or getattr(feature_extractor, "device", "cpu")
        if isinstance(sweep_frames, np.ndarray):
            frames_t = torch.from_numpy(sweep_frames.copy()).float()
        else:
            frames_t = sweep_frames.clone().float()

        if frames_t.max() > 1.0:
            frames_t = frames_t / 255.0

        N = frames_t.shape[0] if frames_t.ndim == 3 else frames_t.shape[1]

        results = {}
        with torch.no_grad():
            features = feature_extractor.extract_all_hierarchy_levels(frames_t)

            # Stage 1: 3D CNN (1, N, 512)
            z_s1 = features.stage1_pooled.squeeze(0).cpu().numpy()

            # Stage 2: Spatial ViT CLS token (1, N, 64)
            z_s2 = features.stage2_vit_cls.squeeze(0).cpu().numpy()

            # Stage 3: Temporal Module Projected (1, N, 512)
            z_s3 = features.stage3_projected.squeeze(0).cpu().numpy()

            # Dense Global Context Encoder across all N frames (N, 512)
            gx = preprocess_frames_for_global_encoder(frames_t, device=dev)
            idx_dense = torch.arange(N, device=dev).unsqueeze(0)
            z_glob = feature_extractor.global_encoder_module(gx, idx_dense).squeeze(0).cpu().numpy()

        if torch.is_tensor(physical_positions):
            pos_np = physical_positions.detach().cpu().numpy()
        else:
            pos_np = np.array(physical_positions)

        stages = {
            "stage1_3d_cnn": (z_s1, "Local Anatomy & Micro-Speckle"),
            "stage2_vit_cls": (z_s2, "Patch Appearance & Semantics"),
            "stage3_temporal": (z_s3, "Sequential Smoothing"),
            "global_context": (z_glob, "Global Trajectory Manifold"),
        }

        for stage_name, (z_mat, desc) in stages.items():
            probe = TrajectoryLinearProbe(alpha=alpha, cv_folds=cv_folds).fit(z_mat, pos_np)
            results[stage_name] = {
                "probe": probe,
                "weights": probe.weights,
                "direction": probe.direction_vector,
                "pearson_r": probe.pearson_r,
                "spearman_rho": probe.spearman_rho,
                "r2_score": probe.r2_score,
                "cv_pearson_r": probe.cv_pearson_r,
                "cv_spearman_rho": probe.cv_spearman_rho,
                "cv_r2_score": probe.cv_r2_score,
                "mse": probe.mse,
                "embedding_dim": z_mat.shape[1],
                "description": desc,
            }

        return results


class TrajectoryDirectionEstimator:
    """Calculates and sign-orients principal trajectory latent directions from reference sweeps."""

    @staticmethod
    def estimate_trajectory_direction(
        embeddings: Union[np.ndarray, torch.Tensor],
        physical_positions: Union[np.ndarray, torch.Tensor],
        standardize: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Computes PC1 of trajectory embeddings and aligns sign with physical progression.
        
        Args:
            embeddings: (N, D) embeddings along reference sweep.
            physical_positions: (N,) or (N, 3) physical positions along arm.
            standardize: Whether to L2-normalize returned direction.
            
        Returns:
            v_trajectory: (D,) unit vector aligned with sweep progression.
            correlation: Pearson correlation r with physical progression.
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        if torch.is_tensor(physical_positions):
            physical_positions = physical_positions.detach().cpu().numpy()

        if physical_positions.ndim == 2:
            # Cumulative 3D distance
            diffs = np.diff(physical_positions, axis=0)
            cum_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=-1))])
        else:
            cum_dist = physical_positions

        # Mean-center embeddings
        Z_mean = np.mean(embeddings, axis=0, keepdims=True)
        Z_centered = embeddings - Z_mean

        # SVD for Principal Component 1
        U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        v_pc1 = Vt[0]  # (D,)

        # Project trajectory onto PC1
        proj = Z_centered @ v_pc1

        # Check correlation with physical progression to resolve sign ambiguity
        r, _ = pearsonr(cum_dist, proj)
        if r < 0:
            v_trajectory = -v_pc1
            r = -r
        else:
            v_trajectory = v_pc1

        if standardize:
            v_trajectory = v_trajectory / (np.linalg.norm(v_trajectory) + 1e-8)

        return v_trajectory, float(r)


def evaluate_cam_faithfulness_deletion(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    cam_mask: torch.Tensor,
    objective_type: str = "trajectory_probe",
    target_dimension: Optional[int] = None,
    direction: Optional[torch.Tensor] = None,
    mask_fractions: List[float] = [0.05, 0.10, 0.20, 0.30, 0.50],
    num_random_trials: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Quantitative Deletion Faithfulness Test.
    
    Measures the drop in objective score when progressively masking top-attributed 
    pixels versus random pixels and least-attributed pixels.
    
    A faithful explainability method shows:
        |Delta S|_top > |Delta S|_random >= |Delta S|_least
    """
    dev = device or next(model.parameters()).device
    model.eval()

    if isinstance(image, np.ndarray):
        x = torch.from_numpy(image).float().to(dev)
    else:
        x = image.clone().to(dev)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(0)

    H, W = x.shape[-2], x.shape[-1]
    cam_np = cam_mask.detach().cpu().numpy()
    if cam_np.ndim == 3:
        cam_np = cam_np[0]

    def get_score(inp_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            out = model(inp_tensor)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim >= 3:
                z = out.mean(dim=(-1, -2)) if out.ndim == 4 else out.mean(dim=(-1, -2)).view(-1, out.shape[2])
            else:
                z = out
            if objective_type == "energy":
                return float(0.5 * torch.sum(z ** 2).item())
            elif objective_type == "latent_dimension":
                return float(z[0, target_dimension].item())
            elif objective_type in ["latent_direction", "trajectory_probe"]:
                d = direction.to(dev) / (torch.norm(direction.to(dev), p=2) + 1e-8)
                return float((z[0] @ d).item())
            else:
                return float(torch.norm(z, p=2).item())

    s_base = get_score(x)

    flat_cam = cam_np.flatten()
    sorted_indices_desc = np.argsort(-flat_cam)  # Highest attribution first
    sorted_indices_asc = np.argsort(flat_cam)    # Lowest attribution first
    total_pixels = len(flat_cam)

    top_drop = []
    least_drop = []
    random_drop = []

    for frac in mask_fractions:
        k = int(frac * total_pixels)

        # 1. Top Attributed Masking
        x_top = x.clone()
        mask_top = np.zeros(total_pixels, dtype=bool)
        mask_top[sorted_indices_desc[:k]] = True
        mask_top_t = torch.from_numpy(mask_top.reshape(H, W)).to(dev)
        x_top[..., mask_top_t] = 0.0
        s_top = get_score(x_top)
        top_drop.append(abs(s_base - s_top))

        # 2. Least Attributed Masking
        x_least = x.clone()
        mask_least = np.zeros(total_pixels, dtype=bool)
        mask_least[sorted_indices_asc[:k]] = True
        mask_least_t = torch.from_numpy(mask_least.reshape(H, W)).to(dev)
        x_least[..., mask_least_t] = 0.0
        s_least = get_score(x_least)
        least_drop.append(abs(s_base - s_least))

        # 3. Random Masking
        r_drops = []
        for _ in range(num_random_trials):
            x_rand = x.clone()
            rand_idx = np.random.choice(total_pixels, size=k, replace=False)
            mask_rand = np.zeros(total_pixels, dtype=bool)
            mask_rand[rand_idx] = True
            mask_rand_t = torch.from_numpy(mask_rand.reshape(H, W)).to(dev)
            x_rand[..., mask_rand_t] = 0.0
            r_drops.append(abs(s_base - get_score(x_rand)))
        random_drop.append(float(np.mean(r_drops)))

    # Compute Area Under Deletion Curve (AUDC) or mean difference
    audc_top = float(np.mean(top_drop))
    audc_rand = float(np.mean(random_drop))
    audc_least = float(np.mean(least_drop))
    is_faithful = bool(audc_top > audc_rand and audc_top > audc_least)

    return {
        "mask_fractions": mask_fractions,
        "top_attribution_drops": top_drop,
        "random_attribution_drops": random_drop,
        "least_attribution_drops": least_drop,
        "audc_top": audc_top,
        "audc_random": audc_rand,
        "audc_least": audc_least,
        "is_faithful": is_faithful,
    }
