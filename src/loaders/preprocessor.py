"""Preprocessing and image resizing for DualTrack feature extraction."""
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def preprocess_frames_for_local_encoder(
    frames: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Preprocess ultrasound frames for the Local Encoder (ViT spatial attention expects 256x256).
    
    Args:
        frames: Array/Tensor of shape (N, H, W) or (B, N, H, W) or (B, N, C, H, W).
        target_size: (height, width), default (256, 256).
        normalize: Convert uint8 [0, 255] to float32 [0, 1].
        device: Output device.
        
    Returns:
        tensor: Shape (B, N, 1, 256, 256) float32 on specified device.
    """
    if isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(frames.copy() if not frames.flags.writeable else frames)
    else:
        tensor = frames.clone()

    if tensor.ndim == 3:  # (N, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(2)  # (1, N, 1, H, W)
    elif tensor.ndim == 4:  # (B, N, H, W)
        tensor = tensor.unsqueeze(2)  # (B, N, 1, H, W)
    elif tensor.ndim == 5:
        if tensor.shape[2] != 1:
            tensor = tensor[:, :, :1]

    if tensor.dtype == torch.uint8 and normalize:
        tensor = tensor.float() / 255.0
    elif normalize and tensor.max() > 1.0:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    B, N, C, H, W = tensor.shape
    if (H, W) != target_size:
        # Reshape to (B*N*C, 1, H, W) for interpolation
        flat = tensor.view(B * N * C, 1, H, W)
        resized = F.interpolate(flat, size=target_size, mode="bilinear", align_corners=False)
        tensor = resized.view(B, N, C, target_size[0], target_size[1])

    return tensor.to(device)


def preprocess_frames_for_global_encoder(
    frames: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Preprocess ultrasound frames for the Global Encoder (expects 224x224).
    
    Args:
        frames: Array/Tensor of shape (N, H, W) or (B, N, H, W).
        target_size: (height, width), default (224, 224).
        normalize: Convert uint8 [0, 255] to float32 [0, 1].
        device: Output device.
        
    Returns:
        tensor: Shape (B, N, 1, 224, 224) float32 on specified device.
    """
    return preprocess_frames_for_local_encoder(
        frames=frames,
        target_size=target_size,
        normalize=normalize,
        device=device,
    )


def prepare_sweep_batch(
    frames: np.ndarray,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Prepare a full sweep for both local and global encoders.
    
    Returns dictionary with:
        'local_encoder_images': (1, N, 1, 256, 256)
        'global_encoder_images': (1, N, 1, 224, 224)
    """
    return {
        "local_encoder_images": preprocess_frames_for_local_encoder(frames, target_size=(256, 256), device=device),
        "global_encoder_images": preprocess_frames_for_global_encoder(frames, target_size=(224, 224), device=device),
    }
