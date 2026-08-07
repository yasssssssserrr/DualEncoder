r"""Bridge loader to safely import and instantiate DualTrack models from D:\DualTrack."""
import os
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

from src.config import CHECKPOINT_PATH, DEVICE, DUALTRACK_ROOT


def setup_dualtrack_path():
    r"""Ensure D:\DualTrack is in sys.path and extend src namespace."""
    root_str = str(DUALTRACK_ROOT.resolve())
    if root_str not in sys.path and DUALTRACK_ROOT.exists():
        sys.path.insert(0, root_str)

    # Extend Python namespace so 'src.models', 'src.utils', etc resolve to D:\DualTrack\src
    import src
    dt_src = str((DUALTRACK_ROOT / "src").resolve())
    if dt_src not in src.__path__:
        src.__path__.append(dt_src)
    for subpkg in ["models", "utils", "datasets"]:
        if hasattr(src, subpkg):
            pkg_obj = getattr(src, subpkg)
            if hasattr(pkg_obj, "__path__"):
                dt_sub = str((DUALTRACK_ROOT / "src" / subpkg).resolve())
                if dt_sub not in pkg_obj.__path__:
                    pkg_obj.__path__.append(dt_sub)


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path | str, strict: bool = False):
    """Load state_dict weights from a checkpoint file into model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    from src.utils import load_model_weights as dt_load_weights
    return dt_load_weights(model, checkpoint_path, strict=strict, handle_size_mismatch=True)


def build_dualtrack_model(
    checkpoint_path: Optional[Path | str] = CHECKPOINT_PATH,
    device: str = DEVICE,
    eval_mode: bool = True,
) -> nn.Module:
    """Instantiate the official DualTrack Fusion Model and load pretrained weights."""
    setup_dualtrack_path()
    from src.models.fusion_model.fusion_model import dualtrack_fusion_model

    model = dualtrack_fusion_model(
        local_encoder_cfg=dict(name="dualtrack_loc_enc_stg3_legacy"),
        decoder_hidden_size=512,
        grid_spacing=16,
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint_weights(model, checkpoint_path, strict=False)

    model.to(device)
    if eval_mode:
        model.eval()

    return model
