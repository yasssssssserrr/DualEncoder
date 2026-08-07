"""Ablation study module isolating the contributions of Local-Only, Global-Only, and Dual-Encoder setups."""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from src.config import DEVICE
from src.diagnostics.ddf_metrics import compute_official_ddf_metrics, DDFMetrics
from src.diagnostics.trajectory_eval import pose_vec_to_mat
from src.loaders.mhd_loader import RobotSweep
from src.loaders.preprocessor import prepare_sweep_batch


@dataclass
class AblationSweepResult:
    sweep_id: str
    fusion_ddf: DDFMetrics
    local_only_ddf: DDFMetrics
    global_only_ddf: DDFMetrics


def run_ablation_on_sweep(
    full_model: nn.Module,
    sweep: RobotSweep,
    device: str = DEVICE,
) -> AblationSweepResult:
    """Run Dual-Encoder, Local-Only, and Global-Only inference on a single sweep."""
    full_model.eval()
    
    # Prepare inputs
    batch = prepare_sweep_batch(sweep.frames, device=device)
    ge = batch["global_encoder_images"]  # (1, N, 1, 224, 224)
    le = batch["local_encoder_images"]   # (1, N, 1, 256, 256)
    N = sweep.num_frames

    with torch.no_grad():
        # 1. Full Dual-Encoder (Fusion)
        full_model.disable_global_encoder = False
        pred_fusion = full_model(ge, le)[0].float().cpu().numpy()  # (N-1, 6)

        # 2. Local-Only (Disable Global Encoder)
        full_model.disable_global_encoder = True
        pred_local = full_model(ge, le)[0].float().cpu().numpy()   # (N-1, 6)
        full_model.disable_global_encoder = False  # Reset

        # 3. Global-Only (Interpolate sparse global encoder predictions or downsampled tracking)
        # In DualTrack, global_encoder predicts sparse global tracking
        sparse_indices = full_model.sampler(ge, full_model.local_encoder(le))
        sub_ge = torch.stack([ge[i][sparse_indices[i]].contiguous() for i in range(ge.shape[0])], dim=0)
        global_features = full_model.global_encoder(sub_ge, sparse_indices)
        
        # Linear projection of global features to relative steps
        # To simulate Global-Only, we run decoder with zeroed local features
        dummy_local = torch.zeros_like(full_model.local_encoder(le))
        global_only_out = full_model.fusion_module(dummy_local, encoder_hidden_states=global_features)
        pred_global_only = full_model.head(global_only_out)[0].float().cpu().numpy()

    # Integrate predicted relative vectors to global 4x4 transforms
    def integrate_to_glob(rel_vecs: np.ndarray) -> np.ndarray:
        T_glob = np.zeros((N, 4, 4), dtype=np.float64)
        T_glob[0] = sweep.transforms[0]
        for k in range(N - 1):
            step_mat = pose_vec_to_mat(rel_vecs[k])
            T_glob[k + 1] = T_glob[k] @ step_mat
        return T_glob

    glob_fusion = integrate_to_glob(pred_fusion)
    glob_local = integrate_to_glob(pred_local)
    glob_global = integrate_to_glob(pred_global_only)

    spacing = (sweep.spacing_mm[0], sweep.spacing_mm[1])
    H, W = sweep.frames.shape[1], sweep.frames.shape[2]

    ddf_fusion = compute_official_ddf_metrics(glob_fusion, sweep.transforms, spacing, (H, W))
    ddf_local = compute_official_ddf_metrics(glob_local, sweep.transforms, spacing, (H, W))
    ddf_global = compute_official_ddf_metrics(glob_global, sweep.transforms, spacing, (H, W))

    return AblationSweepResult(
        sweep_id=sweep.sweep_id,
        fusion_ddf=ddf_fusion,
        local_only_ddf=ddf_local,
        global_only_ddf=ddf_global,
    )
