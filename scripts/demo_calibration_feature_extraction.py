"""Demonstration: Differentiable Layer 2 Feature Extraction & Calibration Compensation.

Demonstrates:
1. Extracting Layer 2 spatial feature maps (128 channels, 28x28) from frozen DualTrack backbone.
2. Generating Bone-Cortex Attention Masks from raw B-Mode ultrasound frames.
3. Evaluating differentiable spatial warping and gradient backpropagation to calibration parameters (tx, ty, theta).
4. Running an Adam calibration optimization loop to recover simulated calibration misalignment.
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.spatial_feature_extractor import DualTrackSpatialFeatureExtractor
from src.calibration.loss_functions import (
    compute_bone_cortex_attention_mask,
    differentiable_spatial_warp_2d,
    BoneWeightedCalibrationLoss,
)
from src.loaders.mhd_loader import load_robot_sweep, list_forearm_phantom_scans
from src.loaders.preprocessor import preprocess_frames_for_global_encoder
from src.config import REPORTS_DIR, FIGURES_DIR, ROBOT_DATA_DIR


def run_calibration_compensation_demo():
    print("=" * 80)
    print(" DUALTRACK DIFFERENTIABLE LAYER 2 CALIBRATION COMPENSATION DEMO")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Initializing DualTrack Spatial Feature Extractor on {device}...")
    spatial_extractor = DualTrackSpatialFeatureExtractor(device=device)

    # 2. Load reference robot sweep
    print("\n[2/5] Loading real robotic sweep from dataset...")
    mhd_files = list_forearm_phantom_scans(ROBOT_DATA_DIR)
    if not mhd_files:
        print("  -> No .mhd files found, using synthetic phantom.")
        frames = np.random.rand(10, 224, 224).astype(np.float32)
    else:
        sweep = load_robot_sweep(mhd_files[0])
        frames = sweep.frames
        print(f"  -> Loaded sweep '{sweep.sweep_id}' ({len(frames)} frames).")

    # Select frame with bone
    mid_idx = len(frames) // 2
    frame_target_np = frames[mid_idx]
    
    # Preprocess
    target_tensor = preprocess_frames_for_global_encoder(frame_target_np[None, None, ...], device=device).squeeze(1) # (1, 1, 224, 224)

    # 3. Extract Layer 2 feature maps
    print("\n[3/5] Extracting Layer 2 Spatial Feature Maps (56x56x128)...")
    target_fmap = spatial_extractor.forward_layer2(target_tensor)
    print(f"  -> Target feature map shape: {target_fmap.shape}")
    print(f"  -> Backbone parameters frozen: {all(not p.requires_grad for p in spatial_extractor.parameters())}")

    # 4. Generate Bone-Cortex Mask
    bone_mask = compute_bone_cortex_attention_mask(frame_target_np).to(device)
    print(f"  -> Computed Bone Cortex Attention Mask (shape: {bone_mask.shape})")

    # 5. Simulate a calibration misalignment Delta_T_true = (tx=0.08, ty=-0.05, theta=5.0 deg)
    print("\n[4/5] Simulating Miscalibration (tx=+0.08, ty=-0.05, theta=+5.0 deg)...")
    tx_true = torch.tensor([0.08], device=device)
    ty_true = torch.tensor([-0.05], device=device)
    theta_true = torch.tensor([np.radians(5.0)], dtype=torch.float32, device=device)

    # Create misaligned source feature map by warping target
    with torch.no_grad():
        source_fmap = differentiable_spatial_warp_2d(target_fmap, tx_true, ty_true, theta_true)

    # 6. Optimize calibration compensation parameter Delta_T_est to align source to target
    print("\n[5/5] Optimizing Calibration Parameter Delta_T_est via Gradient Descent...")
    tx_est = torch.nn.Parameter(torch.zeros(1, device=device))
    ty_est = torch.nn.Parameter(torch.zeros(1, device=device))
    theta_est = torch.nn.Parameter(torch.zeros(1, device=device))

    optimizer = torch.optim.Adam([tx_est, ty_est, theta_est], lr=0.015)
    loss_fn = BoneWeightedCalibrationLoss(metric="cosine", bone_boost_factor=3.0)

    loss_history = []
    for step in range(50):
        optimizer.zero_grad()
        # Warping source with estimated calibration compensation parameters
        warped_est = differentiable_spatial_warp_2d(source_fmap, tx_est, ty_est, theta_est)
        loss = loss_fn(target_fmap, warped_est, bone_weight_mask=bone_mask)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:02d} | Loss: {loss.item():.6f} | tx: {tx_est.item():+.4f} (true: {-tx_true.item():+.4f}) | ty: {ty_est.item():+.4f} (true: {-ty_true.item():+.4f}) | theta: {np.degrees(theta_est.item()):+.2f} deg (true: {-np.degrees(theta_true.item()):+.2f} deg)")

    # Plot visual verification figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    # 1. Target B-Mode
    axes[0].imshow(frame_target_np, cmap="gray")
    axes[0].set_title("Target Ultrasound Slice (B-Mode)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Bone Mask
    axes[1].imshow(bone_mask.squeeze().cpu().numpy(), cmap="hot")
    axes[1].set_title("Bone-Cortex Attention Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Layer 2 Mean Feature Energy
    fmap_energy = target_fmap.squeeze(0).mean(dim=0).detach().cpu().numpy()
    axes[2].imshow(fmap_energy, cmap="viridis")
    axes[2].set_title("Layer 2 Feature Energy (28x28)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Convergence Curve
    axes[3].plot(loss_history, color="#1f77b4", linewidth=2.5, label="Bone-Weighted Cosine Loss")
    axes[3].set_title("Calibration Optimization Curve", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Loss")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend()

    out_fig = FIGURES_DIR / "calibration_layer2_optimization.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\nSaved calibration optimization figure to: {out_fig}")
    print("=" * 80)
    print(" CALIBRATION DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_calibration_compensation_demo()
