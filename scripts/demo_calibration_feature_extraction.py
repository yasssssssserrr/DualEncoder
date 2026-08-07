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
    prepare_joint_binary_mask,
    differentiable_spatial_warp_2d,
    BoneWeightedCalibrationLoss,
)
from src.loaders.mhd_loader import load_robot_sweep, list_forearm_phantom_scans
from src.loaders.preprocessor import preprocess_frames_for_global_encoder
from src.config import REPORTS_DIR, FIGURES_DIR, ROBOT_DATA_DIR


def run_calibration_compensation_demo():
    print("=" * 80)
    print(" DUALTRACK & MDL-UzL/JOINT BINARY MASK CALIBRATION COMPENSATION DEMO")
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
    target_tensor = preprocess_frames_for_global_encoder(frame_target_np[None, None, ...], device=device).squeeze(1)

    # 3. Extract Layer 2 feature maps
    print("\n[3/5] Extracting Layer 2 Spatial Feature Maps (56x56x128)...")
    target_fmap = spatial_extractor.forward_layer2(target_tensor)
    print(f"  -> Target feature map shape: {target_fmap.shape}")
    print(f"  -> Backbone parameters frozen: {all(not p.requires_grad for p in spatial_extractor.parameters())}")

    # 4. Generate & Ingest Masks
    # Heuristic attention mask
    heuristic_mask = compute_bone_cortex_attention_mask(frame_target_np).to(device)

    # Simulate / Ingest high-confidence JOINT binary segmentation mask (cortex surface)
    # Binary segmentation isolates the two cortical arches (Radius and Ulna)
    joint_raw_binary = np.zeros_like(frame_target_np, dtype=np.uint8)
    # Cortex arches in phantom (approximate ground truth segmentation from JOINT)
    norm_img = frame_target_np.astype(np.float32) / 255.0
    joint_raw_binary[(norm_img > 0.45) & (norm_img < 0.98)] = 1
    # Zero top transducer artifact
    joint_raw_binary[:50, :] = 0
    joint_raw_binary[320:, :] = 0

    joint_mask_hard = prepare_joint_binary_mask(joint_raw_binary, target_size=(56, 56), device=device)
    joint_mask_dilated = prepare_joint_binary_mask(
        joint_raw_binary, target_size=(56, 56), dilation_radius=2, smooth_sigma=1.0, device=device
    )
    print(f"  -> Ingested JOINT Binary Mask (Coverage: {(joint_mask_hard > 0.5).float().mean():.1%})")

    # 5. Simulate a calibration misalignment
    print("\n[4/5] Simulating Miscalibration (tx=+0.08, ty=-0.05, theta=+5.0 deg)...")
    tx_true = torch.tensor([0.08], device=device)
    ty_true = torch.tensor([-0.05], device=device)
    theta_true = torch.tensor([np.radians(5.0)], dtype=torch.float32, device=device)

    with torch.no_grad():
        source_fmap = differentiable_spatial_warp_2d(target_fmap, tx_true, ty_true, theta_true)

    # 6. Optimize calibration parameters comparing:
    # A) Baseline Uniform Loss
    # B) Heuristic Attention Mask Loss
    # C) JOINT Binary Hard Mask Loss
    # D) JOINT Soft-Dilated Boost Loss
    experiments = [
        {"name": "1. Uniform Baseline (No Mask)", "fn": BoneWeightedCalibrationLoss(metric="cosine", mask_mode="boost", bone_boost_factor=0.0), "mask": None, "color": "#7f7f7f"},
        {"name": "2. Heuristic Attention Mask", "fn": BoneWeightedCalibrationLoss(metric="cosine", mask_mode="boost", bone_boost_factor=3.0), "mask": heuristic_mask, "color": "#ff7f0e"},
        {"name": "3. JOINT Binary Hard Mask", "fn": BoneWeightedCalibrationLoss(metric="cosine", mask_mode="hard"), "mask": joint_mask_hard, "color": "#2ca02c"},
        {"name": "4. JOINT Dilated Boost (Rec.)", "fn": BoneWeightedCalibrationLoss(metric="cosine", mask_mode="boost", bone_boost_factor=5.0), "mask": joint_mask_dilated, "color": "#1f77b4"},
    ]

    print("\n[5/5] Running Calibration Optimization with Different Weighting Strategies...")
    results = {}

    for exp in experiments:
        tx_est = torch.nn.Parameter(torch.zeros(1, device=device))
        ty_est = torch.nn.Parameter(torch.zeros(1, device=device))
        theta_est = torch.nn.Parameter(torch.zeros(1, device=device))

        optimizer = torch.optim.Adam([tx_est, ty_est, theta_est], lr=0.015)
        loss_hist = []

        for step in range(50):
            optimizer.zero_grad()
            warped_est = differentiable_spatial_warp_2d(source_fmap, tx_est, ty_est, theta_est)
            loss = exp["fn"](target_fmap, warped_est, bone_weight_mask=exp["mask"])
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        results[exp["name"]] = {
            "loss_hist": loss_hist,
            "final_tx": tx_est.item(),
            "final_ty": ty_est.item(),
            "final_theta": np.degrees(theta_est.item()),
            "color": exp["color"],
        }
        print(f"  {exp['name']:<30} | Final Error: d_tx={abs(tx_est.item() - (-tx_true.item())):.4f}, d_ty={abs(ty_est.item() - (-ty_true.item())):.4f}, d_theta={abs(np.degrees(theta_est.item()) - (-np.degrees(theta_true.item()))):.2f} deg")

    # Plot visual verification figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    # 1. Target B-Mode
    axes[0].imshow(frame_target_np, cmap="gray")
    axes[0].set_title("B-Mode Ultrasound", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. JOINT Binary Mask
    axes[1].imshow(joint_raw_binary, cmap="gray")
    axes[1].set_title("MDL-UzL/JOINT Binary Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Dilated Bone Weighting on 56x56
    axes[2].imshow(joint_mask_dilated.squeeze().cpu().numpy(), cmap="hot")
    axes[2].set_title("Resampled Bone Weight (56x56)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Layer 2 Feature Energy
    fmap_energy = target_fmap.squeeze(0).mean(dim=0).detach().cpu().numpy()
    axes[3].imshow(fmap_energy, cmap="viridis")
    axes[3].set_title("Layer 2 Features (56x56)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # 5. Convergence comparison
    for exp_name, data in results.items():
        axes[4].plot(data["loss_hist"], label=exp_name, color=data["color"], linewidth=2.2)
    axes[4].set_title("Optimization Convergence", fontsize=11, fontweight="bold")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Loss")
    axes[4].grid(True, linestyle="--", alpha=0.5)
    axes[4].legend(fontsize=8, loc="upper right")

    out_fig = FIGURES_DIR / "calibration_joint_mask_optimization.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\nSaved JOINT calibration comparison figure to: {out_fig}")
    print("=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_calibration_compensation_demo()
