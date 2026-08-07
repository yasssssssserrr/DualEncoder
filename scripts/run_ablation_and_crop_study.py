"""Comprehensive Ablation, Crop Sensitivity, and Manifold Analysis Benchmark."""
import json
from pathlib import Path
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import CHECKPOINT_PATH, DEVICE, REPORTS_DIR, ROBOT_DATA_DIR
from src.diagnostics.ablation_study import run_ablation_on_sweep
from src.diagnostics.crop_sensitivity import (
    apply_center_crop,
    apply_direct_resize,
    apply_start50_crop,
    evaluate_crop_sensitivity,
)
from src.diagnostics.manifold_learning import compute_tsne_umap_projections
from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep
from src.loaders.preprocessor import prepare_sweep_batch
from src.models.dualtrack_bridge import build_dualtrack_model
from src.models.feature_extractors import DualTrackFeatureExtractor
from src.utils.visualization import (
    plot_ablation_comparison_bar,
    plot_crop_sensitivity_matrix,
    plot_tsne_umap_manifold,
)


def run_ablation_and_crop_study():
    print("=" * 80)
    print(" DualTrack Deep Ablation & Feature Representation Analysis")
    print("=" * 80)

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Scans
    forearm_files = list_forearm_phantom_scans(ROBOT_DATA_DIR)
    print(f"\n[1/4] Found {len(forearm_files)} forearm phantom sweeps.")

    # 2. Build Models
    print("\n[2/4] Initializing DualTrack Model & Feature Extractor...")
    full_model = build_dualtrack_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE, eval_mode=True)
    extractor = DualTrackFeatureExtractor(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)

    # ---------------------------------------------------------
    # PART 1: Ablation Study (Dual-Encoder vs Local-Only vs Global-Only)
    # ---------------------------------------------------------
    print("\n[3/4] Running Ablation Studies across all 10 forearm sweeps...")
    fusion_lpes, local_lpes, global_lpes = [], [], []
    fusion_gpes, local_gpes, global_gpes = [], [], []
    fusion_fdrs, local_fdrs, global_fdrs = [], [], []
    fusion_max_gpes, local_max_gpes, global_max_gpes = [], [], []

    all_ablation_records = []
    all_global_features_list = []
    all_frame_indices_list = []

    for sweep_idx, mhd_file in enumerate(forearm_files):
        sweep = load_robot_sweep(mhd_file)
        ab_res = run_ablation_on_sweep(full_model, sweep, device=DEVICE)

        fusion_lpes.append(ab_res.fusion_ddf.lpe_um)
        local_lpes.append(ab_res.local_only_ddf.lpe_um)
        global_lpes.append(ab_res.global_only_ddf.lpe_um)

        fusion_gpes.append(ab_res.fusion_ddf.gpe_mm)
        local_gpes.append(ab_res.local_only_ddf.gpe_mm)
        global_gpes.append(ab_res.global_only_ddf.gpe_mm)

        fusion_fdrs.append(ab_res.fusion_ddf.final_drift_rate_pct)
        local_fdrs.append(ab_res.local_only_ddf.final_drift_rate_pct)
        global_fdrs.append(ab_res.global_only_ddf.final_drift_rate_pct)

        fusion_max_gpes.append(ab_res.fusion_ddf.max_gpe_mm)
        local_max_gpes.append(ab_res.local_only_ddf.max_gpe_mm)
        global_max_gpes.append(ab_res.global_only_ddf.max_gpe_mm)

        all_ablation_records.append({
            "sweep_id": sweep.sweep_id,
            "fusion": {
                "lpe_um": ab_res.fusion_ddf.lpe_um,
                "gpe_mm": ab_res.fusion_ddf.gpe_mm,
                "fdr_pct": ab_res.fusion_ddf.final_drift_rate_pct,
                "max_gpe_mm": ab_res.fusion_ddf.max_gpe_mm,
            },
            "local_only": {
                "lpe_um": ab_res.local_only_ddf.lpe_um,
                "gpe_mm": ab_res.local_only_ddf.gpe_mm,
                "fdr_pct": ab_res.local_only_ddf.final_drift_rate_pct,
                "max_gpe_mm": ab_res.local_only_ddf.max_gpe_mm,
            },
            "global_only": {
                "lpe_um": ab_res.global_only_ddf.lpe_um,
                "gpe_mm": ab_res.global_only_ddf.gpe_mm,
                "fdr_pct": ab_res.global_only_ddf.final_drift_rate_pct,
                "max_gpe_mm": ab_res.global_only_ddf.max_gpe_mm,
            },
        })

        # Extract global features for manifold visualization
        with torch.no_grad():
            batch = prepare_sweep_batch(sweep.frames, device=DEVICE)
            glob_f, _ = extractor.extract_global_context(batch["global_encoder_images"])
            all_global_features_list.append(glob_f[0].cpu().numpy())
            all_frame_indices_list.append(np.arange(len(glob_f[0])))

        print(f"  Sweep [{sweep_idx+1}/{len(forearm_files)}] {sweep.sweep_id}: "
              f"Fusion GPE={ab_res.fusion_ddf.gpe_mm:.2f}mm (LPE={ab_res.fusion_ddf.lpe_um:.1f}µm) | "
              f"Local-Only GPE={ab_res.local_only_ddf.gpe_mm:.2f}mm | "
              f"Global-Only GPE={ab_res.global_only_ddf.gpe_mm:.2f}mm")

    ablation_summary = {
        "DualTrack (Fusion)": {
            "lpe_um": float(np.mean(fusion_lpes)),
            "gpe_mm": float(np.mean(fusion_gpes)),
            "fdr_pct": float(np.mean(fusion_fdrs)),
            "max_gpe_mm": float(np.mean(fusion_max_gpes)),
        },
        "Local-Only": {
            "lpe_um": float(np.mean(local_lpes)),
            "gpe_mm": float(np.mean(local_gpes)),
            "fdr_pct": float(np.mean(local_fdrs)),
            "max_gpe_mm": float(np.mean(local_max_gpes)),
        },
        "Global-Only": {
            "lpe_um": float(np.mean(global_lpes)),
            "gpe_mm": float(np.mean(global_gpes)),
            "fdr_pct": float(np.mean(global_fdrs)),
            "max_gpe_mm": float(np.mean(global_max_gpes)),
        },
    }

    # Plot ablation comparison
    plot_ablation_comparison_bar(
        ablation_summary,
        save_path=fig_dir / "ablation_study_comparison.png",
    )

    # ---------------------------------------------------------
    # PART 2: Preprocessing & Crop Sensitivity Analysis
    # ---------------------------------------------------------
    print("\n[4/4] Evaluating Preprocessing & Crop Variations...")
    sample_sweep = load_robot_sweep(forearm_files[0])
    
    # Prepare crop variations
    crop_dict = {
        "Start 50 Crop": (
            apply_start50_crop(sample_sweep.frames, target_size=(256, 256)),
            apply_start50_crop(sample_sweep.frames, target_size=(224, 224)),
        ),
        "Direct Resize (Full FOV)": (
            apply_direct_resize(sample_sweep.frames, target_size=(256, 256)),
            apply_direct_resize(sample_sweep.frames, target_size=(224, 224)),
        ),
        "Aspect Padded": (
            prepare_sweep_batch(sample_sweep.frames, device="cpu")["local_encoder_images"],
            prepare_sweep_batch(sample_sweep.frames, device="cpu")["global_encoder_images"],
        ),
    }

    crop_metrics = evaluate_crop_sensitivity(extractor, sample_sweep.frames, crop_dict)

    # Plot Crop Sensitivity Matrix
    plot_crop_sensitivity_matrix(
        crop_metrics,
        save_path=fig_dir / "crop_sensitivity_matrix.png",
    )

    # ---------------------------------------------------------
    # PART 3: t-SNE & UMAP Latent Manifold Projection
    # ---------------------------------------------------------
    print("\n[5/5] Computing t-SNE and UMAP 2D Anatomical Manifolds for Global Encoder...")
    concat_global_f = np.vstack(all_global_features_list)
    concat_indices = np.concatenate(all_frame_indices_list)

    manifold_res = compute_tsne_umap_projections(
        concat_global_f,
        frame_indices=concat_indices,
        perplexity=10.0,
    )

    plot_tsne_umap_manifold(
        manifold_res.tsne_2d,
        manifold_res.umap_2d,
        manifold_res.frame_indices,
        title="Global ResNet3D (512-D) Anatomical Manifold: Forearm Phantom Scans",
        save_path=fig_dir / "global_encoder_manifold_tsne_umap.png",
    )

    # Save full results JSON
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paper_table_1_reference": {
            "DualTrack": {"lpe_um": 122.01, "gpe_mm": 4.93},
            "Local-Only": {"lpe_um": 140.90, "gpe_mm": 7.36},
            "Global-Only": {"lpe_um": 757.64, "gpe_mm": 11.26},
        },
        "robot_phantom_ablation_summary": ablation_summary,
        "crop_sensitivity": [
            {
                "crop_name": cm.crop_name,
                "stage1_3d_cnn": cm.stage1_cosine_sim,
                "stage2_vit_cls": cm.stage2_cosine_sim,
                "stage3_temporal": cm.stage3_cosine_sim,
                "global_resnet3d": cm.global_cosine_sim,
            }
            for cm in crop_metrics
        ],
        "per_sweep_ablation": all_ablation_records,
    }

    out_json = REPORTS_DIR / "ablation_and_crop_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\n" + "=" * 80)
    print(" ABLATION & CROP SENSITIVITY BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f" Summary JSON: {out_json}")
    print(f" Figures Directory: {fig_dir}")
    print("\n--- Ablation Comparison Summary (Forearm Phantom Sweeps) ---")
    print(f"  DualTrack (Fusion): LPE = {ablation_summary['DualTrack (Fusion)']['lpe_um']:.1f} µm | GPE = {ablation_summary['DualTrack (Fusion)']['gpe_mm']:.2f} mm | FDR = {ablation_summary['DualTrack (Fusion)']['fdr_pct']:.1f}%")
    print(f"  Local-Only:         LPE = {ablation_summary['Local-Only']['lpe_um']:.1f} µm | GPE = {ablation_summary['Local-Only']['gpe_mm']:.2f} mm | FDR = {ablation_summary['Local-Only']['fdr_pct']:.1f}%")
    print(f"  Global-Only:        LPE = {ablation_summary['Global-Only']['lpe_um']:.1f} µm | GPE = {ablation_summary['Global-Only']['gpe_mm']:.2f} mm | FDR = {ablation_summary['Global-Only']['fdr_pct']:.1f}%")
    print("\n--- Preprocessing / Crop Cosine Similarity (vs. Center Crop) ---")
    for cm in crop_metrics:
        print(f"  {cm.crop_name:25s}: S1={cm.stage1_cosine_sim:.3f} | S2={cm.stage2_cosine_sim:.3f} | S3={cm.stage3_cosine_sim:.3f} | Glob={cm.global_cosine_sim:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation_and_crop_study()
