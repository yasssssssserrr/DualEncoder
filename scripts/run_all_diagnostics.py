import json
from pathlib import Path
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.config import CHECKPOINT_PATH, DEVICE, REPORTS_DIR, ROBOT_DATA_DIR, TUSREC_TRAIN_DIR
from src.diagnostics.decorrelation_curve import compute_speckle_decorrelation
from src.diagnostics.domain_gap_analysis import compute_pca_embeddings, linear_cka
from src.diagnostics.motion_correlation import evaluate_motion_correlation
from src.diagnostics.trajectory_eval import evaluate_sweep_trajectory
from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep
from src.loaders.tusrec_loader import list_tusrec_hdf5_files, load_tusrec_sweep
from src.models.feature_extractors import DualTrackFeatureExtractor
from src.utils.visualization import (
    plot_3d_trajectory_reconstruction,
    plot_domain_manifold_pca,
    plot_multistage_motion_correlation,
    plot_speckle_decorrelation_curves,
)


def run_diagnostics():
    print("=" * 80)
    print(" DualTrack Encoder Validation & Deep Diagnostics on Robot Ultrasound Scans")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Robot dataset directory: {ROBOT_DATA_DIR}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover forearm phantom scans
    forearm_files = list_forearm_phantom_scans(ROBOT_DATA_DIR)
    print(f"\n[1/5] Found {len(forearm_files)} forearm phantom scans.")
    assert len(forearm_files) > 0, "No forearm phantom scans found!"

    # 2. Instantiate Feature Extractor
    print("\n[2/5] Instantiating DualTrack Feature Extractor...")
    extractor = DualTrackFeatureExtractor(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)

    # 3. Extract Features and Compute Diagnostics across all sweeps
    print("\n[3/5] Extracting Multi-Stage Features & Computing Metrics...")

    all_sweep_metrics = []
    robot_stage3_feats_list = []
    stage_correlations_accum = {
        "Stage 1 (3D CNN)": {"phys_dists": [], "feat_dists": [], "r_list": [], "rho_list": []},
        "Stage 2 (ViT CLS)": {"phys_dists": [], "feat_dists": [], "r_list": [], "rho_list": []},
        "Stage 3 (Temporal)": {"phys_dists": [], "feat_dists": [], "r_list": [], "rho_list": []},
        "Global Context": {"phys_dists": [], "feat_dists": [], "r_list": [], "rho_list": []},
    }
    stage_decorrelations_accum = {}

    for sweep_idx, mhd_file in enumerate(forearm_files):
        sweep = load_robot_sweep(mhd_file)
        print(f"\nProcessing Sweep [{sweep_idx + 1}/{len(forearm_files)}]: {sweep.sweep_id} ({sweep.num_frames} frames)...")

        # Full hierarchy extraction
        extracted = extractor.extract_all_hierarchy_levels(sweep.frames, sweep_id=sweep.sweep_id)

        # Stage features
        s1 = extracted.stage1_pooled[0].detach().cpu().numpy()  # (N, 512)
        s2 = extracted.stage2_vit_cls[0].detach().cpu().numpy()  # (N, 64)
        s3 = extracted.stage3_temporal[0].detach().cpu().numpy()  # (N, 64)
        glob = extracted.global_features[0].detach().cpu().numpy()  # (N_sparse, 512)

        # Expand global features to frame length for lag correlation
        if len(glob) != sweep.num_frames and extracted.sparse_indices is not None:
            from scipy.interpolate import interp1d
            orig_idx = extracted.sparse_indices.cpu().numpy()
            if orig_idx.ndim == 2:
                orig_idx = orig_idx[0]
            f_interp = interp1d(orig_idx, glob, axis=0, fill_value="extrapolate")
            glob_dense = f_interp(np.arange(sweep.num_frames))
        else:
            glob_dense = glob

        robot_stage3_feats_list.append(s3)

        # Motion correlations
        m_s1 = evaluate_motion_correlation(s1, sweep.transforms, stage_name="Stage 1 (3D CNN)")
        m_s2 = evaluate_motion_correlation(s2, sweep.transforms, stage_name="Stage 2 (ViT CLS)")
        m_s3 = evaluate_motion_correlation(s3, sweep.transforms, stage_name="Stage 3 (Temporal)")
        m_glob = evaluate_motion_correlation(glob_dense, sweep.transforms, stage_name="Global Context")

        # Accumulate correlation data
        for stage_name, feat_arr, m_obj in [
            ("Stage 1 (3D CNN)", s1, m_s1),
            ("Stage 2 (ViT CLS)", s2, m_s2),
            ("Stage 3 (Temporal)", s3, m_s3),
            ("Global Context", glob_dense, m_glob),
        ]:
            from src.diagnostics.motion_correlation import compute_feature_distance_matrix, compute_physical_distance_matrix
            fdm = compute_feature_distance_matrix(feat_arr, metric="cosine")
            pdm, _ = compute_physical_distance_matrix(sweep.transforms)
            for lag in range(1, min(15, sweep.num_frames)):
                for i in range(sweep.num_frames - lag):
                    stage_correlations_accum[stage_name]["phys_dists"].append(pdm[i, i + lag])
                    stage_correlations_accum[stage_name]["feat_dists"].append(fdm[i, i + lag])
            stage_correlations_accum[stage_name]["r_list"].append(m_obj.pearson_r)
            stage_correlations_accum[stage_name]["rho_list"].append(m_obj.spearman_rho)

        # Speckle decorrelation
        dec_s1 = compute_speckle_decorrelation(s1, sweep.transforms)
        dec_s2 = compute_speckle_decorrelation(s2, sweep.transforms)
        dec_s3 = compute_speckle_decorrelation(s3, sweep.transforms)
        dec_glob = compute_speckle_decorrelation(glob_dense, sweep.transforms)

        if sweep_idx == 0:
            stage_decorrelations_accum = {
                "Stage 1 (3D CNN)": dec_s1,
                "Stage 2 (ViT CLS)": dec_s2,
                "Stage 3 (Temporal)": dec_s3,
                "Global Context": dec_glob,
            }

        # Trajectory reconstruction
        traj_m = evaluate_sweep_trajectory(
            extracted.pred_rel_poses,
            sweep.transforms,
            sweep_id=sweep.sweep_id,
        )

        sweep_record = {
            "sweep_id": sweep.sweep_id,
            "num_frames": sweep.num_frames,
            "correlations": {
                "stage1_3d_cnn": {"pearson_r": m_s1.pearson_r, "spearman_rho": m_s1.spearman_rho},
                "stage2_vit_cls": {"pearson_r": m_s2.pearson_r, "spearman_rho": m_s2.spearman_rho},
                "stage3_temporal": {"pearson_r": m_s3.pearson_r, "spearman_rho": m_s3.spearman_rho},
                "global_context": {"pearson_r": m_glob.pearson_r, "spearman_rho": m_glob.spearman_rho},
            },
            "decorrelation": {
                "stage1_fwhm_mm": dec_s1.fwhm_mm,
                "stage2_fwhm_mm": dec_s2.fwhm_mm,
                "stage3_fwhm_mm": dec_s3.fwhm_mm,
                "global_fwhm_mm": dec_glob.fwhm_mm,
            },
            "trajectory": {
                "mean_lpe_mm": traj_m.mean_lpe_mm,
                "mean_gpe_mm": traj_m.mean_gpe_mm,
                "max_gpe_mm": traj_m.max_gpe_mm,
                "endpoint_error_mm": traj_m.endpoint_error_mm,
                "mean_rot_error_deg": traj_m.mean_rot_error_deg,
                "drift_percent": traj_m.drift_percent,
            },
        }
        all_sweep_metrics.append(sweep_record)
        print(f"  -> S3 Correlation: r={m_s3.pearson_r:.3f}, rho={m_s3.spearman_rho:.3f} | LPE: {traj_m.mean_lpe_mm:.3f}mm | GPE: {traj_m.mean_gpe_mm:.2f}mm | Drift: {traj_m.drift_percent:.1f}%")

        # Save trajectory 3D plot for first 2 sweeps
        if sweep_idx < 2:
            plot_3d_trajectory_reconstruction(
                traj_m.gt_trajectory,
                traj_m.pred_trajectory,
                sweep_id=sweep.sweep_id,
                save_path=fig_dir / f"trajectory_3d_{sweep.sweep_id}.png",
            )

    # 4. Domain Gap Analysis with TUS-REC Training Sweeps
    print("\n[4/5] Computing Cross-Domain CKA Alignment & PCA Manifold Projection with TUS-REC...")
    tusrec_files = list_tusrec_hdf5_files(TUSREC_TRAIN_DIR)
    tusrec_stage3_feats_list = []

    if tusrec_files:
        for t_file in tusrec_files[:5]:  # Take 5 sample training sweeps
            try:
                t_sweep = load_tusrec_sweep(t_file)
                t_ext = extractor.extract_all_hierarchy_levels(t_sweep.frames[:47], sweep_id=t_sweep.sweep_id)
                tusrec_stage3_feats_list.append(t_ext.stage3_temporal[0].detach().cpu().numpy())
            except Exception as e:
                print(f"  Warning loading {t_file.name}: {e}")

    cka_score = 0.0
    if tusrec_stage3_feats_list and robot_stage3_feats_list:
        t_concat = np.vstack(tusrec_stage3_feats_list)
        r_concat = np.vstack(robot_stage3_feats_list)
        cka_score = linear_cka(t_concat, r_concat)
        print(f"  -> Cross-Domain Linear CKA (TUS-REC vs. Robot Ultrasound): {cka_score:.4f}")

        # PCA Projections
        (t_pca, r_pca), _ = compute_pca_embeddings([t_concat[:200], r_concat[:200]], n_components=2)
        plot_domain_manifold_pca(
            t_pca,
            r_pca,
            save_path=fig_dir / "domain_gap_manifold_pca.png",
            title=f"Latent Manifold: TUS-REC vs. Robot Forearm (CKA = {cka_score:.3f})",
        )

    # 5. Plot Multi-Stage Correlation and Decorrelation Curves
    print("\n[5/5] Generating Diagnostic Figures and Final Metrics Summary...")
    for st_name, d_dict in stage_correlations_accum.items():
        d_dict["phys_dists"] = np.array(d_dict["phys_dists"])
        d_dict["feat_dists"] = np.array(d_dict["feat_dists"])
        d_dict["pearson_r"] = float(np.mean(d_dict["r_list"]))
        d_dict["spearman_rho"] = float(np.mean(d_dict["rho_list"]))

    plot_multistage_motion_correlation(
        stage_correlations_accum,
        save_path=fig_dir / "motion_correlation_multistage.png",
    )

    if stage_decorrelations_accum:
        plot_speckle_decorrelation_curves(
            stage_decorrelations_accum,
            save_path=fig_dir / "speckle_decorrelation_curves.png",
        )

    # Summary Statistics
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_forearm_sweeps": len(all_sweep_metrics),
        "mean_correlations": {
            "stage1_3d_cnn": {
                "pearson_r": float(np.mean([s["correlations"]["stage1_3d_cnn"]["pearson_r"] for s in all_sweep_metrics])),
                "spearman_rho": float(np.mean([s["correlations"]["stage1_3d_cnn"]["spearman_rho"] for s in all_sweep_metrics])),
            },
            "stage2_vit_cls": {
                "pearson_r": float(np.mean([s["correlations"]["stage2_vit_cls"]["pearson_r"] for s in all_sweep_metrics])),
                "spearman_rho": float(np.mean([s["correlations"]["stage2_vit_cls"]["spearman_rho"] for s in all_sweep_metrics])),
            },
            "stage3_temporal": {
                "pearson_r": float(np.mean([s["correlations"]["stage3_temporal"]["pearson_r"] for s in all_sweep_metrics])),
                "spearman_rho": float(np.mean([s["correlations"]["stage3_temporal"]["spearman_rho"] for s in all_sweep_metrics])),
            },
            "global_context": {
                "pearson_r": float(np.mean([s["correlations"]["global_context"]["pearson_r"] for s in all_sweep_metrics])),
                "spearman_rho": float(np.mean([s["correlations"]["global_context"]["spearman_rho"] for s in all_sweep_metrics])),
            },
        },
        "mean_trajectory_metrics": {
            "mean_lpe_mm": float(np.mean([s["trajectory"]["mean_lpe_mm"] for s in all_sweep_metrics])),
            "mean_gpe_mm": float(np.mean([s["trajectory"]["mean_gpe_mm"] for s in all_sweep_metrics])),
            "mean_endpoint_error_mm": float(np.mean([s["trajectory"]["endpoint_error_mm"] for s in all_sweep_metrics])),
            "mean_drift_percent": float(np.mean([s["trajectory"]["drift_percent"] for s in all_sweep_metrics])),
        },
        "domain_alignment": {
            "linear_cka": float(cka_score),
        },
        "per_sweep_details": all_sweep_metrics,
    }

    out_json = REPORTS_DIR / "metrics_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" DIAGNOSTIC BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" Summary Metrics: {out_json}")
    print(f" Figures saved to: {fig_dir}")
    print("=" * 80)
    print(f"  Stage 1 (3D CNN) Mean Pearson r:  {summary['mean_correlations']['stage1_3d_cnn']['pearson_r']:.3f}")
    print(f"  Stage 2 (ViT CLS) Mean Pearson r: {summary['mean_correlations']['stage2_vit_cls']['pearson_r']:.3f}")
    print(f"  Stage 3 (Temporal) Mean Pearson r:{summary['mean_correlations']['stage3_temporal']['pearson_r']:.3f}")
    print(f"  Global Context Mean Pearson r:    {summary['mean_correlations']['global_context']['pearson_r']:.3f}")
    print(f"  Mean Step Local Error (LPE):      {summary['mean_trajectory_metrics']['mean_lpe_mm']:.3f} mm")
    print(f"  Mean Global Position Error (GPE): {summary['mean_trajectory_metrics']['mean_gpe_mm']:.2f} mm")
    print(f"  Mean Trajectory Drift:            {summary['mean_trajectory_metrics']['mean_drift_percent']:.1f} %")
    print(f"  Cross-Domain Linear CKA:          {summary['domain_alignment']['linear_cka']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()
