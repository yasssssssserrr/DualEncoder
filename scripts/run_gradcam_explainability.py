"""Explainability and Grad-CAM Diagnostic Suite for 512-D Ultrasound Feature Encoder.

Comprehensive Research Suite:
    - Experiment 1: Layer-Wise Trajectory Probing across Encoder Hierarchy (3D CNN, ViT, Temporal, Global Context)
    - Experiment 2: Trajectory Information Emergence: Local Micro-Speckle vs. Global Context Anchor
    - Experiment 3: Headline Trajectory Probe Attribution (S_traj = z^T w_traj across sweep progression)
    - Experiment 4: Layer Hierarchy Attribution (Stem -> Layer4, Multi-Scale, Eigen-CAM, Latent Occlusion)
    - Experiment 5: Secondary Explainability Analyses (Unsupervised PC1, Latent Dim #457, Pairwise Cosine Similarity)
    - Experiment 6: Quantitative Faithfulness Validation (Pixel Deletion Test & AUDC)
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.config import CHECKPOINT_PATH, DEVICE, REPORTS_DIR
from src.diagnostics.gradcam import (
    LatentOcclusion,
    MultiScaleGradCAM,
    TrajectoryDirectionEstimator,
    TrajectoryLinearProbe,
    UltrasoundEigenCAM,
    UltrasoundGradCAM,
    UltrasoundGradCAMPlusPlus,
    evaluate_cam_faithfulness_deletion,
)
from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep
from src.models.feature_extractors import DualTrackFeatureExtractor
from src.utils.visualization import (
    overlay_cam_on_image,
    plot_explainability_multipanel,
    plot_faithfulness_deletion_curves,
    plot_local_vs_global_trajectory_attribution,
    set_custom_style,
)


def run_explainability_suite(
    dataset_dir: str = "C:/Users/Ibourk/Downloads/Probe_Calib_Single_Filament_2/Probe_Calib_Single_Filament_2",
    checkpoint_path: str = str(CHECKPOINT_PATH),
    device: str = DEVICE,
    output_dir: Path = REPORTS_DIR,
):
    print("=" * 95)
    print(" DUALTRACK 512-D FEATURE EXPLAINABILITY & TRAJECTORY PROBING SUITE")
    print("=" * 95)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Extractor
    print(f"\n[1/6] Loading DualTrack Feature Extractor on {device}...")
    extractor = DualTrackFeatureExtractor(checkpoint_path=checkpoint_path, device=device)
    extractor.eval()

    # Access local CNN backbone module & global CNN backbone
    local_cnn = extractor.cnn_backbone_module
    global_cnn = extractor.global_encoder_module.backbone[0]

    # 2. Discover and Load Ultrasound Sweep
    print(f"\n[2/6] Discovering robotic sweeps in {dataset_dir}...")
    mhd_files = list_forearm_phantom_scans(dataset_dir)
    if not mhd_files:
        dataset_dir = "C:/Users/Ibourk/Downloads/Probe_Calib_Single_Filament_3/Probe_Calib_Single_Filament_3"
        mhd_files = list_forearm_phantom_scans(dataset_dir)
    print(f"Found {len(mhd_files)} robotic sweeps. Loading reference sweep: {mhd_files[0].name}")
    sweep = load_robot_sweep(mhd_files[0])
    N_frames = sweep.num_frames
    frames = sweep.frames  # (N, H, W) in uint8 [0, 255]
    positions = sweep.transforms[:, :3, 3]  # (N, 3) translations in mm

    # Select representative keyframe (middle of sweep)
    mid_idx = N_frames // 2
    key_frame = frames[mid_idx]  # (H, W)
    key_frame_float = key_frame.astype(np.float32) / 255.0

    # -------------------------------------------------------------------------
    # Experiment 1: Layer-Wise Trajectory Probing across Encoder Hierarchy
    # -------------------------------------------------------------------------
    print("\n[3/6] Running Experiment 1: Layer-Wise Trajectory Probing across Hierarchy...")
    hierarchy_probes = TrajectoryLinearProbe.probe_encoder_hierarchy(
        feature_extractor=extractor,
        sweep_frames=frames,
        physical_positions=positions,
        device=device,
        alpha=10.0,
        cv_folds=5,
    )

    print("\n" + "-" * 100)
    print(f"{'Layer / Encoder Stage':<26} | {'Dim':<5} | {'Train r':<8} | {'5-Fold CV r':<11} | {'CV R^2':<9} | {'Functional Role':<30}")
    print("-" * 100)
    for stage_name, res in hierarchy_probes.items():
        print(
            f"{stage_name:<26} | {res['embedding_dim']:<5} | "
            f"{res['pearson_r']:>+7.4f} | {res['cv_pearson_r']:>+10.4f} | "
            f"{res['cv_r2_score']:>+8.4f} | {res['description']:<30}"
        )
    print("-" * 100)

    # Headline trajectory probe vector from Global Context encoder & Stage 1
    w_traj_global = hierarchy_probes["global_context"]["direction"]  # (512,) normalized
    w_traj_local = hierarchy_probes["stage1_3d_cnn"]["direction"]    # (512,) normalized
    global_cv_r = hierarchy_probes["global_context"]["cv_pearson_r"]

    # -------------------------------------------------------------------------
    # Experiment 2: Trajectory Information Emergence: Local Speckle vs. Global Context
    # -------------------------------------------------------------------------
    print("\n[4/7] Generating Direct Comparison: Local Speckle (CV r=-0.24) vs. Global Context (CV r=+0.98)...")
    local_probe_cams = {}
    global_probe_cams = {}

    for l_name in ["layer1", "layer2", "layer3", "layer4"]:
        with UltrasoundGradCAM(local_cnn, target_layer=l_name, device=device) as gcam:
            c_loc, _ = gcam.explain(key_frame_float, objective="trajectory_probe", direction=w_traj_local, relu=True)
            local_probe_cams[l_name] = c_loc[0].numpy()
        with UltrasoundGradCAM(global_cnn, target_layer=l_name, device=device) as gcam:
            c_glob, _ = gcam.explain(key_frame_float, objective="trajectory_probe", direction=w_traj_global, relu=True)
            global_probe_cams[l_name] = c_glob[0].numpy()

    fig_emergence_path = figures_dir / "gradcam_local_vs_global_emergence.png"
    plot_local_vs_global_trajectory_attribution(
        image_gray=key_frame,
        local_cams=local_probe_cams,
        global_cams=global_probe_cams,
        title="Trajectory Information Emergence: Local Speckle (Overfit/Diffuse) vs. Global Context (Anatomical Anchor)",
        save_path=fig_emergence_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig_emergence_path}")

    # -------------------------------------------------------------------------
    # Experiment 3: Headline Trajectory Probe Objective (S_traj = z^T w_traj across sweep)
    # -------------------------------------------------------------------------
    print("\n[5/7] Running Experiment 3: Headline Trajectory Probe Attribution across Sweep Progression...")
    sweep_sample_indices = [int(i) for i in np.linspace(0, N_frames - 1, 5)]

    set_custom_style()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=200)

    # MultiScale GradCAM across L2+L3+L4 for crisp spatial delineation of anatomy
    ms_gcam_progression = MultiScaleGradCAM(
        global_cnn, layers=["layer2", "layer3", "layer4"], layer_weights=[0.25, 0.40, 0.35], device=device
    )

    for col_idx, f_idx in enumerate(sweep_sample_indices):
        frame_raw = frames[f_idx]
        frame_float = frame_raw.astype(np.float32) / 255.0

        # Row 0: Raw B-mode
        axes[0, col_idx].imshow(frame_raw, cmap="gray")
        pos_str = f"{np.linalg.norm(positions[f_idx] - positions[0]):.1f} mm" if positions is not None else f"Frame {f_idx}"
        axes[0, col_idx].set_title(f"Frame {f_idx} (d={pos_str})\nRaw B-Mode", fontsize=10, fontweight="bold")
        axes[0, col_idx].axis("off")

        # Row 1: MultiScale Trajectory Grad-CAM
        cam_map, _ = ms_gcam_progression.explain(
            frame_float,
            objective="trajectory_probe",
            direction=w_traj_global,
            relu=True,
        )
        overlay = overlay_cam_on_image(
            frame_raw,
            cam_map[0].numpy(),
            alpha=0.70,
            colormap="jet",
            threshold=0.12,
            smooth_sigma=2.0,
            gamma=0.90,
            enhance_contrast=True,
            suppress_borders=10,
        )
        axes[1, col_idx].imshow(overlay)
        axes[1, col_idx].set_title(f"Trajectory Attribution\n(MultiScale L2+L3+L4)", fontsize=10, fontweight="bold")
        axes[1, col_idx].axis("off")

    fig.suptitle(
        f"Headline Trajectory Probe Attribution across Sweep Progression (Global ResNet 5-Fold CV r = {global_cv_r:+.3f})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    fig2_path = figures_dir / "gradcam_trajectory_progression.png"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close("all")
    print(f"  -> Saved {fig2_path}")

    # -------------------------------------------------------------------------
    # Experiment 4: Layer Hierarchy Attribution on Global Context Encoder
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Experiment 4: Layer Hierarchy Attribution on Global Context ResNet...")
    layers = ["layer1", "layer2", "layer3", "layer4"]
    hierarchy_cams = {}

    for layer_name in layers:
        with UltrasoundGradCAM(global_cnn, target_layer=layer_name, device=device) as gcam:
            cam_map, _ = gcam.explain(
                key_frame_float,
                objective="trajectory_probe",
                direction=w_traj_global,
                relu=True,
            )
            hierarchy_cams[f"Grad-CAM ({layer_name})"] = cam_map[0].numpy()

    # Multi-Scale Layer-Fused Grad-CAM (Combines fine boundary L2/L3 with semantic L4)
    ms_gcam = MultiScaleGradCAM(global_cnn, layers=["layer2", "layer3", "layer4"], layer_weights=[0.25, 0.35, 0.40], device=device)
    ms_map, _ = ms_gcam.explain(key_frame_float, objective="trajectory_probe", direction=w_traj_global)
    hierarchy_cams["MultiScale (L2+L3+L4)"] = ms_map[0].numpy()

    # Eigen-CAM (Layer 4)
    with UltrasoundEigenCAM(global_cnn, target_layer="layer4", device=device) as ecam:
        ecam_map, _ = ecam.explain(key_frame_float)
        hierarchy_cams["Eigen-CAM (layer4)"] = ecam_map[0].numpy()

    # Latent Occlusion
    occ = LatentOcclusion(global_cnn, patch_size=(32, 32), stride=(16, 16), metric="cosine", device=device)
    occ_map, _ = occ.explain(key_frame_float, direction=torch.from_numpy(w_traj_global).float().to(device))
    hierarchy_cams["Occlusion Sensitivity"] = occ_map[0].numpy()

    fig1_path = figures_dir / "gradcam_layer_hierarchy.png"
    plot_explainability_multipanel(
        image_gray=key_frame,
        cam_dict=hierarchy_cams,
        title=f"Global Context Layer Hierarchy Attribution for Trajectory Probe (Frame {mid_idx})",
        threshold=0.18,
        colormap="jet",
        alpha=0.65,
        save_path=fig1_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig1_path}")

    # -------------------------------------------------------------------------
    # Experiment 5: Secondary Objectives & Pairwise Similarity
    # -------------------------------------------------------------------------
    print("\n[7/7] Running Experiment 5: Secondary Objectives & Quantitative Faithfulness...")
    # Extract 512-D global embeddings for unsupervised PC1 & variance analysis
    from src.loaders.preprocessor import preprocess_frames_for_global_encoder
    with torch.no_grad():
        gx = preprocess_frames_for_global_encoder(frames, device=device)
        idx_dense = torch.arange(N_frames, device=device).unsqueeze(0)
        sweep_emb = extractor.global_encoder_module(gx, idx_dense).squeeze(0).cpu().numpy()
    dim_variances = np.var(sweep_emb, axis=0)
    top_var_dims = np.argsort(-dim_variances)
    selected_dim = int(top_var_dims[0])

    v_pc1, r_pc1 = TrajectoryDirectionEstimator.estimate_trajectory_direction(
        embeddings=sweep_emb, physical_positions=positions
    )

    # Pairwise representation similarity (displayed each on its own frame)
    idx_a = mid_idx
    idx_b = min(mid_idx + 6, N_frames - 1)
    frame_a_raw = frames[idx_a]
    frame_b_raw = frames[idx_b]
    frame_a_float = frame_a_raw.astype(np.float32) / 255.0
    frame_b_float = frame_b_raw.astype(np.float32) / 255.0

    # Pairwise representation similarity using MultiScale CAM
    ms_gcam_sim = MultiScaleGradCAM(
        global_cnn, layers=["layer2", "layer3", "layer4"], layer_weights=[0.25, 0.40, 0.35], device=device
    )
    cam_a, _ = ms_gcam_sim.explain(frame_a_float, objective="similarity", target_frame=frame_b_float, explain_branch="a")
    cam_b, _ = ms_gcam_sim.explain(frame_b_float, objective="similarity", target_frame=frame_a_float, explain_branch="b")

    ov_a = overlay_cam_on_image(
        frame_a_raw, cam_a[0].numpy(), threshold=0.12, colormap="jet", alpha=0.70, suppress_borders=10, gamma=0.9
    )
    ov_b = overlay_cam_on_image(
        frame_b_raw, cam_b[0].numpy(), threshold=0.12, colormap="jet", alpha=0.70, suppress_borders=10, gamma=0.9
    )

    dist_ab_mm = np.linalg.norm(positions[idx_b] - positions[idx_a])
    set_custom_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    axes[0].imshow(frame_a_raw, cmap="gray"); axes[0].set_title(f"Frame {idx_a} (Raw B-Mode)", fontsize=10, fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(ov_a); axes[1].set_title(f"Frame {idx_a} (Similarity Attribution)", fontsize=10, fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(frame_b_raw, cmap="gray"); axes[2].set_title(f"Frame {idx_b} (Raw B-Mode, +{dist_ab_mm:.1f} mm)", fontsize=10, fontweight="bold"); axes[2].axis("off")
    axes[3].imshow(ov_b); axes[3].set_title(f"Frame {idx_b} (Similarity Attribution)", fontsize=10, fontweight="bold"); axes[3].axis("off")
    fig.suptitle(f"Pairwise Representation Similarity Attribution (Frames {idx_a} & {idx_b}, Δd={dist_ab_mm:.1f} mm)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig3_path = figures_dir / "gradcam_pairwise_similarity.png"
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close("all")
    print(f"  -> Saved {fig3_path}")

    # Secondary Objectives Comparison Panel (MultiScale ResNet)
    cam_traj, _ = ms_gcam_sim.explain(key_frame_float, objective="trajectory_probe", direction=w_traj_global)
    cam_pc1, _ = ms_gcam_sim.explain(key_frame_float, objective="latent_direction", direction=v_pc1)
    cam_dim, _ = ms_gcam_sim.explain(key_frame_float, objective="latent_dimension", target_dimension=selected_dim)

    secondary_cams = {
        "Headline Trajectory Probe": cam_traj[0].numpy(),
        f"Unsupervised PC1 (r={r_pc1:.2f})": cam_pc1[0].numpy(),
        f"Latent Dim #{selected_dim} (Max Var)": cam_dim[0].numpy(),
    }
    fig5_path = figures_dir / "gradcam_secondary_objectives.png"
    plot_explainability_multipanel(
        image_gray=key_frame,
        cam_dict=secondary_cams,
        title="Scalar Objective Comparison: Supervised Trajectory Probe vs. Unsupervised Targets",
        threshold=0.12,
        colormap="jet",
        alpha=0.70,
        save_path=fig5_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig5_path}")

    # Quantitative Faithfulness Deletion Evaluation on Headline Trajectory Objective
    print("\n[Quantitative Faithfulness] Running Deletion Test on Global Context ResNet...")
    with UltrasoundGradCAM(global_cnn, target_layer="layer4", device=device) as gcam:
        cam_eval, _ = gcam.explain(key_frame_float, objective="trajectory_probe", direction=w_traj_global)

    faith_results = evaluate_cam_faithfulness_deletion(
        model=global_cnn,
        image=key_frame_float,
        cam_mask=cam_eval,
        objective_type="trajectory_probe",
        direction=torch.from_numpy(w_traj_global).float(),
        mask_fractions=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        num_random_trials=5,
        device=device,
    )

    fig4_path = figures_dir / "gradcam_faithfulness_deletion.png"
    plot_faithfulness_deletion_curves(
        faithfulness_results=faith_results,
        title="Grad-CAM Faithfulness: Trajectory Probe Score Drop vs. Pixel Deletion",
        save_path=fig4_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig4_path}")

    print(f"\nFaithfulness Summary (Global Context ResNet):")
    print(f"  - Top Attributed AUDC:   {faith_results['audc_top']:.4f}")
    print(f"  - Random Masking AUDC:   {faith_results['audc_random']:.4f}")
    print(f"  - Least Attributed AUDC: {faith_results['audc_least']:.4f}")
    print(f"  - Faithful Criterion:    {faith_results['is_faithful']}")

    # Save comprehensive summary metadata
    probe_summary = {}
    for stage_name, res in hierarchy_probes.items():
        probe_summary[stage_name] = {
            "embedding_dim": res["embedding_dim"],
            "train_pearson_r": float(res["pearson_r"]),
            "train_spearman_rho": float(res["spearman_rho"]),
            "train_r2": float(res["r2_score"]),
            "cv_pearson_r": float(res["cv_pearson_r"]),
            "cv_spearman_rho": float(res["cv_spearman_rho"]),
            "cv_r2": float(res["cv_r2_score"]),
            "description": res["description"],
        }

    summary_metadata = {
        "dataset": sweep.sweep_id,
        "hierarchy_probes": probe_summary,
        "headline_objective": "trajectory_probe",
        "headline_cv_pearson_r": float(global_cv_r),
        "unsupervised_pc1_correlation": float(r_pc1),
        "selected_latent_dimension": selected_dim,
        "faithfulness": faith_results,
        "figures": [
            "gradcam_local_vs_global_emergence.png",
            "gradcam_trajectory_progression.png",
            "gradcam_layer_hierarchy.png",
            "gradcam_pairwise_similarity.png",
            "gradcam_secondary_objectives.png",
            "gradcam_faithfulness_deletion.png",
        ],
    }

    summary_file = output_dir / "explainability_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)

    print("\n" + "=" * 95)
    print(" EXPLAINABILITY SUITE COMPLETED SUCCESSFULLY")
    print(f" Figures exported to : {figures_dir}")
    print(f" Summary written to  : {summary_file}")
    print("=" * 95)
    return summary_metadata


if __name__ == "__main__":
    run_explainability_suite()
