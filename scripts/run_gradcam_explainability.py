"""Explainability and Grad-CAM Diagnostic Suite for 512-D Ultrasound Feature Encoder.

Runs 4 comprehensive experiments across robotic ultrasound sweeps:
    - Experiment 1: Latent Dimensions & Layer Hierarchy (Layer 1..4, Eigen-CAM, Occlusion)
    - Experiment 2: Trajectory Direction Attribution (PCA with sign orientation)
    - Experiment 3: Pairwise Representation Similarity Attribution (Frame t vs. t+1)
    - Experiment 4: Quantitative Faithfulness Deletion Curves & Sanity Checks
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
    TrajectoryDirectionEstimator,
    UltrasoundEigenCAM,
    UltrasoundGradCAM,
    evaluate_cam_faithfulness_deletion,
)
from src.loaders.mhd_loader import list_forearm_phantom_scans, load_robot_sweep
from src.models.feature_extractors import DualTrackFeatureExtractor
from src.utils.visualization import (
    overlay_cam_on_image,
    plot_explainability_multipanel,
    plot_faithfulness_deletion_curves,
)


def run_explainability_suite(
    dataset_dir: str = "C:/Users/Ibourk/Downloads/Probe_Calib_Single_Filament_2/Probe_Calib_Single_Filament_2",
    checkpoint_path: str = str(CHECKPOINT_PATH),
    device: str = DEVICE,
    output_dir: Path = REPORTS_DIR,
):
    print("=" * 80)
    print("[INFO] DUALTRACK 512-D FEATURE EXPLAINABILITY & GRAD-CAM SUITE")
    print("=" * 80)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Extractor
    print(f"\n[1/5] Loading DualTrack Feature Extractor on {device}...")
    extractor = DualTrackFeatureExtractor(checkpoint_path=checkpoint_path, device=device)
    extractor.eval()

    # Access local CNN backbone module
    cnn_model = extractor.cnn_backbone_module

    # 2. Discover and Load Ultrasound Sweep
    print(f"\n[2/5] Discovering robotic sweeps in {dataset_dir}...")
    mhd_files = list_forearm_phantom_scans(dataset_dir)
    if not mhd_files:
        # Fallback to Single_Filament_3
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

    # Extract 512-D embeddings across the entire sweep
    print(f"Extracting 512-D global embeddings across {N_frames} frames...")
    with torch.no_grad():
        sweep_emb = extractor.extract_stage1_cnn(frames, pool=True).squeeze(0).cpu().numpy()  # (N, 512)

    # Calculate variance of each dimension to find prominent latent dimensions
    dim_variances = np.var(sweep_emb, axis=0)
    top_var_dims = np.argsort(-dim_variances)
    selected_dim = int(top_var_dims[0])
    print(f"Top variance latent dimension across sweep: Dim #{selected_dim} (var={dim_variances[selected_dim]:.4f})")

    # -------------------------------------------------------------------------
    # Experiment 1: Layer-by-Layer Attribution & Method Comparison
    # -------------------------------------------------------------------------
    print("\n[3/5] Running Experiment 1: Layer Hierarchy & Method Comparison...")
    layers = ["stem", "layer1", "layer2", "layer3", "layer4"]
    cam_dict = {}

    for layer_name in layers:
        with UltrasoundGradCAM(cnn_model, target_layer=layer_name, device=device) as gcam:
            cam_map, _ = gcam.explain(
                key_frame_float,
                objective="latent_dimension",
                target_dimension=selected_dim,
                relu=True,
            )
            cam_dict[f"Grad-CAM ({layer_name})"] = cam_map[0].numpy()

    # Eigen-CAM (Layer 4)
    with UltrasoundEigenCAM(cnn_model, target_layer="layer4", device=device) as ecam:
        ecam_map, _ = ecam.explain(key_frame_float)
        cam_dict["Eigen-CAM (layer4)"] = ecam_map[0].numpy()

    # Latent Occlusion
    occ = LatentOcclusion(cnn_model, patch_size=(32, 32), stride=(16, 16), metric="cosine", device=device)
    occ_map, _ = occ.explain(key_frame_float)
    cam_dict["Occlusion Sensitivity"] = occ_map[0].numpy()

    fig1_path = figures_dir / "gradcam_layer_hierarchy.png"
    plot_explainability_multipanel(
        image_gray=key_frame,
        cam_dict=cam_dict,
        title=f"DualTrack Layer Attribution (Latent Dim #{selected_dim}, Frame {mid_idx})",
        save_path=fig1_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig1_path}")

    # -------------------------------------------------------------------------
    # Experiment 2: Trajectory Direction Attribution
    # -------------------------------------------------------------------------
    print("\n[4/5] Running Experiment 2: Trajectory Direction Attribution...")
    v_traj, r_corr = TrajectoryDirectionEstimator.estimate_trajectory_direction(
        embeddings=sweep_emb, physical_positions=positions
    )
    print(f"  Estimated Trajectory Direction Vector (PC1 Pearson r with physical sweep: {r_corr:.4f})")

    sweep_sample_indices = [int(i) for i in np.linspace(0, N_frames - 1, 5)]
    traj_cams = {}
    with UltrasoundGradCAM(cnn_model, target_layer="layer4", device=device) as gcam:
        for f_idx in sweep_sample_indices:
            frame_f = frames[f_idx].astype(np.float32) / 255.0
            cam_map, _ = gcam.explain(
                frame_f,
                objective="latent_direction",
                direction=v_traj,
                relu=True,
            )
            dist_mm = np.linalg.norm(positions[f_idx] - positions[0])
            traj_cams[f"Frame {f_idx} (d={dist_mm:.1f}mm)"] = cam_map[0].numpy()

    fig2_path = figures_dir / "gradcam_trajectory_progression.png"
    plot_explainability_multipanel(
        image_gray=frames[mid_idx],
        cam_dict=traj_cams,
        title=f"Trajectory Direction Attribution across Sweep Progression (PC1 r={r_corr:.3f})",
        save_path=fig2_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig2_path}")

    # -------------------------------------------------------------------------
    # Experiment 3: Pairwise Representation Similarity Attribution
    # -------------------------------------------------------------------------
    print("\n[5/5] Running Experiment 3: Pairwise Representation Similarity Attribution...")
    idx_a = mid_idx
    idx_b = min(mid_idx + 5, N_frames - 1)
    frame_a = frames[idx_a].astype(np.float32) / 255.0
    frame_b = frames[idx_b].astype(np.float32) / 255.0

    with UltrasoundGradCAM(cnn_model, target_layer="layer4", device=device) as gcam:
        cam_a, _ = gcam.explain(frame_a, objective="similarity", target_frame=frame_b, explain_branch="a")
        cam_b, _ = gcam.explain(frame_b, objective="similarity", target_frame=frame_a, explain_branch="b")

    sim_cams = {
        f"Frame {idx_a} (Branch A)": cam_a[0].numpy(),
        f"Frame {idx_b} (Branch B)": cam_b[0].numpy(),
    }
    fig3_path = figures_dir / "gradcam_pairwise_similarity.png"
    plot_explainability_multipanel(
        image_gray=frames[idx_a],
        cam_dict=sim_cams,
        title=f"Pairwise Representation Similarity Attribution (Frames {idx_a} & {idx_b})",
        save_path=fig3_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig3_path}")

    # -------------------------------------------------------------------------
    # Experiment 4: Quantitative Faithfulness Deletion Test
    # -------------------------------------------------------------------------
    print("\n[Quantitative Validation] Running Deletion Faithfulness Evaluation...")
    with UltrasoundGradCAM(cnn_model, target_layer="layer4", device=device) as gcam:
        cam_eval, _ = gcam.explain(key_frame_float, objective="latent_dimension", target_dimension=selected_dim)

    faith_results = evaluate_cam_faithfulness_deletion(
        model=cnn_model,
        image=key_frame_float,
        cam_mask=cam_eval,
        objective_type="latent_dimension",
        target_dimension=selected_dim,
        mask_fractions=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        num_random_trials=5,
        device=device,
    )

    fig4_path = figures_dir / "gradcam_faithfulness_deletion.png"
    plot_faithfulness_deletion_curves(
        faithfulness_results=faith_results,
        title="Grad-CAM Faithfulness: Target Latent Score Drop vs. Pixel Deletion",
        save_path=fig4_path,
    )
    plt.close("all")
    print(f"  -> Saved {fig4_path}")

    print(f"\nFaithfulness Summary:")
    print(f"  - Top Attributed AUDC:   {faith_results['audc_top']:.4f}")
    print(f"  - Random Masking AUDC:   {faith_results['audc_random']:.4f}")
    print(f"  - Least Attributed AUDC: {faith_results['audc_least']:.4f}")
    print(f"  - Faithful Criterion:    {faith_results['is_faithful']}")

    # Save summary metadata
    summary = {
        "dataset": sweep.sweep_id,
        "selected_latent_dimension": selected_dim,
        "latent_dimension_variance": float(dim_variances[selected_dim]),
        "trajectory_direction_pc1_correlation": float(r_corr),
        "faithfulness": faith_results,
        "figures": [
            str(fig1_path.name),
            str(fig2_path.name),
            str(fig3_path.name),
            str(fig4_path.name),
        ],
    }

    summary_json_path = output_dir / "explainability_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> Saved metadata summary to {summary_json_path}")
    print("\n[SUCCESS] Explainability Suite Completed Successfully!")


if __name__ == "__main__":
    run_explainability_suite()
