"""Visualization tools for encoder diagnostic benchmarking and calibration analysis."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Use non-interactive backend for server/script execution
plt.switch_backend("Agg")


def set_custom_style():
    """Apply modern clean styling for plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.2,
        "grid.color": "#eeeeee",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_multistage_motion_correlation(
    stage_correlations: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Path | str] = None,
    title: str = "Feature Distance vs. Physical Robot Displacement",
) -> plt.Figure:
    """Plot 4-panel comparison of motion correlation across encoder hierarchy."""
    set_custom_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=200)
    axes = axes.flatten()

    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706"]
    stage_names = list(stage_correlations.keys())

    for idx, (ax, stage) in enumerate(zip(axes, stage_names)):
        data = stage_correlations[stage]
        phys = data["phys_dists"]
        feat = data["feat_dists"]
        r = data.get("pearson_r", 0.0)
        rho = data.get("spearman_rho", 0.0)

        color = colors[idx % len(colors)]
        ax.scatter(phys, feat, alpha=0.45, s=24, color=color, edgecolors="none", label="Frame pairs")

        # Regression line
        if len(phys) > 2 and np.std(phys) > 1e-6:
            slope, intercept, _, _, _ = linregress(phys, feat)
            x_vals = np.linspace(min(phys), max(phys), 100)
            ax.plot(x_vals, slope * x_vals + intercept, color="#dc2626", lw=2.2, label=f"Fit (r={r:.3f})")

        ax.set_title(f"{stage}\nPearson r = {r:.3f} | Spearman ρ = {rho:.3f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Physical Robot Translation (mm)")
        ax.set_ylabel("Cosine Feature Distance")
        ax.grid(True)
        ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_speckle_decorrelation_curves(
    stage_decorrelations: Dict[str, Any],
    save_path: Optional[Path | str] = None,
    title: str = "Elevational Speckle Decorrelation across Encoder Stages",
) -> plt.Figure:
    """Plot speckle decorrelation curves with FWHM markers."""
    set_custom_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    colors = {"Stage 1 (3D CNN)": "#2563eb", "Stage 2 (ViT CLS)": "#7c3aed", "Stage 3 (Temporal)": "#059669", "Global Context": "#d97706"}

    for name, stats in stage_decorrelations.items():
        color = colors.get(name, "#4b5563")
        disps = stats.displacements_mm
        sims = stats.cosine_similarities
        fwhm = stats.fwhm_mm

        label = f"{name} (FWHM = {fwhm:.2f} mm)" if np.isfinite(fwhm) else f"{name}"
        ax.plot(disps, sims, marker="o", markersize=4, lw=2.0, color=color, label=label)

    ax.axhline(0.5, color="#9ca3af", linestyle=":", lw=1.5, label="FWHM threshold (0.5)")
    ax.axhline(1.0 / np.e, color="#ef4444", linestyle="--", lw=1.2, label="1/e Decay threshold (~0.37)")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Elevational Translation Displacement (mm)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_domain_manifold_pca(
    tusrec_pca: np.ndarray,
    robot_pca: np.ndarray,
    save_path: Optional[Path | str] = None,
    title: str = "Latent Space Manifold: TUS-REC Training vs. Forearm Robot Scans",
) -> plt.Figure:
    """Plot 2D PCA projection comparing training distribution with robot scans."""
    set_custom_style()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

    ax.scatter(tusrec_pca[:, 0], tusrec_pca[:, 1], alpha=0.6, s=35, color="#3b82f6", label="TUS-REC In-Vivo Training Sweeps")
    ax.scatter(robot_pca[:, 0], robot_pca[:, 1], alpha=0.8, s=45, color="#ef4444", marker="^", label="Robot Forearm Phantom Sweeps")

    # Connect robot points with line to show trajectory continuity
    ax.plot(robot_pca[:, 0], robot_pca[:, 1], color="#f87171", lw=1.2, linestyle="--", alpha=0.7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_3d_trajectory_reconstruction(
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    sweep_id: str = "sweep",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot 3D spatial trajectory comparison between Robot Ground Truth and DualTrack."""
    set_custom_style()
    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    gt_pos = gt_trajectory[:, :3, 3]
    pred_pos = pred_trajectory[:, :3, 3]

    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label="Robot Tracker Ground Truth", color="#10b981", lw=3.0)
    ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], color="#047857", s=25)

    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], label="DualTrack Pretrained Prediction", color="#6366f1", lw=2.5, linestyle="--")
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], color="#4338ca", s=25)

    ax.set_title(f"3D Trajectory Reconstruction: {sweep_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_crop_sensitivity_matrix(
    crop_metrics: List[Any],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot bar chart of Cosine Similarity across crop variations and encoder stages."""
    set_custom_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    stages = ["Stage 1 (3D CNN)", "Stage 2 (ViT CLS)", "Stage 3 (Temporal)", "Global ResNet3D"]
    x = np.arange(len(stages))
    width = 0.25

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899"]
    for idx, cm in enumerate(crop_metrics):
        sims = [
            cm.stage1_cosine_sim,
            cm.stage2_cosine_sim,
            cm.stage3_cosine_sim,
            cm.global_cosine_sim,
        ]
        offset = (idx - len(crop_metrics) / 2.0 + 0.5) * width
        rects = ax.bar(x + offset, sims, width, label=cm.crop_name, color=colors[idx % len(colors)], alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Cosine Similarity vs. Center Crop", fontsize=11, fontweight="bold")
    ax.set_title("Preprocessing & Crop Sensitivity in Feature Space", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="#6b7280", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_tsne_umap_manifold(
    tsne_2d: np.ndarray,
    umap_2d: np.ndarray,
    frame_indices: np.ndarray,
    title: str = "Global Encoder 512-D Latent Manifold Projection",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot 2D t-SNE and UMAP/PCA latent manifold embeddings colored by sweep progression."""
    set_custom_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)

    # 1. t-SNE Panel
    sc1 = ax1.scatter(
        tsne_2d[:, 0],
        tsne_2d[:, 1],
        c=frame_indices,
        cmap="viridis",
        s=45,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.85,
    )
    ax1.plot(tsne_2d[:, 0], tsne_2d[:, 1], color="#94a3b8", lw=1.2, linestyle="--", alpha=0.6)
    ax1.set_title("t-SNE Projection (2D)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax1.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. UMAP/PCA Panel
    sc2 = ax2.scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=frame_indices,
        cmap="viridis",
        s=45,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.85,
    )
    ax2.plot(umap_2d[:, 0], umap_2d[:, 1], color="#94a3b8", lw=1.2, linestyle="--", alpha=0.6)
    ax2.set_title("UMAP / PCA Projection (2D)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latent Dim 1", fontsize=10)
    ax2.set_ylabel("Latent Dim 2", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation="horizontal", fraction=0.06, pad=0.15)
    cbar.set_label("Sweep Frame Progression (Distal -> Proximal Forearm)", fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_ablation_comparison_bar(
    ablation_summary: Dict[str, Dict[str, float]],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot side-by-side comparison of DualTrack vs Local-Only vs Global-Only."""
    set_custom_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=200)

    models = list(ablation_summary.keys())
    x = np.arange(len(models))
    colors = ["#6366f1", "#10b981", "#f59e0b"]

    # 1. LPE in µm
    lpes = [ablation_summary[m]["lpe_um"] for m in models]
    bars1 = ax1.bar(x, lpes, color=colors, width=0.5, alpha=0.85)
    ax1.set_title("Local Point Error (LPE)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("LPE (µm)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold", rotation=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f} µm", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. GPE in mm
    gpes = [ablation_summary[m]["gpe_mm"] for m in models]
    bars2 = ax2.bar(x, gpes, color=colors, width=0.5, alpha=0.85)
    ax2.set_title("Global Point Error (GPE)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("GPE (mm)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold", rotation=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.2f} mm", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("DualTrack Encoder Ablation Study (Local vs Global Contribution)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig

