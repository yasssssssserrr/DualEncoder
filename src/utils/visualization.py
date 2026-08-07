"""Visualization tools for encoder diagnostic benchmarking and calibration analysis."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
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


def overlay_cam_on_image(
    image_gray: Union[np.ndarray, torch.Tensor],
    cam_mask: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.70,
    colormap: str = "jet",
    threshold: float = 0.12,
    smooth_sigma: float = 2.0,
    gamma: float = 0.90,
    signed: bool = False,
    enhance_contrast: bool = True,
    suppress_borders: int = 12,
) -> np.ndarray:
    """Overlays a 2D attribution heatmap onto a grayscale ultrasound B-mode image with high-contrast medical rendering."""
    if torch.is_tensor(image_gray):
        image_gray = image_gray.detach().cpu().numpy()
    if torch.is_tensor(cam_mask):
        cam_mask = cam_mask.detach().cpu().numpy()

    # Standardize base B-mode image
    img = np.nan_to_num(image_gray.astype(np.float32), nan=0.0)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    if enhance_contrast:
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)

    # Standardize CAM
    cam = np.nan_to_num(cam_mask.astype(np.float32), nan=0.0)
    if smooth_sigma > 0:
        cam = gaussian_filter(cam, sigma=smooth_sigma)

    if signed:
        cam = np.clip(cam, -1.0, 1.0)
        abs_cam = np.abs(cam)
        cam_norm = np.maximum(0.0, (abs_cam - threshold) / (1.0 - threshold + 1e-8)) ** gamma
        cam_scaled = (np.sign(cam) * cam_norm + 1.0) / 2.0
        effective_alpha = alpha * (cam_norm[..., np.newaxis] ** 0.65)
        cmap = plt.get_cmap("coolwarm" if colormap in ("turbo", "jet") else colormap)
    else:
        # Min-max normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam_normed = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_normed = np.zeros_like(cam)

        # Contrast stretch top percentile to make hotspots vivid
        p98_cam = np.percentile(cam_normed, 98)
        if p98_cam > 0.05:
            cam_normed = np.clip(cam_normed / p98_cam, 0.0, 1.0)

        # Soft thresholding
        cam_norm = np.maximum(0.0, (cam_normed - threshold) / (1.0 - threshold + 1e-8)) ** gamma

        # Suppress convolution edge/padding artifacts on borders
        if suppress_borders > 0:
            b = max(int(suppress_borders), 18)
            for k in range(b):
                factor = (0.5 * (1.0 - np.cos(np.pi * float(k) / float(b)))) ** 2  # smooth cosine power ramp
                cam_norm[k, :] *= factor
                cam_norm[-1 - k, :] *= factor
                cam_norm[:, k] *= factor
                cam_norm[:, -1 - k] *= factor
            cam_norm[:8, :] = 0.0
            cam_norm[-8:, :] = 0.0
            cam_norm[:, :8] = 0.0
            cam_norm[:, -8:] = 0.0

        cam_scaled = cam_norm
        effective_alpha = alpha * (cam_norm[..., np.newaxis] ** 0.60)
        cmap = plt.get_cmap(colormap)

    # Apply colormap -> (H, W, 4) RGBA
    heatmap_rgba = cmap(cam_scaled)
    heatmap_rgb = heatmap_rgba[..., :3]  # (H, W, 3) in [0, 1]

    # Convert grayscale image to 3-channel
    img_rgb = np.stack([img, img, img], axis=-1)

    # Clean alpha blending
    blended = (1.0 - effective_alpha) * img_rgb + effective_alpha * heatmap_rgb
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255.0).astype(np.uint8)


def plot_explainability_multipanel(
    image_gray: np.ndarray,
    cam_dict: Dict[str, np.ndarray],
    title: str = "Feature Encoder Attribution Analysis",
    threshold: float = 0.20,
    colormap: str = "jet",
    alpha: float = 0.65,
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plots neutral, publication-ready multi-panel attribution figure."""
    set_custom_style()
    num_panels = 1 + len(cam_dict)
    ncols = min(4, num_panels)
    nrows = int(np.ceil(num_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows), dpi=200)
    axes = np.array(axes).flatten()

    # Panel 0: Original B-mode Ultrasound
    img_disp = np.clip(image_gray, 0.0, 1.0) if image_gray.max() <= 1.0 else image_gray / 255.0
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Input B-Mode Frame", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    for idx, (name, cam_mask) in enumerate(cam_dict.items(), start=1):
        ax = axes[idx]
        overlay = overlay_cam_on_image(
            image_gray,
            cam_mask,
            alpha=alpha,
            colormap=colormap,
            threshold=threshold,
            smooth_sigma=2.0,
            gamma=0.9,
            suppress_borders=10,
        )
        ax.imshow(overlay)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # Hide extra unused subplots
    for ax in axes[num_panels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_local_vs_global_trajectory_attribution(
    image_gray: np.ndarray,
    local_cams: Dict[str, np.ndarray],
    global_cams: Dict[str, np.ndarray],
    title: str = "Trajectory Information Emergence: Local Speckle vs. Global Context",
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plots direct comparison illustrating why trajectory information emerges only in Global Context."""
    set_custom_style()
    num_cols = 1 + max(len(local_cams), len(global_cams))
    fig, axes = plt.subplots(2, num_cols, figsize=(4.2 * num_cols, 8.2), dpi=200)

    # Row 0: Local 3D CNN (Micro-Speckle Encoder, CV r = -0.24)
    axes[0, 0].imshow(image_gray, cmap="gray")
    axes[0, 0].set_title("Input Ultrasound Frame", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.05, 0.05, "Local 3D CNN Path\n(CV r = -0.24, Speckle Only)",
        transform=axes[0, 0].transAxes, fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#dc2626", alpha=0.8),
    )

    for idx, (name, cam) in enumerate(local_cams.items(), start=1):
        if idx < num_cols:
            ov = overlay_cam_on_image(image_gray, cam, threshold=0.12, colormap="jet", alpha=0.70, suppress_borders=10, gamma=0.9)
            axes[0, idx].imshow(ov)
            axes[0, idx].set_title(f"Local: {name}", fontsize=10, fontweight="bold")
            axes[0, idx].axis("off")

    # Row 1: Global ResNet Context (Trajectory Manifold, CV r = +0.98)
    axes[1, 0].imshow(image_gray, cmap="gray")
    axes[1, 0].set_title("Input Ultrasound Frame", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.05, 0.05, "Global Context Path\n(CV r = +0.98, Trajectory Anchor)",
        transform=axes[1, 0].transAxes, fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#16a34a", alpha=0.8),
    )

    for idx, (name, cam) in enumerate(global_cams.items(), start=1):
        if idx < num_cols:
            ov = overlay_cam_on_image(image_gray, cam, threshold=0.12, colormap="jet", alpha=0.70, suppress_borders=10, gamma=0.9)
            axes[1, idx].imshow(ov)
            axes[1, idx].set_title(f"Global: {name}", fontsize=10, fontweight="bold")
            axes[1, idx].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_faithfulness_deletion_curves(
    faithfulness_results: Dict[str, Any],
    title: str = "Grad-CAM Faithfulness Evaluation (Pixel Deletion Test)",
    save_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plots Deletion Faithfulness curves (% masked pixels vs. target score drop)."""
    set_custom_style()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

    fractions = np.array(faithfulness_results["mask_fractions"]) * 100.0
    top_drops = faithfulness_results["top_attribution_drops"]
    rand_drops = faithfulness_results["random_attribution_drops"]
    least_drops = faithfulness_results["least_attribution_drops"]

    ax.plot(fractions, top_drops, "o-", color="#dc2626", lw=2.4, label=f"Top Attributed (AUDC={faithfulness_results['audc_top']:.4f})")
    ax.plot(fractions, rand_drops, "s--", color="#2563eb", lw=2.0, label=f"Random Masking (AUDC={faithfulness_results['audc_random']:.4f})")
    ax.plot(fractions, least_drops, "^:", color="#059669", lw=2.0, label=f"Least Attributed (AUDC={faithfulness_results['audc_least']:.4f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage of Pixels Masked (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Latent Score Change |ΔS|", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    is_faithful = faithfulness_results.get("is_faithful", False)
    status_text = "Faithful: Top > Random >= Least" if is_faithful else "Non-monotonic / Weak Attribution"
    status_color = "#059669" if is_faithful else "#d97706"
    ax.text(
        0.05, 0.92, status_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=status_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor=status_color, alpha=0.9),
    )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig



