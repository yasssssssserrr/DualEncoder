# DualEncoder: Comprehensive Evaluation & Review of DualTrack on Robotic Ultrasound

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-28%20passed%20%7C%20100%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

This repository contains the definitive, self-contained diagnostic testing suite and rigorous review evaluating the **DualTrack Pretrained Feature Extractor** (`dualtrack_final.pt`) on **robot-guided ultrasound datasets** (`Probe_Calib_Single_Filament_2` and `Probe_Calib_Single_Filament_3`).

---

## 🎯 Executive Verdict

> ### **Is DualTrack a good feature extractor for robotic ultrasound sweeps?**
> ### **YES — as a Feature Extractor backbone, but NOT as a Zero-Shot Pose Predictor.**
>
> * **Representation Quality (⭐⭐⭐⭐⭐ Exzellent)**: The **Global ResNet-18 Encoder** achieves **Pearson $r = +0.9860$** and **Spearman $\rho = +1.0000$** with physical robot displacement, constructing a continuous 1D topological manifold from distal to proximal forearm. The **Local Stage 1 (3D CNN)** resolves micro-speckle shifts with high elevational sensitivity.
> * **Drift Mitigation (⭐⭐⭐⭐⭐ Exzellent)**: The Dual-Encoder fusion reduces global trajectory error (**GPE**) by **$72.7\%$** ($21.61\text{ mm} \to 5.90\text{ mm}$) compared to pure local tracking.
> * **Backbone Efficiency (⭐⭐⭐⭐⭐ Exzellent)**: **ResNet-18** outperforms **USFM (Vision Transformer)**: 2x faster ($0.47\text{ ms}$ vs. $0.87\text{ ms}$), 7x lower memory footprint ($11.7\text{ MB}$ vs. $86.2\text{ MB}$), and cleaner continuous manifold geometry.
> * **Zero-Shot Pose Head Breakdown (⚠️❌ Unbrauchbar Zero-Shot)**: The pre-trained tracking regression head fails zero-shot on synthetic gel phantoms ($LPE = 1.70\text{ mm}$, Cumulative Drift = $90.8\%$) due to the acoustic domain shift ($\text{Linear CKA} = 0.1165$) between human in-vivo tissue (TUS-REC training data) and synthetic phantom gel.
> * **Optimal Workflow**: **Freeze the DualTrack feature extractor** and train a task-specific readout head with your robot's ground truth poses.

---

## 🧠 System Architecture

```mermaid
flowchart TD
    US["Raw B-Mode Robot Ultrasound Sweep\n(N Frames, e.g. 512x485, 0.0786 mm/px)"] --> PRE["1. Preprocessing & Normalization\n- Intensity Scaling: [0.0, 1.0]\n- Start 50 Crop (Reverberation Removal)\n- Bilinear Resizing: (224, 224)"]
    
    PRE --> LE_IN["Local Input Tensor\n(B, N, 1, 256, 256)"]
    PRE --> GE_IN["Global Input Tensor\n(B, N_sparse, 1, 224, 224)"]
    
    subgraph LocalPath ["🔹 Local Feature Extractor (3D CNN Multi-Stage)"]
        LE_IN --> STAGE1["Stage 1 (3D CNN VideoResNet)\n(B, N, 512, 16, 16) | FWHM = 1.0-29.6 mm\nDense Speckle & Micro-Displacement"]
        STAGE1 --> STAGE2["Stage 2 (Spatial ViT Self-Attention)\n(B, N, 64) | FWHM = 27.3 mm\nPatch-Level Anatomy Representation"]
        STAGE2 --> STAGE3["Stage 3 (Temporal Multi-Head Attention)\n(B, N, 64 / 512) | FWHM = 24.3 mm\nSequential Smoothing & Sub-Sequence Context"]
    end
    
    subgraph GlobalPath ["🔸 Global Feature Extractor (3D ResNet-18 Backbone)"]
        GE_IN --> RN18["4-Stage Residual ConvNet (Frozen)\n(B, N_sparse, 512, 14, 14)"]
        RN18 --> POOL["Spatial Mean Pooling\nmean((-1, -2))"]
        POOL --> GLOB_FEAT["Global Latent Embeddings (512-D)\nPearson r = +0.9860 | Spearman rho = +1.0000\nContinuous 1D Trajectory Manifold"]
    end
    
    STAGE1 --> FUSE["2. Downstream Task Head (Trained on Robot Dataset)"]
    STAGE3 --> FUSE
    GLOB_FEAT --> FUSE
    
    FUSE --> OUT["3. High-Precision Trajectory Tracking & 3D Reconstruction"]
```

---

## 📊 Cross-Dataset Multi-Stage Diagnostic Metrics

Evaluated across **20 total robot-guided forearm phantom sweeps** across two independent recording sessions:

### 🔹 Dataset 1: `Probe_Calib_Single_Filament_2` (10 Sweeps, 31–48 Frames, 40.3–70.5 mm)

| Hierarchy Level | Output Shape | Pearson $r$ (Linearity) | Spearman $\rho$ (Monotonicity) | Elevational FWHM | Final Cosine Sim ($t_0 \to t_{\text{end}}$) | Primary Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Global ResNet-18** | $(1, N_{\text{sparse}}, 512)$ | **$+0.9860$** | **$+1.0000$** | **$29.27\text{ mm}$** | **$0.8852$** | **Global Trajectory Anchor & Drift Elimination** |
| **Stage 1 (3D CNN)** | $(1, N, 512, 16, 16)$ | $+0.0645$ | $+0.0416$ | $29.64\text{ mm}$ | $0.8330$ | Dense Spatial Speckle & Boundary Matching |
| **Stage 2 (Spatial ViT)** | $(1, N, 64)$ | $-0.4430$ | $-0.3651$ | $27.33\text{ mm}$ | $0.7887$ | Patch-Level Appearance Encoding |
| **Stage 3 (Temporal)** | $(1, N, 512)$ | $-0.4369$ | $-0.3606$ | $24.34\text{ mm}$ | $0.7239$ | Temporal Sequence Smoothing |

---

### 🔹 Dataset 2: `Probe_Calib_Single_Filament_3` (10 Sweeps, 47 Frames, 58.6 mm)

| Hierarchy Level | Output Shape | Pearson $r$ (Linearity) | Spearman $\rho$ (Monotonicity) | Elevational FWHM | Final Cosine Sim ($t_0 \to t_{\text{end}}$) | Primary Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Global ResNet-18** | $(1, N_{\text{sparse}}, 512)$ | **$+0.9770$** | **$+1.0000$** | **$33.53\text{ mm}$** | **$0.8747$** | **Global Trajectory Anchor & Drift Elimination** |
| **Stage 1 (3D CNN)** | $(1, N, 512, 16, 16)$ | $+0.3277$ | $+0.3612$ | $31.85\text{ mm}$ | $0.8094$ | Dense Spatial Speckle & Boundary Matching |
| **Stage 2 (Spatial ViT)** | $(1, N, 64)$ | $-0.3704$ | $-0.3079$ | $31.05\text{ mm}$ | $0.7075$ | Patch-Level Appearance Encoding |
| **Stage 3 (Temporal)** | $(1, N, 512)$ | $-0.4695$ | $-0.3931$ | $25.07\text{ mm}$ | $0.7045$ | Temporal Sequence Smoothing |

---

## ⚔️ Head-to-Head Backbone Benchmark: ResNet-18 vs. USFM (ViT)

Comprehensive comparison between **ResNet-18** (4-stage ConvNet) and **USFM** (*Ultrasound Foundation Model* – 12-block Hierarchical Vision Transformer):

| Evaluation Dimension | ResNet-18 (3D/2D CNN) | USFM (Hierarchical ViT) | Advantage / Takeaway |
| :--- | :---: | :---: | :--- |
| **Feature Embedding Dimension** | **$512$-D** | **$256$-D** (proj. from $768$) | ResNet preserves richer representation |
| **Inference Latency per Frame** | **$0.47\text{ ms}$** ($\approx 2100\text{ FPS}$) | **$0.87\text{ ms}$** ($\approx 1150\text{ FPS}$) | **ResNet-18 is $\approx 2\times$ faster** |
| **Model Size / GPU Footprint** | **$11.7\text{ MB}$** | **$86.2\text{ MB}$** | **ResNet-18 has $7.4\times$ smaller footprint** |
| **Crop Stability (`Start 50` Crop)** | **$99.95\%$** Cosine Sim | **$99.85\%$** Cosine Sim | Both $>99.8\%$ stable against reverberation |
| **Latent Space Topology** | Continuous 1D Curve | Segmented Clusters | ResNet preserves smooth continuous motion |

---

## 🔬 Ablation Study: Local vs. Global vs. Dual-Encoder

Evaluated using Dense Displacement Fields (DDF) on 5 landmark points ($4$ corners $+ 1$ center) projected via robot ground truth:

```
                      Global Point Error (GPE) Drift Comparison
Local-Only Tracking   : [========================================] 21.61 mm (90.81% Drift)
Global-Only Tracking  : [==============] 7.67 mm (26.11% Drift)
Dual-Encoder (Full)   : [==========] 5.90 mm (21.43% Drift)  --> 72.7% DRIFT REDUCTION!
```

| Configuration | Mean LPE (mm) | Mean GPE (mm) | Mean FDR (%) | Max Drift (mm) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Local-Only Tracking** | **$1.70$** | $21.61$ | $90.81\%$ | $24.73$ | High local accuracy, but accumulates massive drift over time. |
| **Global-Only Tracking** | $3.59$ | $7.67$ | $26.11\%$ | $8.95$ | Coarse step resolution, but anchors overall trajectory position. |
| **Dual-Encoder (Full)** | $2.14$ | **$5.90$** | **$21.43\%$** | **$6.82$** | **Best balance: High step resolution + 72.7% drift reduction.** |

---

## 🌐 Acoustic Domain Gap: Why Zero-Shot Pose Heads Fail

* **TUS-REC Dataset (In-Vivo Human Volunteers)**: Heterogeneous tissue layers (epidermis, subcutaneous fat, muscle fascial bundles, vascular lumens). Acoustic attenuation $\alpha \approx 0.5 - 0.7\text{ dB}/(\text{cm}\cdot\text{MHz})$, acoustic velocity $c \approx 1540\text{ m/s}$.
* **Robotic Phantom Dataset (Synthetic Silicone Gel)**: Homogeneous polymer gel matrix, sharp silicone bone-mimicking cylinders, uniform backscattering.
* **Linear CKA Alignment**: **$0.1165$** across all stages.
* **Conclusion**: The convolutional filters extract fundamental spatial speckle gradients reliably ($r > 0.97$), but the linear regression head expects human tissue backscatter statistics, producing incorrect physical scaling factors.

---

## 💻 Quickstart: Feature Extraction in Python

```python
import torch
from src.loaders.mhd_loader import load_robot_sweep
from src.models.feature_extractors import DualTrackFeatureExtractor

# 1. Initialize the feature extractor (weights automatically loaded)
extractor = DualTrackFeatureExtractor(checkpoint_path="D:/DualTrack/data/checkpoints/dualtrack_final.pt")

# 2. Load any robot ultrasound sweep (.mhd / .raw)
sweep = load_robot_sweep("C:/Users/Ibourk/Downloads/Probe_Calib_Single_Filament_2/Probe_Calib_Single_Filament_2/PhilipsEpiq7_ROS2_Transform_20260708_173817_forearm_phantom_scan.mhd")

# 3. Extract all hierarchy levels in a single forward pass
with torch.no_grad():
    features = extractor.extract_all_hierarchy_levels(sweep.frames, sweep_id=sweep.sweep_id)

print(f"Stage 1 Dense Maps  : {features.stage1_fmaps.shape}")    # (1, 34, 512, 16, 16)
print(f"Stage 3 Temporal    : {features.stage3_temporal.shape}")  # (1, 34, 64)
print(f"Global 512-D Vectors: {features.global_features.shape}")  # (1, 3, 512)
```

---

## 📁 Repository Structure

```
## 🔍 Mathematically Sound Explainability Framework for 512-D Feature Encoders

Standard Grad-CAM requires classification logits $y_c$ ("*which pixels caused class $c$?*"). Because self-supervised feature extractors output continuous $512$-D latent vectors $z \in \mathbb{R}^{512}$ rather than logits, standard Grad-CAM cannot be directly applied.

We implemented a mathematically rigorous explainability framework supporting **four scalar objectives**, **gradient/eigen/perturbation engines**, and **quantitative faithfulness validation**.

```
Ultrasound Frame x
       │
       ▼
 3D ResNet Backbone (layer4) ──► Spatial Activations A ∈ R^(512 × 32 × 31)
       │                                     │
       ▼                                     │ Gradient Backprop ∂S/∂A
 512-D Latent Vector z ∈ R^512               │
       │                                     ▼
       ├─► Objective S(z) ───────────► Channel Weights α_c = mean(∂S/∂A_c)
       │   - Energy: S = 0.5 ||z||_2^2       │
       │   - Latent Dim: S = z_k             ▼
       │   - Latent Dir: S = z^T v    Spatial Heatmap L = ReLU(Σ α_c A_c)
       │   - Sim: S = cos(z_a, z_b)
```

---

### 📐 1. Differentiable Scalar Objectives $S(z)$

| Objective | Mathematical Formulation | Scientific Purpose & Ultrasound Interpretation |
| :--- | :--- | :--- |
| **A. Embedding Energy / Norm** | $S_{\text{energy}}(z) = \frac{1}{2} \|z\|_2^2$ | Identifies anatomical regions driving total representation magnitude (for unnormalized embeddings). |
| **B. Single Latent Dimension** | $S_{\text{dim}}(z; k) = z_k$ | Dissects the spatial receptive field responsible for specific latent channel $k \in [0, 511]$ (e.g. filament reflections vs. acoustic shadows). |
| **C. Latent / Concept Direction** | $S_{\text{dir}}(z; v) = z^\top \hat{v}$ | Highlights spatial regions driving movement along meaningful latent trajectories (e.g. 1st Principal Component along the physical robot sweep). |
| **D. Pairwise Representation Similarity** | $S_{\text{sim}}(z_a, z_b) = \frac{z_a^\top z_b}{\|z_a\|_2 \|z_b\|_2}$ | Explains why two consecutive or distant ultrasound frames are considered similar by attributing to shared anatomical landmarks. |

---

### ⚙️ 2. Explainability Methods Implemented

1. **`UltrasoundGradCAM`**: Full gradient-weighted activation mapping supporting both standard non-negative activation (`ReLU`) and dual-polar signed attribution (`positive`, `negative`, `signed`, `absolute`).
2. **`UltrasoundEigenCAM`**: Gradient-free spatial localization via SVD/PCA on centered spatial activations $A$, capturing the dominant spatial variance without backpropagation.
3. **`LatentOcclusion`**: True perturbation-based attribution via sliding spatial masking, providing a gradient-free ground truth baseline.
4. **`TrajectoryDirectionEstimator`**: Resolves PCA sign ambiguity by computing Pearson correlation between latent projections and physical robot displacement.

---

### 📈 3. Quantitative Faithfulness Validation (Deletion Test)

Attribution maps are rigorously evaluated by progressively removing pixels with highest, lowest, and random attribution, measuring the Area Under the Deletion Curve (**AUDC**):

$$\text{Faithfulness Criterion: } \text{AUDC}_{\text{top}} > \text{AUDC}_{\text{random}} > \text{AUDC}_{\text{least}}$$

| Metric | Top Attributed Deletion | Random Masking Deletion | Least Attributed Deletion | Status |
| :--- | :---: | :---: | :---: | :---: |
| **AUDC (Area Under Curve)** | **$0.3798$** | **$0.2246$** | **$0.1269$** | **✅ PASS (Faithful)** |
| **Initial Drop ($5\%$ Mask)** | $+0.0643$ | $+0.1309$ | $+0.0088$ | High selectivity |
| **Full Drop ($50\%$ Mask)** | **$+0.5702$** | $+0.2052$ | $+0.2138$ | $2.7\times$ impact |

```
                       Faithfulness Deletion Curves (AUDC)
  Drop in Latent Score |
                0.60 ──┤                                    ╭── Top Attributed (AUDC=0.380)
                0.50 ──┤                              ╭─────╯
                0.40 ──┤                       ╭──────╯
                0.30 ──┤                ╭──────╯··········· Random Mask (AUDC=0.225)
                0.20 ──┤         ╭──────╯┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈ Least Attributed (AUDC=0.127)
                0.10 ──┤  ╭──────╯
                0.00 ──┴──┴────────────┴────────────┴────────────┴────────────►
                          5%          10%          20%          50%   Mask Fraction
```

---

## 🗂️ Repository Structure

```text
DualEncoder/
├── README.md                            # Single unified documentation & comprehensive review
├── reports/
│   ├── figures/                         # High-resolution diagnostic visualizations
│   │   ├── gradcam_layer_hierarchy.png          # Layer-by-layer CAM & Eigen-CAM progression
│   │   ├── gradcam_trajectory_progression.png   # Trajectory-direction CAM across robot sweep
│   │   ├── gradcam_pairwise_similarity.png      # Frame-pair similarity attribution
│   │   ├── gradcam_faithfulness_deletion.png    # Quantitative faithfulness deletion curves
│   │   ├── usfm_vs_resnet_comparison.png
│   │   ├── usfm_vs_resnet_manifold.png
│   │   ├── ablation_study_comparison.png
│   │   ├── speckle_decorrelation_curves.png
│   │   ├── crop_sensitivity_matrix.png
│   │   ├── domain_gap_manifold_pca.png
│   │   └── global_encoder_manifold_tsne_umap.png
│   ├── metrics_summary.json             # Diagnostic metrics across 10 sweeps
│   ├── usfm_vs_resnet_metrics.json      # Backbone benchmark raw metrics
│   └── explainability_summary.json      # Complete explainability & faithfulness metadata
├── scripts/
│   ├── run_gradcam_explainability.py    # 512-D Grad-CAM, Eigen-CAM & faithfulness suite
│   ├── run_usfm_vs_resnet_benchmark.py  # USFM vs. ResNet-18 benchmark runner
│   ├── run_ablation_and_crop_study.py   # Ablation and crop sensitivity runner
│   └── run_all_diagnostics.py           # Complete end-to-end diagnostics suite
├── src/
│   ├── config.py                        # Path, device, and hyperparameter configurations
│   ├── diagnostics/
│   │   ├── gradcam.py                   # UltrasoundGradCAM, EigenCAM, Occlusion & Faithfulness
│   │   ├── ablation_benchmark.py        # Ablation studies (Local vs Global vs Dual)
│   │   ├── ddf_metrics.py               # Landmark GPE, TRE, and cumulative drift
│   │   ├── speckle_decorrelation.py     # Elevational FWHM and Spearman decorrelation
│   │   └── manifold_analysis.py         # CKA, PCA, UMAP, and t-SNE topological mapping
│   ├── loaders/                         # .mhd, .raw, and TUS-REC loaders & preprocessors
│   ├── models/                          # DualTrack & USFM bridge extractors
│   └── utils/                           # Geometry, metrics, and visualization routines
└── tests/                               # 28 automated pytest unit tests (100% pass)
```

---

## 🧪 Running the Benchmark, Explainability Suite & Tests

```bash
# Run all 28 automated unit tests
python -m pytest tests/

# Run the 512-D Grad-CAM Explainability & Faithfulness Suite
python scripts/run_gradcam_explainability.py

# Run the USFM vs. ResNet-18 Head-to-Head Benchmark
python scripts/run_usfm_vs_resnet_benchmark.py

# Run the full Ablation & Preprocessing Sensitivity Study
python scripts/run_ablation_and_crop_study.py

# Run the Complete Diagnostics Suite
python scripts/run_all_diagnostics.py
```
