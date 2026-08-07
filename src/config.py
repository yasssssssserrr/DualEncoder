"""Configuration and global constants for DualEncoder testing suite."""
from pathlib import Path
import torch

# Directory Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DUALTRACK_ROOT = Path(r"D:\DualTrack")
CHECKPOINT_PATH = Path(r"D:\DualTrack\data\checkpoints\dualtrack_final.pt")
ROBOT_DATA_DIR = Path(r"C:\Users\Ibourk\Downloads\Probe_Calib_Single_Filament_3\Probe_Calib_Single_Filament_3")
TUSREC_TRAIN_DIR = Path(r"C:\Users\Ibourk\Downloads\train_part1")

REPORTS_DIR = WORKSPACE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CACHED_FEATURES_DIR = WORKSPACE_ROOT / "cached_features"

# Ensure output directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Hardware and Model Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (256, 256)
ROBOT_SPACING_MM = 0.0786242  # mm/pixel for Philips Epiq 7 linear probe
TUSREC_SPACING_MM = (0.229389, 0.220980)  # mm/pixel for TUS-REC linear probe
