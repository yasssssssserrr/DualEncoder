"""MetaImage (.mhd/.raw) loader for Robot-Guided Ultrasound Forearm Scans."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RobotSweep:
    """Dataclass holding a single tracked robot ultrasound sweep."""
    sweep_id: str
    mhd_path: Path
    raw_path: Path
    frames: np.ndarray  # Shape (N, H, W) uint8
    transforms: np.ndarray  # Shape (N, 4, 4) float64 homogeneous tool-to-tracker matrices
    timestamps: np.ndarray  # Shape (N,) float64
    spacing_mm: Tuple[float, float, float]  # (sx, sy, sz)
    dimensions: Tuple[int, int, int]  # (W, H, N)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def height(self) -> int:
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        return self.frames.shape[2]


def parse_mhd_header(mhd_path: Path) -> Tuple[Dict[str, str], np.ndarray, np.ndarray, Tuple[float, float, float], Tuple[int, int, int], str]:
    """Parse .mhd metadata header for tracked ultrasound video."""
    tags = {}
    with open(mhd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    frame_transforms = {}
    frame_timestamps = {}
    element_spacing = (0.0786242, 0.0786242, 1.0)
    dim_size = (485, 512, 47)
    raw_filename = ""

    for line in lines:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        tags[key] = val

        if key == "DimSize":
            parts = [int(p) for p in val.split() if p.strip()]
            if len(parts) == 3:
                dim_size = (parts[0], parts[1], parts[2])
        elif key == "ElementDataFile":
            raw_filename = val
        elif key == "ElementSpacing":
            parts = [float(p) for p in val.split() if p.strip()]
            if len(parts) >= 3:
                element_spacing = (parts[0], parts[1], parts[2])
        elif "ROS2ToolToTrackerTransform" in key and not key.endswith("Status"):
            # Format: Seq_Frame0000_ROS2ToolToTrackerTransform
            frame_idx_str = key.split("_")[1].replace("Seq_Frame", "").replace("Frame", "")
            try:
                frame_idx = int(frame_idx_str)
                vals = [float(x) for x in val.split()]
                if len(vals) == 16:
                    mat = np.array(vals, dtype=np.float64).reshape((4, 4))
                    frame_transforms[frame_idx] = mat
            except (ValueError, IndexError):
                pass
        elif "Timestamp" in key and "Station" not in key and "AddFrame" not in key:
            frame_idx_str = key.split("_")[1].replace("Seq_Frame", "").replace("Frame", "")
            try:
                frame_idx = int(frame_idx_str)
                frame_timestamps[frame_idx] = float(val)
            except (ValueError, IndexError):
                pass

    num_frames = dim_size[2]
    # Build ordered transforms array
    transforms_arr = np.zeros((num_frames, 4, 4), dtype=np.float64)
    timestamps_arr = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        if i in frame_transforms:
            transforms_arr[i] = frame_transforms[i]
        else:
            transforms_arr[i] = np.eye(4, dtype=np.float64)
        if i in frame_timestamps:
            timestamps_arr[i] = frame_timestamps[i]
        else:
            timestamps_arr[i] = float(i)

    return tags, transforms_arr, timestamps_arr, element_spacing, dim_size, raw_filename


def load_robot_sweep(mhd_path: Path | str) -> RobotSweep:
    """Load a single .mhd/.raw robot sweep."""
    mhd_path = Path(mhd_path)
    if not mhd_path.exists():
        raise FileNotFoundError(f"MHD file not found: {mhd_path}")

    tags, transforms, timestamps, spacing, dim_size, raw_filename = parse_mhd_header(mhd_path)
    
    # Resolve raw file path
    raw_path = mhd_path.parent / (raw_filename if raw_filename else mhd_path.stem + ".raw")
    if not raw_path.exists():
        raw_path = mhd_path.with_suffix(".raw")
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW image file not found for {mhd_path}")

    w, h, n = dim_size
    expected_bytes = w * h * n
    with open(raw_path, "rb") as f:
        raw_bytes = f.read()

    if len(raw_bytes) < expected_bytes:
        raise ValueError(f"RAW file {raw_path} has {len(raw_bytes)} bytes, expected {expected_bytes}")

    # The raw format stores frames as [N, H, W]
    frames = np.frombuffer(raw_bytes[:expected_bytes], dtype=np.uint8).reshape((n, h, w))

    sweep_id = mhd_path.stem.replace("PhilipsEpiq7_ROS2_Transform_", "").replace(".mhd", "")
    return RobotSweep(
        sweep_id=sweep_id,
        mhd_path=mhd_path,
        raw_path=raw_path,
        frames=frames,
        transforms=transforms,
        timestamps=timestamps,
        spacing_mm=spacing,
        dimensions=dim_size,
    )


def list_forearm_phantom_scans(directory_path: Path | str) -> List[Path]:
    """Find all *forearm_phantom_scan.mhd files in a directory."""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    mhd_files = sorted(dir_path.glob("*forearm_phantom_scan.mhd"))
    return mhd_files


def load_all_forearm_scans(directory_path: Path | str) -> List[RobotSweep]:
    """Load all forearm phantom scans from directory."""
    files = list_forearm_phantom_scans(directory_path)
    sweeps = []
    for f in files:
        sweeps.append(load_robot_sweep(f))
    return sweeps
