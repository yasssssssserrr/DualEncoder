"""Loader for reference TUS-REC training dataset (HDF5 format)."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import h5py
import numpy as np


@dataclass
class TUSRecSweep:
    """Dataclass holding a single TUS-REC ultrasound sweep."""
    sweep_id: str
    file_path: Path
    frames: np.ndarray  # Shape (N, H, W) uint8
    transforms: np.ndarray  # Shape (N, 4, 4) float64
    spacing_mm: tuple = (0.229389, 0.220980, 1.0)

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def list_tusrec_scans(directory_path: Path | str, max_scans: Optional[int] = None) -> List[Path]:
    """Find all .h5 sweep files in TUS-REC directory."""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"TUS-REC directory not found: {dir_path}")
    
    h5_files = sorted(dir_path.glob("*/*.h5"))
    if max_scans:
        h5_files = h5_files[:max_scans]
    return h5_files


def load_tusrec_sweep(h5_path: Path | str) -> TUSRecSweep:
    """Load a single TUS-REC sweep from HDF5 file."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        frames = np.asarray(f["frames"][:], dtype=np.uint8)
        tforms = np.asarray(f["tforms"][:], dtype=np.float64)

    sweep_id = f"{h5_path.parent.name}_{h5_path.stem}"
    return TUSRecSweep(
        sweep_id=sweep_id,
        file_path=h5_path,
        frames=frames,
        transforms=tforms,
    )


# Alias
list_tusrec_hdf5_files = list_tusrec_scans
