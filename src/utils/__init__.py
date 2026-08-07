import sys
from pathlib import Path
import torch
import numpy as np

# Export geometry utils
from .geometry import (
    compute_relative_transforms,
    transform_to_pose_vector,
    pose_vector_to_transform,
    compute_translation_distance,
    compute_rotation_angle_deg,
    integrate_relative_poses,
)


def load_model_weights(model, path_or_state, strict=False, handle_size_mismatch=True, state_dict_prefix=None):
    """Helper function to load model weights into a model."""
    if isinstance(path_or_state, (str, Path)):
        state = torch.load(str(path_or_state), map_location="cpu", weights_only=False)
    else:
        state = path_or_state

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
    consume_prefix_in_state_dict_if_present(state, "_orig_mod.")
    consume_prefix_in_state_dict_if_present(state, "module.")

    if state_dict_prefix:
        state = {
            k[len(state_dict_prefix):]: v for k, v in state.items() if k.startswith(state_dict_prefix)
        }

    if handle_size_mismatch:
        model_state = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state.items()
            if k in model_state and v.size() == model_state[k].size()
        }
        state = filtered_state_dict

    out = model.load_state_dict(state, strict=strict)
    return out


# Dynamically load pose module from DualTrack
dt_pose_path = Path(r"D:\DualTrack\src\utils\pose.py")
if dt_pose_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("src.utils.pose", dt_pose_path)
    if spec and spec.loader:
        pose = importlib.util.module_from_spec(spec)
        sys.modules["src.utils.pose"] = pose
        spec.loader.exec_module(pose)


