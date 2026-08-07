"""Compatibility proxy for src.utils.utils."""
import sys
from pathlib import Path
from src.utils import load_model_weights

# Export SweepData dummy class if needed
class SweepData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
