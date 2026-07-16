"""Small helpers shared across the package."""

from __future__ import annotations

import os
import random
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import torch

PathLike: TypeAlias = str | os.PathLike[str]
MissingStrategy: TypeAlias = Literal["error", "median", "zero"]
ActivationName: TypeAlias = Literal["relu", "gelu", "silu", "leaky_relu"]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def validate_proteome_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.shape[0] < 3:
        raise ValueError(f"{name} needs at least 3 profiles to estimate correlations")
    if frame.shape[1] < 2:
        raise ValueError(f"{name} needs at least 2 protein columns")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{name} contains duplicate protein columns: {duplicates[:5]}")
    if not frame.index.is_unique:
        # Duplicate profile IDs do not invalidate the mathematics, but they make
        # downstream metadata joins ambiguous.
        raise ValueError(f"{name} contains duplicate profile/sample IDs")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.columns = numeric.columns.astype(str)
    if not numeric.columns.is_unique:
        duplicates = numeric.columns[numeric.columns.duplicated()].tolist()
        raise ValueError(
            f"{name} protein IDs are not unique after string conversion: "
            f"{duplicates[:5]}"
        )
    return numeric


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
