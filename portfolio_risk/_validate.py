"""Input validation shared by all modules."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_series(values: ArrayLike, name: str = "returns", min_obs: int = 2) -> NDArray[np.float64]:
    """Return a finite 1-D float64 array with at least ``min_obs`` observations."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size < min_obs:
        raise ValueError(f"{name} needs at least {min_obs} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.max(np.abs(arr)) >= 10.0 or np.mean(np.abs(arr)) > 1.0:
        raise ValueError(
            f"{name} look like prices or percentages (values of 10 or more, or average magnitude above 1); "
            "returns are decimals (0.01 = 1%)"
        )
    return arr


def check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not (1e-12 <= value <= 1.0 - 1e-12):
        raise ValueError(f"alpha must be between 1e-12 and 1 - 1e-12, got {alpha}")
    return value


def check_positive(value: float, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value}")
    return number
