"""Data and tensor sanity checks."""

from __future__ import annotations

import numpy as np


def check_observed_shape(observed: np.ndarray) -> None:
    if observed.ndim != 3:
        raise ValueError(
            f"Observed data must have shape [shot,time,receiver]; got {observed.shape}"
        )


def check_finite_array(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Array '{name}' contains non-finite values.")
