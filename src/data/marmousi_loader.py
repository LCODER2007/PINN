"""Load and preprocess Marmousi velocity models (.npy or .segy)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def _read_segy(path: Path) -> np.ndarray:
    """Read a SEGY file into a 2D ``[nz, nx]`` float32 array.

    Notes
    -----
    This uses ``segyio`` if available. The Marmousi workflow in this project
    defaults to ``.npy`` for robustness/reproducibility.
    """
    try:
        import segyio  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Reading .segy requires 'segyio'. Install it or use .npy input."
        ) from exc

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        data = segyio.tools.cube(f)

    arr = np.asarray(data, dtype=np.float32).squeeze()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Could not parse SEGY as 2D model. Got shape {arr.shape}.")
    return arr


def load_marmousi_vp(path: str | Path, subsample: int = 1) -> np.ndarray:
    """Load Marmousi P-wave velocity from ``.npy`` or ``.segy``.

    Parameters
    ----------
    path:
        Input model path.
    subsample:
        Spatial subsampling factor (``>=1``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Velocity model not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        vp = np.load(p).astype(np.float32)
    elif suffix in {".segy", ".sgy"}:
        vp = _read_segy(p)
    else:
        raise ValueError(f"Unsupported model format: {suffix}. Use .npy or .segy")

    if subsample > 1:
        vp = vp[::subsample, ::subsample]
    return np.ascontiguousarray(vp, dtype=np.float32)


def smooth_model(vp: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Gaussian-smoothed model used as inversion initial guess."""
    return gaussian_filter(vp.astype(np.float32), sigma=sigma).astype(np.float32)


def normalized_coordinate_grids(nz: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized ``x,z`` grids in ``[0,1]`` with shape ``[nz, nx]``."""
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    x_grid, z_grid = np.meshgrid(x, z)
    return x_grid.astype(np.float32), z_grid.astype(np.float32)


def model_stats(vp: np.ndarray) -> Dict[str, float]:
    """Basic model diagnostics for notebook sanity checks."""
    return {
        "vp_min": float(np.min(vp)),
        "vp_max": float(np.max(vp)),
        "vp_mean": float(np.mean(vp)),
        "vp_std": float(np.std(vp)),
        "nz": int(vp.shape[0]),
        "nx": int(vp.shape[1]),
    }


def build_smooth_initial_model(vp_true: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Convenience wrapper to create smooth starting model for FWI."""
    return smooth_model(vp_true, sigma=sigma)
