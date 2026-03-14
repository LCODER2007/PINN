"""Acquisition geometry definition helpers for Marmousi experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AcquisitionGeometry:
    """Stores source/receiver positions (index coordinates) and time axis.

    Notes
    -----
    Coordinates are in **grid indices** (not metres), consistent with Deepwave.
    """

    src_x: np.ndarray
    src_z: np.ndarray
    rec_x: np.ndarray
    rec_z: np.ndarray
    time: np.ndarray

    def __post_init__(self) -> None:
        self.src_x = np.asarray(self.src_x, dtype=np.float32)
        self.src_z = np.asarray(self.src_z, dtype=np.float32)
        self.rec_x = np.asarray(self.rec_x, dtype=np.float32)
        self.rec_z = np.asarray(self.rec_z, dtype=np.float32)
        self.time = np.asarray(self.time, dtype=np.float32)

    @property
    def n_shots(self) -> int:
        return int(len(self.src_x))

    @property
    def n_receivers(self) -> int:
        return int(len(self.rec_x))

    @property
    def nt(self) -> int:
        return int(len(self.time))

    @property
    def dt(self) -> float:
        if len(self.time) < 2:
            return 0.0
        return float(self.time[1] - self.time[0])

    @classmethod
    def from_npz(cls, path: str | Path) -> "AcquisitionGeometry":
        d = np.load(path)
        return cls(d["src_x"], d["src_z"], d["rec_x"], d["rec_z"], d["time"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            src_x=self.src_x,
            src_z=self.src_z,
            rec_x=self.rec_x,
            rec_z=self.rec_z,
            time=self.time,
        )

    def get_normalized_receiver_coords(self, nz: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
        return (self.rec_x / max(nx - 1, 1)).astype(np.float32), (
            self.rec_z / max(nz - 1, 1)
        ).astype(np.float32)

    def get_normalized_source_coords(self, nz: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
        return (self.src_x / max(nx - 1, 1)).astype(np.float32), (
            self.src_z / max(nz - 1, 1)
        ).astype(np.float32)

    def get_normalized_time(self) -> np.ndarray:
        t_max = float(self.time[-1]) if len(self.time) > 0 else 0.0
        if t_max <= 0:
            return self.time.astype(np.float32)
        return (self.time / t_max).astype(np.float32)


def build_surface_acquisition(
    nx: int,
    nz: int,
    n_shots: int,
    nt: int,
    dt: float,
    src_depth_idx: int = 2,
    rec_depth_idx: int = 2,
    pad_x: int = 4,
    receiver_stride: int = 1,
) -> AcquisitionGeometry:
    """Create simple surface acquisition used by notebooks and fast tests."""
    src_x = np.linspace(pad_x, nx - pad_x - 1, n_shots, dtype=np.float32)
    src_z = np.full_like(src_x, fill_value=float(min(max(src_depth_idx, 0), nz - 1)))

    rec_x = np.arange(pad_x, nx - pad_x, receiver_stride, dtype=np.float32)
    rec_z = np.full_like(rec_x, fill_value=float(min(max(rec_depth_idx, 0), nz - 1)))

    time = np.arange(nt, dtype=np.float32) * float(dt)
    return AcquisitionGeometry(src_x=src_x, src_z=src_z, rec_x=rec_x, rec_z=rec_z, time=time)
