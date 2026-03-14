"""Ricker wavelet utilities for acoustic forward/PINN source terms."""

from __future__ import annotations

import numpy as np
import torch


def ricker_wavelet(
    f_peak: float,
    dt: float,
    nt: int,
    delay: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a Ricker wavelet with optional peak normalization."""
    if delay is None:
        delay = 1.5 / f_peak
    t = np.arange(nt, dtype=np.float32) * dt - float(delay)
    a = (np.pi * f_peak * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    if normalize:
        m = np.max(np.abs(w)) + 1e-8
        w = w / m
    return w.astype(np.float32)


def ricker_torch(
    f_peak: float,
    dt: float,
    nt: int,
    delay: float | None = None,
    device: str | torch.device = "cpu",
    normalize: bool = True,
) -> torch.Tensor:
    """Ricker wavelet as ``torch.float32`` tensor of shape ``[nt]``."""
    w = ricker_wavelet(f_peak=f_peak, dt=dt, nt=nt, delay=delay, normalize=normalize)
    return torch.from_numpy(w).to(device=device, dtype=torch.float32)


def analytic_ricker_torch(
    t: torch.Tensor,
    f_peak: float,
    delay: float | None = None,
) -> torch.Tensor:
    """Analytic Ricker for differentiable source evaluation in PINN residual.

    Parameters
    ----------
    t:
        Time tensor in seconds.
    f_peak:
        Peak frequency in Hz.
    delay:
        Time shift in seconds. Defaults to ``1.5 / f_peak``.
    """
    if delay is None:
        delay = 1.5 / f_peak
    tau = t - float(delay)
    a = (torch.pi * f_peak * tau) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)
