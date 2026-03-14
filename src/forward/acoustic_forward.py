"""Acoustic forward modeling backends (Deepwave default + numpy FD fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .acquisition import AcquisitionGeometry
from .ricker import ricker_torch, ricker_wavelet


def _to_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_deepwave_tensors(
    geom: AcquisitionGeometry,
    nt: int,
    f_peak: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Deepwave-compatible tensors.

    Returns
    -------
    src_loc:
        ``[n_shots, 1, 2]`` long tensor (z, x order).
    rec_loc:
        ``[n_shots, n_rec, 2]`` long tensor (z, x order).
    src_amp:
        ``[n_shots, 1, nt]`` float tensor.
    """
    device = _to_device(device)
    n_shots = geom.n_shots
    n_rec = geom.n_receivers

    src_loc = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    src_loc[:, 0, 0] = torch.as_tensor(geom.src_z, dtype=torch.long, device=device)
    src_loc[:, 0, 1] = torch.as_tensor(geom.src_x, dtype=torch.long, device=device)

    rec_loc = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
    rec_loc[:, :, 0] = torch.as_tensor(geom.rec_z[None, :], dtype=torch.long, device=device)
    rec_loc[:, :, 1] = torch.as_tensor(geom.rec_x[None, :], dtype=torch.long, device=device)

    wav = ricker_torch(f_peak=f_peak, dt=geom.dt, nt=nt, device=device)
    src_amp = wav.view(1, 1, nt).repeat(n_shots, 1, 1)
    return src_loc, rec_loc, src_amp


def acoustic_forward_deepwave(
    vp: np.ndarray | torch.Tensor,
    dh: float,
    dt: float,
    src_loc: torch.Tensor,
    rec_loc: torch.Tensor,
    src_amp: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Run acoustic forward modeling with Deepwave.

    Returns
    -------
    receiver_data:
        Tensor ``[shot, time, receiver]``.
    """
    import deepwave

    device = _to_device(device)
    vp_t = (
        vp.to(device=device, dtype=torch.float32)
        if isinstance(vp, torch.Tensor)
        else torch.tensor(vp, dtype=torch.float32, device=device)
    )
    out = deepwave.scalar(
        vp_t,
        dh,
        dt,
        source_amplitudes=src_amp.to(device),
        source_locations=src_loc.to(device),
        receiver_locations=rec_loc.to(device),
        pml_freq=15.0,
    )
    return out[-1].detach().cpu()


def acoustic_forward_fd(
    vp: np.ndarray,
    geom: AcquisitionGeometry,
    dh: float,
    nt: int,
    f_peak: float,
) -> np.ndarray:
    """Simple 2D acoustic finite-difference forward (fallback if Deepwave missing).

    Notes
    -----
    This is intentionally simple and CPU-only. Output shape is ``[shot, time, receiver]``.
    """
    nz, nx = vp.shape
    dt = geom.dt
    rec = np.zeros((geom.n_shots, nt, geom.n_receivers), dtype=np.float32)
    wav = ricker_wavelet(f_peak=f_peak, dt=dt, nt=nt)

    c2 = (vp * dt / dh) ** 2

    for ishot in range(geom.n_shots):
        u_prev = np.zeros((nz, nx), dtype=np.float32)
        u = np.zeros((nz, nx), dtype=np.float32)

        sx = int(round(float(geom.src_x[ishot])))
        sz = int(round(float(geom.src_z[ishot])))

        for it in range(nt):
            lap = np.zeros_like(u)
            lap[1:-1, 1:-1] = (
                u[1:-1, 2:]
                + u[1:-1, :-2]
                + u[2:, 1:-1]
                + u[:-2, 1:-1]
                - 4.0 * u[1:-1, 1:-1]
            )

            u_next = 2.0 * u - u_prev + c2 * lap
            if 0 <= sz < nz and 0 <= sx < nx:
                u_next[sz, sx] += wav[it]

            # very light damping near boundaries (simple sponge)
            damp = 0.995
            u_next[0, :] *= damp
            u_next[-1, :] *= damp
            u_next[:, 0] *= damp
            u_next[:, -1] *= damp

            rz = np.clip(np.round(geom.rec_z).astype(int), 0, nz - 1)
            rx = np.clip(np.round(geom.rec_x).astype(int), 0, nx - 1)
            rec[ishot, it, :] = u_next[rz, rx]

            u_prev, u = u, u_next

    return rec


def generate_observed_data(
    vp: np.ndarray,
    geom: AcquisitionGeometry,
    dh: float,
    f_peak: float,
    output_path: str | Path,
    backend: str = "deepwave",
    device: str = "auto",
) -> np.ndarray:
    """Generate and save observed acoustic gathers ``[shot,time,receiver]``."""
    nt = geom.nt
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: np.ndarray
    if backend == "deepwave":
        try:
            src_loc, rec_loc, src_amp = build_deepwave_tensors(
                geom=geom, nt=nt, f_peak=f_peak, device=device
            )
            data_t = acoustic_forward_deepwave(
                vp=vp,
                dh=dh,
                dt=geom.dt,
                src_loc=src_loc,
                rec_loc=rec_loc,
                src_amp=src_amp,
                device=device,
            )
            data = data_t.numpy().astype(np.float32)
        except Exception:
            data = acoustic_forward_fd(vp=vp, geom=geom, dh=dh, nt=nt, f_peak=f_peak)
    elif backend == "fd":
        data = acoustic_forward_fd(vp=vp, geom=geom, dh=dh, nt=nt, f_peak=f_peak)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    np.save(output_path, data.astype(np.float32))
    return data.astype(np.float32)
