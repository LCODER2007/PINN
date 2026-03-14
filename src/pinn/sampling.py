"""Sampling utilities for collocation, IC/BC, and receiver data points."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..forward.acquisition import AcquisitionGeometry


def _resolve_shot_ids(
    n_shots: int,
    shots_per_batch: int,
    shot_ids: np.ndarray | list[int] | None = None,
    active_shot: int | None = None,
) -> np.ndarray:
    if shot_ids is not None:
        ids = np.asarray(shot_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("shot_ids must not be empty")
        return np.unique(np.clip(ids, 0, n_shots - 1))
    if active_shot is not None:
        return np.array([int(active_shot)], dtype=np.int64)
    if shots_per_batch >= n_shots:
        return np.arange(n_shots, dtype=np.int64)
    return np.random.choice(n_shots, size=min(shots_per_batch, n_shots), replace=False).astype(np.int64)


def _resolve_receiver_ids(
    observed: np.ndarray,
    shot_ids: np.ndarray,
    n_rec_total: int,
    n_receivers: int,
    active_receiver_rms_ratio: float = 0.0,
) -> np.ndarray:
    receiver_ids = np.arange(n_rec_total, dtype=np.int64)
    if active_receiver_rms_ratio > 0.0:
        rms = np.sqrt(np.mean(observed[shot_ids, :, :n_rec_total].astype(np.float64) ** 2, axis=(0, 1)))
        peak = float(rms.max()) if rms.size > 0 else 0.0
        if peak > 0.0:
            threshold = active_receiver_rms_ratio * peak
            active_ids = np.flatnonzero(rms > threshold).astype(np.int64)
            if active_ids.size > 0:
                receiver_ids = active_ids

    if n_receivers >= receiver_ids.size:
        return np.sort(receiver_ids)
    return np.sort(np.random.choice(receiver_ids, size=n_receivers, replace=False).astype(np.int64))


def _rand(n: int, device: str) -> torch.Tensor:
    return torch.rand((n, 1), dtype=torch.float32, device=device)


def sample_collocation_points(
    n_points: int,
    device: str,
    source_xy: tuple[float, float] | None = None,
    source_bias_ratio: float = 0.3,
    source_sigma: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Sample collocation points in normalized space ``x,z,t in [0,1]``.

    Optionally biases part of the samples near source coordinates.
    """
    n_biased = int(n_points * source_bias_ratio) if source_xy is not None else 0
    n_uniform = n_points - n_biased

    x = _rand(n_uniform, device)
    z = _rand(n_uniform, device)
    t = _rand(n_uniform, device)

    if n_biased > 0 and source_xy is not None:
        sx, sz = source_xy
        xb = torch.randn((n_biased, 1), device=device) * source_sigma + float(sx)
        zb = torch.randn((n_biased, 1), device=device) * source_sigma + float(sz)
        tb = _rand(n_biased, device)
        x = torch.cat([x, xb.clamp(0.0, 1.0)], dim=0)
        z = torch.cat([z, zb.clamp(0.0, 1.0)], dim=0)
        t = torch.cat([t, tb], dim=0)

    return {"x": x, "z": z, "t": t}


def sample_initial_points(n_points: int, device: str) -> Dict[str, torch.Tensor]:
    """Sample initial condition points with ``t=0``."""
    return {"x": _rand(n_points, device), "z": _rand(n_points, device), "t": torch.zeros((n_points, 1), device=device)}


def sample_boundary_points(n_points: int, device: str) -> Dict[str, torch.Tensor]:
    """Sample boundary points on x/z edges with random time."""
    n_side = max(1, n_points // 4)
    t = _rand(4 * n_side, device)

    x_left = torch.zeros((n_side, 1), device=device)
    x_right = torch.ones((n_side, 1), device=device)
    x_rand = _rand(n_side, device)
    x = torch.cat([x_left, x_right, x_rand, x_rand], dim=0)

    z_rand = _rand(n_side, device)
    z_top = torch.zeros((n_side, 1), device=device)
    z_bottom = torch.ones((n_side, 1), device=device)
    z = torch.cat([z_rand, z_rand, z_top, z_bottom], dim=0)
    return {"x": x, "z": z, "t": t}


def sample_receiver_data_points(
    geom: AcquisitionGeometry,
    observed: np.ndarray,
    n_samples: int,
    shots_per_batch: int,
    nx: int | None,
    nz: int | None,
    device: str,
    shot_ids: np.ndarray | list[int] | None = None,
    active_shot: int | None = None,
    active_receiver_rms_ratio: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Randomly sample receiver-time observations.

    Parameters
    ----------
    observed:
        Array shape ``[shot, time, receiver]``.
    """
    n_shots, nt, n_rec_obs = observed.shape
    n_rec_geom = len(geom.rec_x)

    n_rec = min(n_rec_obs, n_rec_geom)
    shot_ids = _resolve_shot_ids(
        n_shots=n_shots,
        shots_per_batch=shots_per_batch,
        shot_ids=shot_ids,
        active_shot=active_shot,
    )
    receiver_ids = _resolve_receiver_ids(
        observed=observed,
        shot_ids=shot_ids,
        n_rec_total=n_rec,
        n_receivers=n_rec,
        active_receiver_rms_ratio=active_receiver_rms_ratio,
    )

    per_shot = max(1, n_samples // len(shot_ids))
    xs, zs, ts, ds, sid = [], [], [], [], []

    denom_x = float(max((nx - 1) if nx is not None else np.max(geom.rec_x), 1.0))
    denom_z = float(max((nz - 1) if nz is not None else np.max(geom.rec_z), 1.0))
    rec_x_norm = geom.rec_x / denom_x
    rec_z_norm = geom.rec_z / denom_z
    t_norm = geom.time / max(float(geom.time[-1]), 1e-8)

    for s in shot_ids:
        rec_idx = np.random.choice(receiver_ids, size=per_shot, replace=True)
        t_idx = np.random.randint(0, nt, size=per_shot)
        xs.append(rec_x_norm[rec_idx][:, None])
        zs.append(rec_z_norm[rec_idx][:, None])
        ts.append(t_norm[t_idx][:, None])
        ds.append(observed[s, t_idx, rec_idx][:, None])
        sid.append(np.full((per_shot, 1), s, dtype=np.int64))

    return {
        "x": torch.tensor(np.vstack(xs), dtype=torch.float32, device=device),
        "z": torch.tensor(np.vstack(zs), dtype=torch.float32, device=device),
        "t": torch.tensor(np.vstack(ts), dtype=torch.float32, device=device),
        "d_obs": torch.tensor(np.vstack(ds), dtype=torch.float32, device=device),
        "shot_id": torch.tensor(np.vstack(sid), dtype=torch.long, device=device),
    }


def sample_receiver_trace_batch(
    geom: AcquisitionGeometry,
    observed: np.ndarray,
    shots_per_batch: int,
    n_receivers: int,
    nx: int | None,
    nz: int | None,
    device: str,
    shot_ids: np.ndarray | list[int] | None = None,
    active_shot: int | None = None,
    active_receiver_rms_ratio: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Sample shot/receiver mini-gathers with full time traces.

    Returns flattened coordinates for PINN forward evaluation and a reshaped
    observed tensor for spectral filtering and loss computation.
    """
    n_shots, nt, n_rec_obs = observed.shape
    n_rec_geom = len(geom.rec_x)
    n_rec_total = min(n_rec_obs, n_rec_geom)

    shot_ids = _resolve_shot_ids(
        n_shots=n_shots,
        shots_per_batch=shots_per_batch,
        shot_ids=shot_ids,
        active_shot=active_shot,
    )

    n_receivers = int(max(1, min(n_receivers, n_rec_total)))
    rec_idx = _resolve_receiver_ids(
        observed=observed,
        shot_ids=shot_ids,
        n_rec_total=n_rec_total,
        n_receivers=n_receivers,
        active_receiver_rms_ratio=active_receiver_rms_ratio,
    )

    denom_x = float(max((nx - 1) if nx is not None else np.max(geom.rec_x), 1.0))
    denom_z = float(max((nz - 1) if nz is not None else np.max(geom.rec_z), 1.0))
    rec_x_norm = (geom.rec_x[rec_idx] / denom_x).astype(np.float32)
    rec_z_norm = (geom.rec_z[rec_idx] / denom_z).astype(np.float32)
    t_norm = (geom.time / max(float(geom.time[-1]), 1e-8)).astype(np.float32)

    # Shape convention: [shot_batch, receiver_batch, nt]
    obs_traces = observed[shot_ids][:, :, rec_idx].transpose(0, 2, 1).astype(np.float32)

    s_batch = len(shot_ids)
    r_batch = len(rec_idx)
    tt = np.broadcast_to(t_norm[None, None, :], (s_batch, r_batch, nt))
    xx = np.broadcast_to(rec_x_norm[None, :, None], (s_batch, r_batch, nt))
    zz = np.broadcast_to(rec_z_norm[None, :, None], (s_batch, r_batch, nt))
    sid = np.broadcast_to(shot_ids[:, None, None], (s_batch, r_batch, nt))

    return {
        "x": torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device),
        "z": torch.tensor(zz.reshape(-1, 1), dtype=torch.float32, device=device),
        "t": torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=device),
        "shot_id": torch.tensor(sid.reshape(-1, 1), dtype=torch.long, device=device),
        "d_obs_traces": torch.tensor(obs_traces, dtype=torch.float32, device=device),
        "n_shot_batch": torch.tensor(s_batch, dtype=torch.long, device=device),
        "n_rec_batch": torch.tensor(r_batch, dtype=torch.long, device=device),
        "n_time": torch.tensor(nt, dtype=torch.long, device=device),
    }
