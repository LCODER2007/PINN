"""Loss terms for acoustic PINN-FWI training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def physics_mse_loss(residual: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(residual, torch.zeros_like(residual))


def physics_mse_loss_causal(
    residual: torch.Tensor,
    t: torch.Tensor,
    epsilon: float = 20.0,
    n_chunks: int = 16,
    min_weight: float = 1e-3,
) -> torch.Tensor:
    """Chunked causal PDE loss.

    Residuals are grouped into ordered time bins. Each bin is weighted by the
    cumulative error of all earlier bins so the optimizer must reduce early-time
    residuals before later-time residuals receive material weight.
    """
    t_flat = t.detach().squeeze(-1)
    r_sq = residual.squeeze(-1).pow(2)

    n_chunks = max(int(n_chunks), 1)
    edges = torch.linspace(0.0, 1.0, n_chunks + 1, device=t_flat.device, dtype=t_flat.dtype)
    chunk_losses = []
    zero = torch.zeros((), device=r_sq.device, dtype=r_sq.dtype)

    for idx in range(n_chunks):
        if idx == n_chunks - 1:
            mask = (t_flat >= edges[idx]) & (t_flat <= edges[idx + 1])
        else:
            mask = (t_flat >= edges[idx]) & (t_flat < edges[idx + 1])

        if mask.any():
            chunk_losses.append(r_sq[mask].mean())
        else:
            chunk_losses.append(zero)

    losses = torch.stack(chunk_losses)
    prefix = torch.cat([zero.unsqueeze(0), torch.cumsum(losses.detach(), dim=0)[:-1]], dim=0)
    scale = losses.detach().mean().clamp(min=1e-12)
    exponent = (-epsilon * prefix / scale).clamp(min=-20.0, max=0.0)
    weights = torch.exp(exponent).clamp(min=min_weight).detach()
    return (weights * losses).sum() / weights.sum().clamp(min=1e-8)


def data_mse_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, obs)


def initial_condition_loss(
    u0: torch.Tensor,
    ut0: torch.Tensor,
    weight_u: float = 1.0,
    weight_ut: float = 1.0,
) -> torch.Tensor:
    loss = torch.zeros((), device=u0.device, dtype=u0.dtype)
    if weight_u > 0:
        loss = loss + float(weight_u) * F.mse_loss(u0, torch.zeros_like(u0))
    if weight_ut > 0:
        loss = loss + float(weight_ut) * F.mse_loss(ut0, torch.zeros_like(ut0))
    return loss


def boundary_loss(ub: torch.Tensor) -> torch.Tensor:
    """Soft absorbing boundary by damping field amplitude at boundaries."""
    return F.mse_loss(ub, torch.zeros_like(ub))


def smoothness_regularization(vp_grid: torch.Tensor) -> torch.Tensor:
    """L2 smoothness on spatial gradients of vp (grid shape [nz,nx])."""
    dz = vp_grid[1:, :] - vp_grid[:-1, :]
    dx = vp_grid[:, 1:] - vp_grid[:, :-1]
    return (dz.pow(2).mean() + dx.pow(2).mean())


def total_variation_regularization(vp_grid: torch.Tensor) -> torch.Tensor:
    dz = torch.abs(vp_grid[1:, :] - vp_grid[:-1, :])
    dx = torch.abs(vp_grid[:, 1:] - vp_grid[:, :-1])
    return dz.mean() + dx.mean()


def velocity_bounds_penalty(vp: torch.Tensor, vp_min: float, vp_max: float) -> torch.Tensor:
    low = torch.relu(vp_min - vp)
    high = torch.relu(vp - vp_max)
    return (low.pow(2) + high.pow(2)).mean()


def charbonnier_regularization(
    vp_grid: torch.Tensor,
    epsilon: float = 1.0,
    alpha: float = 0.45,
) -> torch.Tensor:
    """Edge-preserving spatial regularization.

    Compared with L2 smoothness, Charbonnier (alpha < 1) penalizes small
    gradients strongly while allowing larger jumps, which helps preserve
    geological interfaces.
    """
    dz = vp_grid[1:, :] - vp_grid[:-1, :]
    dx = vp_grid[:, 1:] - vp_grid[:, :-1]
    eps_sq = float(epsilon) ** 2
    reg_z = (dz.pow(2) + eps_sq).pow(alpha).mean()
    reg_x = (dx.pow(2) + eps_sq).pow(alpha).mean()
    return reg_z + reg_x
