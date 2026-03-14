"""Autograd-based acoustic PDE residual evaluation."""

from __future__ import annotations

from typing import Callable

import torch


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def acoustic_pde_residual(
    pinn,
    velocity_fn,
    x: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    shot_id: torch.Tensor,
    source_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    *,
    t_max: float = 1.0,
    domain_x: float = 1.0,
    domain_z: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the acoustic PDE residual with proper non-dimensionalization.

    Coordinates (x, z, t) are assumed to be normalized to [0, 1].
    The physical PDE ``u_tt_phys = v^2 * lap_phys(u) + s_phys`` in normalized
    coordinates becomes::

        u_tt = (v * t_max / domain_x)^2 * u_xx
             + (v * t_max / domain_z)^2 * u_zz
             + t_max^2 * s

    This keeps all PDE terms O(1) instead of O(v^2).
    """
    x = x.requires_grad_(True)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)

    u = pinn(x, z, t, shot_id)

    u_x = _grad(u, x)
    u_z = _grad(u, z)
    u_t = _grad(u, t)

    u_xx = _grad(u_x, x)
    u_zz = _grad(u_z, z)
    u_tt = _grad(u_t, t)

    v = velocity_fn(x, z)
    s = source_fn(x, z, t) if source_fn is not None else torch.zeros_like(u)

    # Non-dimensional scaling: converts physical v (m/s) + normalised coords
    # into O(1) coefficients so that l_pde is on the same scale as l_data.
    alpha_x_sq = (t_max / max(domain_x, 1e-8)) ** 2
    alpha_z_sq = (t_max / max(domain_z, 1e-8)) ** 2
    t_sq = t_max ** 2

    r = u_tt - v.pow(2) * (alpha_x_sq * u_xx + alpha_z_sq * u_zz) - t_sq * s

    return {
        "u": u,
        "u_t": u_t,
        "u_tt": u_tt,
        "u_x": u_x,
        "u_xx": u_xx,
        "u_z": u_z,
        "u_zz": u_zz,
        "v": v,
        "s": s,
        "r": r,
    }
