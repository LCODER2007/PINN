"""Loss weight scheduling utilities for stable PINN-FWI optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossWeightScheduler:
    """Two-phase schedule that enforces proper training ordering.

    Phase 1 (warmup, epochs 0..warmup_epochs):
        * PDE weight ramps **up** from ``w_pde_start`` → ``w_pde_end``
          so physics is progressively enforced.
        * Data weight stays at ``w_data_start`` (low).
        * IC weight stays at ``w_ic_start`` (high) to anchor the
          quiescent initial condition.

    Phase 2 (data-fitting, warmup_epochs..total_epochs):
        * PDE weight held at ``w_pde_end``.
        * Data weight ramps from ``w_data_start`` → ``w_data_end``.
        * IC weight anneals from ``w_ic_start`` → ``w_ic_end`` (may decrease
          once the IC is well-learned).
    """

    w_pde_start: float = 1.0
    w_pde_end: float = 100.0
    w_data_start: float = 1.0
    w_data_end: float = 50.0
    w_ic_start: float = 50.0
    w_ic_end: float = 5.0
    warmup_epochs: int = 100
    total_epochs: int = 1000

    def get(self, epoch: int) -> dict[str, float]:
        if self.total_epochs <= 1:
            return {
                "w_pde": float(self.w_pde_end),
                "w_data": float(self.w_data_end),
                "w_ic": float(self.w_ic_end),
            }

        progress = min(max(epoch / max(self.total_epochs - 1, 1), 0.0), 1.0)
        warmup_frac = max(self.warmup_epochs / max(self.total_epochs, 1), 1e-8)

        if progress <= warmup_frac:
            # Phase 1 — ramp PDE up, keep IC high, data low
            alpha = progress / warmup_frac
            w_pde = self.w_pde_start + alpha * (self.w_pde_end - self.w_pde_start)
            w_data = self.w_data_start
            w_ic = self.w_ic_start
        else:
            # Phase 2 — PDE at full, ramp data up, IC can decrease
            beta = (progress - warmup_frac) / max(1.0 - warmup_frac, 1e-8)
            w_pde = self.w_pde_end
            w_data = self.w_data_start + beta * (self.w_data_end - self.w_data_start)
            w_ic = self.w_ic_start + beta * (self.w_ic_end - self.w_ic_start)

        return {"w_pde": float(w_pde), "w_data": float(w_data), "w_ic": float(w_ic)}
