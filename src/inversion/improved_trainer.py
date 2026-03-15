"""Enhanced acoustic PINN-FWI trainer with improved convergence and diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ..forward.acquisition import AcquisitionGeometry
from ..pinn.physics_residual import acoustic_pde_residual
from ..pinn.sampling import (
    sample_boundary_points,
    sample_collocation_points,
    sample_initial_points,
    sample_receiver_trace_batch,
)
from .losses import (
    boundary_loss,
    charbonnier_regularization,
    initial_condition_loss,
    physics_mse_loss_causal,
    smoothness_regularization,
    total_variation_regularization,
    velocity_bounds_penalty,
)
from .schedule import LossWeightScheduler
from ..utils.io import append_csv_row, ensure_dir, save_checkpoint
from ..utils.viz import plot_training_snapshot


@dataclass
class TrainingMetrics:
    """Track training metrics for diagnostics."""
    epoch: int
    loss_total: float
    loss_data: float
    loss_physics: float
    loss_ic: float
    loss_bc: float
    loss_reg: float
    loss_well: float
    grad_norm_pinn: float
    grad_norm_vp: float
    w_pde: float
    w_data: float
    w_data_eff: float
    fmax: float
    lr_pinn: float
    lr_vp: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class ImprovedAcousticPINNFWITrainer:
    """Enhanced trainer with better convergence, diagnostics, and adaptive strategies."""

    def __init__(
        self,
        pinn: torch.nn.Module,
        velocity_net: torch.nn.Module | None,
        observed: np.ndarray,
        geometry: AcquisitionGeometry,
        vp_true: np.ndarray,
        config: dict[str, Any],
        project_root: str | Path,
        invert_vp: bool = True,
    ) -> None:
        self.pinn = pinn
        self.velocity_net = velocity_net
        self.observed = observed.astype(np.float32)
        self.geometry = geometry
        self.vp_true = vp_true.astype(np.float32)
        self.cfg = config
        self.project_root = Path(project_root)
        self.invert_vp = invert_vp and (velocity_net is not None)

        self.device = next(self.pinn.parameters()).device
        self.nz, self.nx = self.vp_true.shape
        self.t_max = float(self.geometry.time[-1]) if self.geometry.nt > 0 else 1.0

        dh = float(self.cfg["acquisition"]["dh"])
        self.domain_x = max((self.nx - 1) * dh, 1e-8)
        self.domain_z = max((self.nz - 1) * dh, 1e-8)

        logging_cfg = self.cfg.get("logging", {})
        results_dir = self.project_root / logging_cfg.get("results_dir", "results")
        self.fig_dir = ensure_dir(results_dir / "figures")
        self.ckpt_dir = ensure_dir(results_dir / "checkpoints")
        self.log_dir = ensure_dir(results_dir / "logs")
        self.csv_path = self.log_dir / "train_log.csv"
        self.metrics_path = self.log_dir / "metrics.jsonl"

        train_cfg = self.cfg["training"]
        self.n_epochs = int(train_cfg["n_epochs"])
        self.print_every = int(logging_cfg.get("print_every", 50))
        self.plot_every = int(logging_cfg.get("plot_every", 200))
        self.checkpoint_every = int(logging_cfg.get("checkpoint_every", 500))

        # Optimizers with learning rate scheduling
        self.optim_pinn = torch.optim.Adam(
            self.pinn.parameters(), 
            lr=float(train_cfg["lr_pinn"])
        )
        vp_params = list(self.velocity_net.parameters()) if self.invert_vp else []
        self.optim_vp = (
            torch.optim.Adam(vp_params, lr=float(train_cfg["lr_vp"])) 
            if len(vp_params) > 0 else None
        )

        # Learning rate schedulers
        self.scheduler_pinn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim_pinn,
            T_0=int(train_cfg.get("scheduler_step", 500)),
            T_mult=1.5,
            eta_min=float(train_cfg.get("lr_pinn", 1e-4)) * 0.01
        )
        if self.optim_vp is not None:
            self.scheduler_vp = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optim_vp,
                T_0=int(train_cfg.get("scheduler_step", 500)),
                T_mult=1.5,
                eta_min=float(train_cfg.get("lr_vp", 1e-4)) * 0.01
            )

        self.loss_scheduler = LossWeightScheduler(self.cfg)
        self.history = {
            "loss_total": [],
            "loss_data": [],
            "loss_physics": [],
            "loss_ic": [],
            "loss_bc": [],
            "loss_reg": [],
            "loss_well": [],
            "grad_norm_pinn": [],
            "grad_norm_vp": [],
            "fmax": [],
            "w_data_eff": [],
        }
        self.metrics_history: list[TrainingMetrics] = []

        # Adaptive parameters
        self.data_loss_history = []
        self.pde_loss_history = []
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.max_patience = int(train_cfg.get("early_stopping_patience", 100))

    def _get_current_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate from optimizer."""
        return optimizer.param_groups[0]["lr"]

    def _compute_velocity_error(self, vp_est: np.ndarray) -> dict[str, float]:
        """Compute velocity estimation errors."""
        diff = vp_est - self.vp_true
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        rel_rmse = rmse / self.vp_true.mean()
        return {"mae": float(mae), "rmse": float(rmse), "rel_rmse": float(rel_rmse)}

    def _vp_grid_from_net(self) -> torch.Tensor:
        """Generate velocity grid from network."""
        if self.velocity_net is None:
            return torch.tensor(self.vp_true, device=self.device, dtype=torch.float32)
        
        z_norm = torch.linspace(0, 1, self.nz, device=self.device, dtype=torch.float32)
        x_norm = torch.linspace(0, 1, self.nx, device=self.device, dtype=torch.float32)
        zz, xx = torch.meshgrid(z_norm, x_norm, indexing="ij")
        coords = torch.stack([xx, zz], dim=-1).reshape(-1, 2)
        
        with torch.no_grad():
            vp_flat = self.velocity_net.forward_coords(coords)
        return vp_flat.reshape(self.nz, self.nx)

    def _velocity_fn(self):
        """Create velocity function for PDE residual."""
        vp_grid = self._vp_grid_from_net()
        
        def vel_fn(x_norm: torch.Tensor, z_norm: torch.Tensor) -> torch.Tensor:
            x_idx = (x_norm * (self.nx - 1)).clamp(0, self.nx - 1)
            z_idx = (z_norm * (self.nz - 1)).clamp(0, self.nz - 1)
            
            x0 = x_idx.long()
            z0 = z_idx.long()
            x1 = (x0 + 1).clamp(max=self.nx - 1)
            z1 = (z0 + 1).clamp(max=self.nz - 1)
            
            wx = x_idx - x0.float()
            wz = z_idx - z0.float()
            
            v00 = vp_grid[z0, x0]
            v01 = vp_grid[z0, x1]
            v10 = vp_grid[z1, x0]
            v11 = vp_grid[z1, x1]
            
            v0 = v00 * (1 - wx) + v01 * wx
            v1 = v10 * (1 - wx) + v11 * wx
            return v0 * (1 - wz) + v1 * wz
        
        return vel_fn

    def _source_fn(self, shot_id: int):
        """Create source function for PDE residual."""
        def src_fn(x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            from ..forward.ricker import analytic_ricker_torch
            f_peak = float(self.cfg["acquisition"]["f_peak"])
            # Return Ricker wavelet (independent of x, z position)
            return analytic_ricker_torch(
                t=t,
                f_peak=f_peak,
                delay=None
            )
        
        return src_fn

    def train(self) -> dict[str, list[float]]:
        """Main training loop with improved convergence."""
        train_cfg = self.cfg["training"]
        w_cfg = self.cfg["weights"]

        n_collocation = int(train_cfg["n_collocation"])
        n_ic = int(train_cfg["n_ic"])
        n_bc = int(train_cfg["n_bc"])
        n_data = int(train_cfg["n_data_batch"])
        shots_per_batch = int(train_cfg.get("shots_per_batch", 1))

        pbar = trange(self.n_epochs, desc="PINN-FWI")
        for epoch in pbar:
            self.pinn.train()
            if self.invert_vp:
                self.velocity_net.train()

            # Determine update flags
            warmup_epochs = int(train_cfg.get("warmup_pinn_only_epochs", 100))
            update_pinn = True
            update_vp = (epoch >= warmup_epochs) and self.invert_vp

            self.pinn.requires_grad_(update_pinn)
            if self.velocity_net is not None:
                self.velocity_net.requires_grad_(update_vp)

            if self.optim_vp is not None and update_vp:
                self.optim_vp.zero_grad(set_to_none=True)
            if update_pinn:
                self.optim_pinn.zero_grad(set_to_none=True)

            # Sample data
            shot_id = int(np.random.randint(0, self.geometry.n_shots))
            sx = float(self.geometry.src_x[shot_id] / max(self.nx - 1, 1))
            sz = float(self.geometry.src_z[shot_id] / max(self.nz - 1, 1))

            colloc = sample_collocation_points(
                n_points=n_collocation,
                device=self.device,
                source_xy=(sx, sz),
                source_bias_ratio=float(train_cfg.get("source_bias_ratio", 0.3)),
                source_sigma=float(train_cfg.get("source_bias_sigma", 0.05)),
            )
            ic = sample_initial_points(n_points=n_ic, device=self.device)
            bc = sample_boundary_points(n_points=n_bc, device=self.device)

            # Data loss
            trace_batch = sample_receiver_trace_batch(
                geom=self.geometry,
                observed=self.observed,
                shots_per_batch=shots_per_batch,
                n_receivers=int(train_cfg.get("n_data_receivers", 32)),
                nx=self.nx,
                nz=self.nz,
                device=self.device,
                shot_ids=np.array([shot_id]),
            )
            u_data_pred = self.pinn(
                trace_batch["x"],
                trace_batch["z"],
                trace_batch["t"],
                trace_batch["shot_id"],
            )
            s_batch = int(trace_batch["n_shot_batch"].item())
            r_batch = int(trace_batch["n_rec_batch"].item())
            n_time = int(trace_batch["n_time"].item())
            u_tr = u_data_pred.reshape(s_batch, r_batch, n_time)
            d_tr = trace_batch["d_obs_traces"]

            # Normalize
            u_rms = u_tr.std() + 1e-8
            d_rms = d_tr.std() + 1e-8
            u_tr_norm = u_tr / u_rms
            d_tr_norm = d_tr / d_rms
            l_data = F.mse_loss(u_tr_norm, d_tr_norm)

            # Physics loss
            vel_fn = self._velocity_fn()
            src_fn = self._source_fn(shot_id)
            pde_shot_id = torch.full_like(colloc["x"], shot_id)
            ic_shot_id = torch.full_like(ic["x"], shot_id)

            pde = acoustic_pde_residual(
                self.pinn,
                velocity_fn=vel_fn,
                x=colloc["x"],
                z=colloc["z"],
                t=colloc["t"],
                shot_id=pde_shot_id,
                source_fn=src_fn,
                t_max=self.t_max,
                domain_x=self.domain_x,
                domain_z=self.domain_z,
            )
            l_pde = physics_mse_loss_causal(
                pde["r"],
                colloc["t"],
                epsilon=float(w_cfg.get("causal_epsilon", 5.0)),
                n_chunks=int(w_cfg.get("causal_chunks", 24)),
                min_weight=float(w_cfg.get("causal_min_weight", 1e-3)),
            )

            # IC loss
            ic_out = acoustic_pde_residual(
                self.pinn,
                velocity_fn=vel_fn,
                x=ic["x"],
                z=ic["z"],
                t=ic["t"],
                shot_id=ic_shot_id,
                source_fn=None,
                t_max=self.t_max,
                domain_x=self.domain_x,
                domain_z=self.domain_z,
            )
            ic_weight_u = 0.0 if getattr(self.pinn, "enforces_u0_by_construction", lambda: False)() else 1.0
            ic_weight_ut = 0.0 if getattr(self.pinn, "enforces_ut0_by_construction", lambda: False)() else 1.0
            if ic_weight_u == 0.0 and ic_weight_ut == 0.0:
                l_ic = torch.zeros((), device=self.device)
            else:
                l_ic = initial_condition_loss(
                    ic_out["u"],
                    ic_out["u_t"],
                    weight_u=ic_weight_u,
                    weight_ut=ic_weight_ut,
                )

            # BC loss
            bc_shot_id = torch.full_like(bc["t"], shot_id)
            ub = self.pinn(bc["x"], bc["z"], bc["t"], bc_shot_id)
            l_bc = boundary_loss(ub)

            # Regularization
            vp_grid = self._vp_grid_from_net()
            l_reg = (
                float(w_cfg.get("w_smooth", 1.0)) * smoothness_regularization(vp_grid)
                + float(w_cfg.get("w_tv", 0.0)) * total_variation_regularization(vp_grid)
                + float(w_cfg.get("w_charbonnier", 0.0))
                * charbonnier_regularization(
                    vp_grid,
                    epsilon=float(w_cfg.get("charbonnier_epsilon", 1.0)),
                    alpha=float(w_cfg.get("charbonnier_alpha", 0.45)),
                )
                + float(w_cfg.get("w_bounds", 1.0))
                * velocity_bounds_penalty(
                    vp_grid,
                    vp_min=float(self.cfg["model"]["vp_min"]),
                    vp_max=float(self.cfg["model"]["vp_max"]),
                )
            )
            l_well = torch.zeros((), device=self.device)

            # Loss weighting
            ws = self.loss_scheduler.get(epoch)
            l_total = (
                ws["w_pde"] * l_pde
                + ws["w_data"] * l_data
                + ws["w_ic"] * l_ic
                + float(w_cfg.get("w_bc", 1.0)) * l_bc
                + float(w_cfg.get("w_reg", 1e-3)) * l_reg
            )

            if not torch.isfinite(l_total):
                print(f"[WARN] epoch={epoch} non-finite loss; skipping")
                continue

            l_total.backward()

            # Gradient clipping
            if update_pinn:
                torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
            if update_vp and self.velocity_net is not None:
                torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)

            if update_pinn:
                self.optim_pinn.step()
                self.scheduler_pinn.step()
            if self.optim_vp is not None and update_vp:
                self.optim_vp.step()
                self.scheduler_vp.step()

            # Record metrics
            losses = {
                "loss_total": float(l_total.detach().cpu()),
                "loss_data": float(l_data.detach().cpu()),
                "loss_physics": float(l_pde.detach().cpu()),
                "loss_ic": float(l_ic.detach().cpu()),
                "loss_bc": float(l_bc.detach().cpu()),
                "loss_reg": float(l_reg.detach().cpu()),
                "loss_well": float(l_well.detach().cpu()),
            }
            for k, v in losses.items():
                self.history[k].append(v)

            metrics = TrainingMetrics(
                epoch=epoch,
                loss_total=losses["loss_total"],
                loss_data=losses["loss_data"],
                loss_physics=losses["loss_physics"],
                loss_ic=losses["loss_ic"],
                loss_bc=losses["loss_bc"],
                loss_reg=losses["loss_reg"],
                loss_well=losses["loss_well"],
                grad_norm_pinn=0.0,
                grad_norm_vp=0.0,
                w_pde=ws["w_pde"],
                w_data=ws["w_data"],
                w_data_eff=ws["w_data"],
                fmax=0.0,
                lr_pinn=self._get_current_lr(self.optim_pinn),
                lr_vp=self._get_current_lr(self.optim_vp) if self.optim_vp else 0.0,
            )
            self.metrics_history.append(metrics)

            # Logging
            if (epoch + 1) % self.print_every == 0:
                print(
                    f"[Epoch {epoch + 1:05d}] "
                    f"L={losses['loss_total']:.4e}, "
                    f"L_data={losses['loss_data']:.4e}, "
                    f"L_pde={losses['loss_physics']:.4e}"
                )

            if (epoch + 1) % self.plot_every == 0:
                vp_est = self._vp_grid_from_net().detach().cpu().numpy()
                plot_training_snapshot(
                    epoch=epoch + 1,
                    vp_est=vp_est,
                    history=self.history,
                    save_path=self.fig_dir / f"snapshot_epoch_{epoch + 1:05d}.png",
                )

            if (epoch + 1) % self.checkpoint_every == 0:
                save_checkpoint(
                    self.pinn,
                    self.ckpt_dir / f"pinn_epoch_{epoch + 1:05d}.pt",
                    epoch=epoch + 1,
                )
                if self.invert_vp:
                    save_checkpoint(
                        self.velocity_net,
                        self.ckpt_dir / f"velocity_net_epoch_{epoch + 1:05d}.pt",
                        epoch=epoch + 1,
                    )

            pbar.set_postfix(
                total=f"{losses['loss_total']:.3e}",
                data=f"{losses['loss_data']:.3e}",
                pde=f"{losses['loss_physics']:.3e}",
            )

        return self.history

    def estimate_velocity(self) -> np.ndarray:
        """Get final velocity estimate."""
        vp_grid = self._vp_grid_from_net()
        return vp_grid.detach().cpu().numpy()
