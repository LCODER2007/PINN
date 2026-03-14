"""Acoustic PINN-FWI training loop with logging/checkpointing/figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ..forward.acquisition import AcquisitionGeometry
from ..forward.ricker import analytic_ricker_torch
from ..pinn.physics_residual import acoustic_pde_residual
from ..pinn.sampling import (
    sample_boundary_points,
    sample_collocation_points,
    sample_initial_points,
    sample_receiver_data_points,
    sample_receiver_trace_batch,
)
from .losses import (
    boundary_loss,
    charbonnier_regularization,
    data_mse_loss,
    initial_condition_loss,
    physics_mse_loss,
    physics_mse_loss_causal,
    smoothness_regularization,
    total_variation_regularization,
    velocity_bounds_penalty,
)
from .schedule import LossWeightScheduler
from ..utils.io import append_csv_row, ensure_dir, save_checkpoint
from ..utils.viz import plot_training_snapshot


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


class AcousticPINNFWITrainer:
    """Joint trainer for AcousticPINN + VelocityNet.

    Observed data convention is **[shot, time, receiver]**.
    """

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
        self.state = TrainerState()

        self.nz, self.nx = self.vp_true.shape
        self.t_max = float(self.geometry.time[-1]) if self.geometry.nt > 0 else 1.0

        # Physical domain extents for PDE non-dimensionalization
        dh = float(self.cfg["acquisition"]["dh"])
        self.domain_x = max((self.nx - 1) * dh, 1e-8)   # metres
        self.domain_z = max((self.nz - 1) * dh, 1e-8)   # metres

        logging_cfg = self.cfg.get("logging", {})
        results_dir = self.project_root / logging_cfg.get("results_dir", "results")
        self.fig_dir = ensure_dir(results_dir / "figures")
        self.ckpt_dir = ensure_dir(results_dir / "checkpoints")
        self.log_dir = ensure_dir(results_dir / "logs")
        self.csv_path = self.log_dir / "train_log.csv"

        train_cfg = self.cfg["training"]
        self.n_epochs = int(train_cfg["n_epochs"])
        self.print_every = int(logging_cfg.get("print_every", 50))
        self.plot_every = int(logging_cfg.get("plot_every", 200))
        self.checkpoint_every = int(logging_cfg.get("checkpoint_every", 500))

        self.optim_pinn = torch.optim.Adam(self.pinn.parameters(), lr=float(train_cfg["lr_pinn"]))
        vp_params = list(self.velocity_net.parameters()) if self.invert_vp else []
        self.optim_vp = (
            torch.optim.Adam(vp_params, lr=float(train_cfg["lr_vp"])) if len(vp_params) > 0 else None
        )

        self.scheduler = LossWeightScheduler(
            w_pde_start=float(self.cfg["weights"].get("w_pde_start", self.cfg["weights"]["w_pde"])),
            w_pde_end=float(self.cfg["weights"].get("w_pde", 100.0)),
            w_data_start=float(self.cfg["weights"].get("w_data_start", 1.0)),
            w_data_end=float(self.cfg["weights"].get("w_data", 50.0)),
            w_ic_start=float(self.cfg["weights"].get("w_ic_start", self.cfg["weights"].get("w_ic", 50.0))),
            w_ic_end=float(self.cfg["weights"].get("w_ic_end", self.cfg["weights"].get("w_ic", 5.0))),
            warmup_epochs=int(train_cfg.get("warmup_epochs", max(1, self.n_epochs // 3))),
            total_epochs=self.n_epochs,
        )

        # Causal weighting & gradient clipping
        self.causal_epsilon = float(self.cfg["weights"].get("causal_epsilon", 20.0))
        self.causal_chunks = int(self.cfg["weights"].get("causal_chunks", 16))
        self.causal_min_weight = float(self.cfg["weights"].get("causal_min_weight", 1e-3))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.grad_clip_value = float(train_cfg.get("grad_clip_value", 0.0))
        self.skip_nonfinite_steps = bool(train_cfg.get("skip_nonfinite_steps", True))
        self.alternating_updates = bool(train_cfg.get("alternating_updates", True))
        self.warmup_pinn_only_epochs = int(train_cfg.get("warmup_pinn_only_epochs", 0))
        self.pinn_steps_per_cycle = int(train_cfg.get("pinn_steps_per_cycle", 1))
        self.vp_steps_per_cycle = int(train_cfg.get("vp_steps_per_cycle", 1))

        # Data fit stabilization: trace-level frequency continuation
        self.use_frequency_continuation = bool(train_cfg.get("use_frequency_continuation", False))
        self.freq_start_hz = float(train_cfg.get("freq_start_hz", 2.0))
        self.freq_end_hz = float(train_cfg.get("freq_end_hz", self.cfg["acquisition"].get("f_peak", 8.0)))
        self.freq_ramp_epochs = int(train_cfg.get("freq_ramp_epochs", self.n_epochs))
        self.n_data_receivers = int(train_cfg.get("n_data_receivers", 24))
        self.data_normalization = str(train_cfg.get("data_normalization", "rms")).lower()
        self.data_loss = str(train_cfg.get("data_loss", "smooth_l1")).lower()
        self.data_gain_match = bool(train_cfg.get("data_gain_match", True))
        self.norm_floor_ratio = float(train_cfg.get("data_norm_floor_ratio", 0.1))
        self.obs_global_rms = float(np.sqrt(np.mean(self.observed.astype(np.float64) ** 2)))
        self.data_batch_all_shots = bool(train_cfg.get("data_batch_all_shots", False))
        self.active_receiver_rms_ratio = float(train_cfg.get("active_receiver_rms_ratio", 0.0))
        self.use_well_prior = bool(train_cfg.get("use_well_prior", False)) and self.invert_vp
        self.well_position_fracs = train_cfg.get("well_position_fracs", [0.5])
        self.well_depth_frac = float(train_cfg.get("well_depth_frac", 0.75))

        # Adaptive balance between PDE and data terms
        self.adaptive_data_weight = bool(self.cfg["weights"].get("adaptive_data_weight", True))
        self.data_weight_min_scale = float(self.cfg["weights"].get("data_weight_min_scale", 0.5))
        self.data_weight_max_scale = float(self.cfg["weights"].get("data_weight_max_scale", 8.0))
        self.data_weight_beta = float(self.cfg["weights"].get("data_weight_beta", 0.5))

        self.history: dict[str, list[float]] = {
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

        self._well_specs: list[tuple[int, int, torch.Tensor]] = []
        if self.use_well_prior:
            depth = max(1, min(self.nz, int(round(self.well_depth_frac * self.nz))))
            vp_min = float(self.cfg["model"]["vp_min"])
            vp_max = float(self.cfg["model"]["vp_max"])
            vp_range = max(vp_max - vp_min, 1e-8)
            for frac in self.well_position_fracs:
                ix = int(round(float(frac) * (self.nx - 1)))
                ix = max(0, min(self.nx - 1, ix))
                target = (self.vp_true[:depth, ix] - vp_min) / vp_range
                target_t = torch.tensor(target[:, None], dtype=torch.float32, device=self.device)
                self._well_specs.append((ix, depth, target_t))

    def _current_fmax(self, epoch: int) -> float:
        ramp = max(self.freq_ramp_epochs, 1)
        alpha = min(max(epoch / ramp, 0.0), 1.0)
        return self.freq_start_hz + alpha * (self.freq_end_hz - self.freq_start_hz)

    def _lowpass_traces(self, traces: torch.Tensor, fmax_hz: float) -> torch.Tensor:
        """Apply simple low-pass in the frequency domain.

        Input shape: [shot_batch, receiver_batch, nt].
        """
        dt = float(self.geometry.dt)
        if dt <= 0.0 or fmax_hz <= 0.0:
            return traces

        nt = traces.shape[-1]
        freqs = torch.fft.rfftfreq(nt, d=dt, device=traces.device)
        spec = torch.fft.rfft(traces, dim=-1)

        # Cosine taper near cutoff for smoother gradients.
        taper_width = max(0.5, 0.2 * fmax_hz)
        pass_band = freqs <= max(fmax_hz - taper_width, 0.0)
        stop_band = freqs >= (fmax_hz + taper_width)
        trans_band = (~pass_band) & (~stop_band)

        mask = torch.zeros_like(freqs)
        mask[pass_band] = 1.0
        if trans_band.any():
            ft = freqs[trans_band]
            arg = (ft - (fmax_hz - taper_width)) / max(2.0 * taper_width, 1e-8)
            mask[trans_band] = 0.5 * (1.0 + torch.cos(np.pi * arg))

        spec = spec * mask.view(1, 1, -1)
        return torch.fft.irfft(spec, n=nt, dim=-1)

    def _normalize_pair(self, pred: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.data_normalization
        if mode == "none":
            return pred, obs

        if mode == "zscore":
            mu = obs.mean(dim=-1, keepdim=True)
            sigma = obs.std(dim=-1, keepdim=True).clamp(min=1e-6)
            return (pred - mu) / sigma, (obs - mu) / sigma

        # Default: RMS normalization per trace.
        rms = torch.sqrt((obs.pow(2)).mean(dim=-1, keepdim=True))
        rms_floor = max(1e-6, self.norm_floor_ratio * self.obs_global_rms)
        rms = rms.clamp(min=rms_floor)
        return pred / rms, obs / rms

    def _match_trace_gain(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Least-squares trace-wise gain matching to reduce amplitude bias."""
        num = (pred * obs).sum(dim=-1, keepdim=True)
        den = pred.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gain = (num / den).clamp(min=0.0, max=10.0)
        return gain * pred

    def _data_loss_value(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if self.data_loss == "l1":
            return F.l1_loss(pred, obs)
        if self.data_loss in ("huber", "smooth_l1"):
            return F.smooth_l1_loss(pred, obs, beta=0.05)
        return data_mse_loss(pred, obs)

    def _effective_data_weight(self, w_data: float, w_pde: float, l_data: torch.Tensor, l_pde: torch.Tensor) -> float:
        if not self.adaptive_data_weight:
            return float(w_data)
        ratio = (w_pde * l_pde.detach()) / (w_data * l_data.detach() + 1e-12)
        scale = torch.clamp(ratio.pow(self.data_weight_beta), self.data_weight_min_scale, self.data_weight_max_scale)
        return float(w_data * scale.item())

    def _zero_optimizers(self) -> None:
        self.optim_pinn.zero_grad(set_to_none=True)
        if self.optim_vp is not None:
            self.optim_vp.zero_grad(set_to_none=True)

    def _set_module_grad(self, module: torch.nn.Module | None, enabled: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _get_update_flags(self, epoch: int) -> tuple[bool, bool]:
        if not self.invert_vp:
            return True, False
        if not self.alternating_updates:
            return True, True
        if epoch < self.warmup_pinn_only_epochs:
            return True, False

        pinn_steps = max(self.pinn_steps_per_cycle, 0)
        vp_steps = max(self.vp_steps_per_cycle, 0)
        cycle = max(pinn_steps + vp_steps, 1)
        offset = (epoch - self.warmup_pinn_only_epochs) % cycle
        update_pinn = offset < pinn_steps
        update_vp = not update_pinn

        # Fallback if one phase length is configured as zero.
        if pinn_steps == 0 and vp_steps > 0:
            update_pinn, update_vp = False, True
        elif vp_steps == 0 and pinn_steps > 0:
            update_pinn, update_vp = True, False
        elif pinn_steps == 0 and vp_steps == 0:
            update_pinn, update_vp = True, True
        return update_pinn, update_vp

    def _select_data_shot_ids(self, shots_per_batch: int, preferred_shot: int | None = None) -> np.ndarray:
        if self.data_batch_all_shots or shots_per_batch >= self.geometry.n_shots:
            shot_ids = np.arange(self.geometry.n_shots, dtype=np.int64)
        else:
            shot_ids = np.random.choice(
                self.geometry.n_shots,
                size=min(shots_per_batch, self.geometry.n_shots),
                replace=False,
            ).astype(np.int64)

        if preferred_shot is not None and preferred_shot not in shot_ids:
            shot_ids[0] = int(preferred_shot)
        return np.unique(shot_ids)

    def _clip_gradients(self, update_pinn: bool, update_vp: bool) -> tuple[bool, dict[str, float]]:
        stats = {"grad_norm_pinn": 0.0, "grad_norm_vp": 0.0}
        modules: list[tuple[str, torch.nn.Module]] = []
        if update_pinn:
            modules.append(("grad_norm_pinn", self.pinn))
        if update_vp and self.invert_vp and self.velocity_net is not None:
            modules.append(("grad_norm_vp", self.velocity_net))

        for key, module in modules:
            params = [param for param in module.parameters() if param.grad is not None]
            if not params:
                continue

            grads_are_finite = all(torch.isfinite(param.grad).all() for param in params)
            if not grads_are_finite:
                if self.skip_nonfinite_steps:
                    self._zero_optimizers()
                    return False, stats
                for param in params:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

            if self.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(params, self.grad_clip_value)

            if self.grad_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    params,
                    self.grad_clip,
                    error_if_nonfinite=False,
                )
            else:
                sq_norm = torch.zeros((), device=self.device)
                for param in params:
                    sq_norm = sq_norm + param.grad.detach().pow(2).sum()
                norm = sq_norm.sqrt()

            stats[key] = float(norm.detach().cpu())

        return True, stats

    def _vp_grid_from_net(self) -> torch.Tensor:
        z = torch.linspace(0.0, 1.0, self.nz, device=self.device)
        x = torch.linspace(0.0, 1.0, self.nx, device=self.device)
        zg, xg = torch.meshgrid(z, x, indexing="ij")
        coords = torch.stack([xg.reshape(-1), zg.reshape(-1)], dim=-1)

        if self.invert_vp:
            vp = self.velocity_net.forward_coords(coords).reshape(self.nz, self.nx)
        else:
            vp = torch.tensor(self.vp_true, device=self.device, dtype=torch.float32)
        return vp

    def _velocity_fn(self):
        if self.invert_vp:
            return lambda x, z: self.velocity_net(x, z)

        vp_t = torch.tensor(self.vp_true, dtype=torch.float32, device=self.device)

        def fixed_vp(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            ix = torch.clamp((x * (self.nx - 1)).long(), 0, self.nx - 1)
            iz = torch.clamp((z * (self.nz - 1)).long(), 0, self.nz - 1)
            return vp_t[iz[:, 0], ix[:, 0]].unsqueeze(-1)

        return fixed_vp

    def _well_loss(self, vp_grid: torch.Tensor) -> torch.Tensor:
        if not self._well_specs:
            return torch.zeros((), device=self.device)

        vp_min = float(self.cfg["model"]["vp_min"])
        vp_max = float(self.cfg["model"]["vp_max"])
        vp_range = max(vp_max - vp_min, 1e-8)
        loss = torch.zeros((), device=self.device)
        for ix, depth, target in self._well_specs:
            pred = (vp_grid[:depth, ix : ix + 1] - vp_min) / vp_range
            loss = loss + F.smooth_l1_loss(pred, target, beta=0.02)
        return loss / max(len(self._well_specs), 1)

    def _source_fn(self, shot_id: int):
        sx = float(self.geometry.src_x[shot_id] / max(self.nx - 1, 1))
        sz = float(self.geometry.src_z[shot_id] / max(self.nz - 1, 1))
        sigma = float(self.cfg["training"].get("source_sigma", 0.02))
        amp = float(self.cfg["training"].get("source_amplitude", 1.0))
        f_peak = float(self.cfg["acquisition"]["f_peak"])

        def source(x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            rr = ((x - sx) ** 2 + (z - sz) ** 2) / (2.0 * sigma**2)
            spatial = torch.exp(-rr)
            wt = analytic_ricker_torch(t * self.t_max, f_peak=f_peak)
            return amp * spatial * wt

        return source

    def _log_epoch(self, epoch: int, losses: dict[str, float], w_pde: float, w_data: float) -> None:
        row = {
            "epoch": epoch,
            "loss_total": losses["loss_total"],
            "loss_data": losses["loss_data"],
            "loss_physics": losses["loss_physics"],
            "loss_ic": losses["loss_ic"],
            "loss_bc": losses["loss_bc"],
            "loss_reg": losses["loss_reg"],
            "w_pde": w_pde,
            "w_data": w_data,
        }
        append_csv_row(self.csv_path, row)

    def train(self) -> dict[str, list[float]]:
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

            update_pinn, update_vp = self._get_update_flags(epoch)
            self._set_module_grad(self.pinn, update_pinn)
            self._set_module_grad(self.velocity_net, update_vp)

            if self.optim_vp is not None and update_vp:
                self.optim_vp.zero_grad(set_to_none=True)
            if update_pinn:
                self.optim_pinn.zero_grad(set_to_none=True)

            # sample one active shot for PDE source consistency
            shot_id = int(np.random.randint(0, self.geometry.n_shots))
            data_shot_ids = self._select_data_shot_ids(shots_per_batch=shots_per_batch, preferred_shot=shot_id)
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

            if self.use_frequency_continuation:
                trace_batch = sample_receiver_trace_batch(
                    geom=self.geometry,
                    observed=self.observed,
                    shots_per_batch=shots_per_batch,
                    n_receivers=self.n_data_receivers,
                    nx=self.nx,
                    nz=self.nz,
                    device=self.device,
                    shot_ids=data_shot_ids,
                    active_receiver_rms_ratio=self.active_receiver_rms_ratio,
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

                fmax = self._current_fmax(epoch)
                u_lp = self._lowpass_traces(u_tr, fmax_hz=fmax)
                d_lp = self._lowpass_traces(d_tr, fmax_hz=fmax)
                if self.data_gain_match:
                    u_lp = self._match_trace_gain(u_lp, d_lp)
                u_lp, d_lp = self._normalize_pair(u_lp, d_lp)
                l_data = self._data_loss_value(u_lp, d_lp)
            else:
                # Random receiver-time points (legacy mode).
                data_batch = sample_receiver_data_points(
                    geom=self.geometry,
                    observed=self.observed,
                    n_samples=n_data,
                    shots_per_batch=shots_per_batch,
                    nx=self.nx,
                    nz=self.nz,
                    device=self.device,
                    shot_ids=data_shot_ids,
                    active_receiver_rms_ratio=self.active_receiver_rms_ratio,
                )
                u_data_pred = self.pinn(data_batch["x"], data_batch["z"], data_batch["t"], data_batch["shot_id"])
                if self.data_gain_match:
                    u_data_pred = self._match_trace_gain(u_data_pred, data_batch["d_obs"])
                u_data_pred, d_obs = self._normalize_pair(u_data_pred, data_batch["d_obs"])
                l_data = self._data_loss_value(u_data_pred, d_obs)

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
                epsilon=self.causal_epsilon,
                n_chunks=self.causal_chunks,
                min_weight=self.causal_min_weight,
            )

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
            elif ic_weight_u == 0.0:
                l_ic = F.mse_loss(ic_out["u_t"], torch.zeros_like(ic_out["u_t"]))
            else:
                l_ic = initial_condition_loss(
                    ic_out["u"],
                    ic_out["u_t"],
                    weight_u=ic_weight_u,
                    weight_ut=ic_weight_ut,
                )

            bc_shot_id = torch.full_like(bc["t"], shot_id)
            ub = self.pinn(bc["x"], bc["z"], bc["t"], bc_shot_id)
            l_bc = boundary_loss(ub)

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
            l_well = self._well_loss(vp_grid)

            ws = self.scheduler.get(epoch)
            w_data_eff = self._effective_data_weight(
                ws["w_data"],
                ws["w_pde"],
                l_data=l_data,
                l_pde=l_pde,
            )
            l_total = (
                ws["w_pde"] * l_pde
                + w_data_eff * l_data
                + ws["w_ic"] * l_ic
                + float(w_cfg.get("w_bc", 1.0)) * l_bc
                + float(w_cfg.get("w_reg", 1e-3)) * l_reg
                + float(w_cfg.get("w_well", 0.0)) * l_well
            )

            if not torch.isfinite(l_total):
                self._zero_optimizers()
                print(f"[WARN] epoch={epoch} produced non-finite loss; skipping optimizer step")
                continue

            if epoch % 10 == 0:
                wd = ws["w_data"] * float(l_data.detach().cpu())
                wp = ws["w_pde"] * float(l_pde.detach().cpu())
                wbc = float(w_cfg.get("w_bc", 1.0)) * float(l_bc.detach().cpu())
                wic = ws["w_ic"] * float(l_ic.detach().cpu())
                wr = float(w_cfg.get("w_reg", 1e-3)) * float(l_reg.detach().cpu())
                ww = float(w_cfg.get("w_well", 0.0)) * float(l_well.detach().cpu())

                print(
                    f"[WEIGHTED] epoch={epoch} "
                    f"data={wd:.3e} "
                    f"pde={wp:.3e} "
                    f"bc={wbc:.3e} "
                    f"ic={wic:.3e} "
                    f"reg={wr:.3e} "
                    f"well={ww:.3e}"
                )

            l_total.backward()

            should_step, grad_stats = self._clip_gradients(update_pinn=update_pinn, update_vp=update_vp)
            if should_step:
                if update_pinn:
                    self.optim_pinn.step()
                if self.optim_vp is not None and update_vp:
                    self.optim_vp.step()
            else:
                print(f"[WARN] epoch={epoch} produced non-finite gradients; skipping optimizer step")

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
            self.history["grad_norm_pinn"].append(grad_stats["grad_norm_pinn"])
            self.history["grad_norm_vp"].append(grad_stats["grad_norm_vp"])

            self._log_epoch(epoch, losses, ws["w_pde"], ws["w_data"])
            self.history["fmax"].append(self._current_fmax(epoch) if self.use_frequency_continuation else float("nan"))
            self.history["w_data_eff"].append(w_data_eff)
            pbar.set_postfix(
                total=f"{losses['loss_total']:.3e}",
                data=f"{losses['loss_data']:.3e}",
                pde=f"{losses['loss_physics']:.3e}",
                mode=("P" if update_pinn else "-") + ("V" if update_vp else "-"),
                fmax=f"{self._current_fmax(epoch):.1f}" if self.use_frequency_continuation else "off",
            )

            if (epoch + 1) % self.print_every == 0:
                print(
                    f"[Epoch {epoch + 1:05d}] "
                    f"L={losses['loss_total']:.4e}, "
                    f"L_data={losses['loss_data']:.4e}, "
                    f"L_pde={losses['loss_physics']:.4e}, "
                    f"L_reg={losses['loss_reg']:.4e}"
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

            # Restore grad flags in case user runs additional code after training.
            self._set_module_grad(self.pinn, True)
            if self.velocity_net is not None:
                self._set_module_grad(self.velocity_net, True)

        return self.history

    def estimate_velocity(self) -> np.ndarray:
        self.pinn.eval()
        if self.invert_vp:
            self.velocity_net.eval()
        with torch.no_grad():
            vp_est = self._vp_grid_from_net().detach().cpu().numpy()
        return vp_est.astype(np.float32)
