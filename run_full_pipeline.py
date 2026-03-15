#!/usr/bin/env python3
"""
Complete acoustic PINN-FWI pipeline: data generation → training → results.
Run from project root: python run_full_pipeline.py --config production.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_marmousi_vp, model_stats
from src.forward.acquisition import AcquisitionGeometry, build_surface_acquisition
from src.forward.acoustic_forward import generate_observed_data
from src.pinn.acoustic_pinn import AcousticPINN, AcousticPINNConfig
from src.pinn.velocity_net import VelocityNet, VelocityNetConfig
from src.inversion.improved_trainer import ImprovedAcousticPINNFWITrainer
from src.utils.io import load_yaml, seed_everything, resolve_path, save_checkpoint
from src.utils.viz import plot_velocity_model, plot_well_log_comparison, plot_losses, plot_true_vs_estimated


def setup_data(cfg: dict, project_root: Path) -> tuple[np.ndarray, AcquisitionGeometry, np.ndarray]:
    """Load/generate data."""
    print("\n" + "="*60)
    print("STEP 1: DATA SETUP")
    print("="*60)
    
    # Validate config
    required_keys = ["data", "model", "acquisition"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")
    
    # Load Marmousi
    vp_path = resolve_path(project_root, cfg["data"]["vp_path"])
    if not vp_path.exists():
        raise FileNotFoundError(f"Velocity model not found: {vp_path}")
    
    print(f"Loading Marmousi from: {vp_path}")
    vp_full = load_marmousi_vp(vp_path, subsample=int(cfg["model"]["subsample"]))
    
    # Validate loaded data
    if vp_full.size == 0:
        raise ValueError("Loaded velocity model is empty")
    if np.any(np.isnan(vp_full)) or np.any(np.isinf(vp_full)):
        raise ValueError("Velocity model contains NaN or Inf values")
    
    print(f"Velocity model shape: {vp_full.shape}")
    print(f"Velocity stats: {model_stats(vp_full)}")
    
    # Setup acquisition
    acq_cfg = cfg["acquisition"]
    
    # Validate acquisition config
    if int(acq_cfg["n_shots"]) <= 0:
        raise ValueError(f"n_shots must be > 0, got {acq_cfg['n_shots']}")
    if int(acq_cfg["nt"]) <= 0:
        raise ValueError(f"nt must be > 0, got {acq_cfg['nt']}")
    if float(acq_cfg["dt"]) <= 0:
        raise ValueError(f"dt must be > 0, got {acq_cfg['dt']}")
    
    vp_full_shape = vp_full.shape
    geom = build_surface_acquisition(
        nx=vp_full_shape[1],
        nz=vp_full_shape[0],
        n_shots=int(acq_cfg["n_shots"]),
        nt=int(acq_cfg["nt"]),
        dt=float(acq_cfg["dt"]),
        src_depth_idx=int(acq_cfg["src_depth_idx"]),
        rec_depth_idx=int(acq_cfg["rec_depth_idx"]),
        pad_x=int(acq_cfg["pad_x"]),
        receiver_stride=int(acq_cfg["receiver_stride"]),
    )
    print(f"Acquisition geometry: {geom.n_shots} shots, {geom.n_receivers} receivers, {geom.nt} time steps")
    
    # Generate observed data
    obs_path = resolve_path(project_root, cfg["data"]["observed_path"])
    geom_path = resolve_path(project_root, cfg["data"]["geometry_path"])
    
    if obs_path.exists():
        print(f"Loading observed data from: {obs_path}")
        observed = np.load(obs_path).astype(np.float32)
        
        # Validate observed data
        if observed.size == 0:
            raise ValueError("Loaded observed data is empty")
        if np.any(np.isnan(observed)) or np.any(np.isinf(observed)):
            raise ValueError("Observed data contains NaN or Inf values")
        if observed.shape[0] != geom.n_shots:
            raise ValueError(f"Observed data shots {observed.shape[0]} != geometry shots {geom.n_shots}")
        if observed.shape[1] != geom.nt:
            raise ValueError(f"Observed data time steps {observed.shape[1]} != geometry nt {geom.nt}")
    else:
        print(f"Generating observed data (backend: {acq_cfg['backend']})...")
        observed = generate_observed_data(
            vp=vp_full,
            geom=geom,
            dh=float(acq_cfg["dh"]),
            f_peak=float(acq_cfg["f_peak"]),
            output_path=obs_path,
            backend=acq_cfg["backend"],
            device=cfg["device"],
        )
        print(f"Observed data shape: {observed.shape}")
    
    # Save geometry
    geom_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        geom_path,
        src_x=geom.src_x,
        src_z=geom.src_z,
        rec_x=geom.rec_x,
        rec_z=geom.rec_z,
        time=geom.time,
    )
    
    return vp_full, geom, observed


def setup_models(cfg: dict, device: str) -> tuple[AcousticPINN, VelocityNet]:
    """Initialize neural networks."""
    print("\n" + "="*60)
    print("STEP 2: MODEL SETUP")
    print("="*60)
    
    # Validate model config
    vp_min = float(cfg["model"]["vp_min"])
    vp_max = float(cfg["model"]["vp_max"])
    if vp_min >= vp_max:
        raise ValueError(f"vp_min ({vp_min}) must be < vp_max ({vp_max})")
    if vp_min <= 0 or vp_max <= 0:
        raise ValueError(f"vp_min and vp_max must be positive")
    
    # Validate device
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # PINN
    pinn_cfg = AcousticPINNConfig(
        in_dim=int(cfg["pinn"]["in_dim"]),
        n_shots=int(cfg["pinn"]["n_shots"]),
        hidden_layers=int(cfg["pinn"]["hidden_layers"]),
        hidden_width=int(cfg["pinn"]["hidden_width"]),
        activation=cfg["pinn"]["activation"],
        first_omega_0=float(cfg["pinn"]["first_omega_0"]),
        hidden_omega_0=float(cfg["pinn"]["hidden_omega_0"]),
        fourier_features=bool(cfg["pinn"]["fourier_features"]),
        n_fourier=int(cfg["pinn"]["n_fourier"]),
        fourier_scale=float(cfg["pinn"]["fourier_scale"]),
        hard_constraint=cfg["pinn"]["hard_constraint"],
        hard_constraint_scale=float(cfg["pinn"]["hard_constraint_scale"]),
        hard_constraint_power=int(cfg["pinn"]["hard_constraint_power"]),
    )
    pinn = AcousticPINN(pinn_cfg).to(device)
    n_params_pinn = sum(p.numel() for p in pinn.parameters())
    print(f"PINN: {n_params_pinn:,} parameters")
    
    # Velocity network
    vp_cfg = VelocityNetConfig(
        in_dim=2,
        hidden_layers=int(cfg["velocity_net"]["hidden_layers"]),
        hidden_width=int(cfg["velocity_net"]["hidden_width"]),
        activation=cfg["velocity_net"]["activation"],
        fourier_features=bool(cfg["velocity_net"]["fourier_features"]),
        n_fourier=int(cfg["velocity_net"]["n_fourier"]),
        fourier_scale=float(cfg["velocity_net"]["fourier_scale"]),
        output_activation=cfg["velocity_net"]["output_activation"],
        vp_min=float(cfg["model"]["vp_min"]),
        vp_max=float(cfg["model"]["vp_max"]),
    )
    velocity_net = VelocityNet(vp_cfg).to(device)
    n_params_vp = sum(p.numel() for p in velocity_net.parameters())
    print(f"VelocityNet: {n_params_vp:,} parameters")
    print(f"Total: {n_params_pinn + n_params_vp:,} parameters")
    
    return pinn, velocity_net


def train_models(
    pinn: AcousticPINN,
    velocity_net: VelocityNet,
    observed: np.ndarray,
    geom: AcquisitionGeometry,
    vp_true: np.ndarray,
    cfg: dict,
    project_root: Path,
) -> dict:
    """Train the models."""
    print("\n" + "="*60)
    print("STEP 3: TRAINING")
    print("="*60)
    
    trainer = ImprovedAcousticPINNFWITrainer(
        pinn=pinn,
        velocity_net=velocity_net,
        observed=observed,
        geometry=geom,
        vp_true=vp_true,
        config=cfg,
        project_root=project_root,
        invert_vp=True,
    )
    
    history = trainer.train()
    
    # Save final models
    final_pinn_path = trainer.ckpt_dir / "pinn_final.pt"
    final_vp_path = trainer.ckpt_dir / "velocity_net_final.pt"
    save_checkpoint(pinn, final_pinn_path, epoch=cfg["training"]["n_epochs"])
    save_checkpoint(velocity_net, final_vp_path, epoch=cfg["training"]["n_epochs"])
    print(f"Saved final PINN to: {final_pinn_path}")
    print(f"Saved final VelocityNet to: {final_vp_path}")
    
    return history, trainer


def evaluate_results(trainer: ImprovedAcousticPINNFWITrainer, project_root: Path):
    """Generate final results and visualizations."""
    print("\n" + "="*60)
    print("STEP 4: RESULTS & VISUALIZATION")
    print("="*60)
    
    # Get final velocity estimate
    vp_est = trainer.estimate_velocity()
    vp_true = trainer.vp_true
    
    # Compute errors
    diff = vp_est - vp_true
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    rel_rmse = rmse / vp_true.mean()
    
    print(f"\nVelocity Estimation Errors:")
    print(f"  MAE:      {mae:.2f} m/s")
    print(f"  RMSE:     {rmse:.2f} m/s")
    print(f"  Rel RMSE: {rel_rmse:.4f} ({rel_rmse*100:.2f}%)")
    
    # Save final estimate
    final_vp_path = trainer.ckpt_dir / "vp_est_final.npy"
    np.save(final_vp_path, vp_est.astype(np.float32))
    print(f"Saved final velocity estimate to: {final_vp_path}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Velocity comparison
    fig_path = trainer.fig_dir / "velocity_comparison_final.png"
    plot_true_vs_estimated(
        vp_true=vp_true,
        vp_est=vp_est,
        save_path=fig_path,
        title="Final Velocity Inversion Result",
    )
    print(f"Saved velocity comparison to: {fig_path}")
    
    # Loss history
    fig_path = trainer.fig_dir / "loss_history_final.png"
    plot_losses(
        history=trainer.history,
        save_path=fig_path,
    )
    print(f"Saved loss history to: {fig_path}")
    
    # Well logs
    fig_path = trainer.fig_dir / "well_logs_final.png"
    plot_well_log_comparison(
        vp_true=vp_true,
        vp_est=vp_est,
        well_x_indices=[vp_true.shape[1] // 4, vp_true.shape[1] // 2, 3 * vp_true.shape[1] // 4],
        save_path=fig_path,
    )
    print(f"Saved well logs to: {fig_path}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Results saved to: {trainer.fig_dir}")
    print(f"Checkpoints saved to: {trainer.ckpt_dir}")
    print(f"Logs saved to: {trainer.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run full acoustic PINN-FWI pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="production.yaml",
        help="Config file (default: production.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, or cuda (default: auto)",
    )
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / "configs" / args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    cfg = load_yaml(config_path)
    if args.device != "auto":
        cfg["device"] = args.device
    
    # Setup device
    if cfg["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]
    cfg["device"] = device
    print(f"Using device: {device}")
    
    # Seed
    seed_everything(int(cfg["seed"]))
    
    # Run pipeline
    try:
        vp_true, geom, observed = setup_data(cfg, PROJECT_ROOT)
        pinn, velocity_net = setup_models(cfg, device)
        history, trainer = train_models(pinn, velocity_net, observed, geom, vp_true, cfg, PROJECT_ROOT)
        evaluate_results(trainer, PROJECT_ROOT)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
