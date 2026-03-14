"""Enhanced visualization utilities for acoustic PINN-FWI results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_velocity_model(
    vp_true: np.ndarray,
    vp_est: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Velocity Model Comparison",
    cmap: str = "viridis",
) -> None:
    """Plot true vs estimated velocity models side-by-side with error."""
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    vmin, vmax = vp_true.min(), vp_true.max()
    
    # True model
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(vp_true, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title("True Velocity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (grid points)")
    ax1.set_ylabel("Z (grid points)")
    plt.colorbar(im1, ax=ax1, label="Vp (m/s)")
    
    # Estimated model
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(vp_est, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax2.set_title("Estimated Velocity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (grid points)")
    ax2.set_ylabel("Z (grid points)")
    plt.colorbar(im2, ax=ax2, label="Vp (m/s)")
    
    # Error
    ax3 = fig.add_subplot(gs[0, 2])
    error = vp_est - vp_true
    im3 = ax3.imshow(error, cmap="RdBu_r", aspect="auto")
    ax3.set_title("Estimation Error", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X (grid points)")
    ax3.set_ylabel("Z (grid points)")
    cbar = plt.colorbar(im3, ax=ax3, label="Error (m/s)")
    
    # Compute statistics
    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean())
    rel_rmse = rmse / vp_true.mean()
    
    fig.suptitle(
        f"{title}\nMAE={mae:.1f} m/s, RMSE={rmse:.1f} m/s, Rel RMSE={rel_rmse:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_well_logs(
    vp_true: np.ndarray,
    vp_est: np.ndarray,
    save_path: Optional[Path] = None,
    n_wells: int = 5,
) -> None:
    """Plot well logs (vertical velocity profiles) at multiple locations."""
    fig, axes = plt.subplots(1, n_wells, figsize=(15, 5), sharey=True)
    if n_wells == 1:
        axes = [axes]
    
    nx = vp_true.shape[1]
    well_indices = np.linspace(0, nx - 1, n_wells, dtype=int)
    
    for idx, ax in enumerate(axes):
        well_idx = well_indices[idx]
        z_axis = np.arange(vp_true.shape[0])
        
        ax.plot(vp_true[:, well_idx], z_axis, "b-", linewidth=2, label="True")
        ax.plot(vp_est[:, well_idx], z_axis, "r--", linewidth=2, label="Estimated")
        ax.set_xlabel("Velocity (m/s)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Depth (grid points)", fontsize=10)
        ax.set_title(f"Well at X={well_idx}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        if idx == n_wells - 1:
            ax.legend(loc="best")
    
    fig.suptitle("Well Log Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_loss_history(
    history: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Plot training loss history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history["loss_total"], linewidth=2, color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Data vs Physics
    ax = axes[0, 1]
    ax.semilogy(history["loss_data"], linewidth=2, label="Data", color="blue")
    ax.semilogy(history["loss_physics"], linewidth=2, label="Physics", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Data vs Physics Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IC and BC
    ax = axes[1, 0]
    ax.semilogy(history["loss_ic"], linewidth=2, label="IC", color="green")
    ax.semilogy(history["loss_bc"], linewidth=2, label="BC", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Initial & Boundary Conditions", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Regularization
    ax = axes[1, 1]
    ax.semilogy(history["loss_reg"], linewidth=2, color="purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Regularization Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("Training Loss History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_data_comparison(
    observed: np.ndarray,
    predicted: np.ndarray,
    shot_id: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """Plot observed vs predicted seismic data for a shot gather."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    obs_shot = observed[shot_id]
    pred_shot = predicted[shot_id]
    
    vmax = max(np.abs(obs_shot).max(), np.abs(pred_shot).max())
    
    # Observed
    ax = axes[0]
    im = ax.imshow(obs_shot.T, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Observed Data", fontweight="bold")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Receiver")
    plt.colorbar(im, ax=ax)
    
    # Predicted
    ax = axes[1]
    im = ax.imshow(pred_shot.T, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Predicted Data", fontweight="bold")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Receiver")
    plt.colorbar(im, ax=ax)
    
    # Difference
    ax = axes[2]
    diff = pred_shot - obs_shot
    im = ax.imshow(diff.T, cmap="RdBu_r", aspect="auto")
    ax.set_title("Residual", fontweight="bold")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Receiver")
    plt.colorbar(im, ax=ax)
    
    fig.suptitle(f"Shot Gather Comparison (Shot {shot_id})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_analysis(
    history: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Detailed convergence analysis with multiple metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Loss components
    ax = fig.add_subplot(gs[0, :2])
    ax.semilogy(history["loss_total"], linewidth=2.5, label="Total", color="black")
    ax.semilogy(history["loss_data"], linewidth=2, label="Data", alpha=0.7)
    ax.semilogy(history["loss_physics"], linewidth=2, label="Physics", alpha=0.7)
    ax.semilogy(history["loss_ic"], linewidth=2, label="IC", alpha=0.7)
    ax.semilogy(history["loss_bc"], linewidth=2, label="BC", alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("All Loss Components", fontweight="bold", fontsize=12)
    ax.legend(loc="best", ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Loss ratio
    ax = fig.add_subplot(gs[0, 2])
    data_loss = np.array(history["loss_data"])
    physics_loss = np.array(history["loss_physics"])
    ratio = data_loss / (physics_loss + 1e-8)
    ax.semilogy(ratio, linewidth=2, color="purple")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Data/Physics Ratio", fontsize=11)
    ax.set_title("Loss Balance", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Gradient norms
    ax = fig.add_subplot(gs[1, 0])
    if history["grad_norm_pinn"]:
        ax.semilogy(history["grad_norm_pinn"], linewidth=2, label="PINN", color="blue")
    if history["grad_norm_vp"]:
        ax.semilogy(history["grad_norm_vp"], linewidth=2, label="VelocityNet", color="red")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Gradient Norm", fontsize=11)
    ax.set_title("Gradient Norms", fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effective data weight
    ax = fig.add_subplot(gs[1, 1])
    if history["w_data_eff"]:
        ax.plot(history["w_data_eff"], linewidth=2, color="green")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Weight", fontsize=11)
        ax.set_title("Effective Data Weight", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Frequency continuation
    ax = fig.add_subplot(gs[1, 2])
    if history["fmax"]:
        fmax_vals = [f for f in history["fmax"] if not np.isnan(f)]
        if fmax_vals:
            ax.plot(fmax_vals, linewidth=2, color="orange")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Fmax (Hz)", fontsize=11)
            ax.set_title("Frequency Continuation", fontweight="bold", fontsize=12)
            ax.grid(True, alpha=0.3)
    
    # Smoothed loss
    ax = fig.add_subplot(gs[2, :])
    window = 50
    total_loss = np.array(history["loss_total"])
    if len(total_loss) > window:
        smoothed = np.convolve(total_loss, np.ones(window)/window, mode="valid")
        epochs = np.arange(len(smoothed)) + window
        ax.semilogy(epochs, smoothed, linewidth=2.5, color="darkblue", label="Smoothed (window=50)")
        ax.semilogy(total_loss, linewidth=0.5, alpha=0.3, color="lightblue", label="Raw")
    else:
        ax.semilogy(total_loss, linewidth=2, color="darkblue")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Total Loss (log scale)", fontsize=11)
    ax.set_title("Smoothed Loss Trajectory", fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("Training Convergence Analysis", fontsize=14, fontweight="bold")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
