#!/usr/bin/env python3
"""
Comprehensive analysis of acoustic PINN-FWI results.
Usage: python analyze_results.py [--results_dir results]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_results(results_dir: Path) -> Dict:
    """Load all results from results directory."""
    results = {
        "vp_true": None,
        "vp_est": None,
        "history": {},
        "metrics": [],
        "checkpoints": [],
    }
    
    # Load velocity models
    vp_est_path = results_dir / "checkpoints" / "vp_est_final.npy"
    if vp_est_path.exists():
        results["vp_est"] = np.load(vp_est_path)
    
    # Load training history
    csv_path = results_dir / "logs" / "train_log.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for col in df.columns:
            results["history"][col] = df[col].values.tolist()
    
    # Load metrics
    metrics_path = results_dir / "logs" / "metrics.jsonl"
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                results["metrics"].append(json.loads(line))
    
    # List checkpoints
    ckpt_dir = results_dir / "checkpoints"
    if ckpt_dir.exists():
        results["checkpoints"] = sorted([f.name for f in ckpt_dir.glob("*.pt")])
    
    return results


def compute_errors(vp_true: np.ndarray, vp_est: np.ndarray) -> Dict[str, float]:
    """Compute velocity estimation errors."""
    diff = vp_est - vp_true
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    rel_rmse = rmse / vp_true.mean()
    max_error = np.abs(diff).max()
    min_error = np.abs(diff).min()
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rel_rmse": float(rel_rmse),
        "max_error": float(max_error),
        "min_error": float(min_error),
        "mean_vp_true": float(vp_true.mean()),
        "mean_vp_est": float(vp_est.mean()),
        "std_vp_true": float(vp_true.std()),
        "std_vp_est": float(vp_est.std()),
    }


def print_summary(results: Dict, vp_true: np.ndarray = None):
    """Print comprehensive results summary."""
    print("\n" + "="*70)
    print("ACOUSTIC PINN-FWI RESULTS ANALYSIS")
    print("="*70)
    
    # Velocity estimation errors
    if results["vp_est"] is not None and vp_true is not None:
        errors = compute_errors(vp_true, results["vp_est"])
        print("\n[VELOCITY ESTIMATION ERRORS]")
        print(f"  MAE:           {errors['mae']:.2f} m/s")
        print(f"  RMSE:          {errors['rmse']:.2f} m/s")
        print(f"  Rel RMSE:      {errors['rel_rmse']:.4f} ({errors['rel_rmse']*100:.2f}%)")
        print(f"  Max Error:     {errors['max_error']:.2f} m/s")
        print(f"  Min Error:     {errors['min_error']:.2f} m/s")
        print(f"\n  True Vp:       mean={errors['mean_vp_true']:.0f}, std={errors['std_vp_true']:.0f} m/s")
        print(f"  Est Vp:        mean={errors['mean_vp_est']:.0f}, std={errors['std_vp_est']:.0f} m/s")
    
    # Training history
    if results["history"]:
        hist = results["history"]
        print("\n[TRAINING HISTORY]")
        if "loss_total" in hist:
            losses = hist["loss_total"]
            print(f"  Total Loss:    initial={losses[0]:.4e}, final={losses[-1]:.4e}")
            print(f"                 reduction={losses[0]/losses[-1]:.2f}x")
        if "loss_data" in hist:
            print(f"  Data Loss:     initial={hist['loss_data'][0]:.4e}, final={hist['loss_data'][-1]:.4e}")
        if "loss_physics" in hist:
            print(f"  Physics Loss:  initial={hist['loss_physics'][0]:.4e}, final={hist['loss_physics'][-1]:.4e}")
        if "loss_reg" in hist:
            print(f"  Reg Loss:      initial={hist['loss_reg'][0]:.4e}, final={hist['loss_reg'][-1]:.4e}")
    
    # Checkpoints
    if results["checkpoints"]:
        print(f"\n[CHECKPOINTS] ({len(results['checkpoints'])} files)")
        for ckpt in results["checkpoints"][-5:]:  # Show last 5
            print(f"  - {ckpt}")
    
    print("\n" + "="*70)


def plot_convergence_analysis(results: Dict, save_path: Path = None):
    """Create comprehensive convergence analysis plot."""
    if not results["history"]:
        print("No history data available")
        return
    
    hist = results["history"]
    epochs = np.arange(len(hist.get("loss_total", [])))
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Total loss
    ax = fig.add_subplot(gs[0, 0])
    if "loss_total" in hist:
        ax.semilogy(epochs, hist["loss_total"], "b-", linewidth=1.5)
        ax.set_title("Total Loss", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    # 2. Data vs Physics
    ax = fig.add_subplot(gs[0, 1])
    if "loss_data" in hist and "loss_physics" in hist:
        ax.semilogy(epochs, hist["loss_data"], "r-", label="Data", linewidth=1.5)
        ax.semilogy(epochs, hist["loss_physics"], "b-", label="Physics", linewidth=1.5)
        ax.set_title("Data vs Physics Loss", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Loss ratio
    ax = fig.add_subplot(gs[0, 2])
    if "loss_data" in hist and "loss_physics" in hist:
        ratio = np.array(hist["loss_data"]) / (np.array(hist["loss_physics"]) + 1e-8)
        ax.semilogy(epochs, ratio, "g-", linewidth=1.5)
        ax.set_title("Data/Physics Ratio", fontweight="bold")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)
    
    # 4. IC loss
    ax = fig.add_subplot(gs[1, 0])
    if "loss_ic" in hist:
        ax.semilogy(epochs, hist["loss_ic"], "m-", linewidth=1.5)
        ax.set_title("Initial Condition Loss", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    # 5. BC loss
    ax = fig.add_subplot(gs[1, 1])
    if "loss_bc" in hist:
        ax.semilogy(epochs, hist["loss_bc"], "c-", linewidth=1.5)
        ax.set_title("Boundary Condition Loss", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    # 6. Regularization loss
    ax = fig.add_subplot(gs[1, 2])
    if "loss_reg" in hist:
        ax.semilogy(epochs, hist["loss_reg"], "orange", linewidth=1.5)
        ax.set_title("Regularization Loss", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    # 7. Gradient norms
    ax = fig.add_subplot(gs[2, 0])
    if "grad_norm_pinn" in hist:
        ax.semilogy(epochs, hist["grad_norm_pinn"], "b-", label="PINN", linewidth=1.5)
        if "grad_norm_vp" in hist:
            ax.semilogy(epochs, hist["grad_norm_vp"], "r-", label="VelocityNet", linewidth=1.5)
        ax.set_title("Gradient Norms", fontweight="bold")
        ax.set_ylabel("Norm")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Frequency continuation
    ax = fig.add_subplot(gs[2, 1])
    if "fmax" in hist:
        fmax = np.array(hist["fmax"])
        valid = fmax[~np.isnan(fmax)]
        if len(valid) > 0:
            ax.plot(epochs[~np.isnan(fmax)], valid, "g-", linewidth=1.5)
            ax.set_title("Frequency Continuation", fontweight="bold")
            ax.set_ylabel("Fmax (Hz)")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    
    # 9. Effective data weight
    ax = fig.add_subplot(gs[2, 2])
    if "w_data_eff" in hist:
        ax.plot(epochs, hist["w_data_eff"], "purple", linewidth=1.5)
        ax.set_title("Effective Data Weight", fontweight="bold")
        ax.set_ylabel("Weight")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Convergence Analysis", fontsize=16, fontweight="bold", y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved convergence analysis to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(vp_true: np.ndarray, vp_est: np.ndarray, save_path: Path = None):
    """Plot error distribution and statistics."""
    diff = vp_est - vp_true
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(diff.flatten(), bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(diff.mean(), color="r", linestyle="--", linewidth=2, label=f"Mean: {diff.mean():.1f}")
    ax.set_xlabel("Error (m/s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Absolute error histogram
    ax = axes[0, 1]
    abs_diff = np.abs(diff)
    ax.hist(abs_diff.flatten(), bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax.axvline(abs_diff.mean(), color="r", linestyle="--", linewidth=2, label=f"MAE: {abs_diff.mean():.1f}")
    ax.set_xlabel("Absolute Error (m/s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Absolute Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error map
    ax = axes[1, 0]
    im = ax.imshow(diff, cmap="RdBu_r", aspect="auto")
    ax.set_title("Error Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.colorbar(im, ax=ax, label="Error (m/s)")
    
    # 4. Relative error
    ax = axes[1, 1]
    rel_error = np.abs(diff) / (np.abs(vp_true) + 1e-8) * 100
    im = ax.imshow(rel_error, cmap="viridis", aspect="auto")
    ax.set_title("Relative Error (%)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.colorbar(im, ax=ax, label="Rel Error (%)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved error distribution to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze acoustic PINN-FWI results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--vp_true",
        type=str,
        default=None,
        help="Path to true velocity model (optional)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots to results/figures/",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load results
    results = load_results(results_dir)
    
    # Load true velocity if provided
    vp_true = None
    if args.vp_true:
        vp_true = np.load(args.vp_true)
    
    # Print summary
    print_summary(results, vp_true)
    
    # Generate plots
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_plots:
        plot_convergence_analysis(
            results,
            save_path=fig_dir / "convergence_analysis.png"
        )
        if vp_true is not None and results["vp_est"] is not None:
            plot_error_distribution(
                vp_true,
                results["vp_est"],
                save_path=fig_dir / "error_distribution.png"
            )
    else:
        plot_convergence_analysis(results)
        if vp_true is not None and results["vp_est"] is not None:
            plot_error_distribution(vp_true, results["vp_est"])


if __name__ == "__main__":
    main()
