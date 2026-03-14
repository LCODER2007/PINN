"""Visualization helpers with output style similar to prior autoencoder project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _save_or_show(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_velocity_model(vp: np.ndarray, title: str = "Velocity Model", save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(vp, cmap="jet", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig.colorbar(im, ax=ax, label="Vp (m/s)")
    _save_or_show(fig, save_path)


def plot_true_vs_estimated(
    vp_true: np.ndarray,
    vp_est: np.ndarray,
    title: str = "True vs Estimated Velocity",
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    vmin = min(float(vp_true.min()), float(vp_est.min()))
    vmax = max(float(vp_true.max()), float(vp_est.max()))

    im0 = axes[0].imshow(vp_true, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("True Model")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    im1 = axes[1].imshow(vp_est, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Estimated Model")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    fig.suptitle(title)
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Vp (m/s)")
    _save_or_show(fig, save_path)


def plot_well_log_comparison(
    vp_true: np.ndarray,
    vp_est: np.ndarray,
    well_x_indices: Iterable[int],
    save_path: str | Path | None = None,
) -> None:
    wells = list(well_x_indices)
    fig, axes = plt.subplots(1, len(wells), figsize=(5 * len(wells), 5), sharey=True)
    if len(wells) == 1:
        axes = [axes]

    z = np.arange(vp_true.shape[0])
    for ax, wx in zip(axes, wells):
        ax.plot(vp_true[:, wx], z, "k-", lw=2, label="True")
        ax.plot(vp_est[:, wx], z, "r--", lw=2, label="Estimated")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Well x={wx}")
        ax.set_xlabel("Vp (m/s)")
        ax.set_ylabel("Depth index")
        ax.legend()

    fig.suptitle("Well-log Comparison: True vs Estimated")
    _save_or_show(fig, save_path)


def plot_losses(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, vals in history.items():
        if len(vals) > 0:
            ax.plot(vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses (Data / Model / Physics)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_or_show(fig, save_path)


def plot_gather(gather: np.ndarray, title: str = "Shot Gather", save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(gather, cmap="seismic", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Time sample")
    fig.colorbar(im, ax=ax)
    _save_or_show(fig, save_path)


def plot_training_snapshot(
    epoch: int,
    vp_est: np.ndarray,
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(vp_est, cmap="jet", aspect="auto")
    axes[0].set_title(f"Estimated Vp at epoch {epoch}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for key, vals in history.items():
        if len(vals) > 0:
            axes[1].plot(vals, label=key)
    axes[1].set_title("Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    _save_or_show(fig, save_path)
