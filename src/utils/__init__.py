"""General utilities for I/O, checks, and visualization."""

from .checks import check_finite_array, check_observed_shape
from .io import (
    append_csv_row,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_yaml,
    resolve_path,
    save_checkpoint,
    seed_everything,
)
from .viz import (
    plot_gather,
    plot_losses,
    plot_training_snapshot,
    plot_true_vs_estimated,
    plot_velocity_model,
    plot_well_log_comparison,
)

__all__ = [
    "ensure_dir",
    "resolve_path",
    "seed_everything",
    "get_device",
    "load_yaml",
    "save_checkpoint",
    "load_checkpoint",
    "append_csv_row",
    "check_observed_shape",
    "check_finite_array",
    "plot_velocity_model",
    "plot_true_vs_estimated",
    "plot_well_log_comparison",
    "plot_losses",
    "plot_gather",
    "plot_training_snapshot",
]
