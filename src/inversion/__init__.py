"""Training, loss functions, and scheduling for acoustic PINN-FWI."""

from .losses import (
    boundary_loss,
    data_mse_loss,
    initial_condition_loss,
    physics_mse_loss,
    smoothness_regularization,
    total_variation_regularization,
    velocity_bounds_penalty,
)
from .schedule import LossWeightScheduler
from .trainer import AcousticPINNFWITrainer

__all__ = [
    "AcousticPINNFWITrainer",
    "LossWeightScheduler",
    "physics_mse_loss",
    "data_mse_loss",
    "initial_condition_loss",
    "boundary_loss",
    "smoothness_regularization",
    "total_variation_regularization",
    "velocity_bounds_penalty",
]
