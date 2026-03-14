"""Acoustic PINN model: maps (x, z, t) -> scalar wavefield u using SIREN."""

from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")

class SineLayer(nn.Module):
    """
    A single SIREN layer: linear followed by sine activation with scaled weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 is_first: bool = False, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # First layer initialization: uniform(-1/in, 1/in)
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layer initialization: uniform(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


@dataclass
class AcousticPINNConfig:
    in_dim: int = 4
    n_shots: int = 8
    hidden_layers: int = 8
    hidden_width: int = 256
    activation: str = "sin"  # Set to "sin" to trigger SIREN behavior
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0
    fourier_features: bool = True
    n_fourier: int = 64
    fourier_scale: float = 1.0
    allow_fourier_with_siren: bool = False
    hard_constraint: str = "exp"
    hard_constraint_scale: float = 1.0
    hard_constraint_power: int = 2


class AcousticPINN(nn.Module):
    """SIREN-based MLP PINN approximating acoustic wavefield u(x,z,t)."""

    def __init__(self, config: AcousticPINNConfig) -> None:
        super().__init__()
        self.config = config
        use_siren = config.activation.lower() == "sin"

        self.use_fourier = bool(config.fourier_features)
        if use_siren and self.use_fourier and not config.allow_fourier_with_siren:
            warnings.warn(
                "Disabling Fourier features for SIREN by default. "
                "This combination can destabilize higher-order PINN derivatives. "
                "Set allow_fourier_with_siren=True to force-enable.",
                RuntimeWarning,
            )
            self.use_fourier = False
        
        # SIREN generally works better with raw coordinates, 
        # but we keep the logic here for config compatibility.
        input_dim = config.in_dim
        # --- NEW: Fourier Feature Mapping Logic ---
        if self.use_fourier:
            # We create a random Gaussian matrix B
            # Mapping: cos(2π * X @ B) and sin(2π * X @ B)
            # This turns input_dim into config.n_fourier
            self.register_buffer(
                'B', 
                torch.randn((input_dim, config.n_fourier // 2)) * config.fourier_scale
            )
            input_dim = config.n_fourier  # The first SineLayer now receives n_fourier features
        
        # Build the network
        layers: list[nn.Module] = []
        
        # Check if we are using Sine or standard activations
        if use_siren:
            # SIREN specific architecture
            layers.append(
                SineLayer(
                    input_dim,
                    config.hidden_width,
                    is_first=True,
                    omega_0=config.first_omega_0,
                )
            )
            for _ in range(config.hidden_layers - 1):
                layers.append(
                    SineLayer(
                        config.hidden_width,
                        config.hidden_width,
                        is_first=False,
                        omega_0=config.hidden_omega_0,
                    )
                )
            
            # Final Linear Layer
            self.net = nn.Sequential(*layers)
            self.final_linear = nn.Linear(config.hidden_width, 1)
            
            # Final layer initialization
            with torch.no_grad():
                self.final_linear.weight.uniform_(
                    -np.sqrt(6 / config.hidden_width) / config.hidden_omega_0,
                    np.sqrt(6 / config.hidden_width) / config.hidden_omega_0
                )
                nn.init.zeros_(self.final_linear.bias)
        else:
            # Fallback to standard MLP logic (Tanh/ReLU/GELU)
            # (Keeping your original logic for backward compatibility)
            act_fn = _get_activation(config.activation)
            layers.append(nn.Linear(input_dim, config.hidden_width))
            layers.append(act_fn)
            for _ in range(config.hidden_layers - 1):
                layers.append(nn.Linear(config.hidden_width, config.hidden_width))
                layers.append(act_fn)
            layers.append(nn.Linear(config.hidden_width, 1))
            self.net = nn.Sequential(*layers)
            self.final_linear = nn.Identity() # The net already has the final output

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, shot_id: torch.Tensor) -> torch.Tensor:
        # 1. Normalize shot_id so the network sees a value between 0 and 1
        n_shots = self.config.n_shots 
        s_norm = shot_id.float() / max(n_shots - 1, 1)
            
        # 2. Concatenate into a 4D vector: [batch, 4]
        coords = torch.cat([x, z, t, s_norm], dim=-1) 
        return self.forward_coords(coords)

    def forward_coords(self, coords: torch.Tensor) -> torch.Tensor:
        # --- NEW: Apply Fourier Mapping if enabled ---
        if self.use_fourier:
             # Map coords to higher freq space: [cos(2πXB), sin(2πXB)]
            proj = 2.0 * np.pi * (coords @ self.B)
            coords = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            
        # Pass the (now expanded) coordinates through the MLP
        features = self.net(coords)

        if isinstance(self.final_linear, nn.Linear):
            output = self.final_linear(features)
        else:
            output = features

        return self._apply_output_constraint(output, coords)

    def _apply_output_constraint(self, output: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        mode = self.config.hard_constraint.lower()
        if mode == "none":
            return output

        t = coords[..., 2:3]
        scale = max(float(self.config.hard_constraint_scale), 1e-8)
        if mode == "exp":
            envelope = 1.0 - torch.exp(-scale * t)
        elif mode == "tanh":
            envelope = torch.tanh(scale * t)
        else:
            raise ValueError(f"Unsupported hard_constraint: {self.config.hard_constraint}")

        power = max(int(self.config.hard_constraint_power), 1)
        if power > 1:
            envelope = envelope.pow(power)
        return envelope * output

    def enforces_u0_by_construction(self) -> bool:
        return self.config.hard_constraint.lower() != "none"

    def enforces_ut0_by_construction(self) -> bool:
        return self.enforces_u0_by_construction() and int(self.config.hard_constraint_power) >= 2