"""Velocity network for vp(x,z) inversion with positivity/range constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class VelocityNetConfig:
    in_dim: int = 2
    hidden_layers: int = 6
    hidden_width: int = 128
    activation: str = "tanh"
    fourier_features: bool = False
    n_fourier: int = 64
    fourier_scale: float = 6.0
    output_activation: str = "sigmoid"
    vp_min: float = 1500.0
    vp_max: float = 4700.0
    use_depth_trend_init: bool = False
    depth_trend_start: float = 0.10
    depth_trend_end: float = 0.70


class VelocityNet(nn.Module):
    """MLP mapping (x,z) -> vp constrained to [vp_min, vp_max]."""

    def __init__(self, config: VelocityNetConfig) -> None:
        super().__init__()
        self.config = config
        self.use_fourier = bool(config.fourier_features)
        feat_dim = int(config.in_dim)

        if self.use_fourier:
            if int(config.n_fourier) < 2 or int(config.n_fourier) % 2 != 0:
                raise ValueError("n_fourier must be an even integer >= 2 when fourier_features=True")
            half = int(config.n_fourier) // 2
            self.register_buffer("B", torch.randn((config.in_dim, half)) * float(config.fourier_scale))
            feat_dim = int(config.n_fourier)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(feat_dim, config.hidden_width))
        layers.append(_act(config.activation))
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(config.hidden_width, config.hidden_width))
            layers.append(_act(config.activation))
        layers.append(nn.Linear(config.hidden_width, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for m in linear_layers[:-1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if linear_layers:
            nn.init.zeros_(linear_layers[-1].weight)
            nn.init.zeros_(linear_layers[-1].bias)

    def _encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.use_fourier:
            return coords
        proj = 2.0 * math.pi * (coords @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def _map_to_velocity(self, raw: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        a = self.config.output_activation.lower()

        if a == "sigmoid":
            if self.config.use_depth_trend_init and z is not None:
                zc = z.clamp(0.0, 1.0)
                base = self.config.depth_trend_start + (
                    self.config.depth_trend_end - self.config.depth_trend_start
                ) * zc
                base = base.clamp(1e-4, 1.0 - 1e-4)
                y = torch.sigmoid(torch.logit(base) + raw)
            else:
                y = torch.sigmoid(raw)
        elif a == "softplus":
            y = torch.tanh(torch.nn.functional.softplus(raw))
        else:
            raise ValueError(f"Unsupported output_activation: {self.config.output_activation}")

        vp = self.config.vp_min + y * (self.config.vp_max - self.config.vp_min)
        return vp

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        coords = torch.cat([x, z], dim=-1)
        raw = self.mlp(self._encode_coords(coords))
        return self._map_to_velocity(raw, z=z)

    def forward_coords(self, coords_xz: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(self._encode_coords(coords_xz))
        return self._map_to_velocity(raw, z=coords_xz[..., 1:2])
