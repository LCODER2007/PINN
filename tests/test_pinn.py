"""Unit tests for PINN architecture."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn.acoustic_pinn import AcousticPINN, AcousticPINNConfig


class TestAcousticPINN:
    """Test suite for AcousticPINN."""
    
    @pytest.fixture
    def pinn_config(self):
        """Create a test PINN config."""
        return AcousticPINNConfig(
            in_dim=4,
            n_shots=8,
            hidden_layers=4,
            hidden_width=64,
            activation="sin",
            first_omega_0=30.0,
            hidden_omega_0=30.0,
            fourier_features=False,
            hard_constraint="exp",
            hard_constraint_scale=1.0,
            hard_constraint_power=2,
        )
    
    @pytest.fixture
    def pinn(self, pinn_config):
        """Create a test PINN."""
        return AcousticPINN(pinn_config)
    
    def test_pinn_initialization(self, pinn):
        """Test PINN initializes correctly."""
        assert pinn is not None
        assert isinstance(pinn, torch.nn.Module)
        
        # Check parameter count
        n_params = sum(p.numel() for p in pinn.parameters())
        assert n_params > 0
    
    def test_pinn_forward_pass(self, pinn):
        """Test PINN forward pass."""
        batch_size = 32
        x = torch.randn(batch_size, 1)
        z = torch.randn(batch_size, 1)
        t = torch.randn(batch_size, 1)
        shot_id = torch.randint(0, 8, (batch_size, 1))
        
        output = pinn(x, z, t, shot_id)
        
        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()
    
    def test_pinn_output_bounds(self, pinn):
        """Test PINN output respects hard constraint."""
        batch_size = 100
        x = torch.randn(batch_size, 1)
        z = torch.randn(batch_size, 1)
        t = torch.zeros(batch_size, 1)  # t=0
        shot_id = torch.zeros(batch_size, 1, dtype=torch.long)
        
        output = pinn(x, z, t, shot_id)
        
        # At t=0, output should be close to 0 (hard constraint)
        assert torch.abs(output).max() < 0.1
    
    def test_pinn_differentiability(self, pinn):
        """Test PINN is differentiable."""
        x = torch.randn(10, 1, requires_grad=True)
        z = torch.randn(10, 1, requires_grad=True)
        t = torch.randn(10, 1, requires_grad=True)
        shot_id = torch.zeros(10, 1, dtype=torch.long)
        
        output = pinn(x, z, t, shot_id)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert z.grad is not None
        assert t.grad is not None
    
    def test_pinn_with_fourier_features(self, pinn_config):
        """Test PINN with Fourier features."""
        pinn_config.fourier_features = True
        pinn_config.n_fourier = 64
        pinn = AcousticPINN(pinn_config)
        
        batch_size = 32
        x = torch.randn(batch_size, 1)
        z = torch.randn(batch_size, 1)
        t = torch.randn(batch_size, 1)
        shot_id = torch.zeros(batch_size, 1, dtype=torch.long)
        
        output = pinn(x, z, t, shot_id)
        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()
    
    def test_pinn_shot_id_normalization(self, pinn):
        """Test shot_id is properly normalized."""
        batch_size = 32
        x = torch.randn(batch_size, 1)
        z = torch.randn(batch_size, 1)
        t = torch.randn(batch_size, 1)
        
        # Test different shot IDs
        for shot_id_val in [0, 1, 4, 7]:
            shot_id = torch.full((batch_size, 1), shot_id_val, dtype=torch.long)
            output = pinn(x, z, t, shot_id)
            assert torch.isfinite(output).all()
    
    def test_pinn_batch_consistency(self, pinn):
        """Test PINN produces consistent results for same input."""
        x = torch.randn(10, 1)
        z = torch.randn(10, 1)
        t = torch.randn(10, 1)
        shot_id = torch.zeros(10, 1, dtype=torch.long)
        
        pinn.eval()
        with torch.no_grad():
            output1 = pinn(x, z, t, shot_id)
            output2 = pinn(x, z, t, shot_id)
        
        assert torch.allclose(output1, output2)
    
    def test_pinn_device_compatibility(self, pinn):
        """Test PINN works on different devices."""
        batch_size = 10
        x = torch.randn(batch_size, 1)
        z = torch.randn(batch_size, 1)
        t = torch.randn(batch_size, 1)
        shot_id = torch.zeros(batch_size, 1, dtype=torch.long)
        
        # CPU
        pinn_cpu = pinn.cpu()
        output_cpu = pinn_cpu(x, z, t, shot_id)
        assert output_cpu.device.type == "cpu"
        
        # CUDA (if available)
        if torch.cuda.is_available():
            pinn_cuda = pinn.cuda()
            x_cuda = x.cuda()
            z_cuda = z.cuda()
            t_cuda = t.cuda()
            shot_id_cuda = shot_id.cuda()
            output_cuda = pinn_cuda(x_cuda, z_cuda, t_cuda, shot_id_cuda)
            assert output_cuda.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
