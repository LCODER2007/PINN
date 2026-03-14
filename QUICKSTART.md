# Quick Start Guide

## Installation

```bash
cd PINN_acoustics_fwi
pip install -r requirements.txt
```

## Run Full Pipeline (Recommended)

### Option 1: Production Run (Best Results)
```bash
python run_full_pipeline.py --config production.yaml --device cuda
```
- **Time**: ~2-4 hours (GPU)
- **Epochs**: 2000
- **Expected Accuracy**: MAE < 100 m/s

### Option 2: Fast Development
```bash
python run_full_pipeline.py --config fastdev.yaml --device cuda
```
- **Time**: ~30-60 minutes (GPU)
- **Epochs**: 1000
- **Good for**: Testing, debugging

### Option 3: CPU (Slow but Works)
```bash
python run_full_pipeline.py --config fastdev.yaml --device cpu
```

## What Happens

The pipeline automatically:

1. **Loads Marmousi velocity model**
   - Subsamples to manageable size
   - Prints model statistics

2. **Generates synthetic data**
   - Creates acquisition geometry
   - Runs forward modeling (Deepwave)
   - Saves observed gathers

3. **Initializes neural networks**
   - PINN: Wavefield approximator
   - VelocityNet: Velocity inversion

4. **Trains models**
   - 200 epochs: PINN warmup
   - 1800 epochs: Joint inversion
   - Saves checkpoints every 200 epochs

5. **Generates results**
   - Velocity comparison plot
   - Loss history
   - Well logs
   - Final velocity estimate

## Output Files

```
results/
├── checkpoints/
│   ├── pinn_final.pt              # Final PINN model
│   ├── velocity_net_final.pt      # Final VelocityNet
│   └── vp_est_final.npy           # Final velocity estimate
├── figures/
│   ├── velocity_comparison_final.png
│   ├── loss_history_final.png
│   ├── well_logs_final.png
│   └── snapshot_epoch_*.png       # Training snapshots
└── logs/
    ├── train_log.csv              # Loss per epoch
    └── metrics.jsonl              # Detailed metrics
```

## Interpret Results

### Velocity Comparison Plot
- **Left**: True velocity model
- **Middle**: Estimated velocity
- **Right**: Estimation error
- **Good**: Error < 100 m/s (MAE)

### Loss History
- **Top-left**: Total loss (should decrease)
- **Top-right**: Data vs Physics (should balance)
- **Bottom-left**: IC/BC losses
- **Bottom-right**: Regularization

### Well Logs
- **Blue line**: True velocity
- **Red dashed**: Estimated velocity
- **Multiple wells**: Different X locations

## Customize Configuration

Edit `configs/production.yaml`:

```yaml
# More shots = better coverage (slower)
acquisition:
  n_shots: 16  # Try 8, 16, 32

# Larger networks = better fit (slower)
pinn:
  hidden_width: 256  # Try 128, 256, 512

# More training = better convergence (slower)
training:
  n_epochs: 2000  # Try 1000, 2000, 5000

# Adjust loss weights
weights:
  w_pde: 15.0    # Physics emphasis
  w_data: 80.0   # Data emphasis
```

Then run:
```bash
python run_full_pipeline.py --config production.yaml
```

## Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU instead
python run_full_pipeline.py --config fastdev.yaml --device cpu

# Or reduce batch sizes in config
```

### "Deepwave not found"
```bash
# Falls back to FD automatically, or install:
pip install deepwave
```

### "Slow convergence"
```bash
# Increase collocation points in config
training:
  n_collocation: 100000  # Was 50000
```

### "Poor velocity estimate"
```bash
# Use production config with more epochs
python run_full_pipeline.py --config production.yaml
```

## Next Steps

1. **Analyze results**: Check `results/figures/` for plots
2. **Load final model**: 
   ```python
   import numpy as np
   vp_est = np.load("results/checkpoints/vp_est_final.npy")
   ```
3. **Modify config**: Tune hyperparameters in `configs/production.yaml`
4. **Run again**: Re-run pipeline with new config
5. **Compare**: Check if results improve

## Performance Tips

| Goal | Action |
|------|--------|
| Faster training | Use `fastdev.yaml`, reduce `n_collocation` |
| Better accuracy | Use `production.yaml`, increase `n_epochs` |
| Lower memory | Reduce `n_data_batch`, use CPU |
| Faster convergence | Increase `lr_pinn`, `lr_vp` |
| Stable training | Decrease `lr_pinn`, `lr_vp` |

## Expected Results

After 2000 epochs (production):
- **MAE**: 50-150 m/s
- **RMSE**: 100-200 m/s
- **Rel RMSE**: 2-5%
- **Training time**: 2-4 hours (GPU)

## Support

- Check `IMPROVEMENTS.md` for detailed enhancements
- See `README.md` for full documentation
- Review code comments for implementation details

---

**Ready to run?**
```bash
python run_full_pipeline.py --config production.yaml --device cuda
```
