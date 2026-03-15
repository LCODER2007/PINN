# PINN-FWI Optimization Guide
## How to Achieve Perfect Ground Truth Reconstruction

This guide explains the key factors and strategies for making the estimated velocity model match the ground truth as closely as possible.

---

## Key Factors for Perfect Reconstruction

### 1. Data Quality and Coverage
- **Use all available shots and receivers**: More data = better constraints
- **High-quality observed data**: Ensure synthetic data is noise-free and accurate
- **Sufficient time sampling**: 1000 time steps provides good temporal resolution
- **Appropriate frequency content**: Start low (2 Hz), ramp up to peak (8 Hz)

### 2. Network Architecture

#### PINN (Wavefield Network)
- **Activation**: SIREN (sin) for smooth, continuous solutions
- **Depth**: 6-8 layers (balance between capacity and overfitting)
- **Width**: 128-256 neurons (sufficient for complex wavefields)
- **Hard constraints**: Enforce IC/BC by construction when possible

#### VelocityNet (Velocity Model)
- **Activation**: tanh (smooth, bounded)
- **Fourier features**: ENABLED (critical for spatial detail)
- **n_fourier**: 128-256 (higher = more spatial detail)
- **Output activation**: sigmoid (ensures bounds)
- **Depth trend initialization**: Start with smooth depth gradient

### 3. Training Strategy

#### Loss Weight Schedule
```
Phase 1 (Warmup, 0-300 epochs):
- w_pde: 0.2 → 4.0 (ramp up physics)
- w_data: 4.0 (constant, high)
- w_ic: 20.0 (constant, very high)

Phase 2 (Data fitting, 300-3000 epochs):
- w_pde: 4.0 (constant)
- w_data: 4.0 → 30.0 (ramp up)
- w_ic: 20.0 → 2.0 (anneal down)
```

#### Learning Rates
- **lr_pinn**: 3e-4 (higher for faster convergence)
- **lr_vp**: 3e-4 (matched to PINN)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=500, T_mult=2)

#### Frequency Continuation
- **Start**: 2 Hz (low frequency captures large-scale structure)
- **End**: 8 Hz (high frequency captures fine details)
- **Ramp**: 70% of total epochs (gradual transition)

### 4. Regularization Balance

```yaml
w_reg: 1.0e-4      # Total regularization weight
w_smooth: 5.0e-7   # Smoothness (very light)
w_tv: 5.0e-6       # Total variation (light)
w_charbonnier: 1.0e-5  # Edge-preserving (light)
w_bounds: 1.0e-4   # Velocity bounds (light)
w_well: 10.0       # Well constraint (strong if available)
```

**Key principle**: Regularization should guide, not dominate. Data fidelity is most important.

### 5. Sampling Strategy

```yaml
n_collocation: 30000   # Physics points (high for good PDE satisfaction)
n_ic: 3000             # Initial condition points
n_bc: 3000             # Boundary condition points
n_data_batch: 8192     # Data points per batch
shots_per_batch: 8     # Use all shots
n_data_receivers: 337  # Use all receivers
```

**Key principle**: Dense sampling of physics and data constraints.

### 6. Convergence Indicators

Monitor these metrics to assess convergence:

1. **Data loss**: Should decrease steadily to < 0.1
2. **PDE loss**: Should decrease to < 0.01
3. **Velocity RMSE**: Should decrease to < 100 m/s
4. **Relative RMSE**: Should decrease to < 5%

### 7. Common Issues and Solutions

#### Issue: High data loss, low PDE loss
**Solution**: Increase w_data, decrease w_pde

#### Issue: High PDE loss, low data loss
**Solution**: Increase w_pde, increase n_collocation

#### Issue: Smooth but inaccurate velocity
**Solution**: 
- Increase Fourier features (n_fourier)
- Decrease smoothness regularization
- Increase training epochs

#### Issue: Noisy velocity with artifacts
**Solution**:
- Increase smoothness regularization
- Use Charbonnier regularization
- Add well constraints

#### Issue: Slow convergence
**Solution**:
- Increase learning rates
- Use frequency continuation
- Increase batch sizes
- Use joint updates (alternating_updates: false)

#### Issue: Training instability
**Solution**:
- Decrease learning rates
- Increase grad_clip
- Enable skip_nonfinite_steps
- Increase warmup_epochs

---

## Recommended Workflow

### Step 1: Quick Test (fastdev.yaml)
- 500 epochs, reduced sampling
- Verify pipeline works
- Check for obvious issues
- Time: 30-60 minutes

### Step 2: Medium Run (production.yaml)
- 2000 epochs, full sampling
- Assess convergence quality
- Tune hyperparameters if needed
- Time: 2-4 hours

### Step 3: Optimized Run (optimized.yaml)
- 3000 epochs, optimized settings
- Best quality reconstruction
- Use for final results
- Time: 4-6 hours

### Step 4: Analysis
```bash
python analyze_results.py --results_dir results --save_plots
```

---

## Expected Results

### Good Convergence
- Final data loss: 0.05-0.15
- Final PDE loss: 0.005-0.015
- Velocity RMSE: 50-150 m/s
- Relative RMSE: 2-6%
- Visual match: 85-95%

### Excellent Convergence
- Final data loss: < 0.05
- Final PDE loss: < 0.005
- Velocity RMSE: < 50 m/s
- Relative RMSE: < 2%
- Visual match: > 95%

---

## Advanced Techniques

### 1. Multi-stage Training
Train in stages with increasing complexity:
1. Low frequency only (2-4 Hz)
2. Medium frequency (2-6 Hz)
3. Full frequency (2-8 Hz)

### 2. Curriculum Learning
Start with easier shots (simple geometry) and gradually add complex shots.

### 3. Adaptive Weighting
Let the algorithm automatically balance loss terms based on their magnitudes.

### 4. Ensemble Methods
Train multiple models with different initializations and average predictions.

---

## Configuration Files

- **fastdev.yaml**: Quick testing (500 epochs, ~30 min)
- **production.yaml**: Standard run (2000 epochs, ~3 hours)
- **optimized.yaml**: Best quality (3000 epochs, ~5 hours)

---

## Key Takeaways

1. **Data is king**: Use all available shots and receivers
2. **Balance is critical**: Physics + Data + Regularization must be balanced
3. **Patience pays off**: More epochs = better results
4. **Frequency continuation**: Essential for multi-scale inversion
5. **Monitor everything**: Watch all loss components, not just total loss
6. **Regularize lightly**: Let data guide the solution
7. **Use Fourier features**: Critical for spatial detail in VelocityNet

---

## Troubleshooting Checklist

- [ ] Observed data matches geometry (shots, receivers, time steps)
- [ ] Velocity bounds are reasonable (vp_min < true_vp < vp_max)
- [ ] Learning rates are not too high (check for NaN/Inf)
- [ ] Loss weights are balanced (no single term dominates)
- [ ] Sufficient training epochs (at least 2000 for good results)
- [ ] Frequency continuation is enabled
- [ ] All shots and receivers are being used
- [ ] Regularization is not too strong
- [ ] Network capacity is sufficient (hidden_width, hidden_layers)
- [ ] Fourier features are enabled for VelocityNet

---

## Contact and Support

For issues or questions:
1. Check ISSUES_FIXED.md for known problems
2. Review training logs for error messages
3. Visualize intermediate results (plot_every)
4. Compare with notebook examples

Good luck achieving perfect reconstruction! 🎯
