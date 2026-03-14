# Acoustic PINN-FWI for Marmousi (Notebook-First)

Research-grade, end-to-end **acoustic PINN full waveform inversion (FWI)** framework for Marmousi, designed for **VS Code Jupyter workflow**.

This project combines:

- PINN wavefield learning: \(u(x,z,t)\)
- Velocity inversion network: \(v_p(x,z)\)
- Acoustic physics residual via `torch.autograd`
- Synthetic observed gather generation (Deepwave default, FD fallback)
- Logging/checkpoint/figure outputs in legacy autoencoder-style reporting

---

## Acoustic Physics

Implemented PDE:

\[
u_{tt} = v(x,z)^2 (u_{xx} + u_{zz}) + s(x,z,t)
\]

Physics residual:

\[
r = u_{tt} - v^2 (u_{xx} + u_{zz}) - s
\]

Autograd derivatives used:

- \(u_t, u_{tt}, u_x, u_{xx}, u_z, u_{zz}\)

---

## Project Structure

```text
pinn_acoustic_fwi/
├── README.md
├── requirements.txt
├── configs/
│   ├── marmousi_acoustic.yaml
│   └── fastdev.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── results/
│   ├── figures/
│   ├── checkpoints/
│   └── logs/
├── notebooks/
│   ├── 01_load_marmousi.ipynb
│   ├── 02_acoustic_forward.ipynb
│   ├── 03_acoustic_pinn_forward.ipynb
│   └── 04_acoustic_pinn_fwi.ipynb
└── src/
    ├── data/
    ├── forward/
    ├── pinn/
    ├── inversion/
    └── utils/
```

---

## Installation

From `pinn_acoustic_fwi/`:

```bash
pip install -r requirements.txt
```

> Notes:
>
> - `deepwave` is the default forward backend.
> - If `deepwave` is unavailable, synthetic generation can fallback to simple FD backend.
> - Optional `.segy` loading requires `segyio`.

---

## Data Inputs

Default config points to your existing Marmousi source:

- `../data/processed/marmousi_vp.npy`

Generated files in this package:

- Geometry: `data/synthetic/acquisition_geometry.npz`
- Observed: `data/synthetic/observed_acoustic.npy`

Observed data shape convention (documented + enforced):

- **`[shot, time, receiver]`**

---

## Notebook-First Workflow

Run in order:

1. **01_load_marmousi.ipynb**
   - Load/visualize Marmousi
   - Print model stats

2. **02_acoustic_forward.ipynb**
   - Build acquisition geometry
   - Generate synthetic observed gathers
   - Plot shot gather

3. **03_acoustic_pinn_forward.ipynb**
   - Fixed `vp` forward sanity test
   - Train `AcousticPINN` only

4. **04_acoustic_pinn_fwi.ipynb**
   - Joint `AcousticPINN + VelocityNet` inversion
   - Save checkpoints/logs/figures
   - Produce final reporting figures

---

## Losses and Scheduling

Total objective:

\[
L = w_{pde}L_{physics} + w_{data}L_{data} + w_{ic}L_{ic} + w_{bc}L_{bc} + w_{reg}L_{reg}
\]

Included:

- Physics residual MSE
- Data MSE at receivers
- Initial condition penalty (`u=0`, `u_t=0` at `t=0`)
- Boundary damping penalty
- Regularization: smoothness + optional TV + bounds

Scheduler gradually shifts from stronger PDE emphasis to stronger data fitting.

---

## Output Style (Legacy Autoencoder Feel)

Training/inversion outputs mirror your previous style:

- Iteration/epoch progress prints
- Intermediate estimated model snapshots
- Final **True vs Estimated** side-by-side
- **Well-log comparison** (True vs Estimated)
- **Data/model/physics** loss panels

Saved to:

- `results/figures/`
- `results/checkpoints/`
- `results/logs/train_log.csv`

---

## Key Modules

- `src/data/marmousi_loader.py`: `.npy/.segy` loading, smoothing, stats
- `src/forward/acoustic_forward.py`: Deepwave + FD forward modeling
- `src/pinn/acoustic_pinn.py`: wavefield MLP PINN
- `src/pinn/velocity_net.py`: bounded velocity inversion network
- `src/pinn/physics_residual.py`: autograd PDE residual
- `src/inversion/trainer.py`: full training loop with logging/checkpointing/plots

---

## Reproducibility

- Seed control in config (`seed`)
- YAML-driven configuration
- Device auto selection (`cuda` if available, else `cpu`)

Use `fastdev.yaml` for quick checks and `marmousi_acoustic.yaml` for longer experiments.
