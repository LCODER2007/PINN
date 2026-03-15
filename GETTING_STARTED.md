# Getting Started - 5 Minute Guide

## What You Have

A complete **acoustic PINN full waveform inversion system** ready to run.

---

## Step 1: Install (2 minutes)

```bash
cd PINN_acoustics_fwi
pip install -r requirements.txt
```

**That's it!** All dependencies are already listed.

---

## Step 2: Run (Choose One)

### Option A: Quick Test (30-60 min) ⚡
```bash
python run_full_pipeline.py --config fastdev.yaml --device cuda
```
✅ Tests everything works  
✅ Generates sample results  
✅ Fast feedback  

### Option B: Full Training (2-4 hours) 🎯
```bash
python run_full_pipeline.py --config production.yaml --device cuda
```
✅ Best accuracy  
✅ Comprehensive results  
✅ Production quality  

### Option C: Interactive (Variable) 📓
```bash
jupyter notebook notebooks/
# Run: 01_load_marmousi.ipynb → 04_acoustic_pinn_fwi.ipynb
```
✅ Step-by-step learning  
✅ Easy debugging  
✅ Interactive exploration  

---

## Step 3: Analyze Results (5 minutes)

```bash
python analyze_results.py --results_dir results --save_plots
```

**Output**: Plots in `results/figures/`
- Velocity comparison
- Loss history
- Well logs
- Convergence analysis

---

## Step 4: Interpret Results

### Check These Files
```
results/
├── figures/
│   ├── velocity_comparison_final.png    ← Main result
│   ├── loss_history_final.png           ← Training progress
│   ├── well_logs_final.png              ← Vertical profiles
│   └── convergence_analysis.png         ← Detailed analysis
├── checkpoints/
│   └── vp_est_final.npy                 ← Final velocity model
└── logs/
    └── train_log.csv                    ← Loss per epoch
```

### Key Metrics
- **MAE**: Mean absolute error (target: < 100 m/s)
- **RMSE**: Root mean square error (target: < 150 m/s)
- **Rel RMSE**: Relative error (target: < 5%)

---

## That's It! 🎉

You now have:
- ✅ Trained PINN model
- ✅ Estimated velocity model
- ✅ Convergence plots
- ✅ Error analysis

---

## Next: Customize (Optional)

### Edit Configuration
```bash
# Edit configs/production.yaml
nano configs/production.yaml
```

**Common tweaks**:
```yaml
# Faster training
training:
  n_epochs: 500          # Was 2000

# Better accuracy
training:
  n_epochs: 3000         # Was 2000

# Larger network
pinn:
  hidden_width: 512      # Was 256
```

### Run Again
```bash
python run_full_pipeline.py --config production.yaml
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
python run_full_pipeline.py --config fastdev.yaml --device cpu
```

### "Training is slow"
```bash
# Use fastdev config
python run_full_pipeline.py --config fastdev.yaml
```

### "Poor results"
```bash
# Use production config with more epochs
python run_full_pipeline.py --config production.yaml
```

---

## Learn More

| Document | Purpose | Time |
|----------|---------|------|
| `QUICKSTART.md` | Detailed guide | 10 min |
| `IMPROVEMENTS.md` | Technical details | 15 min |
| `ROADMAP.md` | Development plan | 10 min |
| `SYSTEM_SUMMARY.md` | Architecture overview | 10 min |
| Notebooks | Interactive learning | Variable |

---

## Command Reference

```bash
# Quick test
python run_full_pipeline.py --config fastdev.yaml --device cuda

# Full training
python run_full_pipeline.py --config production.yaml --device cuda

# Analyze results
python analyze_results.py --results_dir results --save_plots

# Interactive notebooks
jupyter notebook notebooks/

# CPU mode (slow but works)
python run_full_pipeline.py --config fastdev.yaml --device cpu
```

---

## What Happens When You Run

1. **Load Marmousi** - Velocity model from disk
2. **Generate Data** - Synthetic observed gathers
3. **Initialize Networks** - PINN + VelocityNet
4. **Train** - 200 warmup + 1800 joint epochs
5. **Save Results** - Models, plots, metrics
6. **Analyze** - Generate convergence plots

**Total time**: 30 min (fastdev) to 4 hours (production)

---

## Expected Output

### Console Output
```
Project root: /path/to/PINN_acoustics_fwi
Using device: cuda
============================================================
STEP 1: DATA SETUP
============================================================
Loading Marmousi from: ...
Velocity model shape: (176, 681)
Velocity stats: {...}
...
============================================================
STEP 3: TRAINING
============================================================
PINN-FWI: 100%|████████| 1000/1000 [00:45<00:00, 22.00it/s]
...
============================================================
STEP 4: RESULTS & VISUALIZATION
============================================================
Velocity Estimation Errors:
  MAE:      95.23 m/s
  RMSE:     142.15 m/s
  Rel RMSE: 0.0532 (5.32%)
...
```

### Generated Files
```
results/
├── checkpoints/
│   ├── pinn_final.pt
│   ├── velocity_net_final.pt
│   └── vp_est_final.npy
├── figures/
│   ├── velocity_comparison_final.png
│   ├── loss_history_final.png
│   ├── well_logs_final.png
│   └── convergence_analysis.png
└── logs/
    ├── train_log.csv
    └── metrics.jsonl
```

---

## Success Checklist

- [ ] Installed requirements
- [ ] Ran `run_full_pipeline.py`
- [ ] Checked results in `results/figures/`
- [ ] Ran `analyze_results.py`
- [ ] Reviewed convergence plots
- [ ] Understood velocity errors
- [ ] Read `QUICKSTART.md`

---

## Ready? Let's Go! 🚀

```bash
cd PINN_acoustics_fwi
python run_full_pipeline.py --config fastdev.yaml --device cuda
```

**Estimated time**: 30-60 minutes

Then:
```bash
python analyze_results.py --results_dir results --save_plots
```

**Estimated time**: 5 minutes

---

## Questions?

- **How do I run it?** → See "Step 2: Run" above
- **How long does it take?** → 30 min (fastdev) to 4 hours (production)
- **What's the accuracy?** → MAE < 100 m/s (typical)
- **Can I customize it?** → Yes, edit `configs/production.yaml`
- **How do I analyze results?** → Run `analyze_results.py`

---

## Next Steps

1. ✅ Run the pipeline
2. ✅ Analyze results
3. 📖 Read `QUICKSTART.md`
4. 🔧 Customize configuration
5. 📚 Read `IMPROVEMENTS.md`
6. 🗺️ Check `ROADMAP.md`

---

## System Overview

```
Input: Marmousi Velocity Model
  ↓
Forward Modeling: Generate Synthetic Data
  ↓
Neural Networks: PINN + VelocityNet
  ↓
Training: 2000 epochs with adaptive losses
  ↓
Output: Estimated Velocity Model
  ↓
Analysis: Convergence plots + error metrics
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_full_pipeline.py` | Main entry point |
| `analyze_results.py` | Results analysis |
| `configs/fastdev.yaml` | Quick testing |
| `configs/production.yaml` | Full training |
| `notebooks/` | Interactive learning |
| `results/` | Generated outputs |

---

## Performance

| Config | Time | Accuracy | Use Case |
|--------|------|----------|----------|
| fastdev | 30-60 min | Good | Testing |
| production | 2-4 hours | Best | Final results |
| CPU | 5-10x slower | Same | No GPU |

---

## That's All You Need to Know! 🎯

**Start here:**
```bash
python run_full_pipeline.py --config fastdev.yaml --device cuda
```

**Then analyze:**
```bash
python analyze_results.py --results_dir results --save_plots
```

**Then read:**
- `QUICKSTART.md` - Detailed guide
- `IMPROVEMENTS.md` - Technical details
- `ROADMAP.md` - What's next

Good luck! 🚀
