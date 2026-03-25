# Drift vs Diffusion Comparison Experiments

## Overview

Compare Drift Policy (1-NFE) and Diffusion Policy on 2 datasets with 3 seeds each.

| Parameter     | Values                                      |
|---------------|---------------------------------------------|
| Algorithms    | `diffusion`, `drift`                        |
| Datasets      | LIBERO-10, RoboCasa kitchen_pnp             |
| Seeds         | 1, 42, 123                                  |
| Total runs    | 2 × 2 × 3 = **12**                         |
| Epochs        | 5000                                        |
| Batch size    | 256                                         |
| GPUs          | 4 × A100-40GB per job                       |
| WandB project | `drift_vs_diffusion`                        |

### Drift Policy Configuration

Drift runs use the best known configuration:

| Parameter                | Value      |
|--------------------------|------------|
| `kernel_type`            | `rbf`      |
| `dist_scale_mode`        | `sqrt_dim` |
| `rbf_sigma`              | `0.5`      |
| `warmup_epochs`          | `200`      |
| `max_drift_start → end`  | `0.3 → 0.05` (over 2000 epochs) |

## SLURM Array Job

Script: `slurm/train_drift_vs_diffusion.slurm`

Each `SLURM_ARRAY_TASK_ID` (0–11) maps to one experiment:

```
ID = dataset_idx * 6 + algo_idx * 3 + seed_idx
```

| ID   | Dataset  | Algorithm  | Seed | Experiment Name                  |
|------|----------|------------|------|----------------------------------|
| 0    | libero   | diffusion  | 1    | libero10_diffusion_s1            |
| 1    | libero   | diffusion  | 42   | libero10_diffusion_s42           |
| 2    | libero   | diffusion  | 123  | libero10_diffusion_s123          |
| 3    | libero   | drift      | 1    | libero10_drift_s1                |
| 4    | libero   | drift      | 42   | libero10_drift_s42               |
| 5    | libero   | drift      | 123  | libero10_drift_s123              |
| 6    | robocasa | diffusion  | 1    | robocasa_pnp_diffusion_s1        |
| 7    | robocasa | diffusion  | 42   | robocasa_pnp_diffusion_s42       |
| 8    | robocasa | diffusion  | 123  | robocasa_pnp_diffusion_s123      |
| 9    | robocasa | drift      | 1    | robocasa_pnp_drift_s1            |
| 10   | robocasa | drift      | 42   | robocasa_pnp_drift_s42           |
| 11   | robocasa | drift      | 123  | robocasa_pnp_drift_s123          |

## Usage

### Submit all 12 runs (max 4 concurrent)

```bash
sbatch slurm/train_drift_vs_diffusion.slurm
```

### Submit LIBERO-10 only (6 runs)

```bash
sbatch --array=0-5%3 slurm/train_drift_vs_diffusion.slurm
```

### Submit RoboCasa only (6 runs)

```bash
sbatch --array=6-11%3 slurm/train_drift_vs_diffusion.slurm
```

### Submit only diffusion runs

```bash
# LIBERO diffusion (3 runs)
sbatch --array=0-2%3 slurm/train_drift_vs_diffusion.slurm

# RoboCasa diffusion (3 runs)
sbatch --array=6-8%3 slurm/train_drift_vs_diffusion.slurm
```

### Submit only drift runs

```bash
# LIBERO drift (3 runs)
sbatch --array=3-5%3 slurm/train_drift_vs_diffusion.slurm

# RoboCasa drift (3 runs)
sbatch --array=9-11%3 slurm/train_drift_vs_diffusion.slurm
```

### Submit a single experiment

```bash
# libero drift seed=42 (ID=4)
sbatch --array=4 slurm/train_drift_vs_diffusion.slurm
```

### Override via environment variables

```bash
sbatch --array=0 --export=DATASET=libero,ALGO=drift,SEED=42 \
    slurm/train_drift_vs_diffusion.slurm
```

### Override hyperparameters

```bash
sbatch --export=NUM_EPOCHS=1000,BATCH_SIZE=128 slurm/train_drift_vs_diffusion.slurm
```

## Monitoring

```bash
# Check status of all array tasks
squeue -u $USER -r

# Check a specific job array
squeue -j <JOB_ID>

# View logs for a specific task
cat logs/<JOB_ID>_<TASK_ID>.out

# Cancel all tasks
scancel <JOB_ID>

# Cancel a specific task
scancel <JOB_ID>_<TASK_ID>
```

## WandB

All runs log to project `drift_vs_diffusion`. Filter by experiment name prefix:

- `libero10_diffusion_*` — LIBERO diffusion across seeds
- `libero10_drift_*` — LIBERO drift across seeds
- `robocasa_pnp_diffusion_*` — RoboCasa diffusion across seeds
- `robocasa_pnp_drift_*` — RoboCasa drift across seeds

### Key Metrics to Compare

| Metric | Description |
|--------|-------------|
| `Train/Loss` | Training loss (MSE) |
| `Valid/Loss` | Validation loss |
| `evaluate/action_mse` | Action prediction MSE |
| `evaluate/rotation_error/mean` | Rotation prediction error |
| `Drift_Alignment_Mean` | Drift quality indicator (drift only) |
| `Drift_Kernel_Max` | Kernel health (drift only) |
| `Training_Phase` | 0 = warmup, 1 = drift (drift only) |

### Inference Speed

Drift policy uses 1-NFE (single denoising step) vs diffusion's 10 DDIM steps, yielding ~4-10x faster inference.
