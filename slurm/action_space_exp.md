# Action Space Comparison Experiments

## Overview

Compare 3 action spaces on 2 datasets using Diffusion Policy with 3 seeds each.

| Parameter     | Values                                      |
|---------------|---------------------------------------------|
| Action spaces | `pos_euler`, `pos_rot6d`, `pos_axisangle`   |
| Datasets      | LIBERO-10, RoboCasa kitchen_pnp             |
| Seeds         | 1, 42, 123                                  |
| Total runs    | 3 × 2 × 3 = **18**                         |
| Epochs        | 5000                                        |
| Batch size    | 64                                          |
| GPUs          | 4 × A100-40GB per job                       |
| WandB project | `action_space_comparison`                   |

## SLURM Array Job

Script: `slurm/train_action_space_comparison.slurm`

Each `SLURM_ARRAY_TASK_ID` (0–17) maps to one experiment:

```
ID = dataset_idx * 9 + space_idx * 3 + seed_idx
```

| ID   | Dataset  | Action Space    | Seed | Experiment Name                  |
|------|----------|-----------------|------|----------------------------------|
| 0    | libero   | pos_euler       | 1    | libero10_pos_euler_s1            |
| 1    | libero   | pos_euler       | 42   | libero10_pos_euler_s42           |
| 2    | libero   | pos_euler       | 123  | libero10_pos_euler_s123          |
| 3    | libero   | pos_rot6d       | 1    | libero10_pos_rot6d_s1            |
| 4    | libero   | pos_rot6d       | 42   | libero10_pos_rot6d_s42           |
| 5    | libero   | pos_rot6d       | 123  | libero10_pos_rot6d_s123          |
| 6    | libero   | pos_axisangle   | 1    | libero10_pos_axisangle_s1        |
| 7    | libero   | pos_axisangle   | 42   | libero10_pos_axisangle_s42       |
| 8    | libero   | pos_axisangle   | 123  | libero10_pos_axisangle_s123      |
| 9    | robocasa | pos_euler       | 1    | robocasa_pnp_pos_euler_s1        |
| 10   | robocasa | pos_euler       | 42   | robocasa_pnp_pos_euler_s42       |
| 11   | robocasa | pos_euler       | 123  | robocasa_pnp_pos_euler_s123      |
| 12   | robocasa | pos_rot6d       | 1    | robocasa_pnp_pos_rot6d_s1        |
| 13   | robocasa | pos_rot6d       | 42   | robocasa_pnp_pos_rot6d_s42       |
| 14   | robocasa | pos_rot6d       | 123  | robocasa_pnp_pos_rot6d_s123      |
| 15   | robocasa | pos_axisangle   | 1    | robocasa_pnp_pos_axisangle_s1    |
| 16   | robocasa | pos_axisangle   | 42   | robocasa_pnp_pos_axisangle_s42   |
| 17   | robocasa | pos_axisangle   | 123  | robocasa_pnp_pos_axisangle_s123  |

## Usage

### Submit all 18 runs (max 6 concurrent)

```bash
sbatch slurm/train_action_space_comparison.slurm
```

### Submit LIBERO-10 only (9 runs)

```bash
sbatch --array=0-8%3 slurm/train_action_space_comparison.slurm
```

### Submit RoboCasa only (9 runs)

```bash
sbatch --array=9-17%3 slurm/train_action_space_comparison.slurm
```

### Submit a single experiment

```bash
# libero pos_rot6d seed=42 (ID=4)
sbatch --array=4 slurm/train_action_space_comparison.slurm
```

### Override via environment variables

```bash
sbatch --array=0 --export=DATASET=libero,ACTION_SPACE=pos_rot6d,SEED=42 \
    slurm/train_action_space_comparison.slurm
```

### Override hyperparameters

```bash
sbatch --export=NUM_EPOCHS=1000,BATCH_SIZE=128 slurm/train_action_space_comparison.slurm
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

All runs log to project `action_space_comparison`. Filter by experiment name prefix:
- `libero10_pos_euler_*` — LIBERO pos_euler across seeds
- `robocasa_pnp_pos_rot6d_*` — RoboCasa pos_rot6d across seeds
