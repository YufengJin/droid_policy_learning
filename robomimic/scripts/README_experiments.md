# Experiment Scripts 使用指南

本文档介绍如何执行本项目中的实验脚本。

---

## 1. Drift vs Diffusion 对比实验

**脚本**: `robomimic/scripts/run_drift_vs_diffusion.sh`

**实验矩阵**: 2 算法 (diffusion, drift) × 2 数据集 (LIBERO-10, RoboCasa PnP) × 3 seeds = 12 runs

**默认配置**: 5000 epochs, batch_size=64, 4-GPU DDP, WandB project `drift_vs_diffusion`

### 执行方式

```bash
# 全部 12 runs
bash robomimic/scripts/run_drift_vs_diffusion.sh

# 仅 LIBERO-10 (6 runs)
bash robomimic/scripts/run_drift_vs_diffusion.sh libero

# 仅 RoboCasa PnP (6 runs)
bash robomimic/scripts/run_drift_vs_diffusion.sh robocasa

# 按算法过滤 (3 runs)
bash robomimic/scripts/run_drift_vs_diffusion.sh libero drift
bash robomimic/scripts/run_drift_vs_diffusion.sh robocasa diffusion
```

### 日志

- 训练日志: `outputs/drift_vs_diffusion_logs/`
- WandB: `drift_vs_diffusion` project

### Drift 默认配置

对比实验中 drift 使用最佳已知配置:
- `kernel_type=rbf`, `dist_scale_mode=sqrt_dim`, `rbf_sigma=0.5`
- `warmup_epochs=200` (YAML 中的默认值)
- `max_drift_start=0.3 → max_drift_end=0.05` (curriculum, 2000 epochs)

如需修改，编辑脚本中的 `DRIFT_OVERRIDES` 数组。

---

## 2. Action Space 对比实验

**脚本**: `robomimic/scripts/run_action_space_experiments.sh`

**实验矩阵**: 3 action spaces (pos_euler, pos_rot6d, pos_axisangle) × 2 数据集 × 3 seeds = 18 runs

```bash
# 全部 18 runs
bash robomimic/scripts/run_action_space_experiments.sh

# 按数据集
bash robomimic/scripts/run_action_space_experiments.sh libero
bash robomimic/scripts/run_action_space_experiments.sh robocasa

# 按数据集 + action space
bash robomimic/scripts/run_action_space_experiments.sh libero pos_euler
```

---

## 3. Drift Policy 实验 (Kernel / LR 消融)

**脚本**: `robomimic/scripts/run_drift_experiments.sh`

**实验**: 3 个 drift 配置 (Exp 4/5/6)

```bash
# 全部 3 个实验
bash robomimic/scripts/run_drift_experiments.sh

# 指定实验
bash robomimic/scripts/run_drift_experiments.sh 4
bash robomimic/scripts/run_drift_experiments.sh 5 6
```

---

## 4. 单独训练命令

### LIBERO

```bash
# Diffusion Policy (默认)
torchrun --nproc_per_node=4 -m robomimic.scripts.train_libero

# Drift Policy
LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift \
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_libero

# Debug 模式 (快速验证，不需要 GPU)
python -m robomimic.scripts.train_libero debug=true
python -m robomimic.scripts.train_libero algo_name=drift_policy debug=true
```

### RoboCasa

```bash
# Diffusion Policy (默认)
torchrun --nproc_per_node=4 -m robomimic.scripts.train_robocasa

# Drift Policy
torchrun --nproc_per_node=4 -m robomimic.scripts.train_robocasa \
  config_name=train_robocasa_drift
```

### 常用 Hydra 覆盖

```bash
# 修改 seed / epochs / batch_size
... train.seed=42 train.num_epochs=1000 train.batch_size=32

# 修改 action space
... train.action_space=pos_rot6d

# 修改 WandB 项目名
... experiment.logging.wandb_proj_name=my_project experiment.name=my_exp
```

---

## 5. Drift Policy 改进配置

以下改进已集成到 drift policy 中，通过 Hydra 配置控制:

### Two-Phase Training (diffusion warmup)

```bash
# 前 200 epoch 用 diffusion loss 热身，之后切换 drift loss
... algo.drift.warmup_epochs=200

# 禁用 (纯 drift)
... algo.drift.warmup_epochs=0
```

### Multi-Scale Kernel

```bash
# 使用多带宽 RBF kernel
... algo.drift.kernel_type=multiscale_rbf \
    "algo.drift.multiscale_sigmas=[0.25, 0.5, 1.0, 2.0]"
```

### Curriculum Max-Drift

```bash
# 线性退火: 0.3 → 0.05 over 2000 epochs
... algo.drift.max_drift_start=0.3 \
    algo.drift.max_drift_end=0.05 \
    algo.drift.max_drift_anneal_epochs=2000
```

### Scheduled EMA Decay

```bash
# EMA decay 从 0.75 退火到 0.99
... algo.drift.ema_decay_start=0.75 \
    algo.drift.ema_decay_end=0.99 \
    algo.drift.ema_decay_anneal_epochs=3000
```

### Drift Alignment Monitoring

自动启用，随 kernel stats 一起记录。WandB 中查看:
- `Drift_Alignment_Mean`: V 与 (nearest_pos - gen) 的余弦相似度均值
- `Drift_Alignment_Std`: 标准差

---

## 6. 环境要求

- **GPU**: 建议 4× GPU (脚本默认 `NUM_GPUS=4`)
- **数据集路径**:
  - LIBERO: `/workspace/datasets/libero/libero_10`
  - RoboCasa: `/workspace/datasets/robocasa/v0.1/single_stage/kitchen_pnp`
- **WandB**: 确保 `wandb login` 已完成

如需修改 GPU 数量或数据路径，直接编辑脚本头部的变量。

---

## 7. Debug 模式快速验证

所有训练脚本支持 `debug=true`，会:
- 减少 epoch 数和 steps
- 关闭 WandB/TensorBoard
- 使用 `/tmp` 作为输出目录

```bash
# 验证 drift 改进是否正常工作
python -m robomimic.scripts.train_libero algo_name=drift_policy debug=true

# 验证 two-phase training
LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift \
  python -m robomimic.scripts.train_libero debug=true
```
