# Drift Policy（LIBERO / 高维动作）改进计划 — Hydra 可配置实验矩阵

本文档把高维机器人场景下的已知风险整理为**可实施改动**，并约定：**所有行为开关与超参均通过配置（Hydra YAML / CLI 覆盖）暴露**，便于扫实验选最优组合。

**相关代码**：[`drift_policy.py`](../robomimic/algo/drift_policy.py)、[`drift_utils.py`](../robomimic/algo/drift_utils.py)、[`drift_policy_config.py`](../robomimic/config/drift_policy_config.py)、[`train_libero_drift.yaml`](../robomimic/scripts/train_configs/train_libero_drift.yaml)、[`torch_utils.py`](../robomimic/utils/torch_utils.py)（LR scheduler）。

---

## 目标与约束

- **不破坏默认行为**：现有 `train_*_drift.yaml` 在未改字段时行为与当前实现一致。
- **Hydra 为第一界面**：新增项放在 `algo.drift.*`、`algo.optim_params.policy.learning_rate.*`。
- **分阶段落地**：P0（可观测 + 温度尺度） → P1（LR cosine scheduler） → P2/P3（结构改动，暂缓）。

---

## P0：Kernel 可观测性 + 高维温度尺度 ✅ 已实现

### 问题

固定 `temp` 下高维欧氏距离偏大 → Laplacian kernel 趋近 0（"死 kernel"）。即使 `use_adaptive_temp=true`，median 与 kernel 的量纲可能不一致。

### 实现内容

1. **`drift_utils.py`**
   - `compute_drift` 新增参数：`kernel_type`（`laplace`/`rbf`）、`dist_scale_mode`（`none`/`sqrt_dim`）、`rbf_sigma`。
   - `compute_adaptive_temp` 新增 `median_scale` 和 `min_temp` 参数，支持 `dist_scale_mode`。
   - 两个函数均返回 stats dict（`kernel_max`, `kernel_mean`, `dist_median`）供日志使用。

2. **`drift_policy_config.py`**
   - 在 `algo_config()` 中注册所有 `drift.*` 新键及默认值。

3. **`drift_policy.py`**
   - 从 `drift_cfg` 读取所有新字段。
   - `train_on_batch` 中按 `log_kernel_stats` / `log_kernel_every_n_steps` 写入 kernel 统计。
   - 支持 `assert_alive_kernel` 可选警告。

4. **`train_libero_drift.yaml`**
   - 新增 `drift.*` 配置项（默认值保持向后兼容）。

### Hydra 配置键

| 键 | 类型 | 默认 | 说明 |
|----|------|------|------|
| `algo.drift.kernel_type` | str | `laplace` | `laplace` / `rbf` |
| `algo.drift.dist_scale_mode` | str | `none` | `none` / `sqrt_dim`（高维主开关） |
| `algo.drift.adaptive_median_scale` | float | `0.3` | 自适应温度的 median 缩放系数 |
| `algo.drift.adaptive_min_temp` | float | `0.01` | 自适应温度的下界 |
| `algo.drift.rbf_sigma` | float | `1.0` | RBF kernel 的固定 sigma |
| `algo.drift.log_kernel_stats` | bool | `true` | 是否记录 kernel 统计 |
| `algo.drift.log_kernel_every_n_steps` | int | `100` | kernel 统计日志频率 |
| `algo.drift.assert_alive_kernel` | bool | `false` | 是否在 kernel 死亡时警告 |
| `algo.drift.alive_kernel_threshold` | float | `1e-6` | kernel 存活阈值 |

### 实验命令

```bash
# Baseline：观察 kernel 统计
LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift torchrun --nproc_per_node=4 \
  -m robomimic.scripts.train_libero algo.drift.log_kernel_stats=true

# P0-A：启用 sqrt_dim 距离缩放
... algo.drift.dist_scale_mode=sqrt_dim

# P0-B：切换 RBF kernel
... algo.drift.kernel_type=rbf algo.drift.rbf_sigma=0.5
```

---

## P1：学习率 Cosine 衰减 ✅ 已实现

### 实现内容

- **`torch_utils.py`**：`lr_scheduler_from_optim_params` 新增 `cosine` 类型，使用 `CosineAnnealingLR`。
- 配置键通过 `algo.optim_params.policy.learning_rate.*` 暴露。

### Hydra 配置键

| 键 | 类型 | 默认 | 说明 |
|----|------|------|------|
| `scheduler_type` | str | `multistep` | `multistep` / `linear` / `cosine` |
| `cosine_t_max` | int | `100` | cosine 周期（建议与 `num_epochs` 对齐） |
| `cosine_eta_min` | float | `1e-6` | cosine 最小 lr |

### 实验命令

```bash
... algo.optim_params.policy.learning_rate.scheduler_type=cosine \
    algo.optim_params.policy.learning_rate.cosine_t_max=1000 \
    algo.optim_params.policy.learning_rate.cosine_eta_min=1e-6 \
    algo.optim_params.policy.learning_rate.epoch_schedule=[1]
```

---

## P2：Drift 计算结构 — 降维 / 分块（暂缓）

- `algo.drift.pca_dim`：随机投影降维后再算 kernel。
- `algo.drift.chunk_mode`：按时间步分块 `compute_drift`。
- 优先级低于 P0；建议先跑 `sqrt_dim` + RBF 对比实验再决定。

---

## P3：DDP 全局 drift + `noise_samples` > 1（暂缓）

- `algo.drift.global_drift`：多卡 `all_gather` 后算 drift。
- 放宽 `noise_samples=1` 限制。
- 需要额外通信与显存开销分析。

---

## 推荐实验顺序

1. **Baseline**：`log_kernel_stats=true`，观察 `Drift_Kernel_Max`、`Drift_Temp` 曲线。
2. **P0-A**：`dist_scale_mode=sqrt_dim` + `kernel_type=laplace`。
3. **P0-B**：`kernel_type=rbf`，sweep `rbf_sigma`。
4. **P1**：`scheduler_type=cosine`。
5. **P2/P3**：根据 P0/P1 结果决定。

---

## 验收标准

- [x] 单卡 `debug=true` + `assert_alive_kernel=true` 不误报。
- [x] WandB/TensorBoard 可见 `Drift_Kernel_Max`、`Drift_Temp`、`Drift_Norm`。
- [x] 同一命令仅改 Hydra 即可切换 `sqrt_dim` / `rbf` / LR scheduler。
- [ ] DDP：`global_drift=true` 下数值与单卡全 batch 近似（P3）。
