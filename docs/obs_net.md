# Observation Networks（obs_nets）说明

本文档整理 `robomimic/models/obs_nets.py` 及相关模块中的观测编码网络结构、数据流与用法。

---

## 1. 概述

- **用途**：对多模态观测（图像、low-dim、语言等）进行编码，得到统一的向量表示，供 policy / value / diffusion 等网络使用。
- **核心模块**：
  - **ObservationEncoder**：按 obs key 分别编码，再拼接为 `[B, D]`。
  - **ObservationGroupEncoder**：多 observation group（如 `obs`、`goal`）各自编码后 concat，再经固定 MLP 输出 `[B, 512]`。
  - **MIMO_MLP**：encoder（ObservationGroupEncoder）+ MLP + ObservationDecoder，多输入多输出。
- **配套工具**：`obs_encoder_factory`（根据 config 创建 ObservationEncoder）、`TensorUtils.time_distributed`（在 B、T 维上复用 encoder，得到 `[B, T, D]`）。

---

## 2. 模块与文件

| 模块 | 文件 | 说明 |
|------|------|------|
| `obs_nets` | `robomimic/models/obs_nets.py` | ObservationEncoder、ObservationGroupEncoder、MIMO_MLP、ObservationDecoder |
| `obs_core` | `robomimic/models/obs_core.py` | VisualCore、Randomizer（ColorRandomizer、CropRandomizer 等） |
| `base_nets` | `robomimic/models/base_nets.py` | ResNet*Conv、SpatialSoftmax、MLP 等 |
| `obs_utils` | `robomimic/utils/obs_utils.py` | OBS_ENCODER_CORES、DEFAULT_ENCODER_KWARGS、modality 映射 |
| `tensor_utils` | `robomimic/utils/tensor_utils.py` | `time_distributed`、`join_dimensions`、`reshape_dimensions` |

---

## 3. `obs_encoder_factory`

**位置**：`obs_nets.py` L30–127

根据 `obs_shapes` 与 `encoder_kwargs` 创建 **ObservationEncoder**。

- **输入**：
  - `obs_shapes`：`OrderedDict`，obs key → shape（如 `(3, 128, 128)`）。
  - `encoder_kwargs`：per-modality 配置（`core_class`、`core_kwargs`、`obs_randomizer_class`、`obs_randomizer_kwargs` 等）；含 `encoder_kwargs["rgb"]["fuser"]` 时用于 ObservationEncoder 的 `fuser`。
- **逻辑**：
  - 对每个 key，按 modality（rgb / low_dim / …）取配置，`register_obs_key` 注册。
  - RGB 的 `fuser` 可选 `"transformer"` / `"perceiver"`，用于多相机融合；为 `None` 则各 modality 编码后直接 concat。
  - `varied_camera` 系列 key 会 **复用** 第一个 varied_camera 的 net（`share_net_from`），不单独建网。
- **输出**：构造好的 `ObservationEncoder` 实例（已 `make()`）。

---

## 4. ObservationEncoder

**位置**：`obs_nets.py` L131–419

按 obs key 分别编码，再 `final_collation` 成单向量 `[B, D]`。

### 4.1 结构

- **per-key 流程**：  
  `obs_dict[k]` → optional **randomizer forward_in** → **obs_net**（如 VisualCore）→ optional **randomizer forward_out** → `feats[k]`。
- **obs_net**：由 `OBS_ENCODER_CORES[core_class]` 实例化（如 `VisualCore`），典型为 **backbone（ResNet*Conv）+ 可选 pool + 可选 linear**。
- **final_collation**：
  - **fuser is None**：对各 `feats[k]` flatten 后 `torch.cat(..., dim=-1)`；对 `raw` / `lang_fixed/language_raw` 有特殊占位逻辑。
  - **fuser in ("transformer","perceiver")**：图像类 key 走 Transformer / Perceiver 融合为 2048 维，再与 low-dim 等 concat。

### 4.2 输入 / 输出

- **输入**：`obs_dict`，key 与 `obs_shapes` 一致，每个 value 形如 `[B, ...]`（或 `[B, T, ...]` 若外层用 `time_distributed` 压成 `B*T`）。
- **输出**：`[B, D]`，`D = output_shape()[0]`，即各 modality 编码拼接后的总维度。

### 4.3 `output_shape`

见 `obs_nets.py` L379–400：根据各 key 的 shape、randomizer、obs_net 推导 `feat_shape`，累加 `np.prod(feat_shape)`；有 fuser 时图像部分固定 +2048。

---

## 5. ObservationGroupEncoder

**位置**：`obs_nets.py` L496–625

多 **observation group**（如 `obs`、`goal`）各自一个 ObservationEncoder，输出 concat 后再过 **可配置的** combine MLP，得到统一维度的向量。

### 5.1 结构

```
observation_group_shapes = { "obs": { key -> shape }, "goal": { ... }, ... }
        ↓
对每个 group：ObservationEncoder（obs_encoder_factory）→ [B, D_i]
        ↓
concat → [B, D_combo]
        ↓
combine MLP (可配置):
  Linear(D_combo, hidden_dims[0]) → ReLU
  → Linear(hidden_dims[0], hidden_dims[1]) → ReLU
  → ... → Linear(hidden_dims[-1], out_size)
        ↓
out [B, out_size]
```

- **combo_output_shape**：各 group 的 `output_shape()[0]` 之和。第一层 Linear 的输入维度自动适配，
  因此增减 rgb 图片数量时无需手动调整。
- **output_shape**：`[out_size]`（默认 512）。

### 5.2 combine MLP 配置

combine MLP 的维度可通过 `observation.encoder.combine`（YAML）配置：

```yaml
observation:
  encoder:
    combine:
      out_size: 512             # 输出维度 (默认 512)
      hidden_dims: [1024, 512]  # 隐藏层维度列表 (默认 [1024, 512])
```

不同图片数量下 `D_combo` 的参考值（`feature_dimension=512`, `low_dim=9`）：

| 图片数量 | D_combo | 默认 combine MLP |
|----------|---------|------------------|
| 1 张 | 521 | Linear(521,1024)→ReLU→Linear(1024,512)→ReLU→Linear(512,512) |
| 2 张 | 1033 | Linear(1033,1024)→ReLU→Linear(1024,512)→ReLU→Linear(512,512) |
| 3 张 | 1545 | Linear(1545,1024)→ReLU→Linear(1024,512)→ReLU→Linear(512,512) |

第一层 Linear 会根据 `D_combo` 自动适配，默认的 `hidden_dims` 适用于 1~3 张图片。
如需更强表达能力，可增大 `hidden_dims`，例如 `[2048, 1024]`。

### 5.3 forward

- **输入**：`**inputs`，例如 `obs=obs_dict, goal=goal_dict`，与 `observation_group_shapes` 的 key 对应。
- **输出**：`[B, out_size]`（默认 `[B, 512]`）。

### 5.4 在 Diffusion Policy 中的用法

Diffusion Policy 只使用 **单 group `"obs"`**：`observation_group_shapes["obs"] = obs_shapes`，因此仅一个 ObservationEncoder，再经 `combine` 得到 `out_size` 维。UNet 的 `global_cond_dim = out_size * observation_horizon` 会自动适配。

### 5.5 切换图片数量

只需修改 `observation.modalities.obs.rgb` 列表即可：

```yaml
# 1 张图片
rgb: [agentview_image]

# 2 张图片 (默认)
rgb: [agentview_image, robot0_eye_in_hand_image]

# 3 张图片
rgb: [agentview_image, robot0_agentview_right_image, robot0_eye_in_hand_image]
```

可用的图片 key（需在 `ROBOCASA_IMAGE_KEYS` 中定义）：
- `agentview_image` — 第三人称主相机（左）
- `robot0_agentview_right_image` — 第三人称右侧相机
- `robot0_eye_in_hand_image` — 手腕相机

---

## 6. ObservationDecoder

**位置**：`obs_nets.py` L422–494+

从共享的 flat 特征 `[B, D]` 为每个 output modality 接一个线性头，得到 `decode_shapes` 中定义的各模态 shape。MIMO_MLP 等用其生成多输出。

---

## 7. MIMO_MLP

**位置**：`obs_nets.py` L627+

- **encoder**：`ObservationGroupEncoder`，将多 group 的 obs 编码为 `[B, 512]`。
- **mlp**：`MLP(512 → layer_dims)`。
- **decoder**：`ObservationDecoder`，从 MLP 输出生成多模态输出。

BC、TD3-BC、CQL、IQL、GL 等算法的 **ActorNetwork** / **ValueNetwork** 多基于 `MIMO_MLP` 或类似结构，使用同一套 `obs_nets` 编码配置。

---

## 8. `time_distributed` 与 `[B, T, D]` 数据流

**位置**：`tensor_utils.time_distributed`（L932–963）

用于在 **batch 与时间** 维上复用同一 `op`（如 obs_encoder），而不显式写循环。

### 8.1 流程

1. 从 `inputs` 中取第一个 tensor 的 `shape[:2]` 作为 `(batch_size, seq_len)`，即 **B、T**。
2. **join_dimensions(inputs, 0, 1)**：把所有 tensor 的 `[B, T, ...]` 压成 `[B*T, ...]`。
3. **op(**inputs**, **kwargs)**：例如 `obs_encoder(obs=...)`，输入 `[B*T, ...]`，输出 `[B*T, D]`。
4. **reshape_dimensions(..., target_dims=(batch_size, seq_len))**：把第 0 维从 `B*T` 拆回 `(B, T)` → 输出 `[B, T, D]`。

### 8.2 与 obs_encoder 的组合

- `batch["obs"]` 各 key 形如 `[B, T, C, H, W]` 或 `[B, T, D_low]`，`T = observation_horizon`（如 2）。
- `time_distributed({"obs": batch["obs"]}, obs_encoder, inputs_as_kwargs=True)`：
  - 压成 `[B*T, ...]` 送进 `obs_encoder`；
  - `obs_encoder` 输出 `[B*T, 512]`；
  - 再拆成 `[B, T, 512]`。

因此 **`obs_features.ndim == 3`**，即 `[B, T, 512]`。  
若再 `flatten(start_dim=1)`，得 `[B, T*512]`，即 `global_cond_dim = obs_dim * observation_horizon`（如 512*2=1024），供 Diffusion Policy 的 UNet conditioning 使用。

---

## 9. Diffusion Policy 中的 obs_encoder

### 9.1 创建

```python
# diffusion_policy.py _create_networks
observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
obs_encoder = ObsNets.ObservationGroupEncoder(
    observation_group_shapes=observation_group_shapes,
    encoder_kwargs=encoder_kwargs,
)
obs_encoder = replace_bn_with_gn(obs_encoder)  # 为兼容 EMA
obs_dim = obs_encoder.output_shape()[0]  # 512
```

### 9.2 前向

```python
obs_features = TensorUtils.time_distributed(
    {"obs": batch["obs"]},
    self.nets["policy"]["obs_encoder"],
    inputs_as_kwargs=True,
)
# obs_features: [B, T, 512]
obs_cond = obs_features.flatten(start_dim=1)   # [B, T*512]
```

### 9.3 与 UNet 的衔接

`ConditionalUnet1D` 的 `global_cond_dim = obs_dim * observation_horizon`（如 512×2=1024），与 `obs_cond` 的维度一致。

---

## 10. 配置要点

### 10.1 `obs_shapes`

来自 `shape_meta` / config，如：

- `camera/image/varied_camera_1_left_image`: `(3, 128, 128)`
- `robot_state/cartesian_position`: `(6,)`
- `robot_state/gripper_position`: `(1,)`

### 10.2 `encoder_kwargs`

通常从 `observation.encoder` 解析（如 `obs_utils.obs_encoder_kwargs_from_config`），结构类似：

```yaml
encoder:
  low_dim:
    core_class: null
    core_kwargs: {}
    obs_randomizer_class: null
    obs_randomizer_kwargs: {}
  rgb:
    fuser: null  # 或 "transformer" / "perceiver"
    core_class: VisualCore
    core_kwargs:
      feature_dimension: 512
      flatten: true
      backbone_class: ResNet50Conv
      backbone_kwargs: { pretrained: true, ... }
    obs_randomizer_class: [ColorRandomizer, CropRandomizer]
    obs_randomizer_kwargs: [...]
```

### 10.3 多相机与 fuser

- **fuser is None**：每个图像 key 独立用 VisualCore 编码，再与 low-dim 等 concat。
- **fuser in ("transformer","perceiver")**：所有图像 key 的 feature 做序列融合后再与 low-dim concat；`ObservationEncoder` 内会建对应 fusion 子网。

---

## 11. 符号与索引速查

| 符号 | 含义 |
|------|------|
| **B** | batch size |
| **T** | 时间步数（如 `observation_horizon`，常为 2） |
| **D** | 单步 obs 编码维度；ObservationGroupEncoder 的 `out_size`（默认 **512**，可配置） |
| **obs_dim** | `obs_encoder.output_shape()[0]`，即 `out_size`（默认 512） |
| **global_cond_dim** | `obs_dim * observation_horizon`（如 512×2=1024） |
| **obs_features** | `[B, T, out_size]`（默认 `[B, T, 512]`） |
| **obs_cond** | `[B, T*512]`，UNet 的 global condition |

---

## 12. 相关代码索引

| 功能 | 文件 | 行号（约） |
|------|------|------------|
| `obs_encoder_factory` | `obs_nets.py` | 30–127 |
| `ObservationEncoder` | `obs_nets.py` | 131–419 |
| `final_collation`（fuser / no fuser） | `obs_nets.py` | 278–311 |
| `ObservationGroupEncoder` | `obs_nets.py` | 496–625 |
| `MIMO_MLP` | `obs_nets.py` | 627+ |
| `time_distributed` | `tensor_utils.py` | 932–963 |
| Diffusion Policy obs_encoder 创建与调用 | `diffusion_policy.py` | 70–86, 234–237 |
