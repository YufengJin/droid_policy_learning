# 1D Diffusion Conditional UNet 网络架构

> 基于 Diffusion Policy 的 ConditionalUnet1D 实现，用于动作序列 denoising。

## 1. 总体结构

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  DiffusionPolicyUNet                      │
                    └─────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                                                               ▼
┌─────────────────────────────┐                               ┌─────────────────────────────┐
│   obs_encoder                │                               │   noise_pred_net            │
│   ObservationGroupEncoder    │                               │   ConditionalUnet1D         │
│   (vision + lang → 512d)     │                               └─────────────────────────────┘
└─────────────────────────────┘                                               ▲
              │                                                               │ global_cond
              │  obs_features [B, To, 512]                                    │ [B, To×512]
              │  → flatten → obs_cond [B, To×512] ────────────────────────────┘
              ▼
```

---

## 2. ConditionalUnet1D 核心模块

### 2.1 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | ac_dim (如 7) | 动作维度 |
| `global_cond_dim` | obs_dim × observation_horizon | 全局条件维度，通常 512 × 2 = 1024 |
| `diffusion_step_embed_dim` | 256 | 扩散步 t 的位置编码维度 |
| `down_dims` | [256, 512, 1024] | UNet 每层的通道数 |
| `kernel_size` | 5 | Conv1d 卷积核大小 |
| `n_groups` | 8 | GroupNorm 分组数 |

**条件维度：**
```
cond_dim = diffusion_step_embed_dim + global_cond_dim
         = 256 + (512 × To)
         = 1280  (当 To=2 时)
```

### 2.2 子模块一览

| 模块 | 作用 | 代码位置 |
|------|------|----------|
| `SinusoidalPosEmb` | 扩散步 t 的正弦位置编码 | diffusion_policy.py:563 |
| `DiffusionStepEncoder` | t → 256d embedding | diffusion_policy.py:687-692 |
| `Conv1dBlock` | Conv1d → GroupNorm → Mish | diffusion_policy.py:595-610 |
| `ConditionalResidualBlock1D` | 带 FiLM 调制的残差块 | diffusion_policy.py:613-660 |
| `Downsample1d` | Conv1d 下采样 (stride=2) | diffusion_policy.py:577-584 |
| `Upsample1d` | ConvTranspose1d 上采样 | diffusion_policy.py:586-591 |

---

## 3. FiLM 条件调制 (Feature-wise Linear Modulation)

> 参考：https://arxiv.org/abs/1709.07871

每个 `ConditionalResidualBlock1D` 接收条件向量 `cond [B, cond_dim]`，通过全连接层预测每通道的 scale 和 bias：

```python
# cond_encoder: cond [B, cond_dim] → [B, out_channels*2]
embed = self.cond_encoder(cond)
embed = embed.reshape(B, 2, out_channels, 1)
scale = embed[:, 0, :, :]   # [B, C, 1]
bias  = embed[:, 1, :, :]   # [B, C, 1]

# FiLM: out = scale * x + bias
out = self.blocks[0](x)
out = scale * out + bias
out = self.blocks[1](out) + residual_conv(x)
```

---

## 4. ConditionalUnet1D 前向流程

### 4.1 输入

- `sample`: 噪声动作 `(B, T, input_dim)`，内部转置为 `(B, C, T)` 以适配 Conv1d
- `timestep`: 扩散步 `(B,)` 或标量
- `global_cond`: 观测条件 `(B, global_cond_dim)`

### 4.2 条件拼接

```python
global_feature = diffusion_step_encoder(timesteps)   # [B, 256]
if global_cond is not None:
    global_feature = torch.cat([global_feature, global_cond], dim=-1)  # [B, 1280]
```

### 4.3 UNet 结构（以 down_dims=[256,512,1024] 为例）

```
all_dims = [7, 256, 512, 1024]
in_out   = [(7,256), (256,512), (512,1024)]
```

#### Down 路径

| Level | 输入通道 | 输出通道 | 操作 |
|-------|----------|----------|------|
| 0 | 7 | 256 | ResBlock → ResBlock → Downsample |
| 1 | 256 | 512 | ResBlock → ResBlock → Downsample |
| 2 | 512 | 1024 | ResBlock → ResBlock (无下采样) |

#### Mid 路径

| 模块 | 通道 | 说明 |
|------|------|------|
| ResBlock × 2 | 1024 | 保持分辨率 |

#### Up 路径

| Level | 输入 (concat) | 输出 | 操作 |
|-------|---------------|------|------|
| 0 | 1024 + 1024 (skip) | 512 | ResBlock → ResBlock → Upsample |
| 1 | 512 + 512 (skip) | 256 | ResBlock → ResBlock → Upsample |

#### Final

```
Conv1dBlock(256, 256) → Conv1d(256, 7) → (B, T, 7)
```

### 4.4 结构示意

```
输入: (B, 7, T)  noisy_actions
条件: global_feature (B, 1280)

                    noisy_actions (B,7,T)
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │ down_level_0             │                         │
    │  ResBlock(7→256)+cond    │                         │
    │  ResBlock(256→256)+cond  │                         │
    │  Downsample              │                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼ (B,256,T/2)
    ┌─────────────────────────┼─────────────────────────┐
    │ down_level_1             │ h[1]                    │
    │  ResBlock(256→512)+cond  │                         │
    │  ResBlock(512→512)+cond  │                         │
    │  Downsample              │                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼ (B,512,T/4)
    ┌─────────────────────────┼─────────────────────────┐
    │ down_level_2             │ h[0]                    │
    │  ResBlock(512→1024)+cond │                         │
    │  ResBlock(1024→1024)+cond│                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼ (B,1024,T/4)
    ┌─────────────────────────┼─────────────────────────┐
    │ mid_modules (×2)         │                         │
    │  ResBlock(1024→1024)+cond│                         │
    └─────────────────────────┼─────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │ up_level_0               │ concat h[0]            │
    │  ResBlock(2048→512)+cond │                         │
    │  ResBlock(512→512)+cond   │                         │
    │  Upsample                 │                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼
    ┌─────────────────────────┼─────────────────────────┐
    │ up_level_1               │ concat h[1]            │
    │  ResBlock(1024→256)+cond │                         │
    │  ResBlock(256→256)+cond   │                         │
    │  Upsample                 │                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼
    ┌─────────────────────────┼─────────────────────────┐
    │ final_conv               │                         │
    │  Conv1dBlock(256,256)    │                         │
    │  Conv1d(256, 7)          │                         │
    └─────────────────────────┼─────────────────────────┘
                              ▼
                    输出: (B, 7, T)  → moveaxis → (B, T, 7)
```

---

## 5. Global Condition 数据流

### 5.1 来源

- **obs_encoder**: ObservationGroupEncoder，将 vision（RGB）和 lang 等 modality 编码为 512 维
- **time_distributed**: 对每个 observation_horizon 时间步独立编码 → `(B, To, 512)`
- **flatten**: `obs_features.flatten(start_dim=1)` → `(B, To×512)` = `obs_cond`

### 5.2 计算式

```
global_cond_dim = obs_dim × observation_horizon
                = 512 × 2 = 1024  (默认 To=2)
```

### 5.3 代码对应

```python
# diffusion_policy.py:244-247
obs_features = TensorUtils.time_distributed(
    {"obs": inputs["obs"]}, 
    self.nets['policy']['obs_encoder'], 
    inputs_as_kwargs=True
)   # [B, To, 512]
obs_cond = obs_features.flatten(start_dim=1)  # [B, To×512]

# 传入 UNet
noise_pred = self.nets['policy']['noise_pred_net'](
    noisy_actions, timesteps, global_cond=obs_cond
)
```

---

## 6. 训练时条件使用

训练时会对 batch 做 `num_noise_samples` 次噪声采样，条件需复制以匹配：

```python
# diffusion_policy.py:265-266
obs_cond = obs_cond.repeat(num_noise_samples, 1)   # [B*8, 1024]
timesteps = timesteps.repeat(num_noise_samples)    # [B*8]
```

---

## 7. 相关配置 (train_robocasa.yaml)

```yaml
algo:
  horizon:
    observation_horizon: 2
    action_horizon: 8
    prediction_horizon: 16

  unet:
    diffusion_step_embed_dim: 256
    down_dims: [256, 512, 1024]
    kernel_size: 5
    n_groups: 8

  noise_samples: 8
```

---

## 8. 文件索引

| 模块 | 文件 | 行号 |
|------|------|------|
| ConditionalUnet1D | robomimic/algo/diffusion_policy.py | 663-801 |
| ConditionalResidualBlock1D | robomimic/algo/diffusion_policy.py | 613-660 |
| FiLM cond_encoder | robomimic/algo/diffusion_policy.py | 631-635 |
| obs_cond 构建 | robomimic/algo/diffusion_policy.py | 244-247 |
| UNet 实例化 | robomimic/algo/diffusion_policy.py | 89-91 |
