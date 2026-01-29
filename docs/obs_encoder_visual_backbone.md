# 观察编码器与 Visual Encoder 配置

本文说明如何在 `train_rlds.yaml`（以及等效的 JSON/Config）中更换 **visual encoder（视觉骨干）**。  
观察编码逻辑对应上游 robomimic 的 `models/obs_core.py`、`models/obs_nets.py`；本仓库通过配置中的 `observation.encoder` 使用这些模块。

---

## 1. 配置位置

在 **`robomimic/scripts/train_configs/train_rlds.yaml`** 中，视觉编码由以下块控制：

```yaml
observation:
  encoder:
    rgb:
      core_class: VisualCore          # 固定为 VisualCore（多模态视觉包装）
      core_kwargs:
        feature_dimension: 512
        flatten: true
        backbone_class: ResNet50Conv   # ← 在这里切换 visual encoder
        backbone_kwargs:
          pretrained: true
          use_cam: false
          downsample: false
        pool_class: null
        pool_kwargs: null
```

- **`core_class`**：保持为 `VisualCore`，表示“视觉核心”封装（内部会根据 `backbone_class` 创建实际骨干）。
- **`backbone_class`**：实际使用的 **visual encoder 类名**，更换骨干时只改这一项（并视需要改 `backbone_kwargs` / `pool_class`）。

---

## 2. 可用的 Visual Encoder（backbone_class）

根据本仓库中的用法，以下骨干在配置中通过字符串引用（需上游 robomimic 已注册）：

| backbone_class   | 说明 | 常见用途 |
|------------------|------|----------|
| **ResNet18Conv** | ResNet-18 卷积骨干 | 轻量、默认示例（见 `exps/templates/*.json`、`train_bc_rnn.py`） |
| **ResNet50Conv** | ResNet-50 卷积骨干 | 当前 `train_rlds.yaml` 默认，更强表达能力 |
| **R3MConv**      | R3M 预训练视觉编码 | 预训练表示，可配合 `freeze: true` |
| **MVPConv**      | MVP 预训练视觉编码 | 预训练表示，可配合 `freeze: true` |

具体是否可用取决于安装的 robomimic 版本是否包含并注册了对应类（如 `robomimic.models.obs_core` 等）。

---

## 3. 在 train_rlds.yaml 中切换骨干

### 3.1 使用 ResNet18（更轻量）

```yaml
observation:
  encoder:
    rgb:
      core_class: VisualCore
      core_kwargs:
        feature_dimension: 64   # ResNet18 常用 64 维
        flatten: true
        backbone_class: ResNet18Conv
        backbone_kwargs:
          pretrained: false
          input_coord_conv: false
        pool_class: SpatialSoftmax   # 可选，与模板一致
        pool_kwargs:
          num_kp: 32
          learnable_temperature: false
          temperature: 1.0
          noise_std: 0.0
```

### 3.2 使用 ResNet50（当前默认）

即当前 `train_rlds.yaml` 的写法：`backbone_class: ResNet50Conv`，配合 `feature_dimension: 512`、`pool_class: null`。

### 3.3 使用 R3M 预训练骨干

参考 `examples/train_bc_rnn.py` 中的注释：

```yaml
observation:
  encoder:
    rgb:
      core_class: VisualCore
      core_kwargs:
        feature_dimension: 512   # 按 R3M 输出维度设置
        flatten: true
        backbone_class: R3MConv
        backbone_kwargs:
          r3m_model_class: resnet18   # 或 resnet34, resnet50
          freeze: true                # 是否冻结预训练权重
        pool_class: null              # 预训练模型通常不用池化
        pool_kwargs: null
```

### 3.4 使用 MVP 预训练骨干

同样参考 `examples/train_bc_rnn.py`：

```yaml
observation:
  encoder:
    rgb:
      core_class: VisualCore
      core_kwargs:
        feature_dimension: 512   # 按 MVP 输出维度设置
        flatten: true
        backbone_class: MVPConv
        backbone_kwargs:
          mvp_model_class: vitb-mae-egosoup   # 或 vits-mae-hoi, vits-mae-in, vits-sup-in, vitl-256-mae-egosoup
          freeze: true
        pool_class: null
        pool_kwargs: null
```

---

## 4. 重要参数说明

- **`core_class`**  
  保持 `VisualCore`，不要改成 backbone 类名；backbone 只填在 `backbone_class`。

- **`backbone_class`**  
  直接填类名字符串，例如：`ResNet18Conv`、`ResNet50Conv`、`R3MConv`、`MVPConv`。  
  必须与上游 robomimic 中注册的名称一致。

- **`backbone_kwargs`**  
  每个骨干有不同的参数，例如：
  - ResNet*：`pretrained`, `input_coord_conv` / `use_cam`, `downsample`
  - R3M：`r3m_model_class`, `freeze`
  - MVP：`mvp_model_class`, `freeze`

- **`pool_class`**  
  - 使用 ResNet 等可训练骨干时，常用 `SpatialSoftmax` 或 `SpatialMeanPool`，或 `null`。
  - 使用 R3M/MVP 等预训练模型时，通常设为 `null`。

- **`feature_dimension`**  
  视觉特征最终输出维度，需与骨干实际输出维度和后续策略输入一致（例如 Diffusion Policy 的 obs 编码维度）。

---

## 5. 与 config_gen 的关系

若使用 **`droid_runs_language_conditioned_rlds.py`** 生成配置，其内部也会设置：

- `observation.encoder.rgb.core_class = "VisualCore"`
- `observation.encoder.rgb.core_kwargs.backbone_class`（例如 `"ResNet50Conv"`）

生成的 JSON 中同样包含上述 `observation.encoder.rgb` 结构；要换 visual encoder，只需在生成后的 JSON 里修改 `backbone_class` 与对应的 `backbone_kwargs`，或先改 `train_rlds.yaml` 再通过 `load_from` 等方式使用。

---

## 6. 关于 obs_net 文档

本仓库中 **没有** `robomimic/models/obs_net.md` 文件；README 中提到的观察处理对应上游的：

- `robomimic/models/obs_nets.py` — 观察编码器/解码器
- `robomimic/models/obs_core.py` — VisualCore、骨干、池化等

如需更底层的 API 与类说明，需要查看已安装的 robomimic 包内 `models/obs_core.py`、`models/obs_nets.py` 的文档字符串或上游仓库文档。
