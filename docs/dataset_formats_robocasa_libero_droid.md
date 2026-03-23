# RoboCasa / LIBERO / DROID 数据格式与归一化

本文档概括三条训练管线在 **存储格式**、**batch 中的 `obs` / `actions`**，以及 **normalization** 上的差异，便于对齐策略与调试数据。

相关深入文档：[rlds_dataloader.md](rlds_dataloader.md)、[action_normalization_flow.md](action_normalization_flow.md)。

---

## 总览


| 管线           | 底层格式                                | Dataset / 入口                                                | `train.data_format`（典型）                                                     |
| ------------ | ----------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| **RoboCasa** | HDF5（robomimic 风格）                  | `[RoboCasaDataset](../robomimic/utils/robocasa_dataset.py)` | `robomimic`，`[train_robocasa.py](../robomimic/scripts/train_robocasa.py)`   |
| **LIBERO**   | HDF5（LIBERO / robosuite 布局）         | `[LIBERODataset](../robomimic/utils/libero_dataset.py)`     | `robomimic`，`[train_libero.py](../robomimic/scripts/train_libero.py)`       |
| **DROID**    | **RLDS**（Octo）或 **HDF5**（robomimic） | `TorchRLDSDataset` + Octo / `SequenceDataset`               | `droid_rlds` 或 HDF5，`[train_droid.py](../robomimic/scripts/train_droid.py)` |


三条线最终都进入 robomimic：**字典 `obs` + 张量 `actions`**，具体键名由各 yaml 的 `observation.modalities` 与数据集实现决定。

---

## RoboCasa

### 存储格式

- 多 demo 的 HDF5：`data/{demo}/obs/...`、`actions` 等。
- 实现：`[robocasa_dataset.py](../robomimic/utils/robocasa_dataset.py)`。
- 默认配置：`[train_robocasa.yaml](../robomimic/scripts/train_configs/train_robocasa.yaml)`。

### `obs` dict（由 yaml 的 modality 决定）


| 类型  | 典型键                                                                           | 形状 / dtype             | 说明               |
| --- | ----------------------------------------------------------------------------- | ---------------------- | ---------------- |
| RGB | `robot0_agentview_left_image` 等                                               | `(T, H, W, C)`，`uint8` | 以配置为准            |
| 语言  | `lang_fixed/language_distilbert`                                              | `(768,)`，`float32`     | DistilBERT 预计算嵌入 |
| 低维  | `robot0_eef_pos` / `robot0_eef_quat` / `robot0_gripper_qpos` 或 `robot_states` | `(T, ·)`，`float32`     | v0.1 布局可能不同      |


### `actions`

- 一般为 **7 维**（位置 + 旋转 + 夹爪），序列 `**(T, 7)`**，与 `train.action_shapes` 一致。

### Normalization

- **动作**：`RoboCasaDataset(..., normalize_actions=True)` 在数据集内用全集 **min/max 线性映射到 `[-1, 1]`**（`[rescale_array](../robomimic/utils/dataset_utils.py)`）。
- **观测**：`[get_obs_normalization_stats()](../robomimic/utils/robocasa_dataset.py)` 返回 `**None`**；默认 `**hdf5_normalize_obs: false**`，`[train_robocasa.py](../robomimic/scripts/train_robocasa.py)` 中 `**obs_normalization_stats = None**`，即 **不做基于全数据集的 obs mean/std**。图像进入网络前由 **ObsUtils**（如转 float、按模态处理）负责。
- `**action_normalization_stats`**：仍通过 `trainset.get_action_normalization_stats()` 提供，供 `**postprocess_batch_for_training**` 等与 robomimic 训练流程对齐。

---

## LIBERO

### 存储格式

- 目录下多个 `.hdf5`：`data/{demo}/obs/agentview_rgb` 或 JPEG 等。
- 实现：`[libero_dataset.py](../robomimic/utils/libero_dataset.py)`。
- 默认配置：`[train_libero.yaml](../robomimic/scripts/train_configs/train_libero.yaml)`。

### `obs` dict


| 键                                     | 形状 / dtype             | 说明                                                    |
| ------------------------------------- | ---------------------- | ----------------------------------------------------- |
| `agentview_image`、`eye_in_hand_image` | `(T, H, W, C)`，`uint8` | HWC，与 `ObsUtils.get_processed_shape` 一致时需按模态处理为 CHW 等 |
| `robot_states`                        | `(T, 9)`，`float32`     | 常用低维 proprio                                          |


### `actions`

- **7 维**，`**(T, 7)`**。

### Normalization

- **动作**：`normalize_actions=True` 时，用 `**dataset_statistics.json`**（或自定义 `dataset_statistics_path`）中的 min/max **映射到 `[-1, 1]`**。
- **观测**：`get_obs_normalization_stats()` 为 `**None`**；默认 `**hdf5_normalize_obs: false**`，`[train_libero.py](../robomimic/scripts/train_libero.py)` 中 `**obs_normalization_stats = None**`。
- **形状元数据**：`[train_libero.py](../robomimic/scripts/train_libero.py)` 使用 `**ObsUtils.get_processed_shape(modality, v.shape[1:])`** 构建 `shape_meta`，以正确处理 **HWC** 图像。

### Debug 与 OOM（简述）

- `debug=true` 时可通过子目录、`max_demos`、统计文件写到 `/tmp` 等减轻内存压力；详见脚本内注释与环境变量 `LIBERO_DEBUG_DATA_DIR`、`LIBERO_DEBUG_MAX_DEMOS`。

---

## DROID

### A) RLDS（`droid_rlds`）

- 转换：`[robomimic_transform](../robomimic/utils/rlds_utils.py)`（图像 `/255` → `[0,1]` float32；低维与语言来自 RLDS）。
- `**obs`（逻辑键）**：
  - `camera/image/varied_camera_1_left_image`、`camera/image/varied_camera_2_left_image`：`float32`，约 `**[0, 1]`**。
  - `robot_state/cartesian_position`、`robot_state/gripper_position`。
  - `raw_language`、`pad_mask` 等。
- `**actions**`：由 RLDS 与 `[droid_dataset_transform](../robomimic/utils/rlds_utils.py)` 等拼接；维度由配置 `**train.action_shapes**` 决定。
- **Normalization**：
  - Octo `**make_dataset_from_rlds`** 侧配置 **bounds 类 action/proprio 归一化**（见 `[train_droid.py](../robomimic/scripts/train_droid.py)` 中 `BASE_DATASET_KWARGS`）。
  - Robomimic 侧通常 `**obs_normalization_stats = None`**；`**action_normalization_stats**` 由 `**action_stats_to_normalization_stats**` 等与训练循环衔接。

### B) HDF5（robomimic / `droid` 等非 RLDS）

- 使用标准 `**SequenceDataset**` 路径；`obs` / `actions` 由 HDF5 与 `all_obs_keys`、`action_keys` 决定。
- `**train.hdf5_normalize_obs: true**` 时：从 `**trainset.get_obs_normalization_stats()**` 得到 `**obs_normalization_stats**`。
- **动作**：`**trainset.get_action_normalization_stats()`**，与 `**action_config**`（如 min_max）一致。

---

## 对照小结


| 项目                           | RoboCasa | LIBERO  | DROID RLDS        | DROID HDF5                 |
| ---------------------------- | -------- | ------- | ----------------- | -------------------------- |
| 存储                           | HDF5     | HDF5    | RLDS              | HDF5                       |
| 图像常见 dtype                   | `uint8`  | `uint8` | `float32` [0,1]   | 视 HDF5                     |
| 动作维度                         | 7（典型）    | 7       | 由 `action_shapes` | 由配置                        |
| 动作 dataset 内 min→[-1,1]      | 是        | 是       | Octo bounds 管线    | SequenceDataset 行为         |
| `obs_normalization_stats` 默认 | `None`   | `None`  | `None`            | 可因 `hdf5_normalize_obs` 非空 |


---

## 相关源码索引


| 内容                  | 路径                                                                                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RLDS → robomimic 字段 | `[robomimic/utils/rlds_utils.py](../robomimic/utils/rlds_utils.py)`                                                                                                                                                                 |
| RoboCasa 样本构造       | `[robomimic/utils/robocasa_dataset.py](../robomimic/utils/robocasa_dataset.py)`                                                                                                                                                     |
| LIBERO 样本构造         | `[robomimic/utils/libero_dataset.py](../robomimic/utils/libero_dataset.py)`                                                                                                                                                         |
| 统计与 rescale         | `[robomimic/utils/dataset_utils.py](../robomimic/utils/dataset_utils.py)`                                                                                                                                                           |
| Hydra 默认            | `[train_robocasa.yaml](../robomimic/scripts/train_configs/train_robocasa.yaml)`、`[train_libero.yaml](../robomimic/scripts/train_configs/train_libero.yaml)`、`[train_rlds.yaml](../robomimic/scripts/train_configs/train_rlds.yaml)` |


