# RLDS 数据加载器完整分析

本文档详细分析 RLDS (Reinforcement Learning Datasets) 数据加载流程，包括 Octo 数据集函数、Robomimic 工具类以及动作归一化处理。

## 目录

1. [概述](#概述)
2. [Octo 数据集函数](#octo-数据集函数)
   - [代码片段分析 (train.py:111-117)](#代码片段分析-trainpy111-117)
   - [make_dataset_from_rlds 函数](#1-make_dataset_from_rlds-函数)
   - [combine_dataset_statistics 函数](#2-combine_dataset_statistics-函数)
   - [make_interleaved_dataset 函数](#3-make_interleaved_dataset-函数)
3. [动作归一化分析](#动作归一化分析)
   - [动作归一化流程](#动作归一化流程)
   - [action_stats_to_normalization_stats 函数](#action_stats_to_normalization_stats-函数)
   - [归一化方法详解](#归一化方法详解)
4. [Robomimic 工具类](#robomimic-工具类)
   - [TrainUtils - 训练工具类](#trainutils---训练工具类)
   - [ActionUtils - 动作工具类](#actionutils---动作工具类)
   - [FileUtils - 文件工具类](#fileutils---文件工具类)
5. [完整工作流程](#完整工作流程)
6. [源代码位置](#源代码位置)

---

## 概述

RLDS 数据加载流程涉及多个组件：

1. **Octo 库函数**：负责从 RLDS 格式加载和转换数据
2. **Robomimic 工具类**：提供训练、动作处理、文件操作等功能
3. **数据转换流程**：从 RLDS 原始格式 → Octo 标准格式 → Robomimic 格式

---

## Octo 数据集函数

### 代码片段分析 (train.py:111-117)

```python
filter_functions = [[ModuleSpec.create(
                        "robomimic.utils.rlds_utils:filter_success"
                        )] if d_name == "droid" else [] \
                    for d_name in dataset_names]
dataset_kwargs_list = [
    {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS} 
    for d_name, f_functions in zip(dataset_names, filter_functions)
]
```

#### 功能说明

这段代码为每个数据集创建配置字典列表，主要完成以下任务：

1. **创建过滤函数列表** (`filter_functions`)：
   - 使用列表推导式遍历 `dataset_names`
   - 如果数据集名称是 `"droid"`，则创建一个过滤函数 `filter_success`
   - `filter_success` 函数用于过滤出成功轨迹（只保留路径中包含 `/success/` 的轨迹）
   - 对于其他数据集（如 `"role_ros2"`），不应用过滤函数（空列表）

2. **构建数据集配置列表** (`dataset_kwargs_list`)：
   - 为每个数据集创建一个配置字典
   - 包含数据集名称 (`name`)
   - 包含对应的过滤函数列表 (`filter_functions`)
   - 使用 `**BASE_DATASET_KWARGS` 展开基础配置参数

#### filter_success 函数

`filter_success` 函数定义在 `robomimic/utils/rlds_utils.py`：

```python
def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )
```

该函数通过检查轨迹元数据中的文件路径，只保留包含 `/success/` 的轨迹，用于过滤出成功的演示数据。

---

### 1. make_dataset_from_rlds 函数

**源代码位置**: `/opt/third_party/octo/octo/data/dataset.py:201`

#### 功能概述

从 RLDS (Reinforcement Learning Datasets) 格式加载数据集，并将其转换为标准化的 TensorFlow Dataset 格式。

#### 主要功能

1. **加载 RLDS 数据集**：
   - 使用 `tfds.builder(name, data_dir=data_dir)` 加载指定名称的 RLDS 数据集
   - 支持训练/验证集分割

2. **应用过滤函数**：
   - 根据 `filter_functions` 参数过滤轨迹（如只保留成功轨迹）

3. **标准化数据格式**：
   - 如果提供了 `standardize_fn`，首先应用该函数将轨迹转换为标准格式
   - 标准格式必须包含 `"observation"` 和 `"action"` 键

4. **提取观察数据**：
   - **图像观察** (`image_obs_keys`)：从 `observation` 字典中提取 RGB 图像
   - **深度图像** (`depth_obs_keys`)：提取深度图像
   - **状态观察** (`state_obs_keys`)：将多个 1D 状态键连接成单个 `proprio` 数组

5. **归一化处理**：
   - 根据 `action_proprio_normalization_type` 对动作和本体感觉数据进行归一化
   - 支持两种归一化类型：
     - `NORMAL`：零均值、单位方差归一化
     - `BOUNDS`：边界归一化到 [-1, 1]
   - 使用 `dataset_statistics` 参数提供的统计信息

6. **计算数据集统计信息**：
   - 如果未提供 `dataset_statistics`，函数会调用 `get_dataset_statistics` 计算统计信息
   - 统计信息包括：`action` 和 `proprio` 的 `mean`、`std`、`min`、`max`、`num_transitions`、`num_trajectories`

#### 返回值

返回一个元组 `(dataset, dataset_statistics)`：
- **dataset** (`dl.DLataset`)：标准化后的轨迹数据集
- **dataset_statistics** (`dict`)：数据集统计信息字典

#### 在 train.py 中的使用

```python
# 第 120 行：为每个数据集创建数据集并获取统计信息
[make_dataset_from_rlds(**dataset_kwargs, train=True)[1] 
 for dataset_kwargs in dataset_kwargs_list]
```

**注意**：这里只取返回值 `[1]`，即只获取统计信息，不获取数据集本身。这是因为后续会使用 `make_interleaved_dataset` 来创建混合数据集。

---

### 2. combine_dataset_statistics 函数

**源代码位置**: `/opt/third_party/octo/octo/data/utils/data_utils.py:184`

#### 功能概述

合并多个数据集的归一化统计信息，生成统一的统计信息用于归一化。

#### 主要功能

1. **合并统计键**：合并 `"action"` 和 `"proprio"` 两个键的统计信息

2. **计算权重**：根据每个数据集的转移数 (`num_transitions`) 计算权重
   - 权重公式：`weight_i = num_transitions_i / sum(all_num_transitions)`

3. **合并均值**：使用加权平均合并均值
   - 公式：`combined_mean = sum(mean_i * weight_i)`

4. **合并标准差**：使用合并方差的公式计算合并标准差
   ```python
   combined_std = sqrt(
       sum(
           n_i * (std_i² + (mean_i - combined_mean)²)
           for i in range(num_datasets)
       ) / total_transitions
   )
   ```

5. **合并最小值和最大值**：
   - `min`：取所有数据集的最小值
   - `max`：取所有数据集的最大值

#### 在 train.py 中的使用

```python
# 第 119-121 行：合并所有数据集的统计信息
combined_dataset_statistics = combine_dataset_statistics(
    [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] 
     for dataset_kwargs in dataset_kwargs_list]
)
```

**目的**：确保多个数据集使用统一的归一化参数，避免不同数据集分布不一致导致的训练问题。

---

### 3. make_interleaved_dataset 函数

**源代码位置**: `/opt/third_party/octo/octo/data/dataset.py:463`

#### 功能概述

创建交错（混合）数据集，将多个数据集按权重混合，并应用轨迹和帧级别的变换。

#### 主要功能

1. **初始化采样权重**：如果未提供，默认使用均匀权重

2. **获取数据集大小和统计信息**：为每个数据集调用 `make_dataset_from_rlds` 获取统计信息

3. **平衡权重**（可选）：将权重乘以每个数据集的转移数

4. **分配线程资源**：根据采样权重分配线程

5. **构建每个数据集**：
   - 应用统一的 `dataset_statistics`（如果提供）
   - 应用轨迹变换（窗口、子采样等）
   - 将轨迹展平为帧级别

6. **交错采样**：使用 `dl.DLataset.sample_from_datasets` 按权重从多个数据集中采样帧

7. **应用帧变换**：图像 resize、数据增强等

#### 在 train.py 中的使用

```python
# 第 123-150 行：创建混合数据集
dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    config.train.sample_weights,
    train=True,
    shuffle_buffer_size=config.train.shuffle_buffer_size,
    dataset_statistics=combined_dataset_statistics,  # 使用合并的统计信息
    traj_transform_kwargs={...},
    frame_transform_kwargs={...},
    ...
)
```

---

## 动作归一化分析

### 动作归一化流程

在 `train.py:155-156` 中，动作归一化统计信息的计算流程如下：

```python
# 第 153-155 行
rlds_dataset_stats = dataset.dataset_statistics[0] if isinstance(dataset.dataset_statistics, list) else dataset.dataset_statistics
action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats["action"], config.train.action_keys, config.train.action_shapes)
action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
```

#### 流程说明

1. **获取数据集统计信息**：
   - 如果 `dataset_statistics` 是列表，取第一个（假设是 DROID 数据集）
   - 否则直接使用合并的统计信息

2. **提取动作统计信息**：
   - 使用 `ActionUtils.get_action_stats_dict()` 从合并的动作统计信息中分离出各个动作组件的统计信息
   - 根据 `action_keys` 和 `action_shapes` 分割统计信息

3. **计算归一化参数**：
   - 使用 `action_stats_to_normalization_stats()` 根据动作配置和统计信息计算归一化参数（`scale` 和 `offset`）

---

### action_stats_to_normalization_stats 函数

**源代码位置**: `robomimic/utils/dataset.py:1222`

#### 函数签名

```python
def action_stats_to_normalization_stats(action_stats, action_config):
    """
    将动作统计信息转换为归一化参数。
    
    Args:
        action_stats (dict): 动作统计信息字典，每个键对应一个动作组件
            {
                "action/abs_pos": {"mean": [...], "std": [...], "min": [...], "max": [...], "n": ...},
                "action/abs_rot_6d": {...},
                ...
            }
        action_config (dict): 动作配置字典，每个键对应一个动作组件
            {
                "action/abs_pos": {"normalization": "min_max"},
                "action/abs_rot_6d": {"normalization": "min_max"},
                ...
            }
    
    Returns:
        action_normalization_stats (OrderedDict): 归一化参数字典
            {
                "action/abs_pos": {"scale": [...], "offset": [...]},
                "action/abs_rot_6d": {"scale": [...], "offset": [...]},
                ...
            }
    """
```

#### 功能概述

根据动作统计信息和配置，计算每个动作组件的归一化参数（`scale` 和 `offset`）。

归一化公式：
- **归一化**：`normalized_action = (raw_action - offset) / scale`
- **反归一化**：`raw_action = scale * normalized_action + offset`

---

### 归一化方法详解

函数支持三种归一化方法：

#### 1. 无归一化 (`normalization=None`)

**适用场景**：动作已经处于合适的数值范围，不需要归一化

**计算方式**：
```python
scale = np.ones_like(action_stats[action_key]["mean"])
offset = np.zeros_like(action_stats[action_key]["mean"])
```

**效果**：`normalized_action = raw_action`（不进行任何变换）

---

#### 2. Min-Max 归一化 (`normalization="min_max"`)

**适用场景**：将动作值归一化到 [-1, 1] 范围，适用于有界动作空间

**计算方式**：

```python
# 输入范围
input_min = action_stats[action_key]["min"]
input_max = action_stats[action_key]["max"]
input_range = input_max - input_min

# 输出范围（使用略小于 1 的值避免数值不稳定）
output_min = -0.999999
output_max = 0.999999

# 处理过小的范围（避免除零）
range_eps = 1e-4
ignore_dim = input_range < range_eps
input_range[ignore_dim] = output_max - output_min

# 计算 scale 和 offset
# 从两个方程求解：
#   input_max = scale * output_max + offset  ... (1)
#   input_min = scale * output_min + offset  ... (2)
# 
# (1) - (2):
#   input_max - input_min = scale * (output_max - output_min)
#   scale = (input_max - input_min) / (output_max - output_min)
#
# 代入 (2):
#   offset = input_min - scale * output_min

scale = input_range / (output_max - output_min)
offset = input_min - scale * output_min

# 对于过小的维度，使用中心值作为 offset
offset[ignore_dim] = input_min[ignore_dim] - (output_max + output_min) / 2
```

**数学推导**：

给定：
- 输入范围：`[input_min, input_max]`
- 输出范围：`[output_min, output_max]` = `[-0.999999, 0.999999]`

要求线性变换：`normalized = scale * raw + offset`，使得：
- `input_min → output_min`
- `input_max → output_max`

求解：
```
input_max = scale * output_max + offset  ... (1)
input_min = scale * output_min + offset  ... (2)

(1) - (2):
input_max - input_min = scale * (output_max - output_min)
scale = (input_max - input_min) / (output_max - output_min)

代入 (2):
offset = input_min - scale * output_min
```

**示例**：

假设 `action/abs_pos` 的统计信息：
- `min = [0.1, 0.2, 0.3]`
- `max = [0.9, 0.8, 0.7]`

计算：
```python
input_range = [0.8, 0.6, 0.4]
scale = [0.8, 0.6, 0.4] / 1.999998 ≈ [0.4000002, 0.3000001, 0.2000001]
offset = [0.1, 0.2, 0.3] - scale * (-0.999999)
       = [0.1, 0.2, 0.3] + [0.4, 0.3, 0.2] * 0.999999
       ≈ [0.5, 0.5, 0.5]
```

验证：
- `raw = 0.1` → `normalized = (0.1 - 0.5) / 0.4 ≈ -1.0` ✓
- `raw = 0.9` → `normalized = (0.9 - 0.5) / 0.4 ≈ 1.0` ✓

---

#### 3. Gaussian 归一化 (`normalization="gaussian"`)

**适用场景**：将动作值归一化为零均值、单位方差，适用于无界动作空间

**计算方式**：

```python
# 计算均值和标准差
input_mean = action_stats[action_key]["mean"]
input_std = np.sqrt(action_stats[action_key]["sqdiff"] / action_stats[action_key]["n"])

# 处理过小的标准差（避免除零）
std_eps = 1e-6
ignore_dim = input_std < std_eps
input_std[ignore_dim] = 1.0

# 注意：这里 scale 和 offset 的含义与 min_max 不同！
# 对于 gaussian 归一化：
#   normalized = (raw - mean) / std
#   即：normalized = (raw - offset) / scale
#   所以：scale = std, offset = mean

action_normalization_stats[action_key] = {
    "scale": input_std,    # 实际上是标准差
    "offset": input_mean   # 实际上是均值
}
```

**注意**：虽然变量名是 `scale` 和 `offset`，但在这里：
- `scale` 实际上是**标准差** (`std`)
- `offset` 实际上是**均值** (`mean`)

**归一化公式**：
```python
normalized_action = (raw_action - offset) / scale
                  = (raw_action - mean) / std
```

**示例**：

假设 `action/abs_rot_6d` 的统计信息：
- `mean = [0.0, 0.1, 0.2, 0.0, 0.1, 0.2]`
- `std = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7]`

计算：
```python
scale = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7]
offset = [0.0, 0.1, 0.2, 0.0, 0.1, 0.2]
```

验证：
- `raw = [0.0, 0.1, 0.2, 0.0, 0.1, 0.2]` → `normalized = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` ✓
- `raw = [0.5, 0.7, 0.9, 0.5, 0.7, 0.9]` → `normalized ≈ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` ✓

---

### 在 train.py 中的使用

```python
# 第 153-155 行
rlds_dataset_stats = dataset.dataset_statistics[0] if isinstance(dataset.dataset_statistics, list) else dataset.dataset_statistics
action_stats = ActionUtils.get_action_stats_dict(
    rlds_dataset_stats["action"], 
    config.train.action_keys, 
    config.train.action_shapes
)
action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
```

**配置示例**：

在配置文件中，动作归一化方法通过 `action_config` 指定：

```yaml
train:
  action_config:
    action/abs_pos:
      normalization: "min_max"  # 使用 min-max 归一化
    action/abs_rot_6d:
      normalization: "min_max"  # 使用 min-max 归一化
    action/gripper_position:
      normalization: None        # 不归一化（gripper 通常是 0 或 1）
```

---

## Robomimic 工具类

### TrainUtils - 训练工具类

**文件位置**: `robomimic/utils/train_utils.py`

#### 核心函数

1. **`get_exp_dir(config)`**: 创建实验目录结构
2. **`load_data_for_training(config, obs_keys)`**: 加载训练和验证数据集
3. **`run_epoch(model, data_loader, epoch, ...)`**: 运行一个训练或验证周期
4. **`rollout_with_stats(model, env, ...)`**: 在环境中运行策略并收集统计信息
5. **`save_model(model, ckpt_path, ...)`**: 保存模型检查点

详细说明请参考 `docs/robomimic_utils_classes_analysis.md`。

---

### ActionUtils - 动作工具类

**文件位置**: `robomimic/utils/action_utils.py`

#### 核心函数

1. **`get_action_stats_dict(rlds_dataset_stats, action_keys, action_shapes)`**

**功能**: 从 RLDS 数据集统计信息中提取动作统计信息

**作用**:
- 从合并的动作统计信息中分离出各个动作组件的统计信息
- 根据 `action_keys` 和 `action_shapes` 分割统计信息

**在 train.py 中的使用**:
```python
# 第 154 行
action_stats = ActionUtils.get_action_stats_dict(
    rlds_dataset_stats["action"], 
    config.train.action_keys, 
    config.train.action_shapes
)
```

**示例**:
假设 `action_keys = ["action/abs_pos", "action/abs_rot_6d", "action/gripper_position"]`，
`action_shapes = [(1, 3), (1, 6), (1, 1)]`，则函数会将合并的统计信息分割为：
```python
{
    "action/abs_pos": {"mean": [...], "std": [...], "min": [...], "max": [...]},
    "action/abs_rot_6d": {"mean": [...], "std": [...], "min": [...], "max": [...]},
    "action/gripper_position": {"mean": [...], "std": [...], "min": [...], "max": [...]}
}
```

---

### FileUtils - 文件工具类

**文件位置**: `robomimic/utils/file_utils.py`

#### 核心函数

1. **`get_env_metadata_from_dataset(dataset_path, ds_format)`**: 从数据集获取环境元数据
2. **`get_shape_metadata_from_dataset(dataset_path, batch, ...)`**: 从数据集获取形状元数据

详细说明请参考 `docs/robomimic_utils_classes_analysis.md`。

---

## 完整工作流程

以下是 `train.py` 中 RLDS 数据集加载的完整流程：

```
1. 准备基础配置 (BASE_DATASET_KWARGS)
   ├─ data_dir: 数据目录路径
   ├─ image_obs_keys: 图像观察键映射
   ├─ state_obs_keys: 状态观察键列表
   ├─ language_key: 语言指令键
   ├─ action_proprio_normalization_type: "bounds"
   └─ standardize_fn: droid_dataset_transform
   
2. 创建过滤函数列表 (filter_functions)
   ├─ droid 数据集: 使用 filter_success 过滤成功轨迹
   └─ 其他数据集: 不使用过滤
   
3. 构建数据集配置列表 (dataset_kwargs_list)
   └─ 为每个数据集创建配置字典
   
4. 计算合并统计信息
   ├─ 为每个数据集调用 make_dataset_from_rlds 获取统计信息
   └─ 使用 combine_dataset_statistics 合并统计信息
   
5. 创建混合数据集
   ├─ 调用 make_interleaved_dataset
   ├─ 按权重交错采样
   ├─ 应用统一的归一化统计（Octo 层面的归一化）
   ├─ 应用轨迹变换（窗口、子采样等）
   └─ 应用帧变换（resize、增强等）
   
6. 提取动作统计信息
   ├─ 获取数据集统计信息
   └─ 使用 ActionUtils.get_action_stats_dict 分离动作组件
   
7. 计算动作归一化参数
   └─ 使用 action_stats_to_normalization_stats 计算 scale 和 offset
   
8. 应用 robomimic_transform
   └─ 将 Octo 格式转换为 robomimic 格式
   
9. 包装为 PyTorch Dataset
   └─ 使用 TorchRLDSDataset 包装
```

### 数据格式转换流程

```
RLDS 原始格式
    ↓
[standardize_fn] (droid_dataset_transform)
    ↓
Octo 标准格式
    ↓
[make_interleaved_dataset] (轨迹和帧变换 + Octo 层面的归一化)
    ↓
Octo 变换后格式
    ↓
[robomimic_transform]
    ↓
Robomimic 格式
    ↓
[action_normalization_stats] (Robomimic 层面的动作归一化，在训练时应用)
    ↓
最终用于训练的数据
```

### 两层归一化说明

**重要**：RLDS 数据加载流程中存在**两层归一化**：

1. **Octo 层面的归一化**（在 `make_interleaved_dataset` 中）：
   - 使用 `dataset_statistics` 对动作和本体感觉进行归一化
   - 归一化类型由 `action_proprio_normalization_type` 指定（通常是 `"bounds"`）
   - 在数据集层面完成

2. **Robomimic 层面的归一化**（在训练时）：
   - 使用 `action_normalization_stats` 对动作进行归一化
   - 归一化方法由 `action_config` 中的 `normalization` 字段指定（`"min_max"` 或 `"gaussian"`）
   - 在模型训练时应用（通过 `model.postprocess_batch_for_training()`）

这两层归一化可以不同，例如：
- Octo 层面使用 `"bounds"` 归一化到 [-1, 1]
- Robomimic 层面使用 `"min_max"` 再次归一化（通常结果相同）或 `"gaussian"` 归一化

---

## 源代码位置

### Octo 函数

1. **make_dataset_from_rlds**:
   - 文件: `/opt/third_party/octo/octo/data/dataset.py`
   - 行号: 201-462

2. **make_interleaved_dataset**:
   - 文件: `/opt/third_party/octo/octo/data/dataset.py`
   - 行号: 463-576

3. **combine_dataset_statistics**:
   - 文件: `/opt/third_party/octo/octo/data/utils/data_utils.py`
   - 行号: 184-229

### Robomimic 函数

1. **action_stats_to_normalization_stats**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/dataset.py`
   - 行号: 1222-1286

2. **filter_success**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 8-13

3. **droid_dataset_transform**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 27-39

4. **robomimic_transform**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 42-55

5. **ActionUtils.get_action_stats_dict**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/action_utils.py`
   - 行号: 37-51

---

## 关键设计决策

1. **为什么先计算统计信息再创建数据集？**
   - 需要先获取每个数据集的统计信息，然后合并为统一的统计信息
   - 统一的统计信息确保所有数据集使用相同的归一化参数，避免分布不一致

2. **为什么 droid 数据集需要过滤成功轨迹？**
   - DROID 数据集包含成功和失败的轨迹
   - 只使用成功轨迹可以提高训练数据的质量

3. **为什么使用交错采样而不是简单拼接？**
   - 交错采样可以确保训练过程中不同数据集的数据均匀混合
   - 按权重采样可以控制不同数据集的贡献比例

4. **为什么需要两层归一化？**
   - **Octo 层面**：在数据集加载时进行初步归一化，统一不同数据集的分布
   - **Robomimic 层面**：在训练时根据模型需求进行最终归一化，可以灵活选择归一化方法

5. **为什么使用 min_max 归一化而不是 gaussian？**
   - Min-max 归一化将动作值限制在 [-1, 1] 范围内，更适合有界动作空间
   - Gaussian 归一化适合无界动作空间，但可能导致动作值超出合理范围

---

## 参考资料

- Octo 官方文档: https://github.com/octo-models/octo
- RLDS 格式: https://github.com/google-research/rlds
- TensorFlow Datasets: https://www.tensorflow.org/datasets
- Robomimic 官方文档
- 源代码: `/workspace/droid_policy_learning/robomimic/utils/`
