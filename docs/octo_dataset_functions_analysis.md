# Octo 数据集函数详细分析

本文档详细分析 `robomimic/scripts/train.py` 中使用的三个核心 Octo 数据集函数，以及相关的数据集配置代码。

## 目录

1. [代码片段分析 (train.py:111-117)](#代码片段分析)
2. [make_dataset_from_rlds 函数](#1-make_dataset_from_rlds-函数)
3. [combine_dataset_statistics 函数](#2-combine_dataset_statistics-函数)
4. [make_interleaved_dataset 函数](#3-make_interleaved_dataset-函数)
5. [完整工作流程](#完整工作流程)
6. [源代码位置](#源代码位置)

---

## 代码片段分析

### train.py:111-117 行代码解析

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
   - `filter_success` 函数（定义在 `robomimic.utils.rlds_utils`）用于过滤出成功轨迹（只保留路径中包含 `/success/` 的轨迹）
   - 对于其他数据集（如 `"role_ros2"`），不应用过滤函数（空列表）

2. **构建数据集配置列表** (`dataset_kwargs_list`)：
   - 为每个数据集创建一个配置字典
   - 包含数据集名称 (`name`)
   - 包含对应的过滤函数列表 (`filter_functions`)
   - 使用 `**BASE_DATASET_KWARGS` 展开基础配置参数

#### 示例

假设 `dataset_names = ["droid", "role_ros2"]`，则：

```python
filter_functions = [
    [ModuleSpec.create("robomimic.utils.rlds_utils:filter_success")],  # droid 使用过滤
    []  # role_ros2 不使用过滤
]

dataset_kwargs_list = [
    {
        "name": "droid",
        "filter_functions": [ModuleSpec.create("robomimic.utils.rlds_utils:filter_success")],
        "data_dir": "/workspace/dataset",
        "image_obs_keys": {...},
        "state_obs_keys": [...],
        # ... 其他 BASE_DATASET_KWARGS 参数
    },
    {
        "name": "role_ros2",
        "filter_functions": [],
        "data_dir": "/workspace/dataset",
        "image_obs_keys": {...},
        "state_obs_keys": [...],
        # ... 其他 BASE_DATASET_KWARGS 参数
    }
]
```

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

## 1. make_dataset_from_rlds 函数

### 函数签名

```python
def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    state_obs_keys: Sequence[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[Sequence[bool]] = None,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    norm_skip_keys: Optional[Sequence[str]] = None,
    filter_functions: Sequence[ModuleSpec] = (),
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
```

**源代码位置**: `/opt/third_party/octo/octo/data/dataset.py:201`

### 功能概述

从 RLDS (Reinforcement Learning Datasets) 格式加载数据集，并将其转换为标准化的 TensorFlow Dataset 格式。

### 主要功能

1. **加载 RLDS 数据集**：
   - 使用 `tfds.builder(name, data_dir=data_dir)` 加载指定名称的 RLDS 数据集
   - 支持训练/验证集分割

2. **应用过滤函数**：
   - 根据 `filter_functions` 参数过滤轨迹（如只保留成功轨迹）

3. **标准化数据格式**：
   - 如果提供了 `standardize_fn`，首先应用该函数将轨迹转换为标准格式
   - 标准格式必须包含 `"observation"` 和 `"action"` 键

4. **提取观察数据**：
   - **图像观察** (`image_obs_keys`)：从 `observation` 字典中提取 RGB 图像，重命名为 `image_{new_name}`
   - **深度图像** (`depth_obs_keys`)：提取深度图像，重命名为 `depth_{new_name}`
   - **状态观察** (`state_obs_keys`)：将多个 1D 状态键连接成单个 `proprio` 数组
   - 如果某个键为 `None`，则插入填充值（图像为空字符串，状态为 0）

5. **提取任务信息**：
   - 如果提供了 `language_key`，从轨迹中提取语言指令到 `task["language_instruction"]`

6. **归一化处理**：
   - 根据 `action_proprio_normalization_type` 对动作和本体感觉数据进行归一化
   - 支持两种归一化类型：
     - `NORMAL`：零均值、单位方差归一化（需要 `mean` 和 `std`）
     - `BOUNDS`：边界归一化到 [-1, 1]（需要 `min` 和 `max`）
   - 使用 `dataset_statistics` 参数提供的统计信息，如果未提供则自动计算

7. **计算数据集统计信息**：
   - 如果未提供 `dataset_statistics`，函数会调用 `get_dataset_statistics` 计算统计信息
   - 统计信息包括：
     - `action` 和 `proprio` 的 `mean`、`std`、`min`、`max`
     - `num_transitions`：总转移数
     - `num_trajectories`：总轨迹数
   - 统计信息会被缓存到 `dataset_statistics_<hash>.json` 文件中

### 返回值

返回一个元组 `(dataset, dataset_statistics)`：

- **dataset** (`dl.DLataset`)：标准化后的轨迹数据集，每个轨迹包含：
  - `observation`：
    - `image_{name1, name2, ...}`：RGB 图像观察
    - `depth_{name1, name2, ...}`：深度图像观察
    - `proprio`：本体感觉观察数组
    - `timestep`：每帧的时间步
  - `task`：
    - `language_instruction`：语言指令（如果提供了 `language_key`）
  - `action`：动作向量
  - `dataset_name`：数据集名称

- **dataset_statistics** (`dict`)：数据集统计信息字典

### 在 train.py 中的使用

```python
# 第 120 行：为每个数据集创建数据集并获取统计信息
[make_dataset_from_rlds(**dataset_kwargs, train=True)[1] 
 for dataset_kwargs in dataset_kwargs_list]
```

**注意**：这里只取返回值 `[1]`，即只获取统计信息，不获取数据集本身。这是因为后续会使用 `make_interleaved_dataset` 来创建混合数据集。

---

## 2. combine_dataset_statistics 函数

### 函数签名

```python
def combine_dataset_statistics(
    all_dataset_statistics: Sequence[dict],
) -> dict:
```

**源代码位置**: `/opt/third_party/octo/octo/data/utils/data_utils.py:184`

### 功能概述

合并多个数据集的归一化统计信息，生成统一的统计信息用于归一化。

### 主要功能

1. **合并统计键**：
   - 合并 `"action"` 和 `"proprio"` 两个键的统计信息

2. **计算权重**：
   - 根据每个数据集的转移数 (`num_transitions`) 计算权重
   - 权重公式：`weight_i = num_transitions_i / sum(all_num_transitions)`

3. **合并均值**：
   - 使用加权平均合并均值
   - 公式：`combined_mean = sum(mean_i * weight_i)`

4. **合并标准差**：
   - 使用合并方差的公式计算合并标准差
   - 公式基于：`Var(X) = E[X²] - E[X]²`
   - 考虑不同数据集之间的均值差异

5. **合并最小值和最大值**：
   - `min`：取所有数据集的最小值
   - `max`：取所有数据集的最大值

6. **保留原始统计信息**：
   - `num_trajectories`：保留每个数据集的轨迹数列表
   - `num_transitions`：保留每个数据集的转移数列表

### 算法细节

合并标准差的公式（基于合并方差的数学原理）：

```python
combined_std = sqrt(
    sum(
        n_i * (std_i² + (mean_i - combined_mean)²)
        for i in range(num_datasets)
    ) / total_transitions
)
```

这个公式考虑了：
- 每个数据集内部的方差 (`std_i²`)
- 不同数据集之间的均值差异 `(mean_i - combined_mean)²`

### 返回值

返回合并后的统计信息字典：

```python
{
    "action": {
        "min": [...],      # 所有数据集的最小值
        "max": [...],      # 所有数据集的最大值
        "mean": [...],     # 加权平均均值
        "std": [...]       # 合并标准差
    },
    "proprio": {
        "min": [...],
        "max": [...],
        "mean": [...],
        "std": [...]
    },
    "num_trajectories": [n1, n2, ...],  # 每个数据集的轨迹数
    "num_transitions": [t1, t2, ...]    # 每个数据集的转移数
}
```

### 在 train.py 中的使用

```python
# 第 119-121 行：合并所有数据集的统计信息
combined_dataset_statistics = combine_dataset_statistics(
    [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] 
     for dataset_kwargs in dataset_kwargs_list]
)
```

**目的**：确保多个数据集使用统一的归一化参数，避免不同数据集分布不一致导致的训练问题。

---

## 3. make_interleaved_dataset 函数

### 函数签名

```python
def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    dataset_statistics: Optional[Union[dict, str]] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
```

**源代码位置**: `/opt/third_party/octo/octo/data/dataset.py:463`

### 功能概述

创建交错（混合）数据集，将多个数据集按权重混合，并应用轨迹和帧级别的变换。

### 主要功能

1. **初始化采样权重**：
   - 如果未提供 `sample_weights`，默认使用均匀权重 `[1.0, 1.0, ...]`
   - 验证权重数量与数据集数量匹配

2. **获取数据集大小和统计信息**：
   - 为每个数据集调用 `make_dataset_from_rlds` 获取统计信息
   - 记录每个数据集的转移数 (`num_transitions`)

3. **平衡权重**（可选）：
   - 如果 `balance_weights=True`，将权重乘以每个数据集的转移数
   - 这样如果所有权重相等，一次完整迭代会遍历每个数据集一次（期望值）

4. **归一化权重**：
   - 将权重归一化，使总和为 1.0

5. **分配线程资源**：
   - 根据采样权重分配 `traj_transform_threads` 和 `traj_read_threads`
   - 权重越大的数据集分配越多的线程

6. **构建每个数据集**：
   - 为每个数据集调用 `make_dataset_from_rlds` 创建数据集
   - 应用统一的 `dataset_statistics`（如果提供）或使用各自的统计信息
   - 应用轨迹变换 (`apply_trajectory_transforms`)：
     - `window_size`：观察窗口大小
     - `future_action_window_size`：未来动作窗口大小（用于 diffusion policy）
     - `subsample_length`：子采样长度
     - `skip_unlabeled`：跳过没有语言标注的轨迹
   - 将轨迹展平为帧级别

7. **交错采样**：
   - 使用 `dl.DLataset.sample_from_datasets` 按权重从多个数据集中采样帧
   - 应用 `shuffle_buffer_size` 大小的随机打乱

8. **应用帧变换**：
   - 调用 `apply_frame_transforms` 应用帧级别变换：
     - 图像 resize
     - 数据增强（可选）
     - 其他帧级别处理

9. **批处理**（可选）：
   - 如果提供了 `batch_size`，将数据集批处理

10. **内存优化**：
    - 使用 `with_ram_budget(1)` 限制内存使用

### 轨迹变换参数 (traj_transform_kwargs)

在 `train.py` 中的使用：

```python
traj_transform_kwargs=dict(
    window_size=config.algo.horizon.observation_horizon,           # 观察窗口大小
    future_action_window_size=config.algo.horizon.prediction_horizon-1,  # 未来动作窗口
    subsample_length=config.train.subsample_length,                # 子采样长度
    skip_unlabeled=True,                                           # 跳过无语言标注的轨迹
)
```

### 帧变换参数 (frame_transform_kwargs)

在 `train.py` 中的使用：

```python
frame_transform_kwargs=dict(
    image_augment_kwargs=dict(),                                   # 图像增强参数（空）
    resize_size=dict(
        primary=config.observation.image_dim,                      # 主相机图像尺寸
        secondary=config.observation.image_dim,                    # 次相机图像尺寸
    ),
    num_parallel_calls=config.train.num_parallel_calls,            # 并行调用数
)
```

### 返回值

返回一个混合的 `dl.DLataset` 对象，包含：

- **数据集本身**：按权重交错采样的帧序列
- **dataset_statistics**：数据集统计信息（如果提供了统一的统计信息，则为单个字典；否则为列表）
- **sample_weights**：采样权重

### 在 train.py 中的使用

```python
# 第 123-150 行：创建混合数据集
dataset = make_interleaved_dataset(
    dataset_kwargs_list,                    # 多个数据集的配置
    config.train.sample_weights,            # 采样权重
    train=True,
    shuffle_buffer_size=config.train.shuffle_buffer_size,
    dataset_statistics=combined_dataset_statistics,  # 使用合并的统计信息
    traj_transform_kwargs={...},            # 轨迹变换参数
    frame_transform_kwargs={...},          # 帧变换参数
    ...
)
```

---

## 完整工作流程

以下是 `train.py` 中数据集加载的完整流程：

```
1. 准备基础配置 (BASE_DATASET_KWARGS)
   ├─ data_dir: 数据目录路径
   ├─ image_obs_keys: 图像观察键映射
   ├─ state_obs_keys: 状态观察键列表
   ├─ language_key: 语言指令键
   └─ standardize_fn: 标准化函数 (droid_dataset_transform)
   
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
   ├─ 应用统一的归一化统计
   ├─ 应用轨迹变换（窗口、子采样等）
   └─ 应用帧变换（resize、增强等）
   
6. 应用 robomimic_transform
   └─ 将 Octo 格式转换为 robomimic 格式
   
7. 包装为 PyTorch Dataset
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
[make_interleaved_dataset] (轨迹和帧变换)
    ↓
Octo 变换后格式
    ↓
[robomimic_transform]
    ↓
Robomimic 格式 (用于训练)
```

---

## 源代码位置

所有函数的源代码位于 Octo 库中：

1. **make_dataset_from_rlds**:
   - 文件: `/opt/third_party/octo/octo/data/dataset.py`
   - 行号: 201-462

2. **make_interleaved_dataset**:
   - 文件: `/opt/third_party/octo/octo/data/dataset.py`
   - 行号: 463-576

3. **combine_dataset_statistics**:
   - 文件: `/opt/third_party/octo/octo/data/utils/data_utils.py`
   - 行号: 184-229

4. **filter_success**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 8-13

5. **droid_dataset_transform**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 27-39

6. **robomimic_transform**:
   - 文件: `/workspace/droid_policy_learning/robomimic/utils/rlds_utils.py`
   - 行号: 42-55

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

4. **为什么需要轨迹变换和帧变换？**
   - **轨迹变换**：处理轨迹级别的操作（如窗口化、子采样、过滤）
   - **帧变换**：处理帧级别的操作（如图像 resize、数据增强）
   - 分离这两个阶段可以提高并行效率

---

## 参考资料

- Octo 官方文档: https://github.com/octo-models/octo
- RLDS 格式: https://github.com/google-research/rlds
- TensorFlow Datasets: https://www.tensorflow.org/datasets
