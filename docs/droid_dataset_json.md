# DROID 数据集 JSON 文件说明

本文档说明 DROID 数据集目录下的 JSON 文件所保存的信息，以及 `robomimic/scripts/train.py` 在 `droid_rlds` 模式下是否会读取这些文件。

---

## 1. JSON 文件概览

典型 DROID 数据集目录（如 `datasets/droid/1.0.1/`）包含：

| 文件 | 用途 |
|------|------|
| `dataset_info.json` | TFDS/RLDS 数据集元信息（格式、splits、分片） |
| `features.json` | 数据 schema（每步的 key、dtype、shape） |
| `dataset_statistics_<hash>.json` | 归一化统计（action/proprio 的 mean/std/min/max 等） |
| `*.tfrecord-*` | 实际轨迹数据 |

---

## 2. 各 JSON 文件内容

### 2.1 `dataset_info.json`

TFDS/RLDS 的**数据集元信息**，供 TensorFlow Datasets 发现数据、解析分片。

| 字段 | 含义 |
|------|------|
| `fileFormat` | 原始数据格式，如 `"tfrecord"` |
| `moduleName` | 构建脚本所在模块，如 `"__main__"` |
| `name` | 数据集名称，如 `"r2d2_faceblur"` |
| `releaseNotes` | 版本说明，如 `{"0.0.1": "Initial release."}` |
| `splits` | 数据划分与分片信息数组 |
| `splits[].name` | 划分名，如 `"train"` |
| `splits[].filepathTemplate` | 文件名模板，如 `"{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"` |
| `splits[].numBytes` | 该 split 的总字节数 |
| `splits[].shardLengths` | 每个 shard 的轨迹数量（长数组） |
| `version` | 元数据版本，如 `"1.4.0"` |

---

### 2.2 `features.json`

TensorFlow Datasets 的 **Feature 结构（schema）**，描述每个 step 的字段、类型和形状。

- **`episode_metadata`**
  - `file_path`：string
  - `recording_folderpath`：string

- **`steps`**（sequence）中每一步包含：
  - `action`：`float64`，shape `(7,)`
  - `action_dict`：`cartesian_position`(6)、`cartesian_velocity`(6)、`gripper_position`(1)、`gripper_velocity`(1)、`joint_position`(7)、`joint_velocity`(7)
  - `observation`：
    - 低维：`cartesian_position`(6)、`gripper_position`(1)、`joint_position`(7)
    - 图像：`exterior_image_1_left`、`exterior_image_2_left`、`wrist_image_left`，`uint8`，`(180,320,3)`，jpeg
  - `language_instruction` / `language_instruction_2` / `language_instruction_3`：string
  - `is_first`、`is_last`、`is_terminal`、`discount`、`reward`

---

### 2.3 `dataset_statistics_<hash>.json`

由 **Octo** 的 `get_dataset_statistics` 计算得到的**归一化统计**，用于对 action 和 proprio 做归一化。  
`<hash>` 由 `hashlib.sha256(builder.info + proprio_obs_key + standardize_fn + filter_functions + ...)` 的 hex 得到；配置不同（如 `state_obs_keys`、`filter_functions`、`standardize_fn`）会产生不同 hash，因此同一目录下可有多个 `dataset_statistics_*.json`。

典型结构：

- **`action`**：`mean`、`std`、`max`、`min`（Octo 还会算 `p99`、`p01`）
- **`proprio`**：同上；若 `norm_skip_keys` 包含 `"proprio"`，可能全部为 0
- **`num_transitions`**：总转移数
- **`num_trajectories`**：总轨迹数

---

## 3. `train.py` 是否读取这些 JSON

`train.py` **不直接** `open` 或 `json.load` 任何上述 JSON；在 `ds_format == "droid_rlds"` 时，通过 **Octo** 和 **TFDS** 间接使用其中一部分。

### 3.1 调用链

- `train.py` 使用 `octo.data.dataset.make_dataset_from_rlds`、`make_interleaved_dataset` 和 `octo.data.utils.data_utils.combine_dataset_statistics`。
- `make_dataset_from_rlds` 内部：
  - 调用 **`tfds.builder(name, data_dir=data_dir)`**，其中 `data_dir = config.train.data_path`，`name` 来自 `config.train.dataset_names`（如 `"droid"`）。
  - 若未传入 `dataset_statistics`，会调用 **`get_dataset_statistics(..., save_dir=builder.data_dir)`**，在 `builder.data_dir` 下按 hash 查找或写入 `dataset_statistics_<hash>.json`。

### 3.2 各文件是否被读取

| 文件 | 是否被读 | 说明 |
|------|----------|------|
| **`dataset_info.json`** | **会** | 由 `tfds.builder(name, data_dir=data_dir)` 在 `data_dir` 下读取，用于获取 splits、`filepathTemplate`、`shardLengths` 等。 |
| **`dataset_statistics_<hash>.json`** | **会（按需）** | 由 Octo 的 `get_dataset_statistics` 在 `builder.data_dir` 下按 hash 查找；若存在且未设置 `force_recompute`，则直接加载；否则遍历数据计算并写回同名文件。 |
| **`features.json`** | **未发现** | 在 `train.py` 及所查 Octo 代码（`dataset.py`、`data_utils.py`）中均未见引用。TFDS 的 `builder.info` 通常来自 `dataset_info.json` 或 builder 定义；`features.json` 可能用于文档、导出或其它工具。 |

### 3.3 相关代码位置（`train.py`）

- `data_dir` 与 `name` 传入 `make_dataset_from_rlds`：

```python
BASE_DATASET_KWARGS = {
    "data_dir": config.train.data_path,
    # ...
}
dataset_kwargs_list = [
    {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS}
    for d_name, f_functions in zip(dataset_names, filter_functions)
]
```

- 第一次调用 `make_dataset_from_rlds` 时未传入 `dataset_statistics`，因此会走 Octo 的 `get_dataset_statistics`，从而可能读/写 `dataset_statistics_<hash>.json`：

```python
combined_dataset_statistics = combine_dataset_statistics(
    [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list]
)
```

---

## 4. 参考

- Octo: `octo.data.dataset.make_dataset_from_rlds`、`octo.data.utils.data_utils.get_dataset_statistics`
- TensorFlow Datasets：`tfds.builder(name, data_dir=data_dir)` 在 `data_dir` 下的元数据约定
