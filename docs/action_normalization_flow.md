# Action 归一化流程与代码位置

本文档说明 RLDS 训练流程中 action 如何被归一化、统计量从哪里来、多数据集如何统一到同一 action space。

---

## 1. 两层归一化（RLDS 流程）

RLDS 数据流中存在两层与 action 相关的处理：

| 层级 | 谁做 | 何时 | 作用 |
|------|------|------|------|
| **Octo 层** | Octo 库 | 构建/迭代 dataset 时 | 用 `dataset_statistics` 把 **原始 action** 归一化到约 [-1,1]（bounds） |
| **Robomimic 层** | Robomimic | 仅 rollout 时 | 用 `action_normalization_stats` 把 **模型输出** 反归一化回机器人物理空间 |

训练时喂给模型的数据已经是 Octo 归一化后的；Robomimic 的 `action_normalization_stats` 只用于推理时反归一化，不参与训练 batch 的归一化。

---

## 2. Action 归一化的准确代码位置

### 2.1 Octo 层：原始 action → 归一化（在 Octo 库内完成）

- **归一化类型**：由 `action_proprio_normalization_type: "bounds"` 指定，即按 min/max 映射到 [-1, 1]。
- **统计量来源**：`dataset_statistics`（内含 `action` 的 min/max/mean/std 等）。
- **应用位置**：在 **Octo** 的 dataset 构建/迭代中应用（不在本仓库内）。
  - 本仓库中的**调用入口**：`train.py` 里通过 `make_interleaved_dataset(..., dataset_statistics=combined_dataset_statistics)` 传入统一统计量，Octo 用这份统计量对所有数据做归一化。
- **本仓库中设置“用 bounds 归一化”的位置**：
  - **文件**: `robomimic/scripts/train.py`
  - **行号**: 约 104–114
  - **代码**:
    ```python
    BASE_DATASET_KWARGS = {
        ...
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": is_abs_action,
        "action_normalization_mask": is_abs_action,
        "standardize_fn": droid_dataset_transform,
    }
    ```
- **数据流**：每个数据集先通过 `make_dataset_from_rlds(..., train=True)[1]` 得到该数据集的统计量 → 用 `combine_dataset_statistics` 合并 → 将 `combined_dataset_statistics` 传给 `make_interleaved_dataset` → Octo 在迭代时用这份**统一**统计量对 action（及 proprio）做 bounds 归一化。因此 dataloader 吐出的 `trajectory["action"]` 已经是归一化后的。

### 2.2 Robomimic 层：从“统计量”到 offset/scale（用于反归一化）

Robomimic 不在这里对训练 batch 再做一次 action 归一化，而是用同一份统计量推导出 **反归一化** 用的 `offset/scale`，供 rollout 使用。

- **从 Octo 统计量到“按 key 的 action stats”**：
  - **文件**: `robomimic/scripts/train.py`
  - **行号**: 159–162
  - **代码**:
    ```python
    rlds_dataset_stats = dataset.dataset_statistics[0] if isinstance(dataset.dataset_statistics, list) else dataset.dataset_statistics
    action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats["action"], config.train.action_keys, config.train.action_shapes)
    action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
    ```
- **按 key 切分 Octo 的 action 统计量**：
  - **文件**: `robomimic/utils/action_utils.py`
  - **行号**: 37–51  
  - **函数**: `get_action_stats_dict(rlds_dataset_stats, action_keys, action_shapes)`  
  - 作用：把 Octo 的 `rlds_dataset_stats["action"]`（按 action_keys/action_shapes）切成多个 key，每个 key 对应一份 min/max/mean 等，供 `action_stats_to_normalization_stats` 使用。

- **从 action_stats 到 offset/scale（min_max → 线性映射）**：
  - **文件**: `robomimic/utils/dataset.py`
  - **行号**: 1222–1286  
  - **函数**: `action_stats_to_normalization_stats(action_stats, action_config)`
  - 逻辑要点：
    - 若 `action_config[action_key].get("normalization") == "min_max"`：用该 key 的 `min`/`max` 算线性映射到 [-1,1] 的 `scale` 和 `offset`（与 Octo bounds 一致）。
    - 公式：`normalized = (raw - offset) / scale`，反归一化：`raw = scale * normalized + offset`。
  - 这里得到的 `action_normalization_stats` 与 Octo 使用的统计量一致，只用于 **RolloutPolicy** 里把模型输出反归一化回物理空间。

### 2.3 Rollout 时反归一化（使用 action_normalization_stats）

- **文件**: `robomimic/algo/algo.py`
- **行号**: 约 684–688（RolloutPolicy 内）
- **代码**:
  ```python
  if self.action_normalization_stats is not None:
      ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=self.action_normalization_stats)
  ```
- **文件**: `robomimic/utils/obs_utils.py`  
- **函数**: `unnormalize_dict`（约 505–535 行），用 `offset` 和 `scale` 做 `raw = scale * normalized + offset`。

---

## 3. 是否“一个数据集一套归一化参数”？

**不是。** 当前实现是：**多个数据集共用一套归一化参数**。

- 每个数据集单独调用 `make_dataset_from_rlds(**dataset_kwargs, train=True)` 时，只取返回值中的统计量 `[1]`，用于后续合并，**并没有**用“每个数据集自己的统计量”去分别归一化。
- **合并**：`combine_dataset_statistics([...])` 对所有数据集的统计量做合并（见下）。
- **统一应用**：`make_interleaved_dataset(..., dataset_statistics=combined_dataset_statistics)` 时，Octo 用这一份 **combined_dataset_statistics** 对所有数据集的数据做归一化。
- 因此：**所有数据集都被映射到同一个归一化 action space**（同一组 min/max，即同一组 bounds）。

---

## 4. 数据集变化或多种数据集如何“变到正确的 action space”

- **训练阶段**  
  - 每次训练 run 都会根据**当前** `dataset_names` 和 `dataset_kwargs_list` 重新算一遍统计量并合并：
    - `train.py` 中：`combined_dataset_statistics = combine_dataset_statistics([make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list])`
  - 因此：
    - **数据集列表变化**（增删/换数据集）：会重新计算并合并统计量，得到新的 `combined_dataset_statistics`，所有数据仍用这一套参数归一化，保证同一 run 内 action space 一致。
    - **多数据集**：同上，一套合并后的统计量 → 一个统一的归一化 action space。

- **合并规则（Octo 的 combine_dataset_statistics）**  
  - 文档与代码分析（见 `docs/octo_dataset_functions_analysis.md`、`docs/rlds_dataloader.md`）：
    - **min/max**：对所有数据集的 min 取最小、max 取最大，得到全局 bounds。
    - **mean/std**：按各数据集 `num_transitions` 加权平均/合并方差。
  - 这样得到的全局 bounds 保证：任意一个数据集里的 action 落在各自原始范围，归一化后都落在同一 [-1,1] 空间内，多数据集混合训练时尺度一致。

- **推理/部署时**  
  - 必须使用**与训练时相同的** `action_normalization_stats`（通常从训练保存的 checkpoint 里读入），这样模型输出的归一化 action 才能用同一套 offset/scale 反归一化到正确的机器人 action space。
  - 若换了机器人或换了数据采集的 action 定义，需要重新训练或至少用新数据重新算统计量并更新 `action_normalization_stats`。

---

## 5. 小结表

| 问题 | 答案 |
|------|------|
| Action 归一化是谁做的？ | **Octo** 在 dataset 迭代时用 `dataset_statistics` + `action_proprio_normalization_type: "bounds"` 做的。 |
| 设置“bounds”的代码位置？ | `robomimic/scripts/train.py` 约 111 行，`BASE_DATASET_KWARGS["action_proprio_normalization_type"] = "bounds"`。 |
| 统计量从哪里来？ | 各数据集 `make_dataset_from_rlds(..., train=True)[1]` → `combine_dataset_statistics` → `combined_dataset_statistics`。 |
| 一个数据集一套参数吗？ | **否**。多数据集共用一套合并后的 `combined_dataset_statistics`。 |
| 数据集变化/多数据集怎么办？ | 每次 run 按当前数据集列表重新计算并合并统计量，所有数据用同一套 bounds 归一化到同一 action space；部署时用训练保存的 `action_normalization_stats` 反归一化。 |
| Robomimic 的 action_normalization_stats 用途？ | 仅用于 **rollout/推理** 时的反归一化，由 `dataset.py` 的 `action_stats_to_normalization_stats` 从同一套统计量推导出 offset/scale。 |
