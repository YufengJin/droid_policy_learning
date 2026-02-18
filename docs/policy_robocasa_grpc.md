# Policy ↔ RoboCasa gRPC 接口说明

## 1. RoboCasa run_demo / run_eval 的 Action Space

**环境 (PandaMobile 默认)**:

- `env.action_dim == 12`: 7D 手臂 (笛卡尔位姿 + 夹爪) + 5D 移动底座
- `env.action_spec`: (action_low, action_high)，各为 shape=(12,) 的数组

**Policy 输出与客户端处理**:

- Policy 通常输出 **7D**：3D 位置 + 3D 旋转（轴角或四元数等）+ 1D 夹爪
- 当 `action.shape[-1] == 7` 且 `env.action_dim == 12` 时，客户端在 7D 后拼接固定 `mobile_base = [0, 0, 0, 0, -1]`，得到 12D 动作送给环境
- 若 Policy 已输出 12D，客户端直接使用，不做填充

```python
# run_demo.py / run_eval.py 中:
if action.shape[-1] == 7 and env.action_dim == 12:
    mobile_base = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    action = np.concatenate([action, mobile_base])
```

**结论**: RoboCasa 客户端的 action space 为 **12D (PandaMobile)**，Policy 可输出 **7D (手臂)** 或 **12D**，客户端负责 7→12 填充。

---

## 2. 当前 gRPC 传输内容

### Reset

| 字段 | 类型 | 说明 |
|------|------|------|
| task_name | string | 任务名，如 "PnPCounterToCab" |
| task_description | string | 自然语言任务描述 |
| action_dim | int32 | 动作维度 (12) |
| action_low | repeated float | 动作下界，长度=action_dim |
| action_high | repeated float | 动作上界，长度=action_dim |

### GetAction 请求 (ObservationRequest)

| 字段 | 类型 | 说明 |
|------|------|------|
| primary_image | bytes | JPEG，左第三人称相机 (robot0_agentview_left) |
| secondary_image | bytes | JPEG，右第三人称相机 (robot0_agentview_right) |
| wrist_image | bytes | JPEG，手腕相机 (robot0_eye_in_hand) |
| proprio | repeated float | 本体感知：**[gripper_qpos(2), eef_pos(3), eef_quat(4)]** 共 9 维 |
| task_description | string | 自然语言任务描述 |
| image_height, image_width | int32 | 图像原始分辨率 (如 224) |

### GetAction 响应 (ActionResponse)

| 字段 | 类型 | 说明 |
|------|------|------|
| action | repeated float | 动作向量，通常 7D 或 12D |

### proprio 顺序约定（与 RoboCasa prepare_observation 一致）

```
proprio = concat(
    obs["robot0_gripper_qpos"],   # 2D
    obs["robot0_eef_pos"],        # 3D
    obs["robot0_eef_quat"],       # 4D
)
# 总长 = 9
```

---

## 3. Policy Server 与 Checkpoint 的匹配

Policy Server 根据 **checkpoint 的 shape_metadata 与 config** 构建输入：

- **RGB**：仅使用与 checkpoint `rgb` keys 对应的图像，通过 `obs_mapping` 映射 gRPC 字段到 obs key
- **low_dim**：将 `proprio` 按上述顺序拆分为 `robot0_gripper_qpos`、`robot0_eef_pos`、`robot0_eef_quat`，再按 checkpoint 的 `low_dim` keys 填入
- **lang**：从 Reset 的 `task_description` 编码为 DistilBERT 768 维，附加到每个 obs

若 checkpoint 训练时的 obs 模态与 gRPC 约定不一致（例如无 low_dim、不同数量图像），会导致维度不匹配（如 mat1 2331 vs mat2 2313）。应使用与 RoboCasa 观测约定一致的 checkpoint，或在训练时采用相同模态与顺序。
