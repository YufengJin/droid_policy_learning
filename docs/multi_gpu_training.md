# 多 GPU 训练指南

本指南介绍如何在 DROID Policy Learning 中使用多个 GPU 进行训练，并详细说明 `DistributedDataParallel` 和 `DataParallel` 的区别。

## 目录

- [快速开始：直接运行多卡训练](#快速开始直接运行多卡训练)
- [DataParallel vs DistributedDataParallel](#dataparallel-vs-distributeddataparallel)
- [当前实现状态](#当前实现状态)
- [方法 1: DataParallel（简单但效率较低）](#方法-1-dataparallel简单但效率较低)
- [方法 2: DistributedDataParallel（推荐）](#方法-2-distributeddataparallel推荐)
- [方法 3: 使用 torchrun（最简单）](#方法-3-使用-torchrun最简单)
- [配置建议](#配置建议)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 快速开始：直接运行多卡训练

### 最简单的方式（推荐）

**无需任何修改，直接运行训练命令即可使用多卡！**

```bash
# 直接运行，自动使用所有 GPU
python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

代码会自动：
- ✅ 检测所有可用的 GPU
- ✅ 自动使用所有 GPU 进行训练（通过 DataParallel）
- ✅ 无需任何额外配置或修改

### 验证多卡是否在使用

```bash
# 在另一个终端运行，实时监控 GPU 使用
watch -n 1 nvidia-smi
```

如果看到所有 GPU 都在使用（使用率 > 0%），说明多卡训练已启用。

### 指定使用的 GPU

```bash
# 只使用 GPU 0 和 GPU 1
CUDA_VISIBLE_DEVICES=0,1 python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json

# 只使用 GPU 2 和 GPU 3
CUDA_VISIBLE_DEVICES=2,3 python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

### 性能说明

- **DataParallel**：当前实现使用这种方式，适合 2-4 GPU
- **性能**：比单 GPU 快，但不如 DDP 高效
- **适用场景**：快速测试、小规模训练

如果需要更好的性能（4+ GPU），请参考下面的 [DistributedDataParallel 实现](#方法-2-distributeddataparallel推荐)。

---

## DataParallel vs DistributedDataParallel

### 核心区别对比

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| **架构** | 单进程多线程 | 多进程（每个 GPU 一个进程） |
| **通信方式** | 通过 Python GIL，所有操作在 GPU 0 上聚合 | 通过 NCCL，GPU 间直接通信 |
| **梯度聚合** | 在 GPU 0 上聚合所有梯度（瓶颈） | 每个进程独立聚合（AllReduce） |
| **性能** | 较慢，受 GIL 限制 | 更快，接近线性扩展 |
| **可扩展性** | 适合 2-4 GPU | 适合任意数量 GPU（包括多机） |
| **内存效率** | 所有模型副本在 GPU 0 上 | 每个进程只管理自己的 GPU |
| **实现复杂度** | 简单（一行代码） | 需要分布式初始化 |
| **启动方式** | 直接运行 Python 脚本 | 需要 `torchrun` 或 `torch.distributed.launch` |

### 详细技术区别

#### 1. 架构差异

**DataParallel (DP)**:
```
┌─────────────────────────────────────┐
│     Python 主进程 (单进程)           │
│  ┌──────────┐  ┌──────────┐         │
│  │ GPU 0    │  │ GPU 1    │         │
│  │ (主GPU)  │  │ (副本)   │         │
│  │          │  │          │         │
│  │ 聚合梯度 │←─│ 发送梯度 │         │
│  └──────────┘  └──────────┘         │
│       ↑              ↑               │
│       └──────────────┘               │
│    通过 Python GIL 通信              │
└─────────────────────────────────────┘
```

**DistributedDataParallel (DDP)**:
```
┌──────────┐      ┌──────────┐
│ 进程 0   │      │ 进程 1   │
│ GPU 0    │      │ GPU 1    │
│          │      │          │
│ 独立前向 │      │ 独立前向 │
│ 独立反向 │      │ 独立反向 │
│          │      │          │
└────┬─────┘      └────┬─────┘
     │                │
     └──────┬─────────┘
            │
     ┌──────▼──────┐
     │  NCCL AllReduce │
     │  (GPU 间直接通信)│
     └──────────────┘
```

#### 2. 通信机制

**DataParallel**:
- 使用 Python 的全局解释器锁 (GIL)
- 所有 GPU 的梯度必须发送到 GPU 0
- GPU 0 成为通信瓶颈
- 数据传输通过 CPU 内存

**DistributedDataParallel**:
- 使用 NCCL (NVIDIA Collective Communications Library)
- GPU 间直接通信（不经过 CPU）
- 使用高效的 AllReduce 算法（Ring AllReduce）
- 每个 GPU 独立计算，然后同步

#### 3. 性能对比

假设训练一个批次需要的时间：

| GPU 数量 | DataParallel | DistributedDataParallel | 加速比 |
|---------|-------------|------------------------|--------|
| 1 GPU   | 100ms       | 100ms                  | 1.0x   |
| 2 GPU   | 60ms        | 52ms                   | 1.15x  |
| 4 GPU   | 40ms        | 28ms                   | 1.43x  |
| 8 GPU   | 30ms        | 15ms                   | 2.0x   |

**为什么 DDP 更快？**
1. **无 GIL 限制**：多进程避免了 Python GIL 的序列化问题
2. **并行通信**：NCCL 的 AllReduce 是并行的，而 DP 是串行的
3. **无主 GPU 瓶颈**：每个 GPU 独立工作，不需要等待主 GPU

#### 4. 内存使用

**DataParallel**:
- 主模型在 GPU 0 上
- 其他 GPU 只有模型副本
- 梯度聚合在 GPU 0 上进行，需要额外内存

**DistributedDataParallel**:
- 每个进程管理自己的 GPU
- 内存使用更均匀
- 梯度在通信时临时存储，不占用主内存

#### 5. 代码示例对比

**DataParallel 使用**:
```python
# 简单，一行代码
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.cuda()

# 直接运行
python train.py
```

**DistributedDataParallel 使用**:
```python
# 需要初始化分布式环境
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
model = DDP(model, device_ids=[local_rank])

# 需要分布式启动
torchrun --nproc_per_node=4 train.py
```

### 何时使用哪个？

**使用 DataParallel 当：**
- ✅ 快速原型开发
- ✅ 只有 2-4 个 GPU
- ✅ 不想修改现有代码
- ✅ 训练时间不是关键因素

**使用 DistributedDataParallel 当：**
- ✅ 生产环境训练
- ✅ 4+ GPU 训练
- ✅ 需要最佳性能
- ✅ 多机训练
- ✅ 大规模模型训练

---

## 当前实现状态

### 现有支持

代码中已经包含了 `DataParallel` 的实现（在 `robomimic/algo/diffusion_policy.py` 第 97-98 行）：

```python
'obs_encoder': torch.nn.parallel.DataParallel(
    obs_encoder, 
    device_ids=list(range(0, torch.cuda.device_count()))
),
'noise_pred_net': torch.nn.parallel.DataParallel(
    noise_pred_net, 
    device_ids=list(range(0, torch.cuda.device_count()))
)
```

**特点**：
- 自动检测 GPU 数量
- 如果只有 1 个 GPU，不会使用 DataParallel
- 如果有多个 GPU，自动使用所有 GPU

**限制**：
- 使用 DataParallel，效率较低
- 不适合大规模训练（4+ GPU）

---

## 方法 1: DataParallel（简单但效率较低）

### 工作原理

`DataParallel` 是单进程多 GPU 方案：
1. 主进程在 GPU 0 上
2. 将批次数据分割到各个 GPU
3. 每个 GPU 独立计算前向和反向传播
4. 所有梯度在 GPU 0 上聚合
5. 在 GPU 0 上更新模型
6. 将更新后的模型同步到其他 GPU

### 使用方法：直接运行多卡训练

**当前代码已经支持多卡训练！** 无需任何修改，直接运行训练命令即可：

```bash
# 方式 1: 使用 train_rlds.py
python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json

# 方式 2: 使用 train.py
python -m robomimic.scripts.train \
    --config /path/to/config.json
```

**代码会自动：**
- 检测系统中所有可用的 GPU
- 自动使用所有 GPU 进行训练
- 无需任何额外配置

### 验证多 GPU 是否在使用

运行以下命令验证：

```bash
# 1. 检查 GPU 数量
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 2. 在训练时监控 GPU 使用（另开一个终端）
watch -n 1 nvidia-smi

# 3. 或者使用 nvtop（如果已安装）
nvtop
```

**预期结果：**
- 如果有多块 GPU，应该看到所有 GPU 都在使用
- GPU 0 的使用率可能稍高（因为梯度聚合在 GPU 0 上进行）
- 其他 GPU 的使用率应该接近 GPU 0

### 指定使用的 GPU

如果想只使用特定的 GPU，可以设置环境变量：

```bash
# 只使用 GPU 0 和 GPU 1
CUDA_VISIBLE_DEVICES=0,1 python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json

# 只使用 GPU 2 和 GPU 3
CUDA_VISIBLE_DEVICES=2,3 python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

**注意：** `CUDA_VISIBLE_DEVICES` 会重新映射 GPU 索引，所以：
- `CUDA_VISIBLE_DEVICES=2,3` 会将 GPU 2 映射为 `cuda:0`，GPU 3 映射为 `cuda:1`
- 代码会自动使用所有可见的 GPU

### 验证多 GPU 使用

```bash
# 检查 GPU 数量
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 在训练时监控 GPU 使用
watch -n 1 nvidia-smi

# 应该看到所有 GPU 都在使用，但 GPU 0 使用率可能更高
```

### 性能特点

- ⚠️ **单进程限制**：受 Python GIL 限制，无法充分利用多核 CPU
- ⚠️ **GPU 0 瓶颈**：所有梯度聚合在 GPU 0，成为通信瓶颈
- ⚠️ **串行通信**：GPU 间通信是串行的，不是并行的
- ⚠️ **扩展性差**：GPU 数量增加时，性能提升不明显

### 适用场景

- 快速测试多 GPU 训练
- 2-4 GPU 的小规模训练
- 不想修改代码的临时方案

---

## 方法 2: DistributedDataParallel（推荐）

`DistributedDataParallel` (DDP) 是 PyTorch 推荐的多 GPU 训练方法，提供更好的性能和可扩展性。

### 工作原理

1. **多进程架构**：每个 GPU 运行一个独立的 Python 进程
2. **独立计算**：每个进程独立进行前向和反向传播
3. **并行通信**：使用 NCCL 的 AllReduce 算法并行同步梯度
4. **同步更新**：所有进程同步更新模型参数

### 实现步骤

#### 步骤 1: 修改 `train_rlds.py`

在 `train_rlds.py` 中添加分布式训练支持：

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # 使用 NCCL 后端（GPU）
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def run(cfg: "OmegaConf") -> str:
    from robomimic.config import config_factory
    from robomimic.utils import torch_utils as TorchUtils
    from robomimic.scripts.train import train

    # ... 现有代码 ...
    
    # 检查是否使用分布式训练
    use_ddp = int(os.environ.get('WORLD_SIZE', 0)) > 1
    
    if use_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    
    # ... 继续训练 ...
    
    if use_ddp:
        cleanup_distributed()
    
    return "finished run successfully!"
```

#### 步骤 2: 修改 `diffusion_policy.py`

将 `DataParallel` 替换为 `DistributedDataParallel`：

```python
# 在 _create_networks 方法中
def _create_networks(self):
    # ... 创建网络的代码 ...
    
    # 检查是否使用 DDP
    use_ddp = int(os.environ.get('WORLD_SIZE', 0)) > 1
    
    if use_ddp:
        # 使用 DDP 包装网络
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        obs_encoder = DDP(obs_encoder, device_ids=[local_rank])
        noise_pred_net = DDP(noise_pred_net, device_ids=[local_rank])
    else:
        # 回退到 DataParallel（单机多 GPU）
        if torch.cuda.device_count() > 1:
            obs_encoder = torch.nn.parallel.DataParallel(
                obs_encoder, 
                device_ids=list(range(torch.cuda.device_count()))
            )
            noise_pred_net = torch.nn.parallel.DataParallel(
                noise_pred_net, 
                device_ids=list(range(torch.cuda.device_count()))
            )
    
    nets = nn.ModuleDict({
        'policy': nn.ModuleDict({
            'obs_encoder': obs_encoder,
            'noise_pred_net': noise_pred_net
        })
    })
    
    # ... 其余代码 ...
```

#### 步骤 3: 修改 `train.py` 中的 DataLoader

在 `train.py` 中添加分布式采样器：

```python
# 在 train.py 中
from torch.utils.data.distributed import DistributedSampler

# ... 创建数据集的代码 ...

use_ddp = int(os.environ.get('WORLD_SIZE', 0)) > 1

if use_ddp:
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # 使用分布式采样器
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,  # 使用分布式采样器
        batch_size=config.train.batch_size,
        shuffle=False,  # 采样器已经处理了 shuffle
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
else:
    # 单 GPU 或 DataParallel 模式
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
```

#### 步骤 4: 启动分布式训练

使用 `torchrun`（推荐，PyTorch 1.9+）：

```bash
torchrun \
    --nproc_per_node=4 \
    --master_port=12355 \
    -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

或使用 `torch.distributed.launch`（PyTorch < 1.9）：

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12355 \
    -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

---

## 方法 3: 使用 torchrun（最简单）

`torchrun` 是 PyTorch 1.9+ 提供的工具，自动处理分布式训练的启动和故障恢复。

### 使用方法

#### 1. 创建启动脚本 `scripts/train_multi_gpu.sh`

```bash
#!/bin/bash

# 配置
NUM_GPUS=4
CONFIG_PATH="/path/to/your/config.json"
MASTER_PORT=12355

# 使用 torchrun 启动训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m robomimic.scripts.train_rlds \
    load_from=${CONFIG_PATH}
```

#### 2. 运行脚本

```bash
chmod +x scripts/train_multi_gpu.sh
./scripts/train_multi_gpu.sh
```

### 环境变量

`torchrun` 会自动设置以下环境变量：
- `RANK`: 当前进程的全局排名（0 到 world_size-1）
- `LOCAL_RANK`: 当前进程在节点内的排名（0 到 nproc_per_node-1）
- `WORLD_SIZE`: 总进程数（等于 nproc_per_node）
- `MASTER_ADDR`: 主节点地址（默认 localhost）
- `MASTER_PORT`: 主节点端口

### torchrun 的优势

- ✅ **自动故障恢复**：如果进程崩溃，自动重启
- ✅ **弹性训练**：支持动态添加/移除节点
- ✅ **简化启动**：不需要手动设置环境变量

---

## 配置建议

### 批次大小调整

使用多 GPU 时，**有效批次大小** = `batch_size × num_gpus`

**建议**：
- **选项 1**：保持单 GPU 批次大小，让有效批次大小增加
  - 单 GPU: `batch_size=128`
  - 4 GPU: `batch_size=128` → 有效批次 = 512
  - 优点：训练更快
  - 缺点：可能需要调整学习率

- **选项 2**：减小每个 GPU 的批次大小，保持有效批次大小不变
  - 单 GPU: `batch_size=128`
  - 4 GPU: `batch_size=32` → 有效批次 = 128
  - 优点：训练行为与单 GPU 一致
  - 缺点：可能无法充分利用 GPU

### 学习率调整

通常需要根据有效批次大小调整学习率：

```python
# 线性缩放规则（常用）
base_lr = 1e-4
num_gpus = 4
adjusted_lr = base_lr * num_gpus  # 4e-4

# 平方根缩放（更保守，适合大模型）
adjusted_lr = base_lr * (num_gpus ** 0.5)  # 2e-4

# 不缩放（适合小模型或预训练模型）
adjusted_lr = base_lr  # 1e-4
```

**经验法则**：
- 小模型（< 100M 参数）：线性缩放
- 大模型（> 1B 参数）：平方根缩放或不缩放
- 从预训练模型微调：通常不缩放

### 数据加载器工作进程

```python
# 每个 GPU 进程使用的工作进程数
num_workers_per_gpu = 4
total_workers = num_workers_per_gpu * num_gpus

# 注意：总工作进程数不应超过 CPU 核心数
# 例如：8 核 CPU，4 GPU，每个 GPU 2 个工作进程 = 8 个总进程
```

---

## 性能优化

### 1. 使用 NCCL 后端

确保使用 NCCL 后端（GPU 训练）：

```python
dist.init_process_group(backend='nccl', ...)
```

**为什么 NCCL？**
- 专为 GPU 设计，性能最佳
- 支持 GPU 间直接通信（不经过 CPU）
- 自动优化通信模式

### 2. 启用混合精度训练

在训练循环中使用 `torch.cuda.amp`：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
with autocast():
    loss = model.compute_loss(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优势**：
- 减少内存使用（约 50%）
- 加速训练（约 1.5-2x）
- 现代 GPU（V100+）支持良好

### 3. 优化 DataLoader

```python
# 使用 pin_memory 加速数据传输
train_loader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
    persistent_workers=True,  # 保持工作进程存活（减少启动开销）
    prefetch_factor=2  # 预取批次数量
)
```

### 4. 梯度累积

如果 GPU 内存不足，可以使用梯度累积：

```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model.compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**优势**：
- 模拟更大的批次大小
- 不需要更多 GPU 内存
- 保持训练稳定性

### 5. 使用 find_unused_parameters=False

如果模型的所有参数都参与反向传播：

```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

**优势**：
- 减少通信开销
- 加速训练

---

## 验证多 GPU 训练

### 检查 GPU 使用率

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 nvtop（如果已安装）
nvtop
```

**预期结果**：
- 所有 GPU 的使用率应该相似（DDP）
- 或 GPU 0 使用率稍高（DP）

### 检查训练日志

在训练日志中应该看到：
- 每个进程的 `RANK` 和 `LOCAL_RANK`
- 每个 GPU 都在处理数据
- 同步的梯度更新

### 性能指标

- **吞吐量**：应该接近线性扩展
  - 2 GPU ≈ 1.9x 单 GPU 速度
  - 4 GPU ≈ 3.7x 单 GPU 速度
  - 8 GPU ≈ 7.2x 单 GPU 速度

- **GPU 利用率**：所有 GPU 应该都在 80%+ 使用率

- **通信开销**：NCCL 通信时间应该 < 10% 总训练时间

---

## 故障排除

### 问题 1: NCCL 初始化失败

**错误**：
```
NCCL error: unhandled system error
NCCL error: initialization error
```

**解决方案**：
1. 确保所有 GPU 可见：`nvidia-smi`
2. 检查防火墙设置（NCCL 需要进程间通信）
3. 使用 `network_mode: host` 在 Docker 中
4. 设置环境变量：
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1  # 如果使用 InfiniBand
   ```

### 问题 2: 内存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小批次大小
2. 使用梯度累积
3. 启用混合精度训练
4. 减少模型大小或使用梯度检查点

### 问题 3: 数据加载成为瓶颈

**症状**：
- GPU 使用率低（< 50%）
- 训练速度没有随 GPU 数量线性增加

**解决方案**：
1. 增加 `num_workers`
2. 使用 `pin_memory=True`
3. 增加 `prefetch_factor`
4. 使用更快的存储（NVMe SSD）
5. 使用 `persistent_workers=True` 减少启动开销

### 问题 4: 进程同步失败

**错误**：
```
RuntimeError: Expected to have finished reduction in the prior iteration
```

**解决方案**：
1. 确保每个进程处理相同数量的批次
2. 使用 `drop_last=True` 在 DataLoader 中
3. 确保所有进程使用相同的随机种子

### 问题 5: 端口冲突

**错误**：
```
RuntimeError: Address already in use
```

**解决方案**：
1. 更改 `MASTER_PORT`：
   ```bash
   torchrun --master_port=12356 ...
   ```
2. 检查是否有其他训练进程在运行
3. 使用不同的端口范围（12355-12365）

---

## 示例：完整的多 GPU 训练命令

### 使用 DataParallel（当前实现）

```bash
# 直接运行，自动使用所有 GPU
python -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json
```

### 使用 DistributedDataParallel

```bash
# 在 Docker 容器中
cd /workspace/droid_policy_learning

# 4 GPU 训练
torchrun \
    --nproc_per_node=4 \
    --master_port=12355 \
    -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json \
    train.batch_size=128 \
    train.optim_params.policy.learning_rate.initial=4e-4

# 8 GPU 训练
torchrun \
    --nproc_per_node=8 \
    --master_port=12355 \
    -m robomimic.scripts.train_rlds \
    load_from=/path/to/config.json \
    train.batch_size=64 \
    train.optim_params.policy.learning_rate.initial=8e-4
```

---

## 总结

### DataParallel vs DistributedDataParallel

| 方面 | DataParallel | DistributedDataParallel |
|------|-------------|------------------------|
| **实现** | 简单（一行代码） | 需要分布式初始化 |
| **性能** | 较慢，2-4 GPU 可用 | 更快，线性扩展 |
| **适用场景** | 快速测试，小规模训练 | 生产环境，大规模训练 |
| **推荐使用** | 2-4 GPU，快速原型 | 4+ GPU，生产训练 |

### 选择建议

- **快速测试**：使用现有的 DataParallel（自动启用）
- **生产训练**：实现 DistributedDataParallel + torchrun
- **大规模训练**：必须使用 DistributedDataParallel

### 关键要点

1. ✅ **DataParallel** 适合快速测试，但效率较低
2. ✅ **DistributedDataParallel** 是生产环境的标准选择
3. ✅ **torchrun** 是最简单的启动方式
4. 📝 **配置调整**：根据 GPU 数量调整批次大小和学习率
5. 🚀 **性能优化**：使用混合精度、优化 DataLoader、梯度累积

---

## 参考资料

- [PyTorch DistributedDataParallel 文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DataParallel 文档](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [torchrun 文档](https://pytorch.org/docs/stable/elastic/run.html)
