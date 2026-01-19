# Docker 故障排除指南

本文档涵盖了使用 Docker 进行 DROID 策略训练时可能遇到的常见问题和警告。

## 目录

- [GPU 访问问题](#gpu-访问问题)
- [TensorFlow 警告](#tensorflow-警告)
- [CUDA 版本兼容性](#cuda-版本兼容性)
- [常见错误和解决方案](#常见错误和解决方案)

---

## GPU 访问问题

### 检查 GPU 是否可访问

在容器内运行以下命令检查 GPU 状态：

```bash
# 检查 nvidia-smi
docker exec <container_name> nvidia-smi

# 检查 PyTorch GPU 访问
docker exec <container_name> micromamba run -n droid_env python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 预期输出

如果配置正确，应该看到：
- `nvidia-smi` 显示 GPU 信息
- PyTorch 报告 `CUDA available: True` 和 `GPU count: 1`（或更多）

### 如果 GPU 不可访问

1. **检查 Docker 是否支持 NVIDIA 运行时**：
   ```bash
   docker info | grep -i runtime
   ```
   应该看到 `nvidia` 在运行时列表中。

2. **检查 nvidia-container-toolkit**：
   ```bash
   which nvidia-container-runtime
   ```

3. **验证 docker-compose 配置**：
   确保 `docker-compose.yaml` 中包含正确的 GPU 配置：
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu, compute, utility]
   ```

4. **重启 Docker 服务**（如果需要）：
   ```bash
   sudo systemctl restart docker
   ```

---

## TensorFlow 警告

### 常见警告信息

在训练启动时，您可能会看到以下 TensorFlow 警告：

```
Could not find cuda drivers on your machine, GPU will not be used.
Cannot dlopen some GPU libraries.
Unable to register cuDNN/cuFFT/cuBLAS factory
TF-TRT Warning: Could not find TensorRT
```

### 是否需要解决？

**答案：通常不需要立即解决**

#### 原因分析

1. **训练使用 PyTorch，不是 TensorFlow**
   - DROID 策略训练完全基于 PyTorch（`diffusion_policy` 算法）
   - TensorFlow 仅作为 Octo 库的依赖项安装，训练流程不直接使用

2. **PyTorch GPU 正常工作**
   - 已验证 PyTorch 可以正常访问 GPU
   - 训练会在 GPU 上执行，性能不受影响

3. **警告影响评估**

| 警告类型 | 严重性 | 是否影响训练 | 是否需要修复 |
|---------|--------|------------|------------|
| `Could not find cuda drivers` (TensorFlow) | 低 | 否 | 否 |
| `Cannot dlopen GPU libraries` (TensorFlow) | 低 | 否 | 否 |
| `TF-TRT Warning` | 低 | 否 | 否（可选优化） |
| cuDNN/cuFFT/cuBLAS 注册错误 | 低 | 否 | 否 |

### 根本原因

- **CUDA 版本不匹配**：
  - 容器使用 CUDA 11.8（与 PyTorch 2.0.1 兼容）
  - TensorFlow 2.15.0 需要 CUDA 12.x
  - 这导致 TensorFlow 无法访问 GPU，但 PyTorch 正常工作

### 如果将来需要修复（可选）

如果将来需要使用 Octo 的 TensorFlow GPU 功能，可以考虑：

#### 选项 1：升级到 CUDA 12.x

修改 `docker/Dockerfile`：
```dockerfile
FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04
# 更新 PyTorch 安装命令以匹配 CUDA 12.x
RUN pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### 选项 2：降级 TensorFlow

修改 `docker/Dockerfile`：
```dockerfile
# 使用支持 CUDA 11.8 的 TensorFlow 版本
RUN pip install "tensorflow==2.13.0"
```

**注意**：这些更改可能会影响其他依赖项的兼容性，请谨慎操作。

---

## CUDA 版本兼容性

### 当前配置

- **主机 CUDA 版本**：12.4（通过 `nvidia-smi` 查看）
- **容器 CUDA 版本**：11.8（在 Dockerfile 中指定）
- **PyTorch CUDA 版本**：11.8（与容器匹配）
- **TensorFlow CUDA 要求**：12.x（不匹配，导致警告）

### 版本兼容性表

| 组件 | 版本 | CUDA 要求 | 状态 |
|------|------|-----------|------|
| PyTorch | 2.0.1+cu118 | CUDA 11.8 | ✅ 兼容 |
| TensorFlow | 2.15.0 | CUDA 12.x | ⚠️ 不兼容（但不影响训练） |
| 容器基础镜像 | CUDA 11.8 | CUDA 11.8 | ✅ 兼容 |
| 主机驱动 | 550.54.15 | CUDA 12.4 | ✅ 向后兼容 |

### 为什么可以工作？

- NVIDIA 驱动（550.54.15）向后兼容 CUDA 11.8 和 12.4
- PyTorch 使用容器内的 CUDA 11.8 库，与驱动兼容
- TensorFlow 无法找到匹配的 CUDA 库，但训练不使用 TensorFlow

---

## 常见错误和解决方案

### 错误 1：容器启动失败

**症状**：
```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**解决方案**：
1. 确保安装了 `nvidia-container-toolkit`：
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. 验证 Docker 配置：
   ```bash
   cat /etc/docker/daemon.json
   ```
   应该包含：
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### 错误 2：共享内存不足

**症状**：
```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal: Bus error
```

**解决方案**：
确保 `docker-compose.yaml` 中设置了 `ipc: host`：
```yaml
ipc: host
```

如果无法使用 `ipc: host`，增加共享内存大小：
```yaml
shm_size: 32gb
```

### 错误 3：内存锁定失败

**症状**：
```
CUDA error: out of memory
```

**解决方案**：
确保 `docker-compose.yaml` 中设置了：
```yaml
ulimits:
  memlock: -1
```

### 错误 4：文件描述符限制

**症状**：
```
OSError: [Errno 24] Too many open files
```

**解决方案**：
在 `docker-compose.yaml` 中增加文件描述符限制：
```yaml
ulimits:
  nofile:
    soft: 65536
    hard: 65536
```

---

## 验证配置

运行以下命令验证 Docker 配置是否正确：

```bash
# 1. 检查 GPU 访问
docker exec <container_name> nvidia-smi

# 2. 检查 PyTorch
docker exec <container_name> micromamba run -n droid_env python -c "import torch; assert torch.cuda.is_available(), 'GPU not available!'; print('✅ PyTorch GPU: OK')"

# 3. 检查 CUDA 版本
docker exec <container_name> micromamba run -n droid_env python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# 4. 检查共享内存（如果使用 ipc: host，应该显示主机内存）
docker exec <container_name> df -h /dev/shm
```

---

## 性能优化建议

### 1. 使用 Host 网络模式

```yaml
network_mode: host
```

**优势**：
- 消除 Docker 网桥开销
- 对 NCCL 分布式训练至关重要
- 低延迟 API 服务

### 2. 使用 Host IPC 模式

```yaml
ipc: host
```

**优势**：
- 无限制共享内存访问
- PyTorch DataLoader 性能最佳
- 避免共享内存不足问题

### 3. 使用 Host PID 模式

```yaml
pid: host
```

**优势**：
- 监控工具（nvtop, htop, wandb）可以正确显示系统资源
- 便于调试和性能分析

### 4. 挂载 Hugging Face Cache

```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

**优势**：
- 避免重复下载模型
- 节省时间和带宽

---

## 获取帮助

如果遇到其他问题：

1. 检查 [Docker 官方文档](https://docs.docker.com/)
2. 查看 [NVIDIA Container Toolkit 文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
3. 检查项目 [GitHub Issues](https://github.com/droid-dataset/droid_policy_learning/issues)

---

## 总结

- ✅ **PyTorch GPU 访问正常**：训练会在 GPU 上执行
- ⚠️ **TensorFlow 警告可以忽略**：不影响训练性能
- ✅ **当前配置已优化**：遵循 HPC/LLM 训练最佳实践
- 📝 **文档已更新**：包含所有常见问题和解决方案

如果训练正常运行且 PyTorch 可以访问 GPU，您可以安全地忽略 TensorFlow 相关的警告。
