# Apptainer / Singularity 使用说明

构建逻辑与 **`docker/Dockerfile`** 对齐：`uv` + **`pyproject.toml` / `uv.lock`**（`uv sync --frozen`），不再使用 micromamba 与 `requirements-apptainer.txt`。

## 构建镜像

### 前置条件

- **必须在仓库根目录执行** `apptainer build`（`%files` 中的路径相对于构建时的当前工作目录）。
- 构建前确认以下文件存在：
  - `pyproject.toml`
  - `uv.lock`
  - `docker/entrypoint.sh`
  - `apptainer/droid_policy.def`
- **Apptainer ≥ 1.2**（支持 `%arguments` / `--build-arg`）。若使用旧版 Singularity，需升级或手动改 `droid_policy.def` 中的 `INCLUDE_ROBOCASA` 分支。

### 与 Docker 的对应关系

| Docker 构建方式 | Apptainer 等效 |
|----------------|----------------|
| `docker compose -f docker/docker-compose.headless.yaml build` | `apptainer build droid_policy.sif apptainer/droid_policy.def` |
| `docker build --build-arg INCLUDE_ROBOCASA=1 -f docker/Dockerfile .` | `apptainer build --build-arg INCLUDE_ROBOCASA=1 droid_policy_robocasa.sif apptainer/droid_policy.def` |

### 基础构建命令

```bash
cd /path/to/droid_policy_learning

# 与 docker compose 默认一致（不含 RoboCasa 编译变体）
apptainer build droid_policy.sif apptainer/droid_policy.def

# 与 docker build --build-arg INCLUDE_ROBOCASA=1 一致
apptainer build --build-arg INCLUDE_ROBOCASA=1 droid_policy_robocasa.sif apptainer/droid_policy.def
```

### 无特权构建（集群常见）

若本机无法 root，使用 `--fakeroot`：

```bash
cd /path/to/droid_policy_learning
apptainer build --fakeroot droid_policy.sif apptainer/droid_policy.def
```

需集群启用 fakeroot；否则由管理员在有 root 的节点构建，或参考集群文档使用 remote build 服务。

### 网络与代理

构建阶段需访问 `astral.sh`（uv 安装）、`github.com`（Octo / robosuite）、PyPI（`uv sync`）。若集群需代理：

```bash
export http_proxy=http://your-proxy:80
export https_proxy=http://your-proxy:80
cd /path/to/droid_policy_learning
apptainer build --fakeroot droid_policy.sif apptainer/droid_policy.def
```

### 磁盘与缓存

`uv sync` 与 wheel 缓存会占用较多磁盘。若构建节点 /home 配额不足，可把缓存指向大盘：

```bash
export TMPDIR=/scratch/your-user/tmp
export APPTAINER_CACHEDIR=/scratch/your-user/.apptainer_cache
```

### 验证构建成功

```bash
apptainer exec droid_policy.sif /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"
```

或挂载项目后验证 torch + octo：

```bash
apptainer exec -B "$PWD:/workspace/droid_policy_learning" --pwd /workspace/droid_policy_learning \
  droid_policy.sif python -c "import torch, octo; print('torch', torch.__version__, 'octo ok')"
```

### 故障排查

| 现象 | 可能原因 | 处理建议 |
|------|----------|----------|
| `Could not copy` / `No such file` | 构建 cwd 不在仓库根目录 | `cd` 到包含 `pyproject.toml` 的根目录后重试 |
| `git clone` / `curl` 失败 | 无外网或代理未配置 | 设置 `http_proxy`、`https_proxy` 或改用有网节点 |
| `uv sync` 失败 | 网络问题，或本地 `uv.lock` 与仓库不一致 | 检查网络/代理；执行 `uv lock` 后重新构建 |
| 权限错误 / `permission denied` | 无 root | 加 `--fakeroot`，或由管理员构建 |
| `Unknown build argument` | Apptainer 版本过旧 | 升级到 Apptainer ≥ 1.2，或手动改 def 内 `INCLUDE_ROBOCASA` |

---

运行时入口为 **`docker/entrypoint.sh`**（复制到镜像内 `/usr/local/bin/entrypoint.sh`）：启动时会对挂载的仓库执行 `uv pip install -e .`；若镜像以 `INCLUDE_ROBOCASA=1` 构建，还会按 entrypoint 逻辑安装 RoboCasa 并拉取资产。

## 在 Apptainer 中 Bind Datasets

### 基本语法

```bash
apptainer run -B <宿主机路径>:<容器路径>[:ro] image.sif [命令]
```

- `-B src:dst`：将宿主机 `src` 挂载到容器内 `dst`
- `:ro`：只读挂载（可选，推荐用于数据集）
- 多个挂载：`-B src1:dst1 -B src2:dst2` 或 `-B src1:dst1,src2:dst2`

### 路径对应（与 Docker 一致）

| 宿主机路径 | 容器内路径 | 说明 |
|-----------|-----------|------|
| 项目根目录 | `/workspace/droid_policy_learning` | 项目代码 |
| DROID 数据集 | `/workspace/datasets/droid` | RLDS 格式数据 |
| RoboCasa 数据集 | `/workspace/datasets/robocasa` | HDF5 数据 |
| LIBERO 数据集 | `/workspace/datasets/libero` | HDF5（与 slurm 脚本一致） |
| HF 缓存 | `/root/.cache/huggingface` | 避免重复下载模型 |

### 方式一：使用 run_container.sh（推荐）

脚本已预设 bind，修改脚本内变量或通过**环境变量**覆盖：

```bash
cd /path/to/droid_policy_learning/apptainer

# 自定义数据集路径（不存在则不会挂载）
export DROID_DATASET_DIR=/home/xxx/datasets/droid
export ROBOCASA_DATASET_DIR=/home/xxx/datasets/robocasa
export HF_HOME=$HOME/.cache/huggingface

./run_container.sh              # 交互式 bash
./run_container.sh python train.py   # 运行命令
```

### 方式二：手动指定 bind

```bash
apptainer run --nv -C \
  -B /path/to/droid_policy_learning:/workspace/droid_policy_learning \
  -B /path/to/droid_data:/workspace/datasets/droid:ro \
  -B /path/to/robocasa_data:/workspace/datasets/robocasa:ro \
  -B $HOME/.cache/huggingface:/root/.cache/huggingface \
  --pwd /workspace/droid_policy_learning \
  droid_policy.sif bash
```

### 方式三：使用 bind 配置文件

创建 `apptainer-binds.txt`：

```
/path/to/droid_policy_learning /workspace/droid_policy_learning
/path/to/droid_data /workspace/datasets/droid ro
/path/to/robocasa_data /workspace/datasets/robocasa ro
```

运行：

```bash
apptainer run --nv -B apptainer-binds.txt droid_policy.sif bash
```

### 常用选项

- `--nv`：启用 NVIDIA GPU
- `-C` / `--contain`：隔离宿主机环境
- `--pwd <目录>`：容器内工作目录
