# Apptainer / Singularity 使用说明

构建逻辑与 **`docker/Dockerfile`** 对齐：`uv` + **`pyproject.toml` / `uv.lock`**（`uv sync --frozen`），不再使用 micromamba 与 `requirements-apptainer.txt`。

**必须在仓库根目录执行 `apptainer build`**（`%files` 中的路径相对于构建时的当前工作目录）。

需要 **Apptainer ≥ 1.2**（支持 `%arguments` / `--build-arg`）。若版本过旧，请升级或手动改 `droid_policy.def` 中的 `INCLUDE_ROBOCASA` 分支。

## 构建镜像

```bash
cd /path/to/droid_policy_learning

# 与 docker compose 默认一致（不含 RoboCasa 编译变体）
apptainer build droid_policy.sif apptainer/droid_policy.def

# 与 docker build --build-arg INCLUDE_ROBOCASA=1 一致
apptainer build --build-arg INCLUDE_ROBOCASA=1 droid_policy_robocasa.sif apptainer/droid_policy.def
```

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
