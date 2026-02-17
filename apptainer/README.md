# Apptainer / Singularity 使用说明

## 构建镜像

```bash
cd /path/to/droid_policy_learning
apptainer build droid_policy.sif apptainer/droid_policy.def
```

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
