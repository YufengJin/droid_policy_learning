#!/bin/bash -l
#SBATCH --job-name=droid_train       # 任务名称
#SBATCH --output=logs/job_%j.out     # 标准输出日志 (%j 代表 Job ID)
#SBATCH --error=logs/job_%j.err      # 错误日志
#SBATCH --nodes=1                    # 节点数
#SBATCH --ntasks=1                   # 任务数
#SBATCH --cpus-per-task=8            # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:a100:1            # 请求 1 个 A100 GPU (根据集群改为 a40 或 a100)
#SBATCH --time=04:00:00              # 运行时间限制 (hh:mm:ss)
#SBATCH --partition=a100             # 分区名称 (例如 a100, a40, tiny_gpu 等)

# -----------------------------------------------------------------------
# 1. 配置路径变量（与 run_container.sh 一致，支持环境变量覆盖）
# -----------------------------------------------------------------------
CONTAINER_IMAGE="${APPTAINER_IMAGE:-./droid_policy.sif}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 宿主机路径
HOST_CODE_DIR="${DROID_PROJECT_DIR:-$PROJECT_ROOT}"
HOST_DROID_DATA="${DROID_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/droid}"
HOST_ROBOCASA_DATA="${ROBOCASA_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/robocasa}"
HOST_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# 宿主机 TMPDIR 映射到容器 /tmp，避免容器 overlay 空间不足
HOST_TMPDIR="${TMPDIR:-/tmp}"

# 容器内路径
CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_DROID="/workspace/datasets/droid"
CONT_ROBOCASA="/workspace/datasets/robocasa"
CONT_HF="/root/.cache/huggingface"

# -----------------------------------------------------------------------
# 2. 准备 Apptainer Bind 参数
# -----------------------------------------------------------------------
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK:-$HOME}/.apptainer_cache}"

BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR}"
BIND_ARGS="$BIND_ARGS -B ${HOST_TMPDIR}:/tmp"
[ -d "$HOST_DROID_DATA" ]    && BIND_ARGS="$BIND_ARGS -B ${HOST_DROID_DATA}:${CONT_DROID}:ro"
[ -d "$HOST_ROBOCASA_DATA" ] && BIND_ARGS="$BIND_ARGS -B ${HOST_ROBOCASA_DATA}:${CONT_ROBOCASA}:ro"
[ -d "$HOST_HF_CACHE" ]      && BIND_ARGS="$BIND_ARGS -B ${HOST_HF_CACHE}:${CONT_HF}"

echo "Job running on node: $(hostname)"
echo "Binding: $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "Binding: $HOST_TMPDIR -> /tmp (TMPDIR)"
[ -d "$HOST_DROID_DATA" ]    && echo "Binding: $HOST_DROID_DATA -> $CONT_DROID (ro)"
[ -d "$HOST_ROBOCASA_DATA" ] && echo "Binding: $HOST_ROBOCASA_DATA -> $CONT_ROBOCASA (ro)"

# -----------------------------------------------------------------------
# 3. 执行任务
# -----------------------------------------------------------------------
WORK_DIR="/workspace/droid_policy_learning"

srun apptainer exec --nv -C --pwd "$WORK_DIR" $BIND_ARGS $CONTAINER_IMAGE \
    python3 /workspace/droid_policy_learning/train.py \
    --config-name=my_config
