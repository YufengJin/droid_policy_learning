#!/bin/bash
# =======================================================================
# 脚本功能：交互式启动 Droid Policy Learning 容器
# 用法: ./run_container.sh [命令...]  默认进入 bash
#
# Apptainer Bind Mounts 用法（参考 docker-compose）:
#   -B 宿主机路径:容器路径[:ro]   ro=只读
#   多个挂载: -B src1:dst1 -B src2:dst2  或  -B src1:dst1,src2:dst2
#
# 数据集绑定：修改下方 HOST_* 变量即可
# =======================================================================

# 1. 配置路径变量
CONTAINER_IMAGE="${APPTAINER_IMAGE:-./droid_policy.sif}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 宿主机路径 (Host Paths) - 按需修改
HOST_CODE_DIR="${DROID_PROJECT_DIR:-$PROJECT_ROOT}"
HOST_DROID_DATA="${DROID_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/droid}"
HOST_ROBOCASA_DATA="${ROBOCASA_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/robocasa}"
HOST_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# 可选：参考代码仓库（类似 docker headless 中的 cosmos-policy, robocasa）
HOST_COSMOS_POLICY="${COSMOS_POLICY_DIR:-}"
HOST_ROBOCASA_REPO="${ROBOCASA_REPO_DIR:-}"

# Fake Root（用于隔离 $HOME，避免宿主机配置污染 conda 等）
HOST_FAKE_ROOT="${APPTAINER_FAKE_ROOT:-${WORK:-$HOME}/.apptainer_fake_root}"

# 宿主机 TMPDIR 映射到容器 /tmp，避免容器 overlay 空间不足 (No space left on device)
HOST_TMPDIR="${TMPDIR:-/tmp}"

# 容器内路径 (Container Paths)
CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_DROID="/workspace/datasets/droid"
CONT_ROBOCASA="/workspace/datasets/robocasa"
CONT_HF="/root/.cache/huggingface"

# 2. 检查镜像
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Error: 镜像文件 $CONTAINER_IMAGE 未找到！"
    echo "请先构建: apptainer build droid_policy.sif apptainer/droid_policy.def"
    exit 1
fi

# 3. 准备 Apptainer 参数
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK:-$HOME}/.apptainer_cache}"

# 确保 Fake Root、TMPDIR、HF cache 存在
mkdir -p "$HOST_FAKE_ROOT" 2>/dev/null || true
mkdir -p "$HOST_TMPDIR" 2>/dev/null || true
mkdir -p "$HOST_HF_CACHE" 2>/dev/null || true

# 4. 构造 Bind 参数
# 格式: -B 宿主机路径:容器路径 或 -B 宿主机路径:容器路径:ro (只读)
BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR}"
BIND_ARGS="$BIND_ARGS -B ${HOST_FAKE_ROOT}:/root"
BIND_ARGS="$BIND_ARGS -B ${HOST_TMPDIR}:/tmp"

# DROID 数据集（存在则挂载）
if [ -d "$HOST_DROID_DATA" ]; then
    BIND_ARGS="$BIND_ARGS -B ${HOST_DROID_DATA}:${CONT_DROID}:ro"
fi

# RoboCasa 数据集（存在则挂载）
if [ -d "$HOST_ROBOCASA_DATA" ]; then
    BIND_ARGS="$BIND_ARGS -B ${HOST_ROBOCASA_DATA}:${CONT_ROBOCASA}:ro"
fi

# Hugging Face cache（避免重复下载模型）
if [ -d "$HOST_HF_CACHE" ]; then
    BIND_ARGS="$BIND_ARGS -B ${HOST_HF_CACHE}:${CONT_HF}"
fi

# 可选：参考仓库
[ -n "$HOST_COSMOS_POLICY" ] && [ -d "$HOST_COSMOS_POLICY" ] && \
    BIND_ARGS="$BIND_ARGS -B ${HOST_COSMOS_POLICY}:/workspace/cosmos-policy:ro"
[ -n "$HOST_ROBOCASA_REPO" ] && [ -d "$HOST_ROBOCASA_REPO" ] && \
    BIND_ARGS="$BIND_ARGS -B ${HOST_ROBOCASA_REPO}:/workspace/robocasa:ro"

# 5. 启动容器
WORK_DIR="/workspace/droid_policy_learning"
CMD="${@:-bash}"

echo ">> Apptainer bind mounts:"
echo ">>   $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo ">>   $HOST_TMPDIR -> /tmp (TMPDIR)"
[ -d "$HOST_DROID_DATA" ]    && echo ">>   $HOST_DROID_DATA -> $CONT_DROID (ro)"
[ -d "$HOST_ROBOCASA_DATA" ] && echo ">>   $HOST_ROBOCASA_DATA -> $CONT_ROBOCASA (ro)"
[ -d "$HOST_HF_CACHE" ]      && echo ">>   $HOST_HF_CACHE -> $CONT_HF"

exec apptainer run --nv -C --pwd "$WORK_DIR" $BIND_ARGS "$CONTAINER_IMAGE" $CMD
