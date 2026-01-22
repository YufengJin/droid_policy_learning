#!/bin/bash

# =======================================================================
# 脚本功能：交互式启动 Droid Policy Learning 容器
# =======================================================================

# 1. 配置路径变量
CONTAINER_IMAGE="./droid_policy.sif"
HOST_CODE_DIR="/home/hpc/g108ea/g108ea10/repos/droid_policy_learning"
HOST_DATA_DIR="/home/atuin/g108ea/g108ea10/datasets/droid"

# [NEW] 定义宿主机上的 Fake Root 目录
HOST_FAKE_ROOT="$WORK/.apptainer_fake_root"

CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_DATA_DIR="/workspace/datasets/droid"

# 2. 检查镜像
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Error: 镜像文件 $CONTAINER_IMAGE 未找到！"
    exit 1
fi

# 3. 准备 Apptainer 参数
export APPTAINER_CACHEDIR=$WORK/.apptainer_cache

# [NEW] 确保 Fake Root 存在
if [ ! -d "$HOST_FAKE_ROOT" ]; then
    echo ">> [Init] Creating fake root directory at: $HOST_FAKE_ROOT"
    mkdir -p "$HOST_FAKE_ROOT"
fi

# [NEW] 构造挂载参数 (-B src:dst)
# 关键：将宿主机的 FAKE_ROOT 挂载到容器的 /root
BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
           -B ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
           -B ${HOST_FAKE_ROOT}:/root"

# 打印调试信息
echo ">> [DEBUG] Starting Apptainer..."
echo ">> [DEBUG] Fake Root: $HOST_FAKE_ROOT -> /root"

# 4. 启动容器
# --nv: GPU支持
# -C: 隔离宿主机
# --pwd: 指定工作目录
WORK_DIR="/workspace/droid_policy_learning"

# 传递所有参数，默认为 bash
CMD="${@:-bash}"

apptainer run --nv -C --pwd "$WORK_DIR" $BIND_ARGS "$CONTAINER_IMAGE" $CMD