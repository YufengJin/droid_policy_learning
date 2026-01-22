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
# 1. 配置路径变量
# -----------------------------------------------------------------------
# 镜像位置 (建议放绝对路径，或者放在 $WORK 下)
CONTAINER_IMAGE="./droid_policy.sif"

# 宿主机路径 (Host Paths)
HOST_CODE_DIR="/home/hpc/g108ea/g108ea10/repos/droid_policy_learning"
HOST_DATA_DIR="/home/atuin/g108ea/g108ea10/datasets/droid"

# 容器内路径 (Container Paths)
CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_DATA_DIR="/workspace/datasets/droid"

# -----------------------------------------------------------------------
# 2. 准备 Apptainer 参数
# -----------------------------------------------------------------------
# 缓存设置 (根据文档建议 [cite: 73]，指向 $WORK 防止爆 $HOME)
export APPTAINER_CACHEDIR=$WORK/.apptainer_cache

# 构造挂载参数 (-B src:dst)
# 注意：多个挂载点用逗号分隔，或者写多个 -B
BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR} -B ${HOST_DATA_DIR}:${CONT_DATA_DIR}"

# 打印调试信息
echo "Job running on node: $(hostname)"
echo "Using Container: $CONTAINER_IMAGE"
echo "Binding Code: $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "Binding Data: $HOST_DATA_DIR -> $CONT_DATA_DIR"

# -----------------------------------------------------------------------
# 3. 执行任务
# -----------------------------------------------------------------------
# --nv: 启用 NVIDIA GPU 支持 [cite: 61, 67]
# -C (--contain): 隔离宿主机环境，防止 /home 配置污染 
# --pwd: 设置工作目录，避免工作目录警告
# exec: 执行具体命令 (比 run 更灵活)

WORK_DIR="/workspace/droid_policy_learning"

srun apptainer exec --nv -C --pwd "$WORK_DIR" $BIND_ARGS $CONTAINER_IMAGE \
    python3 /workspace/droid_policy_learning/train.py \
    --config-name=my_config