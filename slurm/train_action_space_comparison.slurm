#!/bin/bash -l
#SBATCH --job-name=aspace_%a
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100_40
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yufeng.jin@tu-darmstadt.de
#SBATCH --export=NONE
#SBATCH --array=0-17%6

################################################################################
# Action Space Comparison — SLURM Array Job
#
# 每个 array task 运行一个 (dataset, action_space, seed) 组合。
# 共 18 tasks: 2 datasets × 3 action_spaces × 3 seeds
#
# Task ID 映射:
#   ID = dataset_idx * 9 + space_idx * 3 + seed_idx
#   dataset:  [libero, robocasa]                    (0-1)
#   space:    [pos_euler, pos_rot6d, pos_axisangle]  (0-2)
#   seed:     [1, 42, 123]                          (0-2)
#
# 用法:
#   sbatch slurm/train_action_space_comparison.slurm                          # 全部 18 runs
#   sbatch --array=0-8%3  slurm/train_action_space_comparison.slurm           # 仅 LIBERO (9 runs)
#   sbatch --array=9-17%3 slurm/train_action_space_comparison.slurm           # 仅 RoboCasa (9 runs)
#   sbatch --array=4      slurm/train_action_space_comparison.slurm           # 单次: libero pos_rot6d seed=42
#
# 或用环境变量覆盖（忽略 array ID 的映射）:
#   sbatch --array=0 --export=DATASET=libero,ACTION_SPACE=pos_rot6d,SEED=42 \
#       slurm/train_action_space_comparison.slurm
################################################################################
set -euo pipefail

mkdir -p logs
echo "[$(date +'%F %T')] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}  NODELIST=${SLURM_JOB_NODELIST:-NA}"

trap 'status=$?; now=$(date "+%F %T"); echo "[EXIT $now] code=$status"; [[ $status -ne 0 ]] && env | sort | grep -E "SLURM|CUDA|TMPDIR" || true' EXIT

################################################################################
# 1) 代理设置
################################################################################
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

################################################################################
# 2) 作业环境
################################################################################
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export nointeractive=yes

################################################################################
# 3) 从 SLURM_ARRAY_TASK_ID 解码实验参数
################################################################################
DATASETS=(libero robocasa)
ACTION_SPACES=(pos_euler pos_rot6d pos_axisangle)
SEEDS=(1 42 123)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

dataset_idx=$(( TASK_ID / 9 ))
remainder=$(( TASK_ID % 9 ))
space_idx=$(( remainder / 3 ))
seed_idx=$(( remainder % 3 ))

# 环境变量覆盖（优先级高于 array ID 映射）
DATASET="${DATASET:-${DATASETS[$dataset_idx]}}"
ACTION_SPACE="${ACTION_SPACE:-${ACTION_SPACES[$space_idx]}}"
SEED="${SEED:-${SEEDS[$seed_idx]}}"

NUM_EPOCHS="${NUM_EPOCHS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WANDB_PROJ="action_space_comparison"

echo "[INFO] Experiment: dataset=${DATASET} action_space=${ACTION_SPACE} seed=${SEED} epochs=${NUM_EPOCHS} batch=${BATCH_SIZE}"

################################################################################
# 4) 路径配置
#    镜像与代码在 vault 时：请从仓库根目录 sbatch，使 SLURM_SUBMIT_DIR 指向 vault。
#    亦可 export DROID_PROJECT_DIR=... / APPTAINER_IMAGE=... 覆盖。
################################################################################
SCRIPT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_SLURM_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-/home/hpc/g108ea/g108ea10/repos/droid_policy_learning}"
PROJECT_ROOT="${DROID_PROJECT_DIR:-$WORK_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# 数据集主机路径
LIBERO_HOST_ROOT="${LIBERO_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/libero}"
ROBOCASA_HOST_ROOT="${ROBOCASA_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/robocasa}"

# 容器内路径
CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_LIBERO="/workspace/datasets/libero"
CONT_ROBOCASA="/workspace/datasets/robocasa"
CONT_HF="/root/.cache/huggingface"

# 容器内数据子路径
CONT_LIBERO_DATA="${CONT_LIBERO}/libero_10"
CONT_ROBOCASA_DATA="${CONT_ROBOCASA}/v0.1/single_stage/kitchen_pnp"

################################################################################
# 5) Apptainer 配置
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${SCRIPT_DIR}/.env.wandb" ]] && source "${SCRIPT_DIR}/.env.wandb"
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

CONTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_ROOT/droid_policy.sif}"
HOST_CODE_DIR="${DROID_PROJECT_DIR:-$PROJECT_ROOT}"
HOST_FAKE_ROOT="${APPTAINER_FAKE_ROOT:-${WORK:-$HOME}/.apptainer_fake_root}"
HOST_TMPDIR="${TMPDIR:-/tmp}"
HOST_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "[ERR] Container image not found: $CONTAINER_IMAGE"
    echo "请先构建: apptainer build droid_policy.sif apptainer/droid_policy.def"
    exit 1
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK:-$HOME}/.apptainer_cache}"
mkdir -p "$HOST_FAKE_ROOT" 2>/dev/null || true
mkdir -p "$HOST_HF_CACHE" 2>/dev/null || true

# 基础 bind mounts
BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR}"
BIND_ARGS="$BIND_ARGS -B ${HOST_FAKE_ROOT}:/root"
BIND_ARGS="$BIND_ARGS -B ${HOST_TMPDIR}:/tmp"
[ -d "$HOST_HF_CACHE" ] && BIND_ARGS="$BIND_ARGS -B ${HOST_HF_CACHE}:${CONT_HF}"

[[ -n "${http_proxy:-}" ]] && export APPTAINERENV_http_proxy="${http_proxy}" APPTAINERENV_https_proxy="${https_proxy:-$http_proxy}"

################################################################################
# 6) 按 dataset 分发训练
################################################################################
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

case "$DATASET" in
    libero)
        if [[ ! -d "$LIBERO_HOST_ROOT" ]]; then
            echo "[ERR] LIBERO dataset not found: $LIBERO_HOST_ROOT"; exit 2
        fi
        BIND_ARGS="$BIND_ARGS -B ${LIBERO_HOST_ROOT}:${CONT_LIBERO}:ro"
        EXP_NAME="libero10_${ACTION_SPACE}_s${SEED}"
        TRAIN_MODULE="robomimic.scripts.train_libero"
        DATA_ARG="train.data=[{path: ${CONT_LIBERO_DATA}}]"
        ;;
    robocasa)
        if [[ ! -d "$ROBOCASA_HOST_ROOT" ]]; then
            echo "[ERR] RoboCasa dataset not found: $ROBOCASA_HOST_ROOT"; exit 2
        fi
        BIND_ARGS="$BIND_ARGS -B ${ROBOCASA_HOST_ROOT}:${CONT_ROBOCASA}:ro"
        EXP_NAME="robocasa_pnp_${ACTION_SPACE}_s${SEED}"
        TRAIN_MODULE="robomimic.scripts.train_robocasa"
        DATA_ARG="train.data=[{path: ${CONT_ROBOCASA_DATA}}]"
        ;;
    *)
        echo "[ERR] Unknown dataset: $DATASET (expected libero or robocasa)"; exit 1
        ;;
esac

echo "[INFO] Apptainer bind mounts:"
echo "  $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "  $HOST_TMPDIR -> /tmp"
[ -d "$HOST_HF_CACHE" ] && echo "  $HOST_HF_CACHE -> $CONT_HF"

echo ""
echo "============================================================"
echo "  Experiment: $EXP_NAME"
echo "  Module:     $TRAIN_MODULE"
echo "  Action:     $ACTION_SPACE"
echo "  Seed:       $SEED"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Batch:      $BATCH_SIZE"
echo "  WandB:      $WANDB_PROJ"
echo "============================================================"
echo ""

################################################################################
# 7) 启动训练
################################################################################
echo "[$(date +'%F %T')] Starting training: $EXP_NAME"

apptainer run --nv -C --writable-tmpfs --pwd "$CONT_CODE_DIR" $BIND_ARGS "$CONTAINER_IMAGE" \
    torchrun --nproc_per_node=4 -m "$TRAIN_MODULE" \
    "$DATA_ARG" \
    train.output_dir="${CONT_CODE_DIR}/outputs" \
    train.action_space="$ACTION_SPACE" \
    train.seed="$SEED" \
    train.num_epochs=$NUM_EPOCHS \
    train.batch_size=$BATCH_SIZE \
    experiment.name="$EXP_NAME" \
    experiment.logging.wandb_proj_name=$WANDB_PROJ

echo "[$(date +'%F %T')] Training completed: $EXP_NAME"
