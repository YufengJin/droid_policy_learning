#!/bin/bash -l
#SBATCH --job-name=dvd_%a
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100_40
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yufeng.jin@tu-darmstadt.de
#SBATCH --export=NONE   # 避免把登陆节点的脏环境带入；单任务覆盖参数见下方用法（--export=NONE,...）
#SBATCH --array=0-11%4

################################################################################
# Drift vs Diffusion Comparison — SLURM Array Job
#
# 每个 array task 运行一个 (dataset, algorithm, seed) 组合。
# 共 12 tasks: 2 datasets × 2 algorithms × 3 seeds
#
# Task ID 映射:
#   ID = dataset_idx * 6 + algo_idx * 3 + seed_idx
#   dataset:   [libero, robocasa]        (0-1)
#   algorithm: [diffusion, drift]        (0-1)
#   seed:      [1, 42, 123]             (0-2)
#
# 用法:
#   sbatch slurm/train_drift_vs_diffusion.slurm                              # 全部 12 runs
#   sbatch --array=0-5%3  slurm/train_drift_vs_diffusion.slurm               # 仅 LIBERO (6 runs)
#   sbatch --array=6-11%3 slurm/train_drift_vs_diffusion.slurm               # 仅 RoboCasa (6 runs)
#   sbatch --array=0-2%3  slurm/train_drift_vs_diffusion.slurm               # 仅 LIBERO diffusion (3 runs)
#   sbatch --array=3-5%3  slurm/train_drift_vs_diffusion.slurm               # 仅 LIBERO drift (3 runs)
#   sbatch --array=4      slurm/train_drift_vs_diffusion.slurm               # 单次: libero drift seed=42
#
# 或用环境变量覆盖（忽略 array ID 的映射；export=NONE 时必须把 NONE 写在最前）:
#   sbatch --array=0 --export=NONE,DATASET=libero,ALGO=drift,SEED=42 \
#       slurm/train_drift_vs_diffusion.slurm
################################################################################
set -euo pipefail

mkdir -p logs
echo "[$(date +'%F %T')] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}  NODELIST=${SLURM_JOB_NODELIST:-NA}"

trap 'status=$?; now=$(date "+%F %T"); echo "[EXIT $now] code=$status"; [[ $status -ne 0 ]] && env | sort | grep -E "SLURM|CUDA|TMPDIR" || true' EXIT

################################################################################
# 1) 代理设置（与 train_*_app.slurm 一致；-C 容器内需 APPTAINERENV）
################################################################################
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

################################################################################
# 2) 作业环境（与 train_*_app.slurm 一致）
################################################################################
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export nointeractive=yes

################################################################################
# 3) 从 SLURM_ARRAY_TASK_ID 解码实验参数
################################################################################
DATASETS=(libero robocasa)
ALGOS=(diffusion drift)
SEEDS=(1 42 123)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

dataset_idx=$(( TASK_ID / 6 ))
remainder=$(( TASK_ID % 6 ))
algo_idx=$(( remainder / 3 ))
seed_idx=$(( remainder % 3 ))

# 环境变量覆盖（优先级高于 array ID 映射）
DATASET="${DATASET:-${DATASETS[$dataset_idx]}}"
ALGO="${ALGO:-${ALGOS[$algo_idx]}}"
SEED="${SEED:-${SEEDS[$seed_idx]}}"

NUM_EPOCHS="${NUM_EPOCHS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WANDB_PROJ="drift_vs_diffusion"

echo "[INFO] Experiment: dataset=${DATASET} algo=${ALGO} seed=${SEED} epochs=${NUM_EPOCHS} batch=${BATCH_SIZE}"

################################################################################
# 4) 路径配置（与 train_libero_app / train_robocasa_app 一致）
#    镜像与代码在 vault 时：从仓库根目录 sbatch，使 SLURM_SUBMIT_DIR 指向 vault。
#    覆盖：DROID_PROJECT_DIR、APPTAINER_IMAGE、LIBERO_DATASET_DIR、ROBOCASA_DATASET_DIR、
#          LIBERO_SUBDIR（默认 libero_10）、ROBOCASA_SUBDIR（默认 v0.1/single_stage/kitchen_pnp）
################################################################################
SCRIPT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_SLURM_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-/home/hpc/g108ea/g108ea10/repos/droid_policy_learning}"
PROJECT_ROOT="${DROID_PROJECT_DIR:-$WORK_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# 数据集主机路径（子目录可与 train_*_app 对齐覆盖）
LIBERO_HOST_ROOT="${LIBERO_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/libero}"
ROBOCASA_HOST_ROOT="${ROBOCASA_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/robocasa}"
LIBERO_SUBDIR="${LIBERO_SUBDIR:-libero_10}"
ROBOCASA_SUBDIR="${ROBOCASA_SUBDIR:-v0.1/single_stage/kitchen_pnp}"

# 容器内路径
CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_LIBERO="/workspace/datasets/libero"
CONT_ROBOCASA="/workspace/datasets/robocasa"
CONT_HF="/root/.cache/huggingface"

CONT_LIBERO_DATA="${CONT_LIBERO}/${LIBERO_SUBDIR}"
CONT_ROBOCASA_DATA="${CONT_ROBOCASA}/${ROBOCASA_SUBDIR}"

################################################################################
# 5) Apptainer 配置
################################################################################
WANDB_ENV_FILE="${PROJECT_ROOT}/slurm/.env.wandb"
[[ -f "$WANDB_ENV_FILE" ]] && source "$WANDB_ENV_FILE"
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
# 6) Drift 最佳配置覆盖
################################################################################
DRIFT_OVERRIDES=(
    algo.drift.kernel_type=rbf
    algo.drift.dist_scale_mode=sqrt_dim
    algo.drift.rbf_sigma=0.5
)

################################################################################
# 7) 按 dataset + algo 分发训练
################################################################################
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

# 确定数据集相关参数
case "$DATASET" in
    libero)
        if [[ ! -d "$LIBERO_HOST_ROOT" ]]; then
            echo "[ERR] LIBERO dataset not found: $LIBERO_HOST_ROOT"; exit 2
        fi
        if [[ ! -d "$LIBERO_HOST_ROOT/$LIBERO_SUBDIR" ]]; then
            echo "[ERR] LIBERO subdir not found: $LIBERO_HOST_ROOT/$LIBERO_SUBDIR"; exit 2
        fi
        echo "[$(date +'%F %T')] Using LIBERO at $LIBERO_HOST_ROOT/$LIBERO_SUBDIR (container: $CONT_LIBERO_DATA)"
        ls -1 "$LIBERO_HOST_ROOT/$LIBERO_SUBDIR" | head -n 5 || { echo "[WARN] dataset layout check skipped"; }
        BIND_ARGS="$BIND_ARGS -B ${LIBERO_HOST_ROOT}:${CONT_LIBERO}:ro"
        TRAIN_MODULE="robomimic.scripts.train_libero"
        DATA_ARG="train.data=[{path: ${CONT_LIBERO_DATA}}]"
        EXP_PREFIX="libero10"
        ;;
    robocasa)
        if [[ ! -d "$ROBOCASA_HOST_ROOT" ]]; then
            echo "[ERR] RoboCasa dataset not found: $ROBOCASA_HOST_ROOT"; exit 2
        fi
        if [[ ! -d "$ROBOCASA_HOST_ROOT/$ROBOCASA_SUBDIR" ]]; then
            echo "[ERR] RoboCasa subdir not found: $ROBOCASA_HOST_ROOT/$ROBOCASA_SUBDIR"; exit 2
        fi
        echo "[$(date +'%F %T')] Using RoboCasa at $ROBOCASA_HOST_ROOT/$ROBOCASA_SUBDIR (container: $CONT_ROBOCASA_DATA)"
        ls -1 "$ROBOCASA_HOST_ROOT/$ROBOCASA_SUBDIR" | head -n 5 || { echo "[WARN] dataset layout check skipped"; }
        BIND_ARGS="$BIND_ARGS -B ${ROBOCASA_HOST_ROOT}:${CONT_ROBOCASA}:ro"
        TRAIN_MODULE="robomimic.scripts.train_robocasa"
        DATA_ARG="train.data=[{path: ${CONT_ROBOCASA_DATA}}]"
        EXP_PREFIX="robocasa_pnp"
        ;;
    *)
        echo "[ERR] Unknown dataset: $DATASET (expected libero or robocasa)"; exit 1
        ;;
esac

EXP_NAME="${EXP_PREFIX}_${ALGO}_s${SEED}"

# 构建算法相关的额外参数（与 train_libero_app：algo_name=diffusion_policy|drift_policy 对齐）
ALGO_ARGS=()

case "$ALGO" in
    diffusion)
        # train_libero / train_robocasa 默认 Hydra 已是 diffusion_policy；显式传入与 *_app 一致
        ALGO_ARGS+=("algo_name=diffusion_policy")
        ;;
    drift)
        if [[ "$DATASET" == "libero" ]]; then
            export APPTAINERENV_LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift
        else
            # 避免宿主/上一任务遗留的 LIBERO Hydra 名影响容器（train_libero 会读该变量）
            unset APPTAINERENV_LIBERO_TRAIN_HYDRA_CONFIG || true
            ALGO_ARGS+=("config_name=train_robocasa_drift")
        fi
        ALGO_ARGS+=("${DRIFT_OVERRIDES[@]}")
        ;;
    *)
        echo "[ERR] Unknown algorithm: $ALGO (expected diffusion or drift)"; exit 1
        ;;
esac

# LIBERO + diffusion：勿留 APPTAINERENV_LIBERO_TRAIN_HYDRA_CONFIG（否则 train_libero 会切到 drift yaml）
if [[ "$DATASET" == "libero" && "$ALGO" == "diffusion" ]]; then
    unset APPTAINERENV_LIBERO_TRAIN_HYDRA_CONFIG || true
fi

echo "[INFO] Apptainer bind mounts:"
echo "  $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "  $HOST_TMPDIR -> /tmp"
case "$DATASET" in
    libero)   echo "  $LIBERO_HOST_ROOT -> $CONT_LIBERO (ro)" ;;
    robocasa) echo "  $ROBOCASA_HOST_ROOT -> $CONT_ROBOCASA (ro)" ;;
esac
[ -d "$HOST_HF_CACHE" ] && echo "  $HOST_HF_CACHE -> $CONT_HF"

echo ""
echo "============================================================"
echo "  Experiment: $EXP_NAME"
echo "  Module:     $TRAIN_MODULE"
echo "  Algorithm:  $ALGO"
echo "  Dataset:    $DATASET"
echo "  Seed:       $SEED"
echo "  Epochs:     $NUM_EPOCHS"
echo "  Batch:      $BATCH_SIZE"
echo "  WandB:      $WANDB_PROJ"
[[ "$ALGO" == "drift" ]] && echo "  Drift cfg:  ${DRIFT_OVERRIDES[*]}"
echo "============================================================"
echo ""

################################################################################
# 8) 启动训练
################################################################################
echo "[$(date +'%F %T')] Starting training: $EXP_NAME"

apptainer run --nv -C --writable-tmpfs --pwd "/workspace/droid_policy_learning" $BIND_ARGS "$CONTAINER_IMAGE" \
    torchrun --nproc_per_node=4 -m "$TRAIN_MODULE" \
    "$DATA_ARG" \
    train.output_dir="${CONT_CODE_DIR}/outputs" \
    train.seed="$SEED" \
    train.num_epochs=$NUM_EPOCHS \
    train.batch_size=$BATCH_SIZE \
    experiment.name="$EXP_NAME" \
    experiment.logging.wandb_proj_name=$WANDB_PROJ \
    "${ALGO_ARGS[@]+"${ALGO_ARGS[@]}"}"

echo "[$(date +'%F %T')] Training completed: $EXP_NAME"
