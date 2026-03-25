#!/bin/bash -l
#SBATCH --job-name=train_droid
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100_40
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yufeng.jin@tu-darmstadt.de
#SBATCH --export=NONE   # 避免把登陆节点的脏环境带入

################################################################################
# 1) Bash 安全选项 & 日志
################################################################################
set -euo pipefail

mkdir -p logs
echo "[$(date +'%F %T')] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  NODELIST=${SLURM_JOB_NODELIST:-NA}"

trap 'status=$?; now=$(date "+%F %T"); echo "[EXIT $now] code=$status"; [[ $status -ne 0 ]] && env | sort | grep -E "SLURM|CUDA|TMPDIR" || true' EXIT

################################################################################
# 2) 代理设置
################################################################################
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

################################################################################
# 3) 作业环境
################################################################################
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export nointeractive=yes

################################################################################
# 4) DROID RLDS 数据路径（直接使用源目录，不复制到 TMPDIR）
#    覆盖：export DROID_DATASET_DIR=/path/to/droid_rlds
################################################################################
SCRIPT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_SLURM_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-/home/hpc/g108ea/g108ea10/repos/droid_policy_learning}"
PROJECT_ROOT="${DROID_PROJECT_DIR:-$WORK_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
DATA_ROOT="${DROID_DATASET_DIR:-/home/atuin/g108ea/g108ea10/datasets/droid}"
# RLDS 子集，对应 train_rlds.yaml 中 dataset_names，如 [insert_pin]、[droid_100]
DATASET_NAMES="${DROID_DATASET_NAMES:-insert_pin}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] DROID dataset not found: $DATA_ROOT"; exit 2
fi
echo "[$(date +'%F %T')] Using DROID dataset at $DATA_ROOT (dataset_names=$DATASET_NAMES)"
ls -1 "$DATA_ROOT" | head -n 5 || { echo "[WARN] dataset layout check skipped"; }

################################################################################
# 5) Apptainer 配置（与 train_libero_app.slurm / train_robocasa_app.slurm 一致）
################################################################################
WANDB_ENV_FILE="${PROJECT_ROOT}/slurm/.env.wandb"
[[ -f "$WANDB_ENV_FILE" ]] && source "$WANDB_ENV_FILE"
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

CONTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_ROOT/droid_policy.sif}"
HOST_CODE_DIR="${DROID_PROJECT_DIR:-$PROJECT_ROOT}"
HOST_FAKE_ROOT="${APPTAINER_FAKE_ROOT:-${WORK:-$HOME}/.apptainer_fake_root}"
HOST_TMPDIR="${TMPDIR:-/tmp}"
HOST_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_DROID="/workspace/datasets/droid"
CONT_HF="/root/.cache/huggingface"

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "[ERR] Container image not found: $CONTAINER_IMAGE"
  echo "请先构建: apptainer build droid_policy.sif apptainer/droid_policy.def"
  exit 1
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK:-$HOME}/.apptainer_cache}"
mkdir -p "$HOST_FAKE_ROOT" 2>/dev/null || true
mkdir -p "$HOST_HF_CACHE" 2>/dev/null || true

BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR}"
BIND_ARGS="$BIND_ARGS -B ${HOST_FAKE_ROOT}:/root"
BIND_ARGS="$BIND_ARGS -B ${HOST_TMPDIR}:/tmp"
BIND_ARGS="$BIND_ARGS -B ${DATA_ROOT}:${CONT_DROID}:ro"
[ -d "$HOST_HF_CACHE" ] && BIND_ARGS="$BIND_ARGS -B ${HOST_HF_CACHE}:${CONT_HF}"

# wandb：-C 时需传入代理
[[ -n "${http_proxy:-}" ]] && export APPTAINERENV_http_proxy="${http_proxy}" APPTAINERENV_https_proxy="${https_proxy:-$http_proxy}"

echo "[INFO] Apptainer bind mounts:"
echo "  $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "  $HOST_TMPDIR -> /tmp"
echo "  $DATA_ROOT -> $CONT_DROID (ro)"
[ -d "$HOST_HF_CACHE" ] && echo "  $HOST_HF_CACHE -> $CONT_HF"

################################################################################
# 6) 启动训练（train_droid 使用 train_rlds.yaml，RLDS 格式）
#    覆盖：DROID_DATASET_NAMES="droid_100 role_ros2" 等
################################################################################
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Starting DROID training at $(date +'%F %T')"

apptainer run --nv -C --writable-tmpfs --pwd "/workspace/droid_policy_learning" $BIND_ARGS "$CONTAINER_IMAGE" \
  torchrun --nproc_per_node=4 -m robomimic.scripts.train_droid \
  train.data_path="${CONT_DROID}" \
  train.dataset_names="[${DATASET_NAMES}]" \
  train.output_dir="/workspace/droid_policy_learning/outputs" \
  train.batch_size=128

echo "[INFO] Training completed at $(date +'%F %T')"
