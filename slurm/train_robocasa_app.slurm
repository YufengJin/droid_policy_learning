#!/bin/bash -l
#SBATCH --job-name=train_robocasa
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100_80
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

# 统一退出时打印一些诊断
trap 'status=$?; now=$(date "+%F %T"); echo "[EXIT $now] code=$status"; [[ $status -ne 0 ]] && env | sort | grep -E "SLURM|CUDA|TMPDIR" || true' EXIT

################################################################################
# 2) 代理设置
################################################################################
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

################################################################################
# 3) 作业环境
################################################################################
# 线程限制，避免与 DataLoader 抢 CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 非交互模式
export nointeractive=yes

################################################################################
# 4) RoboCasa 数据路径（直接使用源目录，不复制到 TMPDIR）
#    覆盖：ROBOCASA_DATASET_DIR；ROBOCASA_SUBDIR（子集路径，默认 v0.1/single_stage）
#    可选 subtask 示例：
#      v0.1/single_stage         所有单阶段任务（默认，25 个任务，一个 model 全覆盖）
#      v0.1/multi_stage          多阶段任务（5 个任务）
#      v0.1                      single+multi 全部 30 个任务
#      v0.1/single_stage/kitchen_coffee
#      v0.1/single_stage/kitchen_doors
#      v0.1/single_stage/kitchen_drawer
#      v0.1/single_stage/kitchen_microwave
#      v0.1/single_stage/kitchen_navigate
#      v0.1/single_stage/kitchen_pnp
#      v0.1/single_stage/kitchen_sink
#      v0.1/single_stage/kitchen_stove
################################################################################
SCRIPT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_SLURM_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-/home/hpc/g108ea/g108ea10/vault/vla-wam/droid_policy_learning}"
PROJECT_ROOT="${DROID_PROJECT_DIR:-$WORK_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
DATA_ROOT="${ROBOCASA_DATASET_DIR:-/home/hpc/g108ea/g108ea10/atuin/datasets/robocasa}"
ROBOCASA_SUBDIR="${ROBOCASA_SUBDIR:-v0.1/single_stage}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] RoboCasa dataset not found: $DATA_ROOT"; exit 2
fi
if [[ ! -d "$DATA_ROOT/$ROBOCASA_SUBDIR" ]]; then
  echo "[ERR] RoboCasa subdir not found: $DATA_ROOT/$ROBOCASA_SUBDIR"; exit 2
fi
echo "[$(date +'%F %T')] Using RoboCasa dataset at $DATA_ROOT/$ROBOCASA_SUBDIR (no copy to TMPDIR)"
ls -1 "$DATA_ROOT/$ROBOCASA_SUBDIR" | head -n 5 || { echo "[WARN] dataset layout check skipped"; }

################################################################################
# 5) Apptainer 配置（参考 run_container.sh）
################################################################################
WANDB_ENV_FILE="${PROJECT_ROOT}/slurm/.env.wandb"
[[ -f "$WANDB_ENV_FILE" ]] && source "$WANDB_ENV_FILE"
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

CONTAINER_IMAGE="${APPTAINER_IMAGE:-${WORK:-$PROJECT_ROOT}/sif/droid.sif}"
HOST_CODE_DIR="${DROID_PROJECT_DIR:-$PROJECT_ROOT}"
HOST_FAKE_ROOT="${APPTAINER_FAKE_ROOT:-${WORK:-$HOME}/.apptainer_fake_root}"
HOST_TMPDIR="${TMPDIR:-/tmp}"
HOST_HF_CACHE="${HF_HOME:-$HOME/atuin/huggingface}"

CONT_CODE_DIR="/workspace/droid_policy_learning"
CONT_ROBOCASA="/workspace/datasets/robocasa"
CONT_HF="/root/.cache/huggingface"

# 检查镜像
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "[ERR] Container image not found: $CONTAINER_IMAGE"
  echo "请先构建 droid.sif（docker→sif，见 docs/quick_start_apptainer.md；或设 APPTAINER_IMAGE 指向已有镜像/.def 产物）"
  exit 1
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORK:-$HOME}/.apptainer_cache}"
mkdir -p "$HOST_FAKE_ROOT" 2>/dev/null || true
mkdir -p "$HOST_HF_CACHE" 2>/dev/null || true

# Bind：代码、fake root、TMPDIR、RoboCasa（使用 $TMPDIR 中已复制的数据）、HF cache
BIND_ARGS="-B ${HOST_CODE_DIR}:${CONT_CODE_DIR}"
BIND_ARGS="$BIND_ARGS -B ${HOST_FAKE_ROOT}:/root"
BIND_ARGS="$BIND_ARGS -B ${HOST_TMPDIR}:/tmp"
BIND_ARGS="$BIND_ARGS -B ${DATA_ROOT}:${CONT_ROBOCASA}:ro"
[ -d "$HOST_HF_CACHE" ] && BIND_ARGS="$BIND_ARGS -B ${HOST_HF_CACHE}:${CONT_HF}"

# wandb：-C 时需传入代理
[[ -n "${http_proxy:-}" ]] && export APPTAINERENV_http_proxy="${http_proxy}" APPTAINERENV_https_proxy="${https_proxy:-$http_proxy}"
# wandb client<->service 走 localhost；http_proxy 下必须 bypass，否则 wandb.init 反复重试卡死（见 eval_app.slurm bug #3）
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export APPTAINERENV_no_proxy="$no_proxy" APPTAINERENV_NO_PROXY="$NO_PROXY"

# Alex 修复（见 eval_app.slurm bug #1/#4）：
#  - GLIBC_2.34：apptainer 注入 host GL 库与容器旧 glibc 冲突 -> 容器库优先（cv2 import 否则崩）
#  - 缓存 No space：--writable-tmpfs overlay 仅 16MiB -> 各类缓存重定向到 /tmp（绑定到大容量 host TMPDIR）
APPTAINER_ENV_FIX="--env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
APPTAINER_ENV_FIX="$APPTAINER_ENV_FIX --env NUMBA_CACHE_DIR=/tmp/numba_cache --env XDG_CACHE_HOME=/tmp/xdg_cache --env MPLCONFIGDIR=/tmp/mpl"
APPTAINER_ENV_FIX="$APPTAINER_ENV_FIX --env HF_HOME=${CONT_HF} --env TRANSFORMERS_CACHE=${CONT_HF}/hub"

echo "[INFO] Apptainer bind mounts:"
echo "  $HOST_CODE_DIR -> $CONT_CODE_DIR"
echo "  $HOST_TMPDIR -> /tmp"
echo "  $DATA_ROOT -> $CONT_ROBOCASA (ro)"
[ -d "$HOST_HF_CACHE" ] && echo "  $HOST_HF_CACHE -> $CONT_HF"

################################################################################
# 6) 启动训练（默认：单算法 diffusion_policy，一个 model 覆盖 single_stage 全部任务）
#    数据路径：/workspace/datasets/robocasa/${ROBOCASA_SUBDIR}（默认 v0.1/single_stage，25 个任务）
#    覆盖：ROBOCASA_ALGO=drift_policy sbatch ...（换单算法）
#         ROBOCASA_MULTIRUN=1 sbatch ...（恢复 drift_policy+diffusion_policy 双算法）
################################################################################
CONT_ROBOCASA_SUB="${CONT_ROBOCASA}/${ROBOCASA_SUBDIR}"
ROBOCASA_MULTIRUN="${ROBOCASA_MULTIRUN:-0}"
ROBOCASA_ALGOS="${ROBOCASA_ALGOS:-drift_policy diffusion_policy}"
ROBOCASA_ALGO="${ROBOCASA_ALGO:-diffusion_policy}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

if [[ "${ROBOCASA_MULTIRUN}" == "1" ]]; then
  ALGO_LIST="$ROBOCASA_ALGOS"
else
  ALGO_LIST="$ROBOCASA_ALGO"
fi

echo "[INFO] Starting RoboCasa training at $(date +'%F %T') (MULTIRUN=$ROBOCASA_MULTIRUN: $ALGO_LIST, SUBDIR=$ROBOCASA_SUBDIR)"

# 观测 modality 消融开关（robot_state vs vision-only）：
#   ROBOCASA_LOW_DIM 未设置  -> 用 yaml 默认 low_dim=[]（纯视觉+语言）
#   ROBOCASA_LOW_DIM=robot_states -> 加入本体 proprio (9D)
#   EXP_NAME 可覆盖 wandb/实验名，便于区分对比任务
#
# ⚠️ 数据规模 / RAM：RoboCasaDataset 会把所有图像 *eager-load 进 RAM*（每个 DDP rank
#    各一份，忽略 hdf5_cache_mode）。单 task 的 mg 子集有 ~3000 demos，4 卡全量约 862GB
#    -> 直接 OUT_OF_MEMORY。务必用 ROBOCASA_FILTER_KEY 限制 demo 数：
#   ROBOCASA_FILTER_KEY=300_demos  -> 只取 300 个 demo（mask key，文件内有 10/50/100/.../3000_demos）
#   ROBOCASA_VALIDATE=false        -> 关验证（mg 文件无 valid mask，开验证会再次全量加载 -> OOM；
#                                     带验证只能用 human-demo 子集 .../<date>_2，含 train/valid mask）
EXTRA_OVERRIDES=()
if [[ -n "${ROBOCASA_LOW_DIM+x}" ]]; then
  EXTRA_OVERRIDES+=("observation.modalities.obs.low_dim=[${ROBOCASA_LOW_DIM}]")
fi
if [[ -n "${ROBOCASA_FILTER_KEY:-}" ]]; then
  EXTRA_OVERRIDES+=("train.hdf5_filter_key=${ROBOCASA_FILTER_KEY}")
fi
if [[ -n "${ROBOCASA_VALIDATE:-}" ]]; then
  EXTRA_OVERRIDES+=("experiment.validate=${ROBOCASA_VALIDATE}")
fi
if [[ -n "${EXP_NAME:-}" ]]; then
  EXTRA_OVERRIDES+=("experiment.name=${EXP_NAME}")
fi
echo "[INFO] EXTRA_OVERRIDES=${EXTRA_OVERRIDES[*]:-<none>}"

# A100-80GB 实测：BS=448/GPU 仍可跑(76GB)但余量小；320/GPU(~55GB)留约 24GB 余量，稳定首选。
# 有效 batch = 320 * 4 GPU = 1280。可用 ROBOCASA_BATCH 覆盖。
ROBOCASA_BATCH="${ROBOCASA_BATCH:-320}"

for algo in $ALGO_LIST; do
  echo "[INFO] algo_name=$algo"
  apptainer run --nv -C --writable-tmpfs $APPTAINER_ENV_FIX --pwd "/workspace/droid_policy_learning" $BIND_ARGS "$CONTAINER_IMAGE" \
    torchrun --nproc_per_node=4 -m robomimic.scripts.train_robocasa \
    "train.data=[{path: ${CONT_ROBOCASA_SUB}}]" \
    train.output_dir="/workspace/droid_policy_learning/outputs" \
    train.batch_size=${ROBOCASA_BATCH} \
    "algo_name=${algo}" \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
  echo "[INFO] 完成 algo_name=$algo"
done

echo "[INFO] Training completed at $(date +'%F %T')"
