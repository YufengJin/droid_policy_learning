#!/bin/bash -l
#SBATCH --job-name=train_rlds
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
# 2) Modules & Conda
################################################################################
module load python
# 推荐：在集群侧做一次 conda init bash，保证非交互 shell 能 activate
source ~/.bashrc 2>/dev/null || true
conda activate droid_policy_learning_env

################################################################################
# 3) 代理设置
################################################################################
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

################################################################################
# 4) 作业环境
################################################################################
# 线程限制，避免与 DataLoader 抢 CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 非交互模式
export nointeractive=yes

################################################################################
# 5) 数据缓存到 $TMPDIR（节点本地盘）
################################################################################
echo "[$(date +'%F %T')] Staging dataset to TMPDIR=${TMPDIR}"

WORK_DIR="/home/hpc/g108ea/g108ea10/repos/droid_policy_learning"
OUTPUT_DIR="/home/hpc/g108ea/g108ea10/repos/droid_policy_learning/outputs"
# DATA_SOURCE="$WORK/datasets/droid"
DATA_SOURCE="/home/atuin/g108ea/g108ea11/datasets/droid/1.0.1"
DATA_ROOT="$TMPDIR/datasets/droid/1.0.1"

# 复制前检查源存在
if [[ ! -d "$DATA_SOURCE" ]]; then
  echo "[ERR] DATA_SOURCE not found: $DATA_SOURCE"; exit 2
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Copying dataset from $DATA_SOURCE to $DATA_ROOT..."
  mkdir -p "$(dirname "$DATA_ROOT")"
  start_time=$(date +%s)
  if command -v rsync >/dev/null 2>&1; then
    rsync -av --progress "$DATA_SOURCE/" "$DATA_ROOT/"
  else
    cp -r "$DATA_SOURCE" "$DATA_ROOT"
  fi
  end_time=$(date +%s)
  echo "Dataset copied in $((end_time - start_time)) seconds."
else
  echo "Dataset already staged at $DATA_ROOT (skip copy)."
fi

# 快速 sanity check
if [[ -d "$DATA_ROOT" ]]; then
  echo "Dataset verified at $DATA_ROOT"
  ls -1 "$DATA_ROOT" | head -n 5 || { echo "[WARN] dataset layout check skipped"; }
else
  echo "[ERR] dataset not found at $DATA_ROOT"; exit 2
fi

################################################################################
# 6) 代码目录
################################################################################
cd $WORK_DIR

################################################################################
# 7) 启动训练
################################################################################
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Starting training at $(date +'%F %T')"

python -m robomimic.scripts.train_rlds \
  train.data_path="$TMPDIR/datasets" \
  train.output_dir="${OUTPUT_DIR}" \
  train.dataset_names=[droid] \
  train.batch_size=1024 

echo "[INFO] Training completed at $(date +'%F %T')"
