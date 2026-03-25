#!/usr/bin/env bash
# ============================================================================
# Action Space Comparison Experiments
# 对比 pos_euler / pos_rot6d / pos_axisangle 在 LIBERO-10 和 RoboCasa PnP 上的表现
#
# 用法:
#   bash robomimic/scripts/run_action_space_experiments.sh              # 全部 18 runs
#   bash robomimic/scripts/run_action_space_experiments.sh libero       # 仅 LIBERO-10 (9 runs)
#   bash robomimic/scripts/run_action_space_experiments.sh robocasa     # 仅 RoboCasa PnP (9 runs)
#   bash robomimic/scripts/run_action_space_experiments.sh libero pos_euler    # 仅 LIBERO pos_euler (3 seeds)
#   bash robomimic/scripts/run_action_space_experiments.sh robocasa pos_rot6d  # 仅 RoboCasa pos_rot6d (3 seeds)
#
# 配置:
#   - 算法: Diffusion Policy (默认)
#   - 3 action spaces x 2 datasets x 3 seeds = 18 runs
#   - 5000 epochs, batch_size=64, 4-GPU DDP
#   - wandb project: action_space_comparison
# ============================================================================
set -euo pipefail

# ── 训练参数 ──────────────────────────────────────────────────
NUM_EPOCHS=5000
BATCH_SIZE=64
NUM_GPUS=4
SEEDS=(1 42 123)
ACTION_SPACES=(pos_euler pos_rot6d pos_axisangle)

# ── 数据集路径 ────────────────────────────────────────────────
LIBERO_DATA="/workspace/datasets/libero/libero_10"
ROBOCASA_DATA="/workspace/datasets/robocasa/v0.1/single_stage/kitchen_pnp"

# ── 日志目录 ──────────────────────────────────────────────────
LOG_DIR="/workspace/droid_policy_learning/outputs/action_space_exp_logs"
mkdir -p "$LOG_DIR"

# ── 公共 wandb 项目 ──────────────────────────────────────────
WANDB_PROJ="action_space_comparison"

# ── 训练命令 ──────────────────────────────────────────────────
LIBERO_CMD="torchrun --nproc_per_node=$NUM_GPUS -m robomimic.scripts.train_libero"
ROBOCASA_CMD="torchrun --nproc_per_node=$NUM_GPUS -m robomimic.scripts.train_robocasa"

# ── 颜色输出 ──────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── 计数器 ────────────────────────────────────────────────────
TOTAL=0
DONE=0
FAILED=0

# ============================================================================
# 核心训练函数
# ============================================================================

run_libero() {
    local action_space=$1
    local seed=$2
    local exp_name="libero10_${action_space}_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${BLUE}[LIBERO-10]${NC} action_space=${action_space} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    $LIBERO_CMD \
        "train.data=[{path: $LIBERO_DATA}]" \
        train.action_space="$action_space" \
        train.seed="$seed" \
        train.num_epochs=$NUM_EPOCHS \
        train.batch_size=$BATCH_SIZE \
        experiment.name="$exp_name" \
        experiment.logging.wandb_proj_name=$WANDB_PROJ \
        2>&1 | tee "$log_file" \
    && { DONE=$((DONE + 1)); echo -e "${GREEN}[DONE]${NC} $exp_name"; } \
    || { FAILED=$((FAILED + 1)); echo "[FAILED] $exp_name — see $log_file"; }

    echo ""
}

run_robocasa() {
    local action_space=$1
    local seed=$2
    local exp_name="robocasa_pnp_${action_space}_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${BLUE}[RoboCasa-PnP]${NC} action_space=${action_space} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    $ROBOCASA_CMD \
        "train.data=[{path: $ROBOCASA_DATA}]" \
        train.action_space="$action_space" \
        train.seed="$seed" \
        train.num_epochs=$NUM_EPOCHS \
        train.batch_size=$BATCH_SIZE \
        experiment.name="$exp_name" \
        experiment.logging.wandb_proj_name=$WANDB_PROJ \
        2>&1 | tee "$log_file" \
    && { DONE=$((DONE + 1)); echo -e "${GREEN}[DONE]${NC} $exp_name"; } \
    || { FAILED=$((FAILED + 1)); echo "[FAILED] $exp_name — see $log_file"; }

    echo ""
}

# ============================================================================
# 批量运行函数
# ============================================================================

run_libero_all() {
    local filter_space="${1:-}"
    for action_space in "${ACTION_SPACES[@]}"; do
        if [ -n "$filter_space" ] && [ "$filter_space" != "$action_space" ]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            run_libero "$action_space" "$seed"
        done
    done
}

run_robocasa_all() {
    local filter_space="${1:-}"
    for action_space in "${ACTION_SPACES[@]}"; do
        if [ -n "$filter_space" ] && [ "$filter_space" != "$action_space" ]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            run_robocasa "$action_space" "$seed"
        done
    done
}

# ============================================================================
# 参数解析
# ============================================================================

echo "============================================================"
echo "  Action Space Comparison Experiments"
echo "  Epochs: $NUM_EPOCHS | Batch: $BATCH_SIZE | GPUs: $NUM_GPUS"
echo "  Seeds: ${SEEDS[*]}"
echo "  Action spaces: ${ACTION_SPACES[*]}"
echo "  Logs: $LOG_DIR/"
echo "============================================================"
echo ""

DATASET="${1:-all}"
SPACE_FILTER="${2:-}"

case "$DATASET" in
    libero)
        echo "Running LIBERO-10 experiments..."
        run_libero_all "$SPACE_FILTER"
        ;;
    robocasa)
        echo "Running RoboCasa PnP experiments..."
        run_robocasa_all "$SPACE_FILTER"
        ;;
    all)
        echo "Running ALL experiments (LIBERO-10 + RoboCasa PnP)..."
        run_libero_all "$SPACE_FILTER"
        run_robocasa_all "$SPACE_FILTER"
        ;;
    *)
        echo "Usage: $0 [libero|robocasa|all] [pos_euler|pos_rot6d|pos_axisangle]"
        exit 1
        ;;
esac

# ============================================================================
# 汇总
# ============================================================================
echo "============================================================"
echo "  Experiments finished"
echo "  Total: $TOTAL | Done: $DONE | Failed: $FAILED"
echo "  Logs: $LOG_DIR/"
echo "  WandB project: $WANDB_PROJ"
echo "============================================================"
