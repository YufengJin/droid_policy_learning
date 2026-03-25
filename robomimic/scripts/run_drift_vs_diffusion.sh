#!/usr/bin/env bash
# ============================================================================
# Drift vs Diffusion Policy Comparison Experiments
# 对比 drift_policy 和 diffusion_policy 在 LIBERO-10 和 RoboCasa PnP 上的表现
#
# 用法:
#   bash robomimic/scripts/run_drift_vs_diffusion.sh                        # 全部 12 runs
#   bash robomimic/scripts/run_drift_vs_diffusion.sh libero                 # 仅 LIBERO-10 (6 runs)
#   bash robomimic/scripts/run_drift_vs_diffusion.sh robocasa               # 仅 RoboCasa PnP (6 runs)
#   bash robomimic/scripts/run_drift_vs_diffusion.sh libero drift           # 仅 LIBERO drift (3 seeds)
#   bash robomimic/scripts/run_drift_vs_diffusion.sh robocasa diffusion     # 仅 RoboCasa diffusion (3 seeds)
#
# 配置:
#   - 2 algorithms (diffusion, drift) x 2 datasets x 3 seeds = 12 runs
#   - 5000 epochs, batch_size=64, 4-GPU DDP
#   - Drift 使用最佳已知配置: RBF + sqrt_dim + sigma=0.5
#   - wandb project: drift_vs_diffusion
# ============================================================================
set -euo pipefail

# ── 训练参数 ──────────────────────────────────────────────────
NUM_EPOCHS=5000
BATCH_SIZE=64
NUM_GPUS=4
SEEDS=(1 42 123)
ALGOS=(diffusion drift)

# ── 数据集路径 ────────────────────────────────────────────────
LIBERO_DATA="/workspace/datasets/libero/libero_10"
ROBOCASA_DATA="/workspace/datasets/robocasa/v0.1/single_stage/kitchen_pnp"

# ── 日志目录 ──────────────────────────────────────────────────
LOG_DIR="/workspace/droid_policy_learning/outputs/drift_vs_diffusion_logs"
mkdir -p "$LOG_DIR"

# ── 公共 wandb 项目 ──────────────────────────────────────────
WANDB_PROJ="drift_vs_diffusion"

# ── 训练命令 ──────────────────────────────────────────────────
LIBERO_CMD="torchrun --nproc_per_node=$NUM_GPUS -m robomimic.scripts.train_libero"
ROBOCASA_CMD="torchrun --nproc_per_node=$NUM_GPUS -m robomimic.scripts.train_robocasa"

# ── Drift 最佳配置覆盖 ──────────────────────────────────────
DRIFT_OVERRIDES=(
    algo.drift.kernel_type=rbf
    algo.drift.dist_scale_mode=sqrt_dim
    algo.drift.rbf_sigma=0.5
)

# ── 颜色输出 ──────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

# ── 计数器 ────────────────────────────────────────────────────
TOTAL=0
DONE=0
FAILED=0

# ============================================================================
# 核心训练函数
# ============================================================================

run_libero_diffusion() {
    local seed=$1
    local exp_name="libero10_diffusion_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${BLUE}[LIBERO-10 / diffusion]${NC} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    $LIBERO_CMD \
        "train.data=[{path: $LIBERO_DATA}]" \
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

run_libero_drift() {
    local seed=$1
    local exp_name="libero10_drift_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${YELLOW}[LIBERO-10 / drift]${NC} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    # 使用子 shell 设置环境变量，避免影响后续 diffusion runs
    (
        export LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift
        $LIBERO_CMD \
            "train.data=[{path: $LIBERO_DATA}]" \
            train.seed="$seed" \
            train.num_epochs=$NUM_EPOCHS \
            train.batch_size=$BATCH_SIZE \
            "${DRIFT_OVERRIDES[@]}" \
            experiment.name="$exp_name" \
            experiment.logging.wandb_proj_name=$WANDB_PROJ \
            2>&1 | tee "$log_file"
    ) \
    && { DONE=$((DONE + 1)); echo -e "${GREEN}[DONE]${NC} $exp_name"; } \
    || { FAILED=$((FAILED + 1)); echo "[FAILED] $exp_name — see $log_file"; }

    echo ""
}

run_robocasa_diffusion() {
    local seed=$1
    local exp_name="robocasa_pnp_diffusion_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${BLUE}[RoboCasa-PnP / diffusion]${NC} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    $ROBOCASA_CMD \
        "train.data=[{path: $ROBOCASA_DATA}]" \
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

run_robocasa_drift() {
    local seed=$1
    local exp_name="robocasa_pnp_drift_s${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"

    echo -e "${YELLOW}[RoboCasa-PnP / drift]${NC} seed=${seed} → ${exp_name}"
    TOTAL=$((TOTAL + 1))

    $ROBOCASA_CMD \
        config_name=train_robocasa_drift \
        "train.data=[{path: $ROBOCASA_DATA}]" \
        train.seed="$seed" \
        train.num_epochs=$NUM_EPOCHS \
        train.batch_size=$BATCH_SIZE \
        "${DRIFT_OVERRIDES[@]}" \
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
    local filter_algo="${1:-}"
    for algo in "${ALGOS[@]}"; do
        if [ -n "$filter_algo" ] && [ "$filter_algo" != "$algo" ]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            if [ "$algo" = "diffusion" ]; then
                run_libero_diffusion "$seed"
            else
                run_libero_drift "$seed"
            fi
        done
    done
}

run_robocasa_all() {
    local filter_algo="${1:-}"
    for algo in "${ALGOS[@]}"; do
        if [ -n "$filter_algo" ] && [ "$filter_algo" != "$algo" ]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            if [ "$algo" = "diffusion" ]; then
                run_robocasa_diffusion "$seed"
            else
                run_robocasa_drift "$seed"
            fi
        done
    done
}

# ============================================================================
# 参数解析
# ============================================================================

echo "============================================================"
echo "  Drift vs Diffusion Comparison Experiments"
echo "  Epochs: $NUM_EPOCHS | Batch: $BATCH_SIZE | GPUs: $NUM_GPUS"
echo "  Seeds: ${SEEDS[*]}"
echo "  Algorithms: ${ALGOS[*]}"
echo "  Drift config: ${DRIFT_OVERRIDES[*]}"
echo "  Logs: $LOG_DIR/"
echo "============================================================"
echo ""

DATASET="${1:-all}"
ALGO_FILTER="${2:-}"

case "$DATASET" in
    libero)
        echo "Running LIBERO-10 experiments..."
        run_libero_all "$ALGO_FILTER"
        ;;
    robocasa)
        echo "Running RoboCasa PnP experiments..."
        run_robocasa_all "$ALGO_FILTER"
        ;;
    all)
        echo "Running ALL experiments (LIBERO-10 + RoboCasa PnP)..."
        run_libero_all "$ALGO_FILTER"
        run_robocasa_all "$ALGO_FILTER"
        ;;
    *)
        echo "Usage: $0 [libero|robocasa|all] [diffusion|drift]"
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
