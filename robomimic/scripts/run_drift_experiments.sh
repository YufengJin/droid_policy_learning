#!/usr/bin/env bash
# ============================================================================
# Drift Policy Experiments 4-6: RBF + sqrt_dim 正式训练
# 用法:
#   bash robomimic/scripts/run_drift_experiments.sh          # 跑全部 4/5/6
#   bash robomimic/scripts/run_drift_experiments.sh 4        # 只跑 Exp 4
#   bash robomimic/scripts/run_drift_experiments.sh 5 6      # 跑 Exp 5 和 6
# ============================================================================
set -euo pipefail

export LIBERO_TRAIN_HYDRA_CONFIG=train_libero_drift
TRAIN_CMD="python -m robomimic.scripts.train_libero"
NUM_EPOCHS=5000
BATCH_SIZE=64
LOG_DIR="/tmp/drift_exp_logs"
mkdir -p "$LOG_DIR"

# 公共 drift 参数
DRIFT_COMMON=(
  algo.drift.kernel_type=rbf
  algo.drift.dist_scale_mode=sqrt_dim
  algo.drift.log_kernel_stats=true
  algo.drift.log_kernel_every_n_steps=10
  train.num_epochs=$NUM_EPOCHS
  train.batch_size=$BATCH_SIZE
)

# 决定跑哪些实验
if [ $# -eq 0 ]; then
  EXPS=(4 5 6)
else
  EXPS=("$@")
fi

run_exp4() {
  echo "========== Exp 4: RBF + sqrt_dim + default LR =========="
  $TRAIN_CMD \
    experiment.name=exp4_rbf_sqrtdim_full \
    "${DRIFT_COMMON[@]}" \
    algo.drift.rbf_sigma=0.5 \
    2>&1 | tee "$LOG_DIR/exp4.log"
}

run_exp5() {
  echo "========== Exp 5: RBF + sqrt_dim + cosine LR =========="
  $TRAIN_CMD \
    experiment.name=exp5_rbf_sqrtdim_cosine \
    "${DRIFT_COMMON[@]}" \
    algo.drift.rbf_sigma=0.5 \
    algo.optim_params.policy.learning_rate.scheduler_type=cosine \
    "algo.optim_params.policy.learning_rate.epoch_schedule=[1]" \
    algo.optim_params.policy.learning_rate.cosine_t_max=$NUM_EPOCHS \
    algo.optim_params.policy.learning_rate.cosine_eta_min=1e-6 \
    2>&1 | tee "$LOG_DIR/exp5.log"
}

run_exp6() {
  echo "========== Exp 6: RBF + sqrt_dim + sigma=1.0 =========="
  $TRAIN_CMD \
    experiment.name=exp6_rbf_sigma1.0 \
    "${DRIFT_COMMON[@]}" \
    algo.drift.rbf_sigma=1.0 \
    2>&1 | tee "$LOG_DIR/exp6.log"
}

for exp in "${EXPS[@]}"; do
  case $exp in
    4) run_exp4 ;;
    5) run_exp5 ;;
    6) run_exp6 ;;
    *) echo "Unknown experiment: $exp (valid: 4, 5, 6)" ;;
  esac
  echo ""
done

echo "All requested experiments finished. Logs in $LOG_DIR/"
