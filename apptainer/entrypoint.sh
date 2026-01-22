#!/bin/bash
set -e

function log() {
    echo ">> [Entrypoint] $1"
}

log "Starting initialization..."
log "User: $(whoami)"
log "HOME: $HOME"

# --- 1. 动态查找 micromamba ---
MAMBA_EXE=""
if [ -x "/usr/local/bin/micromamba" ]; then
    MAMBA_EXE="/usr/local/bin/micromamba"
elif command -v micromamba >/dev/null 2>&1; then
    MAMBA_EXE=$(command -v micromamba)
fi

if [ -z "$MAMBA_EXE" ]; then
    log "ERROR: Could not find micromamba executable!"
    exit 1
fi

log "Found micromamba at: $MAMBA_EXE"

# --- 2. 初始化与激活 (关键修改点) ---
export MAMBA_ROOT_PREFIX=/opt/conda

# A. 生成 Shell Hook (定义 micromamba 函数)
eval "$($MAMBA_EXE shell hook --shell bash)"

# B. 激活环境
# [修复] 必须直接调用 'micromamba' (这是刚才 eval 生成的函数)
# 不要使用 $MAMBA_EXE activate，那样会调用二进制文件导致报错
log "Activating environment 'droid_env'..."
micromamba activate droid_env

# --- 3. 验证 Python ---
CURRENT_PYTHON=$(which python)
log "Active Python path: $CURRENT_PYTHON"

if [[ "$CURRENT_PYTHON" != *"/envs/droid_env/"* ]]; then
    log "WARNING: Environment activation failed! Python is: $CURRENT_PYTHON"
    # 尝试备选方案：如果 activate 失败，将 bin 目录强行加入 PATH
    export PATH="/opt/conda/envs/droid_env/bin:$PATH"
    log "Fallback: Added env bin to PATH manually."
fi

# --- 4. 挂载项目的 Editable Install ---
PROJECT_DIR="/workspace/droid_policy_learning"

if [ -d "$PROJECT_DIR" ]; then
    if [ -f "$PROJECT_DIR/setup.py" ] || [ -f "$PROJECT_DIR/pyproject.toml" ]; then
        log "Installing project in editable mode..."
        cd "$PROJECT_DIR"
        # 使用 python -m pip 确保使用当前环境的 pip
        python -m pip install -e . --no-build-isolation
        log "Project installed."
        cd - > /dev/null
    fi
fi

log "Executing command: $@"
exec "$@"
