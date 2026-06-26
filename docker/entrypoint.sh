#!/bin/bash
set -e

# uv 安装在 /root/.local/bin（见 Dockerfile）；缺了会导致 runtime editable install 报 uv: not found
# /usr/local/bin 含构建阶段复制的 uv（见 Dockerfile）；bind 覆盖 /root 时仍可用
export PATH="/opt/venv/bin:/usr/local/bin:/root/.local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"
PY="${PY:-/opt/venv/bin/python}"

# Editable install。源码已在 build 时烤入并 editable 安装（见 Dockerfile §7），故默认已可 import。
# 仅当 robomimic 不可导入时才装（例如运行时 bind 了一份未安装的源码覆盖）——这样 .sif 在集群无外网/只读
# 场景下直接跳过，不触发 uv 联网。editable 路径固定为 /workspace/droid_policy_learning，bind 覆盖同路径仍生效。
if ! "$PY" -c "import robomimic" 2>/dev/null; then
    if [ -f "/workspace/droid_policy_learning/pyproject.toml" ]; then
        echo ">> robomimic 不可导入，editable 安装中..."
        cd /workspace/droid_policy_learning && uv pip install -e . && cd - > /dev/null
    fi
else
    echo ">> robomimic 已安装（镜像烤入），跳过 editable 安装。"
fi

# Robocasa (when built with INCLUDE_ROBOCASA=1)
if [ "${INCLUDE_ROBOCASA}" = "1" ] && [ -f "/workspace/droid_policy_learning/benchmarks/robocasa/setup.py" ]; then
    echo ">> Installing robocasa (editable)..."
    cd /workspace/droid_policy_learning/benchmarks/robocasa && uv pip install -e . --no-deps && cd - > /dev/null

    ROBOCASA_ASSETS="${ROBOCASA_ASSETS:-/workspace/droid_policy_learning/benchmarks/robocasa/robocasa/models/assets}"
    if [ ! -d "${ROBOCASA_ASSETS}/textures" ] || [ -z "$(ls -A "${ROBOCASA_ASSETS}/textures" 2>/dev/null)" ]; then
        echo ">> Downloading robocasa kitchen assets (~5GB)..."
        yes | "$PY" /workspace/droid_policy_learning/benchmarks/robocasa/robocasa/scripts/download_kitchen_assets.py
    fi

    if ! "$PY" -c "import robocasa.macros_private" 2>/dev/null; then
        echo ">> Setting up robocasa macros..."
        "$PY" /workspace/droid_policy_learning/benchmarks/robocasa/robocasa/scripts/setup_macros.py
    fi
fi

# Claude Code CLI (optional: set INSTALL_CLAUDE_CODE=1 to install)
if [ "${INSTALL_CLAUDE_CODE}" = "1" ]; then
    echo ">> Installing Claude Code CLI..."
    if curl -fsSL https://claude.ai/install.sh | bash 2>/dev/null; then
        export PATH="${HOME}/.local/bin:${PATH}"
        echo ">> Claude Code CLI installed. Run 'claude' to start."
    else
        echo ">> Claude Code CLI install skipped (network or install script unavailable)."
    fi
fi

echo ">> Ready."
exec "$@"
