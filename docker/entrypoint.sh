#!/bin/bash
set -e

export PATH="/opt/venv/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"

# Editable install (project mounted at /workspace/droid_policy_learning)
if [ -f "/workspace/droid_policy_learning/pyproject.toml" ]; then
    echo ">> Installing editable package (pyproject)..."
    cd /workspace/droid_policy_learning && uv pip install -e . && cd - > /dev/null
elif [ -f "/workspace/droid_policy_learning/setup.py" ]; then
    echo ">> Installing editable package (setup.py, no-deps)..."
    cd /workspace/droid_policy_learning && uv pip install -e . --no-deps && cd - > /dev/null
fi

# Robocasa (when built with INCLUDE_ROBOCASA=1)
if [ "${INCLUDE_ROBOCASA}" = "1" ] && [ -f "/workspace/droid_policy_learning/benchmarks/robocasa/setup.py" ]; then
    echo ">> Installing robocasa (editable)..."
    cd /workspace/droid_policy_learning/benchmarks/robocasa && uv pip install -e . --no-deps && cd - > /dev/null

    ROBOCASA_ASSETS="${ROBOCASA_ASSETS:-/workspace/droid_policy_learning/benchmarks/robocasa/robocasa/models/assets}"
    if [ ! -d "${ROBOCASA_ASSETS}/textures" ] || [ -z "$(ls -A "${ROBOCASA_ASSETS}/textures" 2>/dev/null)" ]; then
        echo ">> Downloading robocasa kitchen assets (~5GB)..."
        yes | python /workspace/droid_policy_learning/benchmarks/robocasa/robocasa/scripts/download_kitchen_assets.py
    fi

    if ! python -c "import robocasa.macros_private" 2>/dev/null; then
        echo ">> Setting up robocasa macros..."
        python /workspace/droid_policy_learning/benchmarks/robocasa/robocasa/scripts/setup_macros.py
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
