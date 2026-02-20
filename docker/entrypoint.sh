#!/bin/bash
set -e

export PATH="/usr/local/bin:${PATH:-/usr/bin:/bin}"
eval "$(micromamba shell hook --shell bash --root-prefix ${MAMBA_ROOT_PREFIX:-/opt/conda})"
micromamba activate droid_env

# Editable install (project mounted at /workspace/droid_policy_learning)
if [ -f "/workspace/droid_policy_learning/setup.py" ] || [ -f "/workspace/droid_policy_learning/pyproject.toml" ]; then
    echo ">> Installing editable package..."
    cd /workspace/droid_policy_learning && pip install -e . --no-deps > /dev/null 2>&1 && cd - > /dev/null
fi

# Robocasa (when built with INCLUDE_ROBOCASA=1)
if [ "${INCLUDE_ROBOCASA}" = "1" ] && [ -f "/workspace/droid_policy_learning/benchmarks/robocasa/setup.py" ]; then
    echo ">> Installing robocasa (editable)..."
    cd /workspace/droid_policy_learning/benchmarks/robocasa && pip install -e . --no-deps && cd - > /dev/null

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

echo ">> Ready."
exec "$@"
