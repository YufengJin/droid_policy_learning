#!/bin/bash
set -e

# 0. 确保 micromamba 在 PATH 中（/usr/local/bin 通常已在 PATH）
export PATH="/usr/local/bin:${PATH:-/usr/bin:/bin}"

# 1. 初始化 Micromamba shell hook
# 这允许我们在脚本中使用 'micromamba activate'
eval "$(micromamba shell hook --shell bash --root-prefix ${MAMBA_ROOT_PREFIX:-/opt/conda})"

# 2. 激活环境
micromamba activate droid_env

# 3. 安装/刷新 Editable Install (关键步骤)
# 项目由 volume 挂载 ../:/workspace/droid_policy_learning，构建阶段无法预先安装。
# 每次容器启动时执行 pip install -e . --no-deps 确保环境正确链接到挂载的代码。
if [ -f "/workspace/droid_policy_learning/setup.py" ] || [ -f "/workspace/droid_policy_learning/pyproject.toml" ]; then
    echo ">> Detected project in /workspace/droid_policy_learning. Installing editable package..."
    cd /workspace/droid_policy_learning
    pip install -e . --no-deps > /dev/null 2>&1
    cd - > /dev/null
fi

echo ">> Environment 'droid_env' activated."
echo ">> Ready."

# 4. 执行用户命令 (默认为 CMD ["/bin/bash"])
exec "$@"