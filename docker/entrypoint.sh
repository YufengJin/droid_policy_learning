#!/bin/bash
set -e

# 1. 初始化 Micromamba shell hook
# 这允许我们在脚本中使用 'micromamba activate'
eval "$(micromamba shell hook --shell bash)"

# 2. 激活环境
micromamba activate droid_env

# 3. 修复 Editable Install (关键步骤)
# 当我们将宿主机目录挂载到 /workspace/droid_policy_learning 时，构建阶段生成的 egg-link 可能会失效。
# 这里重新执行一次 install -e . --no-deps 确保环境正确链接到挂载的代码。
if [ -f "/workspace/droid_policy_learning/setup.py" ] || [ -f "/workspace/droid_policy_learning/pyproject.toml" ]; then
    echo ">> Detected local project in /workspace/droid_policy_learning. Refreshing editable install..."
    cd /workspace/droid_policy_learning
    # --no-deps 确保不重新下载庞大的依赖，只修复链接
    pip install -e . --no-deps > /dev/null 2>&1
    cd - > /dev/null
fi

echo ">> Environment 'droid_env' activated."
echo ">> Ready."

# 4. 执行用户命令 (默认为 CMD ["/bin/bash"])
exec "$@"