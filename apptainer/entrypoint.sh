#!/bin/bash
# 已弃用：当前 apptainer/droid_policy.def 使用 docker/entrypoint.sh（构建时打入镜像 /usr/local/bin/entrypoint.sh）。
# 若需在宿主机包装启动命令，请直接调用：
#   apptainer run ... image.sif /usr/local/bin/entrypoint.sh bash
# 或使用本目录下的 run_container.sh
echo "[WARN] apptainer/entrypoint.sh is obsolete; use docker/entrypoint.sh inside the .sif or run_container.sh" >&2
exec "$@"
