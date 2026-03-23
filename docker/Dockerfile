# =============================================================================
# Base: CUDA 11.8
# =============================================================================
ARG CUDA_VERSION=11.8
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-devel-ubuntu20.04
ARG INCLUDE_ROBOCASA=0

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake unzip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "${INCLUDE_ROBOCASA}" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 libhidapi-libusb0 \
        lsof psmisc \
    && rm -rf /var/lib/apt/lists/*; fi

# -----------------------------------------------------------------------------
# 2. uv (Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# 运行时常见把宿主机目录 bind 到 /root，会遮住镜像内 /root/.local/bin；复制 uv 到 PATH 固定位置
RUN cp /root/.local/bin/uv /usr/local/bin/uv && chmod +x /usr/local/bin/uv

# -----------------------------------------------------------------------------
# 3. Python venv (3.10) via uv
# -----------------------------------------------------------------------------
RUN uv venv --python 3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
# uv 把 Python 装在 /root/.local/share/uv/python/...；bind /root 时软链接失效。
# 复制到 /opt/python310 并重链，保证任何 bind 场景下 venv Python 可用。
RUN _py=$(find /root/.local/share/uv/python -maxdepth 3 -name 'python3.10' -type f 2>/dev/null | head -1) && \
    if [ -n "$_py" ]; then \
        _dir=$(dirname "$(dirname "$_py")") && \
        cp -a "$_dir" /opt/python310 && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python3 && \
        ln -sf /opt/python310/bin/python3.10 /opt/venv/bin/python3.10; \
    fi

# -----------------------------------------------------------------------------
# 4. Third-party: Octo (not on PyPI; installed after locked deps)
# -----------------------------------------------------------------------------
WORKDIR /opt/third_party
RUN git clone https://github.com/octo-models/octo.git && \
    cd octo && git checkout 85b83fc19657ab407a7f56558a5384ae56fe453b

RUN if [ "${INCLUDE_ROBOCASA}" = "1" ]; then \
    cd /opt/third_party && git clone https://github.com/ARISE-Initiative/robosuite.git robosuite && \
    cd robosuite && git checkout v1.5.2 && uv pip install -e .; fi

# -----------------------------------------------------------------------------
# 5. Locked runtime stack (PyTorch cu118, JAX CUDA, TF, etc.) via uv sync
# -----------------------------------------------------------------------------
WORKDIR /build
COPY pyproject.toml uv.lock ./

# Sync into /opt/venv (--active). CMAKE_POLICY_VERSION_MINIMUM for egl_probe builds.
# Optional Robocasa stack: `uv sync --extra robocasa` (numba/protobuf resolved in uv.lock).
RUN if [ "${INCLUDE_ROBOCASA}" = "1" ]; then \
      CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --frozen --no-install-project --active --extra robocasa; \
    else \
      CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --frozen --no-install-project --active; \
    fi

ENV INCLUDE_ROBOCASA=${INCLUDE_ROBOCASA}

# -----------------------------------------------------------------------------
# 6. Octo package (editable)
# -----------------------------------------------------------------------------
RUN uv pip install -e /opt/third_party/octo

# -----------------------------------------------------------------------------
# 7. Workspace (project mounted at runtime; entrypoint does pip install -e .)
# -----------------------------------------------------------------------------
WORKDIR /workspace/droid_policy_learning
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
