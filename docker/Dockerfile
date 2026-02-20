# =============================================================================
# Base: CUDA 11.8
# =============================================================================
ARG CUDA_VERSION=11.8
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-devel-ubuntu20.04
ARG INCLUDE_ROBOCASA=0
ARG ENV_NAME=droid_env

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
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
# 2. Micromamba
# -----------------------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# -----------------------------------------------------------------------------
# 3. Python environment (3.10)
# -----------------------------------------------------------------------------
RUN micromamba create -y -n ${ENV_NAME} python=3.10.* -c conda-forge && micromamba clean -a -y
RUN micromamba shell init --shell bash
RUN echo '' >> /root/.bashrc && \
    echo '# Micromamba' >> /root/.bashrc && \
    echo 'if [ -f /opt/conda/etc/profile.d/mamba.sh ]; then . /opt/conda/etc/profile.d/mamba.sh; fi' >> /root/.bashrc && \
    echo "micromamba activate ${ENV_NAME}" >> /root/.bashrc
RUN echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> /root/.bash_profile

# -----------------------------------------------------------------------------
# 4. Third-party: Octo
# -----------------------------------------------------------------------------
WORKDIR /opt/third_party
RUN git clone https://github.com/octo-models/octo.git && \
    cd octo && git checkout 85b83fc19657ab407a7f56558a5384ae56fe453b

RUN if [ "${INCLUDE_ROBOCASA}" = "1" ]; then \
    cd /opt/third_party && git clone https://github.com/ARISE-Initiative/robosuite.git robosuite && \
    cd robosuite && git checkout v1.5.2 && micromamba run -n ${ENV_NAME} pip install -e .; fi

# -----------------------------------------------------------------------------
# 5. PyTorch & JAX (GPU)
# -----------------------------------------------------------------------------
RUN micromamba run -n ${ENV_NAME} pip install --upgrade pip
RUN micromamba run -n ${ENV_NAME} pip install \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
RUN micromamba run -n ${ENV_NAME} pip install --upgrade \
    "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# -----------------------------------------------------------------------------
# 6. Python dependencies (Droid + Octo)
# -----------------------------------------------------------------------------
RUN micromamba run -n ${ENV_NAME} pip install cmake
RUN micromamba run -n ${ENV_NAME} pip install \
    "numpy==1.24.3" "protobuf>=3.20.3,<4.25" h5py psutil tqdm termcolor tensorboard tensorboardX \
    imageio imageio-ffmpeg matplotlib "diffusers==0.11.1" opencv-python "transformers==4.34.0" \
    gym "ml_dtypes==0.2.0" "chex==0.1.85" "optax==0.1.5" "tensorflow_probability==0.23.0" \
    "tensorflow==2.15.0" "distrax==0.1.5" "flax==0.7.5" ml_collections absl-py scipy \
    wandb einops moviepy "pre-commit==3.3.3" tensorflow_hub tensorflow_text \
    "tensorflow_datasets==4.9.2" "tensorflow_graphics==2021.12.3" \
    "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972" \
    plotly "hydra-core>=1.1.0"

RUN micromamba run -n ${ENV_NAME} bash -c \
    "export CMAKE_POLICY_VERSION_MINIMUM=3.5 && pip install --no-build-isolation 'egl_probe>=1.0.1'"

RUN if [ "${INCLUDE_ROBOCASA}" = "1" ]; then \
    micromamba run -n ${ENV_NAME} pip install \
        "numba==0.56.4" "mujoco==3.2.6" pygame Pillow pyyaml pynput lxml hidapi \
        "tianshou==0.4.10" "policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git"; fi

ENV INCLUDE_ROBOCASA=${INCLUDE_ROBOCASA}

# -----------------------------------------------------------------------------
# 7. Octo package
# -----------------------------------------------------------------------------
RUN micromamba run -n ${ENV_NAME} pip install -e /opt/third_party/octo

# -----------------------------------------------------------------------------
# 8. Workspace (project mounted at runtime; entrypoint does pip install -e .)
# -----------------------------------------------------------------------------
WORKDIR /workspace/droid_policy_learning
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
