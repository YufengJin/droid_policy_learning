# ------------------------------------------------------------
# Base Image: CUDA 11.8
# ------------------------------------------------------------
    ARG CUDA_VERSION=11.8
    FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-devel-ubuntu20.04
    
    # Basic Env Setup
    ENV DEBIAN_FRONTEND=noninteractive
    ENV MAMBA_ROOT_PREFIX=/opt/conda
    ENV PATH=/opt/conda/bin:$PATH
    ENV LC_ALL=C.UTF-8
    ENV LANG=C.UTF-8
    
    # ------------------------------------------------------------
    # 1. System Dependencies
    # ------------------------------------------------------------
    RUN apt-get update && apt-get install -y \
        curl \
        wget \
        bzip2 \
        ca-certificates \
        git \
        build-essential \
        cmake \
        unzip \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # ------------------------------------------------------------
    # 2. Install Micromamba
    # ------------------------------------------------------------
    RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
        tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
    
    # ------------------------------------------------------------
    # 3. Create Python Environment (Python 3.10)
    # ------------------------------------------------------------
    ARG ENV_NAME=droid_env
    RUN micromamba create -y -n ${ENV_NAME} python=3.10.* -c conda-forge && \
        micromamba clean -a -y
    
    # Setup shell hook for micromamba
    # Initialize micromamba shell hook so it works in all shells
    # Note: MAMBA_ROOT_PREFIX is already set via ENV, so no need for --prefix
    RUN micromamba shell init --shell bash
    # Ensure mamba.sh is sourced and environment is activated in .bashrc
    RUN echo '' >> /root/.bashrc && \
        echo '# Micromamba initialization' >> /root/.bashrc && \
        echo 'if [ -f /opt/conda/etc/profile.d/mamba.sh ]; then' >> /root/.bashrc && \
        echo '    . /opt/conda/etc/profile.d/mamba.sh' >> /root/.bashrc && \
        echo 'fi' >> /root/.bashrc && \
        echo "micromamba activate ${ENV_NAME}" >> /root/.bashrc
    # Ensure .bashrc is sourced in login shells (for docker exec)
    RUN echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> /root/.bash_profile
    
    # ------------------------------------------------------------
    # 4. Third-Party Source Dependencies (Octo Source)
    # ------------------------------------------------------------
    WORKDIR /opt/third_party
    # 根据您提供的 snippet，此处使用 v1.5 tag
    RUN git clone https://github.com/octo-models/octo.git && \
        cd octo && \
        git checkout 85b83fc19657ab407a7f56558a5384ae56fe453b
    
    # ------------------------------------------------------------
    # 5. CRITICAL: Install PyTorch & JAX (GPU Versions)
    # ------------------------------------------------------------
    RUN micromamba run -n ${ENV_NAME} pip install --upgrade pip
    
    # [FIX]: Explicitly install Torch 2.0.1 + CUDA 11.8 via pip index
    # using pip is safer than conda for exact CUDA version matching here.
    RUN micromamba run -n ${ENV_NAME} pip install \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Install JAX (CUDA 11.8) for Octo
    RUN micromamba run -n ${ENV_NAME} pip install --upgrade \
        "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # ------------------------------------------------------------
    # 6. Install Merged Python Dependencies (Droid + Octo)
    # ------------------------------------------------------------
    # 注意：这里去掉了 torch/torchvision，因为上面已经安装了特定版本
    # Install cmake Python package (required for egl_probe build)
    # Note: System cmake (apt) is for C/C++ compilation, while Python cmake
    # package (pip) provides Python API that egl_probe's build system needs
    RUN micromamba run -n ${ENV_NAME} pip install cmake
    # Install dependencies with specific versions preserved, >= versions install latest
    RUN micromamba run -n ${ENV_NAME} pip install \
        # --- Droid Core ---
        "numpy==1.24.3" \
        h5py \
        psutil \
        tqdm \
        termcolor \
        tensorboard \
        tensorboardX \
        imageio \
        imageio-ffmpeg \
        matplotlib \
        "diffusers==0.11.1" \
        opencv-python \
        "transformers==4.34.0" \
        # --- Octo Core ---
        gym \
        "ml_dtypes==0.2.0" \
        "chex==0.1.85" \
        "optax==0.1.5" \
        "tensorflow_probability==0.23.0" \
        "tensorflow==2.15.0" \
        "distrax==0.1.5" \
        "flax==0.7.5" \
        ml_collections \
        absl-py \
        scipy \
        wandb \
        einops \
        moviepy \
        "pre-commit==3.3.3" \
        tensorflow_hub \
        tensorflow_text \
        "tensorflow_datasets==4.9.2" \
        "tensorflow_graphics==2021.12.3" \
        "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972" \
        plotly
    # Install egl_probe separately with --no-build-isolation to access installed cmake
    # Set CMAKE_POLICY_VERSION_MINIMUM to work around CMake compatibility issue
    # (egl_probe's CMakeLists.txt requires old CMake version, but system has newer CMake)
    RUN micromamba run -n ${ENV_NAME} bash -c "export CMAKE_POLICY_VERSION_MINIMUM=3.5 && pip install --no-build-isolation 'egl_probe>=1.0.1'"
    
    # ------------------------------------------------------------
    # 7. Install Octo Package
    # ------------------------------------------------------------
    RUN micromamba run -n ${ENV_NAME} pip install -e /opt/third_party/octo
    
    # ------------------------------------------------------------
    # 8. Project Setup
    # ------------------------------------------------------------
    WORKDIR /workspace/droid_policy_learning
    COPY . /workspace/droid_policy_learning
    
    # Install current project
    RUN micromamba run -n ${ENV_NAME} pip install -e . --no-deps
    
    # ------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------
    COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
    RUN chmod +x /usr/local/bin/entrypoint.sh
    
    ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
    CMD ["/bin/bash"]