# Docker Setup for DROID Policy Learning

## Prerequisites

- Docker (Compose v2+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

## Dependencies (uv)

Runtime versions are pinned in `pyproject.toml` and `uv.lock`. Refresh the lockfile after editing dependencies:

```bash
uv lock
```

Docker installs the locked environment with `uv sync --frozen --no-install-project` (see `docker/Dockerfile`).

## Build

From project root:

```bash
# Base (DROID + Octo only)
docker compose -f docker/docker-compose.headless.yaml build

# With Robocasa (~5GB extra; assets downloaded on first run)
docker compose -f docker/docker-compose.headless.yaml build --build-arg INCLUDE_ROBOCASA=1
```

### Import smoke test (after build)

```bash
docker build -f docker/Dockerfile -t droid-policy:uv-test .
docker run --rm -v "$PWD:/workspace/droid_policy_learning" -w /workspace/droid_policy_learning droid-policy:uv-test \
  python -u scripts/test_train_imports.py
```

| Build arg       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `INCLUDE_ROBOCASA` | 0     | Set to `1` for Robocasa simulation   |
| `CUDA_VERSION`  | 11.8    | Base image CUDA version              |

## Volume mounts

Update paths in the compose file to match your system.

| Container path                  | Purpose                  |
|---------------------------------|--------------------------|
| `../` → `/workspace/droid_policy_learning` | Project (editable install) |
| `/path/to/droid` → `/workspace/datasets/droid` | DROID dataset (ro)   |
| `/path/to/robocasa` → `/workspace/datasets/robocasa` | Robocasa data (ro) |
| `${HOME}/.cache/huggingface`    | HF model cache           |

## Usage

### Headless (training)

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it droid-dev-headless bash
# Inside: cd /workspace/droid_policy_learning
```

### X11 (GUI / Robocasa)

```bash
xhost +local:
docker compose -f docker/docker-compose.x11.yaml up -d
docker exec -it droid-dev-gui bash
```

### One-off

```bash
docker run --rm --gpus all \
  -v /path/to/droid_policy_learning:/workspace/droid_policy_learning \
  -v /path/to/datasets:/workspace/datasets \
  droid-policy:latest python -c "print('hello')"
```

## Entrypoint

On each start:

1. Uses uv-managed venv at `/opt/venv` (Python 3.10)
2. When `pyproject.toml` is present: `uv pip install -e .` (deps already synced in the image). Legacy `setup.py`-only trees use `uv pip install -e . --no-deps`
3. If `INCLUDE_ROBOCASA=1`: installs Robocasa, downloads assets if needed, runs `setup_macros`
4. If `INSTALL_CLAUDE_CODE=1`: installs [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/setup) (`claude`). Requires network. Run `claude` after install; use `claude login` on first use.

## Robocasa

When built with `INCLUDE_ROBOCASA=1`:

- Robocasa installed from `benchmarks/robocasa` at container start
- Kitchen assets (~5GB) downloaded on first run to `benchmarks/robocasa/robocasa/models/assets/`
- Ensure `benchmarks/robocasa` exists (e.g. submodule) and project is bind-mounted
