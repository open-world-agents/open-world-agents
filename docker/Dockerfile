# Base Image - Miniforge + uv/vuv
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# Build arguments
ARG MINIFORGE_VERSION=latest
ARG CONDA_INSTALL_PATH=/opt/conda

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive 
ENV USER=root UID=0 GID=0 HOME=/root

# Install system dependencies first (rarely changes)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential

# Install miniforge
RUN curl -fsSL https://raw.githubusercontent.com/MilkClouds/devcontainer-feature-installer/main/install-feature-cli.sh | bash
RUN feature-install ghcr.io/MilkClouds/devcontainer-features/miniforge
ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv tool install virtual-uv
ENV VUV_ALLOW_BASE=true

# Default configuration
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
