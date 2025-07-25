# Base Image - Miniforge + uv/vuv
# Can be built on Ubuntu (lightweight) or CUDA (for training pipeline)
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
COPY docker/setup_miniforge.sh /tmp/setup_miniforge.sh
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    chmod +x /tmp/setup_miniforge.sh && \
    /tmp/setup_miniforge.sh ${MINIFORGE_VERSION} ${CONDA_INSTALL_PATH} && \
    rm -f /tmp/setup_miniforge.sh

ENV VUV_ALLOW_BASE=true PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

# Default configuration
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]
