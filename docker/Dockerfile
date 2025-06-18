# OWA Base Image - CUDA + Miniforge
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Build arguments
ARG MINIFORGE_VERSION=latest
ARG CONDA_INSTALL_PATH=/opt/conda

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

# Install system dependencies first (rarely changes)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates

# Install miniforge (copy script last to maximize cache hits)
COPY ./setup_miniforge.sh /tmp/setup_miniforge.sh
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    chmod +x /tmp/setup_miniforge.sh && \
    /tmp/setup_miniforge.sh ${MINIFORGE_VERSION} ${CONDA_INSTALL_PATH} && \
    rm -f /tmp/setup_miniforge.sh

# Default configuration
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]