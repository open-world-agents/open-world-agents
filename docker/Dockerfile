# OWA Base Image - CUDA + Miniforge
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Build arguments
ARG MINIFORGE_VERSION=latest
ARG CONDA_INSTALL_PATH=/opt/conda

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive

# Install miniforge
COPY ./setup_miniforge.sh /tmp/setup_miniforge.sh
RUN chmod +x /tmp/setup_miniforge.sh && \
    /tmp/setup_miniforge.sh ${MINIFORGE_VERSION} ${CONDA_INSTALL_PATH} && \
    rm /tmp/setup_miniforge.sh

# Add conda/mamba to PATH
ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"

# Default configuration
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]