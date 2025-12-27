# Devcontainer Image - Add Development Tools to Any Base Image
ARG BASE_IMAGE=owa/train:latest
FROM ${BASE_IMAGE}

ARG CONDA_INSTALL_PATH=/opt/conda

# Install necessary tools
RUN --mount=type=cache,target=${CONDA_INSTALL_PATH}/pkgs,sharing=locked \
    . activate owa && \
    mamba install -y ffmpeg open-world-agents::gstreamer-bundle
