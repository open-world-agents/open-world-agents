# Devcontainer Image - Add Development Tools to Any Base Image
ARG BASE_IMAGE=owa/train:latest
FROM ${BASE_IMAGE}

# User setup arguments
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ARG CONDA_INSTALL_PATH=/opt/conda

# System setup (user creation, packages, sudoers)
COPY ./.devcontainer/devcontainer_system.sh /tmp/devcontainer_system.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    chmod +x /tmp/devcontainer_system.sh && \
    /tmp/devcontainer_system.sh ${USERNAME} ${USER_UID} ${USER_GID} && \
    rm /tmp/devcontainer_system.sh

# Install useful tools
RUN --mount=type=cache,target=${CONDA_INSTALL_PATH}/pkgs,sharing=locked \
    . activate owa && \
    mamba install -y ffmpeg open-world-agents::gstreamer-bundle
RUN uv pip install --system --no-cache-dir glances gpustat

# Switch to user and setup environment
USER ${USERNAME}
ENV USER=${USERNAME} UID=${USER_UID} GID=${USER_GID} HOME=/home/${USERNAME}
WORKDIR ${HOME}

# User environment setup (shell, conda config)
COPY --chown=${USER}:${USER} ./.devcontainer/devcontainer_user.sh /tmp/devcontainer_user.sh
COPY --chown=${USER}:${USER} ./.devcontainer/aliases.sh ${HOME}/.devcontainer/aliases.sh
RUN --mount=type=cache,target=${HOME}/.cache,sharing=locked,uid=${USER_UID},gid=${USER_GID} \
    chmod +x /tmp/devcontainer_user.sh && \
    /tmp/devcontainer_user.sh && \
    rm /tmp/devcontainer_user.sh

# Default configuration
SHELL ["/bin/zsh", "-c"]
CMD ["/bin/zsh"]
