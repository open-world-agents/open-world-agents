#!/bin/bash
set -e

USERNAME=${1:-vscode}
USER_UID=${2:-1000}
USER_GID=${3:-1000}
DOCKER_GID=${4:-998}

echo "Setting up devcontainer for user: $USERNAME (UID: $USER_UID, GID: $USER_GID)"

# Set locale and environment
export DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8

# Create user and groups
groupadd -g "$USER_GID" "$USERNAME"
useradd -m -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME"

# Remove passwords for passwordless sudo
passwd -d root
passwd -d "$USERNAME"

# Install essential development packages
apt-get update && apt-get install -y --no-install-recommends \
    sudo git curl wget vim tmux bash-completion htop tree nano jq unzip \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    zsh locales rsync

# Generate locale and configure user permissions
locale-gen en_US.UTF-8
usermod -aG sudo "$USERNAME"
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"
chmod 0440 "/etc/sudoers.d/$USERNAME"

# Setup docker group
groupadd -g "$DOCKER_GID" docker 2>/dev/null || true
usermod -aG docker "$USERNAME"

# Setup conda group for efficient permission management (if conda exists)
if [ -d "/opt/conda" ]; then
    groupadd -g 2000 conda 2>/dev/null || true
    usermod -aG conda "$USERNAME"
    # Only change group ownership (faster than chown -R)
    chgrp -R conda /opt/conda
    chmod -R g+w /opt/conda
    echo "Conda group permissions configured"
fi

echo "Devcontainer setup completed successfully"