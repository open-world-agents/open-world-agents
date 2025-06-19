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
    # Terminal and shell environment
    sudo bash-completion zsh tmux vim nano htop tree \
    # Network and file utilities
    git curl wget openssh-client jq unzip rsync xz-utils locales \
    # Compilation and build dependencies
    build-essential libssl-dev zlib1g-dev libbz2-dev liblzma-dev \
    # Language runtime libraries
    libreadline-dev libsqlite3-dev libncursesw5-dev tk-dev libxml2-dev libxmlsec1-dev libffi-dev

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
    usermod -aG conda "$USERNAME"
    echo "Conda group permissions configured"
fi

echo "Devcontainer setup completed successfully"