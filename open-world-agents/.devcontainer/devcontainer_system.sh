#!/bin/bash
set -e

USERNAME=${1:-vscode}
USER_UID=${2:-1000}
USER_GID=${3:-1000}

echo "Setting up devcontainer for user: $USERNAME (UID: $USER_UID, GID: $USER_GID)"

# Set locale and environment
export DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8

# Create or rename group
EXISTING_GROUP=$(getent group "$USER_GID" | cut -d: -f1)
if [ -n "$EXISTING_GROUP" ]; then
    if [ "$EXISTING_GROUP" != "$USERNAME" ]; then
        echo "Renaming existing group: $EXISTING_GROUP (GID: $USER_GID) -> $USERNAME"
        groupmod -n "$USERNAME" "$EXISTING_GROUP"
    else
        echo "Group $USERNAME (GID: $USER_GID) already exists"
    fi
else
    groupadd -g "$USER_GID" "$USERNAME"
fi

# Create or rename user
EXISTING_USER=$(getent passwd "$USER_UID" | cut -d: -f1)
if [ -n "$EXISTING_USER" ]; then
    if [ "$EXISTING_USER" != "$USERNAME" ]; then
        echo "Renaming existing user: $EXISTING_USER (UID: $USER_UID) -> $USERNAME"
        usermod -l "$USERNAME" "$EXISTING_USER"
        usermod -d "/home/$USERNAME" -m "$USERNAME"
        usermod -g "$USER_GID" "$USERNAME"
    else
        echo "User $USERNAME (UID: $USER_UID) already exists"
    fi
else
    useradd -m -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME"
fi

# Remove passwords for passwordless sudo
passwd -d root || true
passwd -d "$USERNAME" || true

# Update package lists
apt-get update

# Install system utilities and shell tools
apt-get install -y --no-install-recommends \
    sudo bash-completion zsh tmux \
    htop tree locales

# Install editors and text processing
apt-get install -y --no-install-recommends \
    vim vim-runtime nano

# Install network and download tools
apt-get install -y --no-install-recommends \
    git curl wget aria2 openssh-client \
    jq unzip rsync xz-utils

# Install development tools and libraries
apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev zlib1g-dev libffi-dev \
    libreadline-dev libsqlite3-dev libncursesw5-dev

# Install X11 applications and utilities
apt-get install -y --no-install-recommends \
    x11-apps xauth

# Generate locale and configure sudoers
locale-gen en_US.UTF-8
usermod -aG sudo "$USERNAME"
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"
chmod 0440 "/etc/sudoers.d/$USERNAME"

echo "Devcontainer setup completed successfully"