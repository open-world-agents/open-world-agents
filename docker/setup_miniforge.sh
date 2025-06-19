#!/bin/bash
set -e

MINIFORGE_VERSION=${1:-latest}
CONDA_INSTALL_PATH=${2:-/opt/conda}
CONDA_GID=2000

export DEBIAN_FRONTEND=noninteractive

# Create conda group and directory structure
groupadd -f -g "$CONDA_GID" conda
mkdir -p "$(dirname "$CONDA_INSTALL_PATH")"

# Download and install Miniforge
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
[ "$MINIFORGE_VERSION" != "latest" ] && \
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh"

wget -O miniforge.sh "$MINIFORGE_URL"
bash miniforge.sh -b -p "$CONDA_INSTALL_PATH"
rm miniforge.sh

# Set group ownership and permissions before configuration
chgrp -R conda "$CONDA_INSTALL_PATH"
# Add full group permissions and setgid on directories
chmod -R g+rw "$CONDA_INSTALL_PATH"
find "$CONDA_INSTALL_PATH" -type d -exec chmod g+xs {} \;

# Configure conda
"$CONDA_INSTALL_PATH/bin/conda" init --all
"$CONDA_INSTALL_PATH/bin/conda" config --set channel_priority strict \
    --set always_yes true \
    --set show_channel_urls true

# Install additional tools and clean up
"$CONDA_INSTALL_PATH/bin/pip" install uv virtual-uv
"$CONDA_INSTALL_PATH/bin/conda" clean -afy

echo "Miniforge installation completed at: $CONDA_INSTALL_PATH"