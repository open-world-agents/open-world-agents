#!/bin/bash
set -e

# Set locale for user session
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Change shell to zsh
sudo chsh -s "$(which zsh)" "$(whoami)"

# Install Oh My Zsh
export RUNZSH=no CHSH=no
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Ensure custom plugins directory exists
mkdir -p ~/.oh-my-zsh/custom/plugins

# Install zsh plugins
git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions 2>/dev/null || true
git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting 2>/dev/null || true

# Configure .zshrc
if [ -f ~/.zshrc ]; then
    sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc
    sed -i 's/^ZSH_THEME=.*/ZSH_THEME="agnoster"/' ~/.zshrc
    grep -q "export LC_ALL=C.UTF-8" ~/.zshrc || {
        echo -e "\n# Locale settings\nexport LC_ALL=C.UTF-8\nexport LANG=C.UTF-8" >> ~/.zshrc
    }
fi

# Initialize conda for zsh
conda init --all
conda config --set channel_priority strict \
    --set always_yes true \
    --set show_channel_urls true

conda config --set auto_activate_base false
echo ". activate owa" >> ~/.zshrc

# Install useful tools
pip3 install glances gpustat
