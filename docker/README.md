# 🐳 OWA Docker Images

Simple Docker build for Open World Agents.

## 🏗️ What Gets Built

Three images in sequence:
```
owa/base:latest     ← Ubuntu 24.04 + Python + Miniforge
    ↓
owa/runtime:latest  ← + Project dependencies
    ↓
owa/train:latest    ← CUDA + PyTorch + ML packages
```

## 🚀 Quick Start

```bash
make build
# or
./build.sh
```

Then run:
```bash
docker run -it owa/train:latest
```

## 📋 Commands

```bash
make build     # Build all images
make clean     # Remove all images
make list      # Show built images
```

## 📦 What's Inside

- **owa/base:latest** (765MB) - Ubuntu 24.04 + Python + Miniforge
- **owa/runtime:latest** (1.8GB) - + project dependencies
- **owa/train:latest** (14.6GB) - CUDA 12.6 + PyTorch + flash-attention

## 🔧 Development

For development environment, see `.devcontainer/` directory.


