# 🐳 OWA Docker Images

Simple Docker build for Open World Agents.

## 🏗️ What Gets Built

Two images in sequence:
```
owa/base:latest     ← Ubuntu 24.04 + Python + Miniforge
    ↓
owa/runtime:latest  ← + Project dependencies
```

## 🚀 Quick Start

```bash
make build
# or
./build.sh
```

Then run:
```bash
docker run -it owa/runtime:latest
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

## 🔧 Development

For development environment, see `.devcontainer/` directory.

