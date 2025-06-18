# 🐳 OWA Docker Images

Simple Docker setup for Open World Agents development and training.

## 🏗️ What You Get

Four images that build on each other:

```
owa/base:latest     ← CUDA + Python foundation
    ↓
owa/base:dev        ← + Development tools
    ↓
owa/runtime:dev     ← + Project dependencies
    ↓
owa/train:dev       ← + ML packages (PyTorch, etc.)
```

## 🚀 Quick Start

**Just want to train models?**
```bash
make train
docker run -it owa/train:dev
```

**Want everything?**
```bash
make all
```

**Want to customize?**
```bash
# Build training image directly from base (smaller, faster)
./build.sh train --from owa/base:latest

# Custom output name and tag (Docker-style)
./build.sh train -t my-train:minimal
# → Builds: my-train:minimal

# Use your own base with custom output
./build.sh train --from my-custom:tag -t my-train:v1.0
# → Builds: my-train:v1.0 (from my-custom:tag)
```

## 📋 Simple Commands

### Make (Recommended)
```bash
make base      # Build foundation
make dev       # Build dev environment
make project   # Build project environment
make train     # Build training environment
make all       # Build everything

make clean     # Remove all images
make list      # Show built images
```

### Build Script (More Options)
```bash
./build.sh train                                        # Build owa/train:dev
./build.sh train --from owa/base:latest                 # Build from custom base
./build.sh train -t my-train:minimal                    # Build my-train:minimal
./build.sh --registry ghcr.io/user --push all          # Build and push all
```

### Make with Custom Options
```bash
make train FROM=owa/base:latest TAG=my-train:minimal
# → Builds: my-train:minimal (from owa/base:latest)

make dev TAG=my-dev:custom
# → Builds: my-dev:custom
```

## 🎯 Common Use Cases

**I want to develop:**
```bash
make dev
docker run -it owa/base:dev
```

**I want to run the project:**
```bash
make project
docker run -it owa/runtime:dev
```

**I want to train models:**
```bash
make train
docker run -it owa/train:dev
# Working directory: /workspace/projects/nanovlm
# Branch: feature/data
# Packages: torch, transformers, wandb, etc.
```

**I want minimal training (no project deps):**
```bash
./build.sh train --from owa/base:latest -t owa/train:minimal
# → Builds: owa/train:minimal (from owa/base:latest)
```

## 📦 What's Inside

- **owa/base:latest** (~8GB) - CUDA 12.1 + Python 3.11 + Miniforge
- **owa/base:dev** (~9GB) - + zsh, git, development tools
- **owa/runtime:dev** (~10GB) - + project dependencies
- **owa/train:dev** (~12GB) - + PyTorch, transformers, wandb, datasets

## 🔧 Advanced Options

```bash
# Custom registry
make all REGISTRY=ghcr.io/myuser

# Custom tag
make all TAG=v1.0

# Build and push
make all PUSH=true

# No cache
./build.sh --no-cache all

# Complex custom build
./build.sh train --from owa/base:latest -t ghcr.io/myuser/my-trainer:v2.0 --push
# → Builds and pushes: ghcr.io/myuser/my-trainer:v2.0 (from owa/base:latest)
```

## 🆘 Need Help?

```bash
make help        # Show make targets
./build.sh -h    # Show build script options
```

**Problems?** The build script automatically handles dependencies - just run what you need!
