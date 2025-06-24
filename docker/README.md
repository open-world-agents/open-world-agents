# 🐳 OWA Docker Images

Simple Docker setup for Open World Agents development and training.

## 🏗️ What You Get

Two separate workflows:

**Main Workflow (Production):**
```
owa/base:latest     ← CUDA + Python foundation
    ↓
owa/runtime:latest  ← + Project dependencies
    ↓
owa/train:latest    ← + ML packages (PyTorch, etc.)
```

**Devcontainer Workflow (Development):**
```
Any base image → Add dev tools → Dev variant
owa/base:latest    → owa/base:dev
owa/runtime:latest → owa/runtime:dev
owa/train:latest   → owa/train:dev
```

## 🚀 Quick Start

**Just want to train models? (Production)**
```bash
make train
docker run -it owa/train:latest
```

**Want development environment?**
```bash
# Use VS Code with devcontainer extension
# or build manually in .devcontainer/ directory
```

**Want everything?**
```bash
make all      # Build main workflow (base, runtime, train)
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
make runtime   # Build runtime environment
make train     # Build training environment
make all       # Build all images

make clean     # Remove all images
make list      # Show built images
```

### Build Script (More Options)
```bash
./build.sh train                                        # Build owa/train:latest
./build.sh train --from owa/base:latest                 # Build from custom base
./build.sh train -t my-train:minimal                    # Build my-train:minimal
./build.sh --registry ghcr.io/user --push all          # Build and push all images
```

### Make with Custom Options
```bash
make train FROM=owa/base:latest TAG=my-train:minimal
# → Builds: my-train:minimal (from owa/base:latest)

make dev TAG=my-dev:custom
# → Builds: my-dev:custom
```

## 🎯 Common Use Cases

**I want to develop (with VS Code devcontainer):**
```bash
# Open in VS Code with devcontainer extension
# Uses .devcontainer/devcontainer.json automatically
```

**I want to run the project:**
```bash
make runtime
docker run -it owa/runtime:latest
```

**I want to train models:**
```bash
make train
docker run -it owa/train:latest
# Working directory: /workspace/projects/nanoVLM
# Branch: feature/data
# Packages: torch, transformers, wandb, etc.
```

**I want minimal training (no project deps):**
```bash
./build.sh train --from owa/base:latest -t owa/train:minimal
# → Builds: owa/train:minimal (from owa/base:latest)
```

## 📦 What's Inside

- **owa/base:latest** (3.05GB) - CUDA 12.6 + Python 3.11 + Miniforge
- **owa/runtime:latest** (4GB) - + project dependencies
- **owa/train:latest** (9.68GB) - + PyTorch, transformers, wandb, datasets

For development environments, see `.devcontainer/` directory.

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

## 📁 Directory Structure

```
docker/                     # Main Docker files
├── Dockerfile              # Base image (CUDA + Miniforge)
├── Dockerfile.runtime      # Runtime image (+ project deps)
├── Dockerfile.train        # Training image (+ ML packages)
├── setup_runtime.sh        # Runtime setup script
├── setup_miniforge.sh      # Miniforge installation
├── build.sh               # Build script
├── Makefile               # Make targets
└── README.md              # This file

.devcontainer/             # Devcontainer files
├── devcontainer.json      # VS Code devcontainer config
├── Dockerfile             # Single devcontainer Dockerfile (adds dev tools to any base)
├── devcontainer_system.sh # System setup for devcontainer
└── devcontainer_user.sh   # User setup for devcontainer
```

The separation allows:
- **Main workflow**: Clean production images without dev tools
- **Single devcontainer**: One Dockerfile that adds dev tools to any base image
- **Flexibility**: Transform any image (base, runtime, train) into a dev variant
