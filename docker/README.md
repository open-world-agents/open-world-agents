# OWA Docker Setup

This directory contains Docker configurations for Open World Agents development environment.

## Architecture

The Docker setup follows a 3-tier architecture:

1. **owa/base:latest** - Base image with CUDA and Miniforge
2. **owa/base:dev** - Development image with user environment and tools
3. **owa/runtime:dev** - Complete project environment with dependencies

## Quick Start

### Using Make (Recommended)

```bash
# Build all images
make build-all

# Build with custom registry and tag
make build-all REGISTRY=ghcr.io/myuser TAG=v1.0

# Build without cache
make build-all CACHE=false

# Build and push to registry
make build-all REGISTRY=ghcr.io/myuser PUSH=true

# Clean up images
make clean
```

### Using Build Script

```bash
# Build all images
./build.sh

# Build specific images
./build.sh base dev

# Build with custom options
./build.sh -r ghcr.io/myuser -t v1.0 -p all

# Show help
./build.sh --help
```

## Files

- `Dockerfile` - Base image with CUDA and Miniforge
- `Dockerfile.dev` - Development image extending base
- `Dockerfile.project-dev` - Project image with full environment
- `setup_miniforge.sh` - Miniforge installation script
- `setup_devcontainer.sh` - Development tools setup script
- `setup_project.sh` - Project dependencies setup script
- `build.sh` - Comprehensive build script
- `Makefile` - Simple build interface

## Build Options

### Build Script Options

- `-r, --registry` - Docker registry prefix
- `-t, --tag` - Image tag (default: latest)
- `-p, --push` - Push images after build
- `--no-cache` - Disable build cache
- `--platform` - Target platform (default: linux/amd64)
- `--build-arg` - Pass build arguments
- `--user-uid` - User UID for dev containers (default: current user)
- `--user-gid` - User GID for dev containers (default: current group)
- `--docker-gid` - Docker group GID (default: docker group or 998)

### Make Variables

- `REGISTRY` - Docker registry prefix
- `TAG` - Image tag (default: latest)
- `PUSH` - Push after build (default: false)
- `CACHE` - Enable Docker build cache (default: true)
- `USER_UID` - User UID for dev containers (default: current user)
- `USER_GID` - User GID for dev containers (default: current group)
- `DOCKER_GID` - Docker group GID (default: docker group or 998)

## Examples

```bash
# Development workflow
make build-dev                    # Build base and dev images
docker run -it owa/base:dev

# Full project build
make build-all REGISTRY=ghcr.io/owa TAG=latest

# Custom user/group IDs
make build-dev USER_UID=1001 USER_GID=1001 DOCKER_GID=999

# Custom Python version
./build.sh --build-arg PYTHON_VERSION=3.12 base

# Custom user IDs with build script
./build.sh --user-uid 1001 --user-gid 1001 --docker-gid 999 dev

# Multi-platform build
./build.sh --platform linux/amd64,linux/arm64 base
```

## Image Sizes

Approximate image sizes:
- owa/base:latest: ~8GB (CUDA + Miniforge)
- owa/base:dev: ~9GB (+ dev tools)
- owa/runtime:dev: ~10GB (+ project dependencies)
