#!/bin/bash
set -e

# OWA Docker Build Script
export DOCKER_BUILDKIT=1
cd "$(dirname "$0")/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to build Docker image with timing and error handling
build_image() {
    local name="$1"
    local tag="$2"
    local build_command="$3"
    
    log_info "🏗️  Building $name image..."
    local start_time=$(date +%s)
    
    if eval "$build_command"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local size=$(docker images --format "{{.Size}}" "$tag" | head -1)
        log_success "✅ Built $tag (${duration}s, $size)"
    else
        log_error "❌ Failed to build $name image"
        exit 1
    fi
}

# Build all images
log_info "🚀 Building all Docker images..."

# Base image
build_image "base" "owa/base:latest" "docker build --build-arg BASE_IMAGE=ubuntu:24.04 -f docker/Dockerfile -t owa/base:latest ."

# Runtime image
build_image "runtime" "owa/runtime:latest" "docker build --build-arg BASE_IMAGE=owa/base:latest -f docker/Dockerfile.runtime -t owa/runtime:latest ."

# Training image (from CUDA runtime)
build_image "base-cuda" "owa/base:cuda" "docker build --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 -f docker/Dockerfile -t owa/base:cuda ."
build_image "runtime-cuda" "owa/runtime:cuda" "docker build --build-arg BASE_IMAGE=owa/base:cuda -f docker/Dockerfile.runtime -t owa/runtime:cuda ."
build_image "training" "owa/train:latest" "docker build --build-arg BASE_IMAGE=owa/runtime:cuda -f docker/Dockerfile.train -t owa/train:latest ."
docker image rm owa/base:cuda owa/runtime:cuda

log_success "🎉 All images built successfully!"