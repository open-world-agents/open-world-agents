#!/bin/bash
set -e

# OWA Docker Build Script
# Builds the 3-tier Docker image architecture for Open World Agents

# Default values
REGISTRY=""
CUSTOM_TAG=""
PUSH=false
CACHE=true
CUSTOM_BASE_IMAGE=""

# User/Group defaults
USER_UID="${USER_UID:-$(id -u)}"
USER_GID="${USER_GID:-$(id -g)}"
DOCKER_GID="${DOCKER_GID:-$(getent group docker | cut -d: -f3 2>/dev/null || echo 998)}"

# Image names
BASE_IMAGE="owa/base"
DEV_IMAGE="owa/base"
PROJECT_IMAGE="owa/runtime"
TRAIN_IMAGE="owa/train"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
🐳 OWA Docker Build Script

Usage: $0 [OPTIONS] [IMAGES...]

IMAGES:
    base        Build owa/base:latest
    runtime     Build owa/runtime:latest
    train       Build owa/train:latest
    all         Build all images (base, runtime, train)

OPTIONS:
    -t, --tag NAME:TAG         Output image name and tag (like docker build -t)
    --from IMAGE               Base image to build from
    --registry REGISTRY        Docker registry prefix
    --push                     Push after build
    --no-cache                 Disable cache
    -h, --help                 Show help

EXAMPLES:
    $0 train                                        # Build owa/train:latest
    $0 train --from owa/base:latest                 # Build from custom base
    $0 train -t my-train:minimal                    # Build my-train:minimal
    $0 --registry ghcr.io/user --push all          # Build and push all images

EOF
}

# Parse arguments
IMAGES_TO_BUILD=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag) CUSTOM_TAG="$2"; shift 2 ;;
        --from) CUSTOM_BASE_IMAGE="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --push) PUSH=true; shift ;;
        --no-cache) CACHE=false; shift ;;
        -h|--help) show_help; exit 0 ;;
        base|runtime|train|all) IMAGES_TO_BUILD+=("$1"); shift ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Default to all if no images specified
[ ${#IMAGES_TO_BUILD[@]} -eq 0 ] && IMAGES_TO_BUILD=("all")

# Expand "all"
[[ " ${IMAGES_TO_BUILD[*]} " =~ " all " ]] && IMAGES_TO_BUILD=("base" "runtime" "train")

# Warn if using custom tag with multiple images
if [ -n "$CUSTOM_TAG" ] && [ ${#IMAGES_TO_BUILD[@]} -gt 1 ]; then
    log_warning "Custom -t tag ignored when building multiple images. Use -t with single image only."
fi

# Setup
REGISTRY_PREFIX="${REGISTRY:+$REGISTRY/}"
CACHE_OPTS=""
[ "$CACHE" = false ] && CACHE_OPTS="--no-cache"

# Get default output name and tag for image type
get_defaults() {
    local type="$1"
    case $type in
        base) echo "owa/base:latest" ;;
        runtime) echo "owa/runtime:latest" ;;
        train) echo "owa/train:latest" ;;
    esac
}

# Parse docker-style tag (name:tag format)
parse_tag() {
    local input="$1"
    if [[ "$input" == *":"* ]]; then
        echo "$input"  # Already has tag
    else
        # Get default tag for the image type being built
        local type="$2"
        local default_tag
        case $type in
            base) default_tag="latest" ;;
            *) default_tag="dev" ;;
        esac
        echo "$input:$default_tag"
    fi
}

# Build an image
build_image() {
    local type="$1" dockerfile="$2" base="$3"
    local start_time=$(date +%s)

    # Determine output name:tag
    local full_name
    # Only use custom tag for single image builds, not for "all"
    if [ -n "$CUSTOM_TAG" ] && [ ${#IMAGES_TO_BUILD[@]} -eq 1 ]; then
        full_name=$(parse_tag "$CUSTOM_TAG" "$type")
    else
        full_name=$(get_defaults "$type")
    fi

    # Add registry prefix
    full_name="${REGISTRY_PREFIX}${full_name}"

    log_info "🏗️  Building $type: $full_name"
    [ -n "$base" ] && log_info "📦 From: $base"

    # Build command
    local cmd="docker build $CACHE_OPTS"

    # Add base image arg
    if [ -n "$base" ]; then
        case $type in
            runtime) cmd="$cmd --build-arg BASE_IMAGE=$base" ;;
            train) cmd="$cmd --build-arg RUNTIME_IMAGE=$base" ;;
        esac
    fi

    cmd="$cmd -f $dockerfile -t $full_name ."

    log_info "🔨 $cmd"

    # Execute build with error handling
    if eval $cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "✅ Built $full_name (${duration}s)"

        # Get image size
        local size=$(docker images --format "table {{.Size}}" "$full_name" | tail -n 1)
        log_info "📏 Size: $size"
    else
        log_error "❌ Failed to build $full_name"
        return 1
    fi

    if [ "$PUSH" = true ]; then
        log_info "📤 Pushing..."
        if docker push "$full_name"; then
            log_success "✅ Pushed $full_name"
        else
            log_error "❌ Failed to push $full_name"
            return 1
        fi
    fi

    # Store built image name for final output
    BUILT_IMAGES+=("$full_name")
}

# Enable BuildKit for cache mounts
export DOCKER_BUILDKIT=1

# Change to docker directory
cd "$(dirname "$0")"

# Track built images for final output
BUILT_IMAGES=()

# Get default base for image type
get_base() {
    case $1 in
        runtime) echo "${REGISTRY_PREFIX}owa/base:latest" ;;
        train) echo "${REGISTRY_PREFIX}owa/runtime:latest" ;;
    esac
}

# Check if image exists
exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Build dependencies if needed (only when using defaults)
ensure_deps() {
    case $1 in
        runtime)
            local base="${REGISTRY_PREFIX}owa/base:latest"
            exists "$base" || build_image "base" "Dockerfile" ""
            ;;
        train)
            local runtime="${REGISTRY_PREFIX}owa/runtime:latest"
            if ! exists "$runtime"; then
                ensure_deps "runtime"
                build_image "runtime" "Dockerfile.runtime" "${REGISTRY_PREFIX}owa/base:latest"
            fi
            ;;
    esac
}

# Build images
for image in "${IMAGES_TO_BUILD[@]}"; do
    case $image in
        base)
            build_image "base" "Dockerfile" ""
            ;;
        runtime)
            base="${CUSTOM_BASE_IMAGE:-$(get_base runtime)}"
            [ -z "$CUSTOM_BASE_IMAGE" ] && ensure_deps "runtime"
            build_image "runtime" "Dockerfile.runtime" "$base"
            ;;
        train)
            base="${CUSTOM_BASE_IMAGE:-$(get_base train)}"
            [ -z "$CUSTOM_BASE_IMAGE" ] && ensure_deps "train"
            build_image "train" "Dockerfile.train" "$base"
            ;;
    esac
done

log_success "🎉 Build completed!"

# Show built images
log_info "📋 Built images:"
for image in "${BUILT_IMAGES[@]}"; do
    echo "  $image"
done
