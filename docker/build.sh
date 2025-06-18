#!/bin/bash
set -e

# OWA Docker Build Script
# Builds the 3-tier Docker image architecture for Open World Agents

# Default values
REGISTRY=""
TAG="latest"
PUSH=false
CACHE=true
PLATFORM="linux/amd64"
BUILD_ARGS=""

# User/Group defaults from environment
USER_UID="${USER_UID:-$(id -u)}"
USER_GID="${USER_GID:-$(id -g)}"
DOCKER_GID="${DOCKER_GID:-$(getent group docker | cut -d: -f3 2>/dev/null || echo 998)}"

# Image names
BASE_IMAGE="owa/base"
DEV_IMAGE="owa/base"
PROJECT_IMAGE="owa/runtime"

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
OWA Docker Build Script

Usage: $0 [OPTIONS] [IMAGES...]

IMAGES:
    base        Build owa/base:latest image only
    dev         Build owa/base:dev image only
    project     Build owa/runtime:dev image only
    all         Build all images (default)

OPTIONS:
    -r, --registry REGISTRY    Docker registry prefix (e.g., ghcr.io/user)
    -t, --tag TAG             Tag for images (default: latest)
    -p, --push                Push images to registry after build
    --no-cache                Disable Docker build cache
    --platform PLATFORM       Target platform (default: linux/amd64)
    --build-arg KEY=VALUE     Pass build arguments to Docker
    --user-uid UID            User UID for dev containers (default: current user)
    --user-gid GID            User GID for dev containers (default: current group)
    --docker-gid GID          Docker group GID (default: docker group or 998)
    -h, --help                Show this help message

EXAMPLES:
    $0                                    # Build all images with default settings
    $0 base dev                          # Build only base and dev images
    $0 -r ghcr.io/myuser -t v1.0 -p all  # Build all, tag as v1.0, and push
    $0 --build-arg PYTHON_VERSION=3.12   # Build with custom Python version
    $0 --user-uid 1001 --user-gid 1001   # Build with custom user/group IDs

EOF
}

# Parse command line arguments
IMAGES_TO_BUILD=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --user-uid)
            USER_UID="$2"
            shift 2
            ;;
        --user-gid)
            USER_GID="$2"
            shift 2
            ;;
        --docker-gid)
            DOCKER_GID="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        base|dev|project|all)
            IMAGES_TO_BUILD+=("$1")
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to building all images if none specified
if [ ${#IMAGES_TO_BUILD[@]} -eq 0 ]; then
    IMAGES_TO_BUILD=("all")
fi

# Expand "all" to individual images
if [[ " ${IMAGES_TO_BUILD[*]} " =~ " all " ]]; then
    IMAGES_TO_BUILD=("base" "dev" "project")
fi

# Build registry prefix
if [ -n "$REGISTRY" ]; then
    REGISTRY_PREFIX="${REGISTRY}/"
else
    REGISTRY_PREFIX=""
fi

# Build cache options
if [ "$CACHE" = false ]; then
    CACHE_OPTS="--no-cache"
else
    CACHE_OPTS=""
fi

# Function to build an image
build_image() {
    local image_type="$1"
    local dockerfile="$2"
    local image_name="$3"
    local base_image="$4"
    local image_tag="$5"

    # Use provided tag or default to TAG
    local tag_to_use="${image_tag:-$TAG}"
    local full_image_name="${REGISTRY_PREFIX}${image_name}:${tag_to_use}"

    log_info "Building $image_type image: $full_image_name"

    local build_cmd="docker build"
    build_cmd="$build_cmd --platform $PLATFORM"
    build_cmd="$build_cmd $CACHE_OPTS"
    build_cmd="$build_cmd $BUILD_ARGS"

    # Add user/group arguments for dev and project images
    if [[ "$image_type" == "dev" || "$image_type" == "project" ]]; then
        build_cmd="$build_cmd --build-arg USER_UID=$USER_UID"
        build_cmd="$build_cmd --build-arg USER_GID=$USER_GID"
        build_cmd="$build_cmd --build-arg DOCKER_GID=$DOCKER_GID"
    fi

    if [ -n "$base_image" ]; then
        build_cmd="$build_cmd --build-arg BASE_IMAGE=$base_image"
    fi

    build_cmd="$build_cmd -f $dockerfile"
    build_cmd="$build_cmd -t $full_image_name"
    build_cmd="$build_cmd ."

    log_info "Running: $build_cmd"
    eval $build_cmd

    log_success "Built $full_image_name"

    if [ "$PUSH" = true ]; then
        log_info "Pushing $full_image_name"
        docker push "$full_image_name"
        log_success "Pushed $full_image_name"
    fi
}

# Change to docker directory
cd "$(dirname "$0")"

# Build images in dependency order
for image in "${IMAGES_TO_BUILD[@]}"; do
    case $image in
        base)
            build_image "base" "Dockerfile" "$BASE_IMAGE" "" "latest"
            ;;
        dev)
            # Check if base image exists or needs to be built
            base_full_name="${REGISTRY_PREFIX}${BASE_IMAGE}:latest"
            if ! docker image inspect "$base_full_name" >/dev/null 2>&1; then
                log_warning "Base image $base_full_name not found, building it first"
                build_image "base" "Dockerfile" "$BASE_IMAGE" "" "latest"
            fi
            build_image "dev" "Dockerfile.dev" "$DEV_IMAGE" "$base_full_name" "dev"
            ;;
        project)
            # Check if dev image exists or needs to be built
            dev_full_name="${REGISTRY_PREFIX}${DEV_IMAGE}:dev"
            if ! docker image inspect "$dev_full_name" >/dev/null 2>&1; then
                log_warning "Dev image $dev_full_name not found, building dependency chain"

                # Check base image
                base_full_name="${REGISTRY_PREFIX}${BASE_IMAGE}:latest"
                if ! docker image inspect "$base_full_name" >/dev/null 2>&1; then
                    build_image "base" "Dockerfile" "$BASE_IMAGE" "" "latest"
                fi

                build_image "dev" "Dockerfile.dev" "$DEV_IMAGE" "$base_full_name" "dev"
            fi
            build_image "project" "Dockerfile.project-dev" "$PROJECT_IMAGE" "$dev_full_name" "dev"
            ;;
    esac
done

log_success "Build completed successfully!"

# Show built images
log_info "Built images:"
for image in "${IMAGES_TO_BUILD[@]}"; do
    case $image in
        base)
            echo "  ${REGISTRY_PREFIX}${BASE_IMAGE}:latest"
            ;;
        dev)
            echo "  ${REGISTRY_PREFIX}${DEV_IMAGE}:dev"
            ;;
        project)
            echo "  ${REGISTRY_PREFIX}${PROJECT_IMAGE}:dev"
            ;;
    esac
done
