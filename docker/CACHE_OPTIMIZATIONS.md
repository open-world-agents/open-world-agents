# Docker Cache Optimizations

This document outlines the cache optimizations applied to the OWA Docker build system to improve build performance using BuildKit cache mounts.

## Summary of Changes

### 1. BuildKit Cache Mounts
- **APT Cache Mounts**: `--mount=type=cache,target=/var/cache/apt,sharing=locked` and `/var/lib/apt`
- **Conda Package Cache**: `--mount=type=cache,target=/opt/conda/pkgs,sharing=locked`
- **User Cache Directories**: `--mount=type=cache,target=/home/${USERNAME}/.cache,sharing=locked,uid=${USER_UID},gid=${USER_GID}`
- **Root Cache**: `--mount=type=cache,target=/root/.cache,sharing=locked`

### 2. Layer Ordering Optimization
- **Moved frequently changing operations later** in Dockerfiles to maximize cache hits
- **Separated system dependencies** from application-specific setup
- **Optimized COPY operations** to minimize cache invalidation

### 3. Package Manager Optimizations

#### APT (System Packages)
- Persistent cache mounts eliminate need for manual cleanup
- Shared cache across builds with `sharing=locked`

#### Conda/Mamba
- Persistent package cache at `/opt/conda/pkgs`
- Optimized conda configuration:
  - `--set always_yes true` (reduces interactive prompts)
  - `--set show_channel_urls true` (better debugging)

#### Git Operations
- Used shallow clones (`--depth=1`) for:
  - Oh My Zsh plugins
  - Project repository clone
- Reduces download time and disk usage

#### Python/Pip
- Persistent cache directories for pip, uv, and pipx
- User-specific cache mounts with proper ownership

### 4. Environment Variable Optimization
- **Combined ENV statements** to reduce layers
- **Moved PATH updates** to single ENV instruction

### 5. Script Simplification
- **Removed manual cache cleanup** from scripts (handled by cache mounts)
- **Kept scripts focused** on installation logic only
- **Maintained shallow git clones** for performance

## File-by-File Changes

### `Dockerfile` (Base Image)
- **Added cache mounts**: APT cache and conda package cache
- **Combined ENV statements** for PATH and DEBIAN_FRONTEND
- **Simplified cleanup**: Only remove script files, cache handled by mounts

### `Dockerfile.dev` (Development Image)
- **Added cache mounts**: APT cache, conda packages, and user cache directories
- **User-specific cache ownership**: Proper UID/GID for user cache mounts
- **Improved layer organization** with cache mount integration

### `Dockerfile.project-dev` (Project Image)
- **Added comprehensive cache mounts**: Conda, user cache, and root cache
- **Added missing ARG declarations**: USER_UID and USER_GID for cache ownership
- **Streamlined with cache mounts**: No manual cleanup needed

### `setup_miniforge.sh`
- **Simplified script**: Removed manual cache cleanup (handled by mounts)
- **Kept conda configuration optimization**
- **Focused on installation logic only**

### `devcontainer_system.sh`
- **Simplified APT operations**: No manual cleanup needed
- **Kept essential package installation**
- **Cache persistence handled by mounts**

### `devcontainer_user.sh`
- **Kept shallow git clones** for performance
- **Simplified pip installations**: Cache handled by mounts
- **Removed manual cache cleanup**

### `setup_project.sh`
- **Kept shallow git clone** for project repository
- **Simplified package installations**: Cache handled by mounts
- **Removed manual cache cleanup operations**

## Expected Benefits

1. **Significantly Faster Builds**: Persistent cache mounts eliminate repeated downloads
2. **Consistent Performance**: Cache shared across builds and developers
3. **Better Layer Reuse**: Optimized layer ordering maximizes cache hits
4. **Reduced Network Usage**: Shallow clones and persistent package caches
5. **Cleaner Scripts**: Simplified logic without manual cache management

## Build Performance Tips

- **Enable BuildKit**: Required for cache mounts (`DOCKER_BUILDKIT=1`)
- **Use shared caches**: Cache mounts persist across builds
- **Monitor cache usage**: `docker system df` shows cache sizes
- **Clean when needed**: `docker builder prune` to clear build cache

## Cache Mount Details

### APT Cache Mounts
```dockerfile
--mount=type=cache,target=/var/cache/apt,sharing=locked
--mount=type=cache,target=/var/lib/apt,sharing=locked
```

### Conda Package Cache
```dockerfile
--mount=type=cache,target=/opt/conda/pkgs,sharing=locked
```

### User-Specific Caches
```dockerfile
--mount=type=cache,target=/home/${USERNAME}/.cache,sharing=locked,uid=${USER_UID},gid=${USER_GID}
```

## Requirements

- **Docker BuildKit**: Must be enabled (`DOCKER_BUILDKIT=1`)
- **Docker 18.09+**: For BuildKit support
- **Proper permissions**: User cache mounts require correct UID/GID

## Future Optimization Opportunities

1. **Multi-stage builds** for further size reduction
2. **Registry cache imports** for CI/CD pipelines
3. **Base image updates** to newer Ubuntu/CUDA versions
4. **Dependency pinning** for more predictable builds
