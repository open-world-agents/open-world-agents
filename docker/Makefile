# OWA Docker Makefile
# Simple interface for building Docker images

.PHONY: help build-base build-dev build-project build-all clean push

# Default values
REGISTRY ?=
TAG ?= latest
PUSH ?= false
CACHE ?= true

# User/Group defaults from environment
USER_UID ?= $(shell id -u)
USER_GID ?= $(shell id -g)
DOCKER_GID ?= $(shell getent group docker | cut -d: -f3 2>/dev/null || echo 998)

# Image names
BASE_IMAGE = owa/base
DEV_IMAGE = owa/base
PROJECT_IMAGE = owa/runtime

# Registry prefix
ifdef REGISTRY
	REGISTRY_PREFIX = $(REGISTRY)/
else
	REGISTRY_PREFIX =
endif

help: ## Show this help message
	@echo "OWA Docker Build Commands"
	@echo ""
	@echo "Usage: make [target] [REGISTRY=...] [TAG=...] [PUSH=true] [USER_UID=...] [USER_GID=...] [DOCKER_GID=...]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  REGISTRY    Docker registry prefix (e.g., ghcr.io/user)"
	@echo "  TAG         Tag for images (default: latest)"
	@echo "  PUSH        Push images after build (default: false)"
	@echo "  CACHE       Enable Docker build cache (default: true)"
	@echo "  USER_UID    User UID for dev containers (default: current user)"
	@echo "  USER_GID    User GID for dev containers (default: current group)"
	@echo "  DOCKER_GID  Docker group GID (default: docker group or 998)"
	@echo ""
	@echo "Examples:"
	@echo "  make build-all                           # Build all images"
	@echo "  make build-all REGISTRY=ghcr.io/user    # Build with registry"
	@echo "  make build-all TAG=v1.0 PUSH=true       # Build, tag, and push"
	@echo "  make build-all CACHE=false               # Build without cache"
	@echo "  make build-all USER_UID=1001 USER_GID=1001  # Build with custom user/group"

build-base: ## Build base image only
	./build.sh $(if $(REGISTRY),-r $(REGISTRY)) -t $(TAG) $(if $(filter true,$(PUSH)),-p) $(if $(filter false,$(CACHE)),--no-cache) base

build-dev: ## Build base and dev images
	./build.sh $(if $(REGISTRY),-r $(REGISTRY)) -t $(TAG) $(if $(filter true,$(PUSH)),-p) $(if $(filter false,$(CACHE)),--no-cache) --user-uid $(USER_UID) --user-gid $(USER_GID) --docker-gid $(DOCKER_GID) base dev

build-project: ## Build all images including project
	./build.sh $(if $(REGISTRY),-r $(REGISTRY)) -t $(TAG) $(if $(filter true,$(PUSH)),-p) $(if $(filter false,$(CACHE)),--no-cache) --user-uid $(USER_UID) --user-gid $(USER_GID) --docker-gid $(DOCKER_GID) all

build-all: build-project ## Build all images (alias for build-project)

clean: ## Remove all OWA Docker images
	@echo "Removing OWA Docker images..."
	-docker rmi $(REGISTRY_PREFIX)$(PROJECT_IMAGE):dev 2>/dev/null || true
	-docker rmi $(REGISTRY_PREFIX)$(DEV_IMAGE):dev 2>/dev/null || true
	-docker rmi $(REGISTRY_PREFIX)$(BASE_IMAGE):latest 2>/dev/null || true
	@echo "Cleanup completed"

push: ## Push all images to registry
	@if [ -z "$(REGISTRY)" ]; then \
		echo "Error: REGISTRY must be set to push images"; \
		exit 1; \
	fi
	docker push $(REGISTRY_PREFIX)$(BASE_IMAGE):latest
	docker push $(REGISTRY_PREFIX)$(DEV_IMAGE):dev
	docker push $(REGISTRY_PREFIX)$(PROJECT_IMAGE):dev

list: ## List built OWA images
	@echo "OWA Docker Images:"
	@docker images --filter "reference=$(REGISTRY_PREFIX)$(BASE_IMAGE)" --filter "reference=$(REGISTRY_PREFIX)$(DEV_IMAGE)" --filter "reference=$(REGISTRY_PREFIX)$(PROJECT_IMAGE)" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
