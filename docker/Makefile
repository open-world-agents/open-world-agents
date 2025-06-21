# OWA Docker Makefile - Simple Build Interface

.PHONY: help base runtime train all clean list

# Default values
REGISTRY ?=
TAG ?= latest
PUSH ?= false

help: ## Show help
	@echo "🐳 OWA Docker Build"
	@echo ""
	@echo "Main workflow (base -> runtime -> train):"
	@echo "  make base      # Build owa/base:latest"
	@echo "  make runtime   # Build owa/runtime:latest"
	@echo "  make train     # Build owa/train:latest"
	@echo "  make all       # Build all images"
	@echo ""
	@echo "Custom builds:"
	@echo "  make train FROM=owa/base:latest TAG=my-train:minimal"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean     # Remove all images"
	@echo "  make list      # List built images"
	@echo ""
	@echo "Variables:"
	@echo "  FROM=...       # Base image to build from"
	@echo "  TAG=...        # Output image name:tag (like docker -t)"
	@echo "  REGISTRY=...   # Docker registry prefix"
	@echo "  PUSH=true      # Push after build"
	@echo ""
	@echo "For devcontainer builds, use .devcontainer/ directory"

base: ## Build base image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) base

runtime: ## Build runtime image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) runtime

train: ## Build training image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) train

all: ## Build all images
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) all

clean: ## Remove all images
	@echo "🧹 Cleaning up..."
	-docker rmi owa/train:latest owa/runtime:latest owa/base:latest 2>/dev/null || true

list: ## List built images
	@echo "📋 Built images:"
	@docker images --filter "reference=owa/*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
