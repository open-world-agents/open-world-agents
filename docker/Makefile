# OWA Docker Makefile - Simple Build Interface

.PHONY: help base dev project train all clean list

# Default values
REGISTRY ?=
TAG ?= latest
PUSH ?= false

help: ## Show help
	@echo "🐳 OWA Docker Build"
	@echo ""
	@echo "Simple targets:"
	@echo "  make base      # Build owa/base:latest"
	@echo "  make dev       # Build owa/base:dev"
	@echo "  make project   # Build owa/runtime:dev"
	@echo "  make train     # Build owa/train:dev"
	@echo "  make all       # Build all images"
	@echo ""
	@echo "Custom builds:"
	@echo "  make train FROM=owa/base:latest TAG=my-train:minimal"
	@echo "  make dev TAG=my-dev:custom"
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

base: ## Build base image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) base

dev: ## Build dev image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) dev

project: ## Build project image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) project

train: ## Build training image
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(TAG),-t $(TAG)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) train

all: ## Build all images
	./build.sh $(if $(REGISTRY),--registry $(REGISTRY)) $(if $(FROM),--from $(FROM)) $(if $(filter true,$(PUSH)),--push) all

clean: ## Remove all images
	@echo "🧹 Cleaning up..."
	-docker rmi owa/train:dev owa/runtime:dev owa/base:dev owa/base:latest 2>/dev/null || true

list: ## List built images
	@echo "📋 Built images:"
	@docker images --filter "reference=owa/*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
