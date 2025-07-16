# OWA Docker Makefile

.PHONY: help build clean list

help: ## Show help
	@echo "🐳 OWA Docker Build"
	@echo ""
	@echo "Commands:"
	@echo "  make build     # Build all images (base → runtime → train)"
	@echo "  make clean     # Remove all images"
	@echo "  make list      # List built images"
	@echo ""
	@echo "Images built:"
	@echo "  owa/base:latest     - Ubuntu + Python"
	@echo "  owa/runtime:latest  - base + project deps"
	@echo "  owa/train:latest    - CUDA + ML packages"

build: ## Build all images
	./build.sh

clean: ## Remove all images
	@echo "🧹 Cleaning up..."
	-docker rmi owa/train:latest owa/runtime:latest owa/base:latest 2>/dev/null || true

list: ## List built images
	@echo "📋 Built images:"
	@docker images --filter "reference=owa/*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
