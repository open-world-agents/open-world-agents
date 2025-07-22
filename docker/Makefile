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
	@echo "  owa/base:ubuntu     - Ubuntu + Python"
	@echo "  owa/base:cuda       - CUDA + Python"
	@echo "  owa/base:latest     - Ubuntu + Python (alias)"
	@echo "  owa/runtime:ubuntu  - Ubuntu + project deps"
	@echo "  owa/runtime:cuda    - CUDA + project deps"
	@echo "  owa/runtime:latest  - Ubuntu + project deps (alias)"
	@echo "  owa/train:latest    - CUDA + ML packages"

build: ## Build all images
	./build.sh

clean: ## Remove all images
	@echo "🧹 Cleaning up..."
	-docker rmi owa/train:latest owa/runtime:cuda owa/runtime:ubuntu owa/runtime:latest owa/base:cuda owa/base:ubuntu owa/base:latest 2>/dev/null || true

list: ## List built images
	@echo "📋 Built images:"
	@docker images --filter "reference=owa/*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
