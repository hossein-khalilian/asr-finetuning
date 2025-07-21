.PHONY: help create-manifests upload-manifests

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

create-manifests: ## Generate dataset manifests using the specified dataset name
	@if [ -z "$(dataset)" ]; then \
		echo "Error: dataset is not set. Use 'make create-manifests dataset=your_dataset_name'"; \
		exit 1; \
	fi
	@echo "Generating manifests for dataset: $(dataset)"
	@python -m scripts.create_manifests --dataset $(dataset)

upload-manifests: ## Upload manifests using the specified manifests-path
	@if [ -z "$(manifests-path)" ]; then \
		echo "Error: manifests-path is not set. Use 'make upload-manifests manifests-path=your_path'"; \
		exit 1; \
	fi
	@echo "Uploading manifests from: $(manifests-path)"
	@./scripts/upload_nemo_dataset.sh $(manifests-path)

