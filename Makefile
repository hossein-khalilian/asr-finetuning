.PHONY: help create-manifests upload-manifests create-tokenizer combine-manifests full-tokenizer-pipeline

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

create-tokenizer: ## Process ASR tokenizer. Required: manifest, vocab_size. Optional: tokenizer (default: wpe), spe_type (only for spe), data_root (default: ~/.cache/asr-finetuning/tokenizers)
	@if [ -z "$(manifest)" ]; then \
		echo "Error: manifest is not set. Use 'make create-tokenizer manifest=path1,path2,...'"; \
		exit 1; \
	fi
	@if [ -z "$(vocab_size)" ]; then \
		echo "Error: vocab_size is not set. Use 'make create-tokenizer vocab_size=1024'"; \
		exit 1; \
	fi
	@data_root=$(if $(data_root),$(data_root),~/.cache/asr-finetuning/tokenizers); \
	tokenizer=$(if $(tokenizer),$(tokenizer),wpe); \
	if [ "$$tokenizer" = "spe" ] && [ -z "$(spe_type)" ]; then \
		echo "Error: spe_type is required when tokenizer=spe"; \
		exit 1; \
	fi; \
	echo "Using data_root: $$data_root"; \
	echo "Using tokenizer: $$tokenizer"; \
	cmd="python scripts/process_asr_text_tokenizer.py \
		--manifest=$(manifest) \
		--data_root=$$data_root \
		--vocab_size=$(vocab_size) \
		--tokenizer=$$tokenizer"; \
	if [ "$$tokenizer" = "spe" ]; then \
		cmd="$$cmd --spe_type=$(spe_type)"; \
	fi; \
	cmd="$$cmd --log"; \
	eval "$$cmd"

combine-manifests: ## Combine multiple manifests. Usage: make combine-manifests manifests="path1 path2 ..."
	@if [ -z "$(manifests)" ]; then \
		echo "Error: manifests is not set. Use 'make combine-manifests manifests=\"path1 path2 ...\"'"; \
		exit 1; \
	fi
	@echo "Combining manifests:"; \
	for f in $(manifests); do echo "  $$f"; done; \
	python scripts/combine_manifests.py $(manifests)

full-tokenizer-pipeline: ## Run the full pipeline: create, combine manifests, and create tokenizer
	@$(MAKE) create-manifests dataset=hsekhalilian/fleurs
	@$(MAKE) create-manifests dataset=hsekhalilian/sorted_commonvoice
	@$(MAKE) combine-manifests manifests="~/.cache/asr-finetuning/datasets/hsekhalilian___fleurs/manifests/train_manifest.json ~/.cache/asr-finetuning/datasets/hsekhalilian___sorted_commonvoice/manifests/train_manifest.json"
	@$(MAKE) create-tokenizer manifest=~/.cache/asr-finetuning/datasets/combined/train_manifest.json vocab_size=1024 tokenizer=spe spe_type=bpe
