.PHONY: lint fetch-slurm-jobs

BRANCH ?= main

default:
	./run_dashboard.sh

lint:
	uvx pre-commit run --all-files
	uvx ty check

fetch-slurm-jobs:
	@echo "📥 Fetching slurm job scripts from dynamo repo (branch: $(BRANCH))..."
	@rm -rf .tmp-dynamo
	@git clone --depth 1 --filter=blob:none --sparse --branch $(BRANCH) https://github.com/ai-dynamo/dynamo.git .tmp-dynamo
	@cd .tmp-dynamo && git sparse-checkout set examples/backends/sglang/slurm_jobs
	@mkdir -p slurm_jobs
	@cp -r .tmp-dynamo/examples/backends/sglang/slurm_jobs/* slurm_jobs/
	@rm -rf .tmp-dynamo
	@echo "✅ Slurm job scripts copied to ./slurm_jobs/"

