.PHONY: help train-low train-full

# Defaults (override at runtime, e.g. make train-full HEAD=linear)
HEAD ?= mlp

help:
	@printf "Targets:\n"
	@printf "  make train-low    # reduced dataset, single GPU\n"
	@printf "  make train-full   # full dataset, single GPU\n"
	@printf "\nOptions:\n"
	@printf "  HEAD=mlp|linear (default: %s)\n" "$(HEAD)"

train-low:
	DATA_MODE=reduced HEAD_TYPE=$(HEAD) PYTHONPATH=. python -u src/train.py

train-full:
	DATA_MODE=full HEAD_TYPE=$(HEAD) PYTHONPATH=. python -u src/train.py
