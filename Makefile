.PHONY: help train-low-1gpu train-low-mgpu train-full-1gpu train-full-mgpu

# Defaults (override at runtime, e.g. make train-full-mgpu NPROC=4 HEAD=linear)
NPROC ?= 8
HEAD ?= mlp

help:
	@printf "Targets:\n"
	@printf "  make train-low-1gpu    # reduced dataset, single GPU\n"
	@printf "  make train-low-mgpu    # reduced dataset, multi GPU (torchrun)\n"
	@printf "  make train-full-1gpu   # full dataset, single GPU\n"
	@printf "  make train-full-mgpu   # full dataset, multi GPU (torchrun)\n"
	@printf "\nOptions:\n"
	@printf "  NPROC=<num_gpus> (default: %s)\n" "$(NPROC)"
	@printf "  HEAD=mlp|linear (default: %s)\n" "$(HEAD)"

train-low-1gpu:
	DATA_MODE=reduced HEAD_TYPE=$(HEAD) PYTHONPATH=. python -u src/train.py

train-low-mgpu:
	DATA_MODE=reduced HEAD_TYPE=$(HEAD) PYTHONPATH=. torchrun --nproc_per_node=$(NPROC) -u src/train.py

train-full-1gpu:
	DATA_MODE=full HEAD_TYPE=$(HEAD) PYTHONPATH=. python -u src/train.py

train-full-mgpu:
	DATA_MODE=full HEAD_TYPE=$(HEAD) PYTHONPATH=. torchrun --nproc_per_node=$(NPROC) -u src/train.py
