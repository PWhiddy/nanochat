#!/bin/bash
set -eo pipefail

# Run from the repository root: bash runs/speedrun_80m_nope.sh
# Same tokenizer, data, FP8 training, evaluation, and SFT pipeline as speedrun.sh.
# 15 layers, width 512, 4 attention heads (128 dims), MLP width 2048.
# 80,740,408 total parameters at vocab_size=32768:
#   47,185,920 attention/MLP + 33,554,432 embedding/lm_head + 56 scalars/gate.
# No per-layer value embedding tables or gates, and no positional encoding (NoPE).
SPEEDRUN_BASE_ARGS=(
    --depth=15
    --aspect-ratio=32
    --head-dim=128
    --no-value-embeddings
    --position-encoding=nope
)
# Per-device batch size for pretraining/evaluation; SFT inherits it from the checkpoint.
SPEEDRUN_DEVICE_BATCH_SIZE=32
# Pass the tag to every stage so existing larger checkpoints cannot be selected.
SPEEDRUN_MODEL_TAG=80m_nope
source "$(dirname "${BASH_SOURCE[0]}")/speedrun.sh"
