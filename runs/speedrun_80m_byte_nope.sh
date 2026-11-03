#!/bin/bash
set -eo pipefail

# Run from the repository root: bash runs/speedrun_80m_byte_nope.sh
# Separate byte experiment; existing speedrun scripts/artifacts stay unchanged.
# 78,970,938 parameters: 16 layers, width 640, 5 heads, MLP width 2560.
# NoPE, no value embeddings, exactly 256 vocabulary IDs (including chat markers).
# 4096 context x batch 8 = the same tokens/micro-batch as 2048 x 16.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer_byte256"
export NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix_byte256"
model_tag=80m_byte_nope
experiment_dir="$NANOCHAT_BASE_DIR/experiments/$model_tag"
mkdir -p "$NANOCHAT_BASE_DIR" "$NANOCHAT_DATA_DIR"
# Reuse existing cached files without adding shards to another experiment's data.
shared_data_dir="$(cd "$NANOCHAT_BASE_DIR" && pwd)/base_data_climbmix"
if [ -d "$shared_data_dir" ]; then
    for shard in "$shared_data_dir"/shard_*.parquet; do
        [ -f "$shard" ] || continue
        destination="$NANOCHAT_DATA_DIR/${shard##*/}"
        if [ ! -e "$destination" ]; then
            ln -s "$shard" "$destination"
        fi
    done
fi

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu --group dev
source .venv/bin/activate
run="${WANDB_RUN:-dummy}"

python -m nanochat.dataset -n 8
# The shorter context in source bytes crops more long documents. Add fresh shards
# in this isolated directory to leave headroom for the calibrated ~27.6B-byte run.
python -m nanochat.dataset -n 200 &
dataset_download_pid=$!
python -m scripts.tok_train --tokenizer-type=byte --vocab-size=256

# Read the existing BPE tokenizer for calibration. On a fresh setup, train a
# separate reference inside this experiment's directory, never in the old path.
reference_tokenizer_dir="$NANOCHAT_BASE_DIR/tokenizer"
if [ ! -f "$reference_tokenizer_dir/tokenizer.pkl" ]; then
    reference_tokenizer_dir="$NANOCHAT_TOKENIZER_DIR/reference_bpe"
    if [ ! -f "$reference_tokenizer_dir/tokenizer.pkl" ]; then
        NANOCHAT_TOKENIZER_DIR="$reference_tokenizer_dir" python -m scripts.tok_train
    fi
fi
python -m scripts.tok_eval
wait "$dataset_download_pid"

# Measure the actual packers' byte/token densities to match the BPE experiment's
# source-text volume. Calibration is a sampled estimate; details are saved to JSON.
python -m scripts.prepare_byte_experiment \
    --reference-tokenizer-dir="$reference_tokenizer_dir" \
    --output-dir="$experiment_dir" \
    --calibration-tokens="${BYTE_CALIBRATION_TOKENS:-4194304}"
mapfile -t training_args < "$experiment_dir/training_args.txt"
mapfile -t base_eval_args < "$experiment_dir/base_eval_args.txt"
mapfile -t chat_sft_args < "$experiment_dir/chat_sft_args.txt"
mapfile -t chat_eval_args < "$experiment_dir/chat_eval_args.txt"

python -m scripts.base_train \
    --depth=16 --aspect-ratio=40 --head-dim=128 \
    --no-value-embeddings --position-encoding=nope \
    --max-seq-len=4096 --device-batch-size=8 \
    --target-param-data-ratio=8 --fp8 --run="$run" --model-tag="$model_tag" \
    "${training_args[@]}"
python -m scripts.base_eval --device-batch-size=8 --model-tag="$model_tag" "${base_eval_args[@]}"
# SFT inherits the 4096 context and per-device batch size 8 from pretraining.
python -m scripts.chat_sft --run="$run" --model-tag="$model_tag" "${chat_sft_args[@]}"
python -m scripts.chat_eval -i sft --model-tag="$model_tag" "${chat_eval_args[@]}"
