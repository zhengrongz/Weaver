#!/bin/bash
set -x

# ──────────────────────────────────────────────────────────────────────────────
# eval_interleave.sh  —  Launch interleave eval for LongVideoReason benchmark
#
# Usage:
#   bash scripts/eval_interleave.sh
#
# All variables below can be overridden from the environment, e.g.:
#   MODEL_ID=/path/to/ckpt bash scripts/eval_interleave.sh
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
# Path to the HuggingFace model checkpoint to evaluate
MODEL_ID=${MODEL_ID:-"path/to/your/model_checkpoint"}

# Path to the test JSONL annotation file
DATA_DIR=${DATA_DIR:-"path/to/test.jsonl"}

# Root directory containing the video files
VIDEO_DIR=${VIDEO_DIR:-"path/to/longvideo_eval"}

# Directory where predictions.json and metrics.json will be saved
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/interleave"}

# ── Dataset ───────────────────────────────────────────────────────────────────
# Number of samples to randomly evaluate (leave empty or 0 to use all samples)
NUM_SAMPLES=${NUM_SAMPLES:-""}

# DataLoader worker processes
NUM_WORKERS=${NUM_WORKERS:-4}

# ── Model inference ───────────────────────────────────────────────────────────
# Number of frames sampled per video
NFRAMES=${NFRAMES:-128}

# Max pixels per frame (default: 448*448 = 200704)
MAX_PIXELS=${MAX_PIXELS:-200704}

# Max new tokens generated per turn
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

# Max tool-call turns per sample
MAX_TURNS=${MAX_TURNS:-10}

# ── Distributed ───────────────────────────────────────────────────────────────
# Number of GPUs to use
WORLD_SIZE=${WORLD_SIZE:-4}

# Optionally restrict which GPUs are visible
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ──────────────────────────────────────────────────────────────────────────────
# Build optional arguments
# ──────────────────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    EXTRA_ARGS="${EXTRA_ARGS} --num_samples ${NUM_SAMPLES}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
python eval/lvreason/interleave.py \
    --model_id      "${MODEL_ID}"      \
    --data_dir      "${DATA_DIR}"      \
    --video_dir     "${VIDEO_DIR}"     \
    --output_dir    "${OUTPUT_DIR}"    \
    --num_workers   "${NUM_WORKERS}"   \
    --nframes       "${NFRAMES}"       \
    --max_pixels    "${MAX_PIXELS}"    \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_turns     "${MAX_TURNS}"     \
    --world_size    "${WORLD_SIZE}"    \
    ${EXTRA_ARGS}
