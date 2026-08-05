#!/bin/bash
# setup_tools_env.sh
# ──────────────────────────────────────────────────────────────────────────────
# One-shot script to fully reproduce the tools conda environment.
# This env is used to run tool services (GroundedSAM2, GroundingDINO, etc.)
#
# Usage (on a GPU machine with CUDA 12.x):
#   bash scripts/setup_tools_env.sh
#
# Steps:
#   1. Create conda env from tools/environment.yml  (CPU packages)
#   2. Install GPU packages (PyTorch, flash-attn, triton, NVIDIA wheels)
#   3. Install local editable packages (SAM-2, GroundingDINO)
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # REVPT/
TOOLS_DIR="$REPO_DIR/tools"

PYTHON_BIN="$(conda run -n tools which python 2>/dev/null || echo '')"

# ── 1. Create conda environment ───────────────────────────────────────────────
echo "==> [1/3] Creating conda environment 'tools' from tools/environment.yml ..."
conda env create -f "$TOOLS_DIR/environment.yml"

# ── 2. Install GPU packages ───────────────────────────────────────────────────
echo ""
echo "==> [2/3] Installing GPU packages ..."

conda run -n tools bash -c "
set -e
PIP='python -m pip'

echo '  --> PyTorch 2.6.0 (CUDA 12.4)'
\$PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo '  --> NVIDIA CUDA runtime wheels'
\$PIP install \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cuda-cupti-cu12==12.4.127 \
    nvidia-cuda-nvrtc-cu12==12.4.127 \
    nvidia-cuda-runtime-cu12==12.4.127 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.2.1.3 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-cusolver-cu12==11.6.1.9 \
    nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-cusparselt-cu12==0.6.2 \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-nvjitlink-cu12==12.4.127 \
    nvidia-nvtx-cu12==12.4.127

echo '  --> triton 3.2.0'
\$PIP install triton==3.2.0

echo '  --> flash-attention 2.7.4.post1'
\$PIP install flash_attn==2.7.4.post1 --no-build-isolation \
    || echo '  [WARN] Prebuilt wheel not found, trying to build from source (slow)...'
"

# ── 3. Install local editable packages ───────────────────────────────────────
echo ""
echo "==> [3/3] Installing local editable packages (SAM-2, GroundingDINO) ..."

GROUNDEDSAM2_DIR="$TOOLS_DIR/GroundedSAM2"

if [ -d "$GROUNDEDSAM2_DIR" ]; then
    # SAM-2 (root of GroundedSAM2 dir)
    conda run -n tools pip install -e "$GROUNDEDSAM2_DIR"

    # GroundingDINO (sub-package inside GroundedSAM2)
    if [ -d "$GROUNDEDSAM2_DIR/grounding_dino" ]; then
        conda run -n tools pip install -e "$GROUNDEDSAM2_DIR/grounding_dino"
    else
        echo "  [WARN] $GROUNDEDSAM2_DIR/grounding_dino not found, skipping."
    fi
else
    echo "  [WARN] $GROUNDEDSAM2_DIR not found, skipping local installs."
fi

echo ""
echo "==> Tools environment setup complete!"
echo "==> Activate with: conda activate tools"
echo "==> Verify GPU:    conda run -n tools python -c \"import torch; print(torch.cuda.is_available())\""
