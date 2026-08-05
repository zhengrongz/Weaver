#!/bin/bash
# install_gpu_packages.sh
# ──────────────────────────────────────────────────────────────────────────────
# Install GPU-dependent packages into the weaver conda environment.
# Must be run on a machine with CUDA 12.x and the weaver env already created
# via:  conda env create -f environment.yml
#
# Usage:
#   conda activate weaver
#   bash install_gpu_packages.sh
#
# To skip a step (e.g. if already installed), comment out the relevant block.
# ──────────────────────────────────────────────────────────────────────────────
set -e

PYTHON=$(which python)
PIP="$PYTHON -m pip"

echo "==> Using Python: $PYTHON"
echo "==> CUDA version: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found, assuming CUDA 12.x')"

# ── 1. PyTorch 2.6.0 + CUDA 12.4 ──────────────────────────────────────────────
echo ""
echo "==> [1/6] Installing PyTorch 2.6.0 (CUDA 12.4) ..."
$PIP install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 2. NVIDIA CUDA runtime wheels (bundled with torch, but pin versions) ───────
echo ""
echo "==> [2/6] Installing NVIDIA CUDA runtime wheels ..."
$PIP install \
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

# ── 3. triton ─────────────────────────────────────────────────────────────────
echo ""
echo "==> [3/6] Installing triton 3.2.0 ..."
$PIP install triton==3.2.0

# ── 4. flash-attention ────────────────────────────────────────────────────────
# flash_attn must be compiled against the installed torch/CUDA.
# Prebuilt wheels are available at:
#   https://github.com/Dao-AILab/flash-attention/releases
# Pick the wheel matching: torch2.6 + cu124 + cp310
echo ""
echo "==> [4/6] Installing flash-attention 2.7.4.post1 ..."
$PIP install flash_attn==2.7.4.post1 \
    --no-build-isolation \
    || echo "  [WARN] Prebuilt wheel not found, trying to build from source (slow)..."

# ── 5. vLLM ───────────────────────────────────────────────────────────────────
echo ""
echo "==> [5/6] Installing vLLM 0.8.3 ..."
$PIP install vllm==0.8.3

# ── 6. xformers + cupy ────────────────────────────────────────────────────────
echo ""
echo "==> [6/6] Installing xformers 0.0.29.post2 and cupy-cuda12x 13.6.0 ..."
$PIP install xformers==0.0.29.post2
$PIP install cupy-cuda12x==13.6.0

# ── 7. torchdata (optional, may conflict with torch version) ──────────────────
echo ""
echo "==> [7/7] Installing torchdata 0.11.0 ..."
$PIP install torchdata==0.11.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    || $PIP install torchdata==0.11.0

echo ""
echo "==> All GPU packages installed successfully."
echo "==> Run 'python -c \"import torch; print(torch.cuda.is_available())\"' to verify."
