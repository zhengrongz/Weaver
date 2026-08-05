#!/bin/bash
# setup_env.sh
# ──────────────────────────────────────────────────────────────────────────────
# One-shot script to fully reproduce the weaver conda environment.
#
# Usage (on a GPU machine with CUDA 12.x):
#   bash setup_env.sh
#
# Steps:
#   1. Create conda env from environment.yml  (CPU packages)
#   2. Install GPU packages via install_gpu_packages.sh
#   3. Install local editable packages (verl, qwen-vl-utils)
# ──────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Create conda environment ───────────────────────────────────────────────
echo "==> [1/3] Creating conda environment from environment.yml ..."
conda env create -f "$SCRIPT_DIR/environment.yml"

# ── 2. Install GPU packages ───────────────────────────────────────────────────
echo ""
echo "==> [2/3] Installing GPU packages ..."
conda run -n weaver bash "$SCRIPT_DIR/install_gpu_packages.sh"

# ── 3. Install local editable packages ───────────────────────────────────────
echo ""
echo "==> [3/3] Installing local editable packages ..."

# verl (the main REVPT package)
conda run -n weaver pip install -e "$SCRIPT_DIR"

# qwen-vl-utils (local fork under REVPT/qwen-vl-utils/)
if [ -d "$SCRIPT_DIR/qwen-vl-utils" ]; then
    conda run -n weaver pip install -e "$SCRIPT_DIR/qwen-vl-utils"
else
    echo "  [WARN] $SCRIPT_DIR/qwen-vl-utils not found, skipping."
fi

echo ""
echo "==> Environment setup complete!"
echo "==> Activate with: conda activate weaver"
