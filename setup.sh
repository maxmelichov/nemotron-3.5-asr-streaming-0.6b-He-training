#!/usr/bin/env bash
# Environment setup for Nemotron 3.5 ASR Hebrew fine-tuning (uv).
# Requires: uv, Python >= 3.11, CUDA GPU, ffmpeg, sox, libsndfile
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="${NEMO_DIR:-$ROOT/NeMo}"
NEMO_BRANCH="${NEMO_BRANCH:-main}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv &>/dev/null; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "==> System dependencies (requires sudo for apt)"
if command -v apt-get &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq libsndfile1 ffmpeg sox libsox-fmt-all git
fi

echo "==> uv sync (PyTorch CUDA + data-prep deps)"
cd "$ROOT"
uv sync

echo "==> NeMo ASR toolkit + PyTorch Lightning"
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_BRANCH}" lightning

echo "==> Pin numba (0.66 breaks RNNT CUDA kernels on Blackwell / CUDA 13)"
uv pip install "numba==0.60.0"

if [[ ! -d "$NEMO_DIR" ]]; then
  echo "==> Cloning NeMo examples (for training/inference scripts)"
  git clone --depth 1 -b "$NEMO_BRANCH" https://github.com/NVIDIA/NeMo.git "$NEMO_DIR"
fi

mkdir -p "$ROOT/data/manifests" "$ROOT/checkpoints" "$ROOT/exp"

echo ""
echo "Setup complete."
echo "  export NEMO_ROOT=$NEMO_DIR"
echo ""
echo "Next steps:"
echo "  uv run scripts/check_ready.py --fix"
echo "  uv run scripts/download_model.py"
echo "  uv run scripts/finetune.py"
