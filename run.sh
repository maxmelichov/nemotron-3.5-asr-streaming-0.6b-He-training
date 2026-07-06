#!/usr/bin/env bash
# End-to-end Hebrew Nemotron 3.5 ASR fine-tuning pipeline (uv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export NEMO_ROOT="${NEMO_ROOT:-$ROOT/NeMo}"

run_py() {
  uv run --no-sync "$@"
}

STEP="${1:-all}"
shift || true

case "$STEP" in
  setup)
    bash "$ROOT/setup.sh"
    ;;
  ready)
    run_py scripts/check_ready.py "$@"
    ;;
  download-musan)
    run_py scripts/download_musan.py "$@"
    run_py scripts/build_noise_manifest.py "$@"
    ;;
  synthetic)
    run_py scripts/prepare_synthetic.py "$@"
    ;;
  data)
    run_py scripts/build_dataset.py --source mixed "$@"
    ;;
  noise)
    run_py scripts/build_noise_manifest.py "$@"
    ;;
  eval-data)
    run_py scripts/build_eval_benchmarks.py "$@"
    ;;
  tarred)
    run_py scripts/convert_to_tarred.py --manifest data/manifests/train.json "$@"
    ;;
  download)
    run_py scripts/download_model.py "$@"
    ;;
  train)
    run_py scripts/finetune.py "$@"
    ;;
  eval)
    run_py scripts/build_eval_benchmarks.py
    run_py scripts/eval_streaming.py --compare-base "$@"
    ;;
  infer)
    run_py scripts/inference.py "$@"
    ;;
  all)
    run_py scripts/check_ready.py --fix
    run_py scripts/download_model.py
    run_py scripts/finetune.py
    run_py scripts/eval_streaming.py --compare-base
    ;;
  *)
    echo "Usage: $0 {setup|ready|data|synthetic|download-musan|noise|eval-data|tarred|download|train|eval|infer|all} [args...]"
    exit 1
    ;;
esac
