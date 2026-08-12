#!/bin/bash
set -u
: "${HF_TOKEN:?set HF_TOKEN in the environment (never commit it)}"
export NEMO_ROOT=${NEMO_ROOT:-/root/NeMo-main}
# PREPEND, never default: `${PYTHONPATH:-/root/stubs}` keeps an inherited PYTHONPATH
# instead, and NeMo then dies on `import nv_one_logger` (NVIDIA-internal, not on PyPI,
# stubbed under /root/stubs).
export STUB_DIR=${STUB_DIR:-/root/stubs}
export PYTHONPATH=${STUB_DIR}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Score val_wer on words only. 20.9% of dev reference words carry punctuation and it
# inflated the base model's WER by 8.1 points (50.55 -> 42.45), so leaving this off means
# checkpoint selection and early stopping are partly deciding on comma placement.
export NEMO_WER_NORMALIZE=1
cd "$(dirname "$0")"

# NoamHoldAnnealing with an explicit 2.5e-4 peak reached at step 1000, held to step
# 16000, then decay_rate 0.25 -- a language-acquisition schedule, not a fine-tune one.
#
# Starts from run 2's step-2000 weights (val_wer 48.95%, already better than the 50.55%
# base) with a FRESH optimizer and LR schedule. --base-model points at a snapshot copy
# rather than exp/hebrew_ft/best.nemo so exp_manager's top-k pruning cannot delete the
# file we are initializing from partway through the run.
BASE_MODEL=${BASE_MODEL:-/root/nemo-he/checkpoints/run2-step2000-wer4895.nemo}

python scripts/finetune.py \
  --base-model "$BASE_MODEL" \
  --no-resume \
  --train-manifest /root/data/manifests/train.json \
  --dev-manifest /root/data/manifests/dev.json \
  --val-interval 1000 \
  --exp-dir /root/exp
