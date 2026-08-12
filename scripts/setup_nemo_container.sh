#!/usr/bin/env bash
# Prepare an NVIDIA NeMo container (e.g. nvcr.io/nvidia/nemo:25.07) to run this recipe.
#
# The shipped container pins an older NeMo (2.4.1) that predates the prompt-conditioned
# cache-aware streaming support this fine-tune needs: it has no
# fastconformer_transducer_bpe_streaming_prompt.yaml and no
# nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py. Rather than reinstall
# NeMo (which would drag in a different torch than the container's tuned build), we clone
# NeMo main and shadow the installed package via PYTHONPATH.
#
# Usage:
#   bash scripts/setup_nemo_container.sh
#   export NEMO_ROOT=/root/NeMo-main PYTHONPATH=/root/stubs
set -euo pipefail

NEMO_MAIN="${NEMO_MAIN:-/root/NeMo-main}"
STUBS="${STUBS:-/root/stubs}"

echo "==> NeMo main (provides the streaming *prompt* config + prompt dataset)"
[ -d "$NEMO_MAIN" ] || git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_MAIN"

echo "==> Dependencies NeMo main needs that the container predates"
# lhotse: main imports ClippingTransform, absent from the container's pinned lhotse.
pip install -q -U lhotse
pip install -q kaldialign editdistance jiwer

echo "==> Stub nv_one_logger"
# NeMo main imports NVIDIA-internal training telemetry unconditionally, and it is not
# published on PyPI (nor on pypi.nvidia.com). These inert shims satisfy the imports;
# nothing here affects training numbers.
mkdir -p "$STUBS"/nv_one_logger/api \
         "$STUBS"/nv_one_logger/training_telemetry/api \
         "$STUBS"/nv_one_logger/training_telemetry/integration

cat > "$STUBS/nv_one_logger/__init__.py" <<'PY'
"""Inert stand-in for NVIDIA internal training telemetry."""
PY
: > "$STUBS/nv_one_logger/api/__init__.py"
cat > "$STUBS/nv_one_logger/api/config.py" <<'PY'
class OneLoggerConfig:
    def __init__(self, *a, **k): pass
PY
: > "$STUBS/nv_one_logger/training_telemetry/__init__.py"
: > "$STUBS/nv_one_logger/training_telemetry/api/__init__.py"
cat > "$STUBS/nv_one_logger/training_telemetry/api/callbacks.py" <<'PY'
def on_app_start(*a, **k): return None
def on_app_end(*a, **k): return None
PY
cat > "$STUBS/nv_one_logger/training_telemetry/api/config.py" <<'PY'
class TrainingTelemetryConfig:
    def __init__(self, *a, **k): pass
PY
cat > "$STUBS/nv_one_logger/training_telemetry/api/training_telemetry_provider.py" <<'PY'
class _Chain:
    def __getattr__(self, _): return self
    def __call__(self, *a, **k): return self
class TrainingTelemetryProvider:
    instance = _Chain()
    def __getattr__(self, _): return _Chain()
PY
: > "$STUBS/nv_one_logger/training_telemetry/integration/__init__.py"
cat > "$STUBS/nv_one_logger/training_telemetry/integration/pytorch_lightning.py" <<'PY'
try:
    from lightning.pytorch import Callback
except Exception:
    class Callback:
        pass
class TimeEventCallback(Callback):
    def __init__(self, *a, **k): pass
PY

echo "==> Verifying"
PYTHONPATH="$STUBS:$NEMO_MAIN" python -c "
import nemo.collections.asr  # noqa: F401
from nemo.collections.asr.data import audio_to_text_lhotse_prompt_index  # noqa: F401
print('NeMo main import OK (prompt dataset available)')
"

cat <<EOF

Done. Export these before training:
  export NEMO_ROOT=$NEMO_MAIN
  export PYTHONPATH=$STUBS
EOF
