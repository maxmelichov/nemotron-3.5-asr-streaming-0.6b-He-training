#!/bin/bash
# Full provisioning for a fresh NVIDIA NeMo container: system packages, NeMo main
# (the container's 2.4.1 lacks the streaming prompt support), and the base checkpoint.
set -eu

: "${HF_TOKEN:?set HF_TOKEN in the environment (never commit it)}"
export DEBIAN_FRONTEND=noninteractive

echo "==> System packages"
apt-get update -qq
# ffmpeg is not optional: most source audio is mp3/m4a and silently fails to decode
# without it. p7zip for the .7z dataset archives.
apt-get install -y -qq ffmpeg p7zip-full tmux

echo "==> Python packages"
pip install -q soundfile fsspec py7zr kaldialign editdistance jiwer
pip install -q -U lhotse                     # container's lhotse predates ClippingTransform

echo "==> NeMo main + telemetry stub"
bash /root/nemo-he/scripts/setup_nemo_container.sh

echo "==> Base checkpoint"
mkdir -p /root/nemo-he/checkpoints
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
dest = "/root/nemo-he/checkpoints/nemotron-3.5-asr-base.nemo"
if not os.path.exists(dest):
    p = hf_hub_download("nvidia/nemotron-3.5-asr-streaming-0.6b",
                        "nemotron-3.5-asr-streaming-0.6b.nemo", token=os.environ["HF_TOKEN"])
    shutil.copy(p, dest)
print("base checkpoint:", dest, os.path.getsize(dest) / 1e9, "GB")
PY

echo "==> Verify"
NEMO_ROOT=/root/NeMo-main PYTHONPATH=/root/stubs:/root/NeMo-main python -c "
import torch, nemo.collections.asr  # noqa: F401
from nemo.collections.asr.data import audio_to_text_lhotse_prompt_index  # noqa: F401
print('GPUs:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('NeMo main + prompt dataset OK')
"
echo "PROVISION_OK"
