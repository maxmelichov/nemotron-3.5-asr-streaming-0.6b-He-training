#!/bin/bash
# Stage every usable Hebrew ASR source, uncapped, in parallel.
# crowd_transcribe (294.5 h) is already staged under /root/data.
#
# No --max-hours-per-source: sources run until exhausted or until --min-free-gb is hit.
# 16 kHz mono wav is ~115 MB/h, so this 707 GB volume tops out near ~5,500 h.
: "${HF_TOKEN:?set HF_TOKEN in the environment (never commit it)}"
export PYTHONUNBUFFERED=1
cd /root/nemo-he

python scripts/build_hf_manifests.py \
  --source voxknesset --source ivrit30s --source eval_whatsapp \
  --out /root/data2 --dev-hours 2 \
  --min-confidence 0.5 --min-vad-ratio 0.6 \
  --workers 24 --shard-workers 32 --min-free-gb 70
