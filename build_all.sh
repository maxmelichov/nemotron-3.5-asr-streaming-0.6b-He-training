#!/bin/bash
# Stage every usable Hebrew ASR source, uncapped, in parallel, onto the 2.4 TB volume.
#
# Not included: SASPEECH_AUTO_clean and RanLevi40h ship phoneme text only (IPA), no
# Hebrew column -- there is no known index-based recovery for them the way there was for
# SententicDataTTS (joined to Phonikud/phonikud-data by clip-id line number; see
# notmax123/SententicDataTTS README and hebrew_text.parquet). `synthetic` IS that
# recovered source, so it is not listed separately.
: "${HF_TOKEN:?set HF_TOKEN in the environment (never commit it)}"
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
cd /root/nemo-he

# --shard-workers 20: validated on this box specifically. 64 divided this box's route to
# HF's CDN too thin for any single connection to finish before curl's --max-time; every
# first-wave shard failed. 20 is the number that was actually measured working.
python scripts/build_hf_manifests.py \
  --source crowd_transcribe --source voxknesset --source crowd_recital \
  --source podcast286 --source lhs_synthetic --source ivrit30s --source eval_whatsapp \
  --out /root/data --dev-hours 2 \
  --min-confidence 0.5 --min-vad-ratio 0.6 \
  --workers 24 --shard-workers 20 --min-free-gb 120
