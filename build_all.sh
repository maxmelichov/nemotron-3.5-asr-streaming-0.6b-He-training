#!/bin/bash
# Stage every usable Hebrew ASR source, uncapped, in parallel, onto the 2.4 TB volume.
#
# Not included: SASPEECH_AUTO_clean, RanLevi40h and SententicDataTTS. All three ship
# phoneme text only (filename,phonemes / original_phonemes,whisper_phonemes) and no
# Hebrew column, so they yield zero ASR rows -- SententicDataTTS alone is a 350 GB
# download for nothing. Their audio can be whisper-transcribed later if wanted.
: "${HF_TOKEN:?set HF_TOKEN in the environment (never commit it)}"
export PYTHONUNBUFFERED=1
cd /root/nemo-he

python scripts/build_hf_manifests.py \
  --source crowd_transcribe --source voxknesset --source ivrit30s --source eval_whatsapp \
  --out /root/data --dev-hours 2 \
  --min-confidence 0.5 --min-vad-ratio 0.6 \
  --workers 24 --shard-workers 32 --min-free-gb 120
