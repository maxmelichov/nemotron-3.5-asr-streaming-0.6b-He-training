# Fine-Tune Nemotron 3.5 ASR for Hebrew (he-IL)

End-to-end implementation of NVIDIA's [Nemotron 3.5 ASR fine-tuning workflow](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr), using your **VoxKnesset Hebrew data** from `/mnt/windows_nvme/asr_transcition`.

Hebrew is an **adaptation-ready** locale in [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — the tokenizer recognizes it, but in-domain fine-tuning unlocks production-quality transcription.

## Data (Hugging Face native)

`scripts/build_hf_manifests.py` pulls everything from the Hub — no `/mnt/windows_nvme`
or `/home/maxm` paths required:

| Source | Repo | Notes |
|---|---|---|
| `ivrit30s` | `notmax123/ivirits-audio-v2-30s` | 2–30 s VAD segments + whisper text |
| `voxknesset` | `notmax123/voxknesset-hebrew-ipa` ⋈ `ivrit-ai/VoxKnesset` | transcripts joined to audio on `source_filename`, cut at `[start_sec, end_sec]` |
| `crowd_transcribe` | `ivrit-ai/crowd-transcribe-v5` | human-corrected `sentence` |
| `saspeech` | `notmax123/SASPEECH_AUTO_clean` | 7z archive |
| `ranlevi` | `notmax123/RanLevi40h` | 7z archive |
| `synthetic` | `notmax123/SententicDataTTS` | 7z archive; uses the `text` column, never the phoneme columns |
| `eval_whatsapp` | `ivrit-ai/eval-whatsapp` | the only held-out benchmark |

```bash
uv run scripts/build_hf_manifests.py --source all
uv run scripts/build_hf_manifests.py --source ivrit30s --max-hours-per-source 500
```

Dev is a small **stratified** sample of the training sources (`--dev-hours`, default 2 h),
so every source is represented without burning eval time.

Two repos in the original request are intentionally absent:

- `ivrit-ai/VoxKnesset` alone has **no transcript column** (`speaker_id`/`age`/`gender`/`audio`
  only — it is an age/demographics corpus). The Hebrew text lives in
  `notmax123/voxknesset-hebrew-ipa`, which the `voxknesset` source joins against.
- `ivrit-ai/audio-transcripts` has **no audio column** — it is the transcript layer for
  `audio-v2`, already represented by `ivrit30s`.

## Your data

| Dataset | Path | Clips | Notes |
|---------|------|-------|-------|
| **VoxKnesset Hebrew IPA** (default) | `asr_transcition/ivrit/VoxKnesset-hebrew-ipa/` | ~902K | Punctuated `transcript`, 16 kHz wav in `clips/` |
| VoxKnesset clips | `asr_transcition/ivrit/VoxKnesset-clips/` | ~102K | Earlier clip set, same schema |
| **Synthetic TTS** | [notmax123/large-he-synthetic-tts-dataset](https://huggingface.co/datasets/notmax123/large-he-synthetic-tts-dataset) | ~581K | Vocalized text → unvocalized for ASR; 24 kHz → 16 kHz resample |
| **Podcast segments** | `/home/maxm/aunikud/aunikud-data-harvest/podcast_segments` | ~195K | Whisper transcripts from music-clean pass; YouTube podcast speech |

Manifests use `target_lang=he-IL` and speaker-disjoint train/dev/test splits.

## Quick start

```bash
chmod +x setup.sh run.sh
./setup.sh
export NEMO_ROOT=$PWD/NeMo

# Verify data + env, build manifests if needed
uv run scripts/check_ready.py --fix

# Download base model
uv run scripts/download_model.py

# Fine-tune (telephony augmentations on by default)
uv run scripts/finetune.py

# Evaluate on ivrit.ai benchmark suite
uv run scripts/build_eval_benchmarks.py
uv run scripts/eval_streaming.py --compare-base

# Transcribe phone audio
uv run scripts/inference.py --audio call.wav --phone
```

Or: `./run.sh ready --fix` then `./run.sh train`

All commands use **`uv run`** — no manual venv activation needed.

## Pipeline

## Training control

- **Cache-aware context is now applied during training**, not just at eval:
  `training.att_context_size: [56, 13]` (1120 ms) becomes
  `model.encoder.att_context_size=[56,13]`. Previously the model was fine-tuned at the
  checkpoint's default context and then measured at another — a silent train/serve mismatch.
- **No fixed step budget.** `training.max_steps: null`; the run continues while `val_wer`
  improves and is ended by early stopping (`patience: 5` on `val_wer`).
- **Validation cadence**: `val_check_interval: 100` for the shakedown, then
  `--val-interval 1000` for the long run.

```bash
uv run scripts/finetune.py --val-interval 100    # shakedown
uv run scripts/finetune.py --val-interval 1000   # long run
uv run scripts/finetune.py --att-context "[56,0]" --no-early-stopping
```

## Running inside the NVIDIA NeMo container

`nvcr.io/nvidia/nemo:25.07` ships NeMo 2.4.1, which predates the prompt-conditioned
cache-aware streaming support this recipe needs. `scripts/setup_nemo_container.sh` clones
NeMo main alongside it and shadows the installed package via `PYTHONPATH` (so the
container's tuned torch build is left alone), upgrades `lhotse`, and stubs the
unpublished `nv_one_logger` telemetry package:

```bash
bash scripts/setup_nemo_container.sh
export NEMO_ROOT=/root/NeMo-main PYTHONPATH=/root/stubs
```

## Requirements

- **GPU**: NVIDIA GPU with bf16 support (tested recipe: 1× GPU, 24 GB+ VRAM)
- **Python**: ≥ 3.11
- **System**: `ffmpeg`, `sox`, `libsndfile`
- **NeMo**: 26.06+ (installed by `setup.sh` from GitHub `main`)

## Data options

### VoxKnesset (default)

Reads your existing JSONL manifests — **no wav copying**, absolute paths into `clips/`:

```bash
# Full ~902K clip build (takes a few minutes to scan)
python scripts/build_dataset.py --source voxknesset

# Smaller 102K clip set
python scripts/build_dataset.py --source voxknesset --dataset voxknesset_clips

# Tune quality filters in config.yaml:
#   min_whisper_confidence, min_vad_speech_ratio, test/dev speaker fractions
```

### Synthetic TTS (notmax123/large-he-synthetic-tts-dataset)

~581K clips / 859 h of Hebrew TTS (WER &lt; 0.25 tier). One-time prep downloads a 132 GB `tar.zst` from Hugging Face, extracts metadata + wav, and resamples to 16 kHz for NeMo:

```bash
# One-time (~132 GB download + extract + resample; use --jobs for parallel ffmpeg)
python scripts/prepare_synthetic.py all --jobs 16

# Build manifests from synthetic only
python scripts/build_dataset.py --source synthetic

# Default: mix VoxKnesset + synthetic
python scripts/build_dataset.py --source mixed
```

Configure in `config.yaml` under `data.synthetic`:
- `local_dir` — extract location (default: `asr_transcition/large-he-synthetic-tts-dataset`)
- `metadata_file` — `metadata_wer_025.csv` (recommended), `metadata_wer_02.csv`, or `metadata_wer0.csv`
- `text_mode: unvocalized` — strips niqqud from vocalized TTS labels

### Podcast segments (aunikud harvest)

~195K segmented clips from Hebrew YouTube podcasts. Audio lives under `podcast_segments/segments/`; Whisper transcripts are joined from the music-clean pass:

```bash
python scripts/build_dataset.py --source podcasts
python scripts/build_dataset.py --source mixed   # includes podcasts by default
```

Configure in `config.yaml` under `data.podcasts` (`root`, `transcripts_tsv`, `min_token_confidence`).

### FLEURS (optional benchmark)

```bash
python scripts/build_dataset.py --source fleurs
```

### Common Voice (optional)

```bash
python scripts/build_dataset.py --source common_voice --cv-dir /path/to/cv-corpus/he
```

### Tarred datasets (large scale)

For multi-thousand-hour training (blog Step 4):

```bash
python scripts/convert_to_tarred.py --manifest data/manifests/train.json --num-shards 16
python scripts/finetune.py --tarred
```

## Training

Fine-tuning follows the official NeMo recipe:

- Script: `NeMo/examples/asr/speech_to_text_finetune.py`
- Config: `fastconformer_transducer_bpe_streaming_prompt.yaml`
- Init: `init_from_nemo_model` from base `.nemo`
- Optimizer: AdamW + NoamAnnealing (`lr=0.1`, `warmup_steps=500`)
- Schedule: **step budget** (`max_steps=8000`), not epochs
- Tokenizer: reused from base model (recommended for < 50 h)

Key overrides vs NeMo defaults:

```bash
# NeMo caps training at 1000 batches/epoch by default — remove for full data:
python scripts/finetune.py  # uses ~trainer.limit_train_batches

# Large datasets (millions of lines):
python scripts/finetune.py --limit-train-batches 30000
```

Edit defaults in [`config.yaml`](config.yaml).

### Phone-call augmentations (telephony preset)

Training applies **online** NeMo augmentations tuned for phone audio. The primary augmentor simulates **narrowband phone channels** by resampling each clip **16 kHz → 8 kHz → 16 kHz**, picking a **different resampling method** for the down and up steps each time:

| Method | Backend |
|--------|---------|
| `kaiser_fast` / `kaiser_best` | librosa/resampy |
| `scipy` / `fft` | librosa |
| `polyphase` | scipy `resample_poly` (exact 2:1 ratio) |
| `soxr_hq` / `soxr_mq` | librosa SoXR |

Optional add-ons: `gain`, `white_noise`, and `noise` (if a noise manifest is built).

```bash
# Optional: background noise manifest
uv run scripts/build_noise_manifest.py --musan /data/musan

# Train with 8 kHz round-trip augmentations (default)
uv run scripts/finetune.py

# Disable augmentations
uv run scripts/finetune.py --no-augment
```

Configure under `training.augmentation.telephony` in `config.yaml`.

## Evaluation

Evaluation uses the **ivrit.ai-style Hebrew benchmark suite** (~13 h total), not the training hold-out split:

| Benchmark | Source | ~Size |
|-----------|--------|-------|
| eval-d1 | [ivrit-ai/eval-d1](https://huggingface.co/datasets/ivrit-ai/eval-d1) | 2 h |
| eval-whatsapp | [ivrit-ai/eval-whatsapp](https://huggingface.co/datasets/ivrit-ai/eval-whatsapp) | 1.2 h |
| SASpeech | [upai-inc/saspeech](https://huggingface.co/datasets/upai-inc/saspeech) | 4 h |
| FLEURS | `google/fleurs` he_il test | 2 h |
| Common Voice | `mozilla-foundation/common_voice_17_0` he validated | 2 h |
| Kan | [imvladikon/hebrew_speech_kan](https://huggingface.co/datasets/imvladikon/hebrew_speech_kan) validation | 1.7 h |

```bash
# Cache eval manifests + 16 kHz wav (once)
python scripts/build_eval_benchmarks.py

# Base vs fine-tuned at 80 ms on all benchmarks
python scripts/eval_streaming.py --compare-base

# Subset or legacy single manifest
python scripts/eval_streaming.py --benchmark fleurs saspeech
python scripts/eval_streaming.py --test-manifest data/manifests/test.json
```

`eval-whatsapp` is gated on Hugging Face — accept access and run `hf auth login`. Common Voice falls back to a local `cv-corpus-17.0/he` dir if the HF copy is unavailable.

Evaluate at the **same latency you will deploy** — the blog uses ultra-low latency with no lookahead:

| `att_context_size` | Latency | Use case |
|--------------------|---------|----------|
| `[56, 0]` | 80 ms | Voice agents |
| `[56, 1]` | 160 ms | Interactive agents |
| `[56, 3]` | 320 ms | Live captioning |
| `[56, 6]` | 560 ms | Higher accuracy |
| `[56, 13]` | 1120 ms | Highest accuracy |

```bash
# Full latency ladder on fine-tuned model
python scripts/eval_streaming.py --ladder
```

## Inference

Known language (best accuracy):

```bash
python scripts/inference.py \
  --model checkpoints/hebrew-finetuned.nemo \
  --audio clip.wav \
  --target-lang he-IL \
  --att-context "[56,3]"
```

Uses NeMo's cache-aware streaming infer script — same path as production deployment.

Direct NeMo command (equivalent):

```bash
python $NEMO_ROOT/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path=checkpoints/hebrew-finetuned.nemo \
  dataset_manifest=data/manifests/test.json \
  output_path=exp/infer \
  target_lang=he-IL \
  att_context_size="[56,3]" \
  strip_lang_tags=true
```

## Protecting other languages (replay)

When specializing a multilingual checkpoint, blend a slice of other locales during training so you don't erode them. The blog recommends **replay** data from Granary / FLEURS other languages mixed into the train pool, then re-check WER on those locales.

This repo focuses on Hebrew-only fine-tuning. For multilingual preservation:

1. Add replay manifests with correct `target_lang` tags per locale
2. Concatenate train manifests or use Lhotse multiplexing
3. Re-evaluate held-out clips for replay languages after fine-tuning

See [nvidia/Granary](https://huggingface.co/datasets/nvidia/Granary) for large-scale multilingual data.

## Project layout

```
├── config.yaml              # Hebrew defaults (he-IL, hyperparams, latency)
├── setup.sh                 # Install NeMo + dependencies
├── run.sh                   # Pipeline orchestrator
├── scripts/
│   ├── build_dataset.py     # Step 1: training manifests (VoxKnesset / synthetic / podcasts)
│   ├── build_eval_benchmarks.py  # Step 1b: ivrit.ai eval suite manifests
│   ├── build_noise_manifest.py   # Noise corpus for train-time augmentation
│   ├── telephony_perturb.py      # 8 kHz round-trip resample perturbation
│   ├── convert_to_tarred.py # Step 1c: tarred shards
│   ├── download_model.py    # Fetch base .nemo from HF
│   ├── finetune.py          # Step 2: training wrapper
│   ├── eval_streaming.py    # Step 3: streaming WER
│   └── inference.py         # Step 5: transcribe audio
├── data/manifests/          # Generated train/dev/test JSONL
└── checkpoints/             # Base + fine-tuned .nemo
```

## References

- [Blog: How to Fine-Tune Nemotron 3.5 ASR](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr)
- [Model: nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Official fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb)
- [NeMo streaming infer example](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/asr/asr_cache_aware_streaming)

## License

Training code: MIT. Base model: [OpenMDW-1.1](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). Check dataset licenses (FLEURS: CC-BY 4.0, Common Voice: CC0).
