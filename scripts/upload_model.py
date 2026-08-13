#!/usr/bin/env python3
"""Publish the fine-tuned checkpoint to the Hugging Face Hub and tag the revision.

  HF_TOKEN=... python scripts/upload_model.py \
      --checkpoint /root/exp/hebrew_ft/best.nemo \
      --repo notmax123/nemotron-3.5-asr-hebrew-streaming-0.6b \
      --tag "56,13" \
      --results /root/nemo-he/exp/eval/streaming_result.json

The tag names the cache-aware latency the model was trained AND evaluated at, which is
the number that actually matters for a streaming model: a checkpoint tuned at one
att_context_size and served at another silently loses accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_card(repo: str, tag: str, results: dict | None, base_wer: float | None,
               train_hours: float, train_clips: int) -> str:
    rows = ""
    if results:
        raw = results.get("raw", {}).get("finetuned", {})
        norm = results.get("normalized", {}).get("finetuned", {})
        for bench in sorted(set(raw) | set(norm)):
            r = raw.get(bench)
            n = norm.get(bench)
            rows += f"| {bench} | {r:.2f}% | {n:.2f}% |\n" if r is not None and n is not None else ""
    table = (
        "| Benchmark | WER | WER (words only) |\n|---|---|---|\n" + rows
        if rows else "_Evaluation pending._\n"
    )
    base_note = f"\nBase model on the same dev set: **{base_wer:.2f}%** (words only).\n" if base_wer else ""
    return f"""---
language: he
license: cc-by-4.0
library_name: nemo
tags:
- automatic-speech-recognition
- speech
- streaming
- hebrew
- nemo
- fastconformer
- rnnt
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
---

# Nemotron 3.5 ASR Streaming 0.6B — Hebrew

Cache-aware streaming FastConformer-Transducer fine-tuned for Hebrew (`he-IL`).

**Latency setting: `att_context_size = [{tag.replace(',', ', ')}]`** — the model was
fine-tuned *and* evaluated at this context, so serve it at the same setting.

## Results

{table}{base_note}
WER is reported two ways. "Words only" removes punctuation and niqqud before scoring:
WER marks a whole token wrong when a comma differs, and ~21% of Hebrew reference words
carry punctuation, so the raw figure measures punctuation as much as recognition.
Word-internal quotes are preserved, since Hebrew gershayim marks acronyms (צה"ל is not
צהל) and geresh marks modified consonants (ג'ון).

## Training data

{train_clips:,} clips / {train_hours:,.0f} hours of Hebrew speech, drawn from
`ivrit-ai` corpora, Knesset recordings, podcasts, and synthetic TTS. Every example
carries `target_lang: he-IL` for the model's prompt conditioning.

## Usage

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("{repo}")
model.encoder.set_default_att_context_size([{tag.replace(',', ', ')}])
model.eval()
print(model.transcribe(["audio.wav"])[0].text)
```

For streaming inference use NeMo's
`examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
with `att_context_size="[{tag}]"` and `target_lang=he-IL`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--repo", default="notmax123/nemotron-3.5-asr-hebrew-streaming-0.6b")
    ap.add_argument("--tag", default="56,13")
    ap.add_argument("--results", type=Path, help="exp/eval/streaming_result.json")
    ap.add_argument("--base-wer", type=float, help="base model WER for the card")
    ap.add_argument("--train-hours", type=float, default=13464.0)
    ap.add_argument("--train-clips", type=int, default=5133571)
    ap.add_argument("--filename", default="hebrew-streaming-56-13.nemo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (never hardcode it)")
    if not args.checkpoint.is_file():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    results = None
    if args.results and args.results.is_file():
        results = json.loads(args.results.read_text(encoding="utf-8"))

    card = build_card(args.repo, args.tag, results, args.base_wer,
                      args.train_hours, args.train_clips)
    size_gb = args.checkpoint.stat().st_size / 1e9
    print(f"repo      {args.repo}\ncheckpoint {args.checkpoint} ({size_gb:.2f} GB)"
          f"\nas        {args.filename}\ntag       {args.tag}")
    if args.dry_run:
        print("\n--- README.md ---\n" + card)
        print("dry run -- nothing uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(args.checkpoint), path_in_repo=args.filename,
                    repo_id=args.repo, repo_type="model")
    print("uploaded checkpoint")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    print("uploaded model card")

    # Tag last, so the tag points at a revision that already has both files.
    try:
        api.create_tag(args.repo, tag=args.tag, repo_type="model")
        print(f"tagged {args.tag}")
    except Exception as exc:  # tag may already exist from a previous upload
        print(f"tag {args.tag}: {exc}")
        api.delete_tag(args.repo, tag=args.tag, repo_type="model")
        api.create_tag(args.repo, tag=args.tag, repo_type="model")
        print(f"re-tagged {args.tag}")
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
