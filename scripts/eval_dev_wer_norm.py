#!/usr/bin/env python3
"""Compute dev WER with and without punctuation on the training val subset.

Matches NeMo finetune validation: dev.json in order, batch_size=2,
limit_val_batches=3000 → first 6000 clips (~14 h).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jiwer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ensure_cuda_home, load_config, read_nemo_manifest, repo_root

_PUNCT_RE = re.compile(r"[^\w\s\u0590-\u05FF]", re.UNICODE)


def strip_punct(text: str) -> str:
    return " ".join(_PUNCT_RE.sub("", text).split())


def training_val_subset(
    dev_rows: list[dict],
    *,
    val_batches: int,
    batch_size: int,
) -> list[dict]:
    n_clips = val_batches * batch_size
    return dev_rows[:n_clips]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=repo_root() / "data/manifests/dev.json")
    ap.add_argument("--val-batches", type=int, default=3000, help="trainer.limit_val_batches")
    ap.add_argument("--batch-size", type=int, default=2, help="model.validation_ds.batch_size")
    ap.add_argument("--infer-batch-size", type=int, default=32)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--out-manifest", type=Path, help="Write val subset manifest (optional)")
    args = ap.parse_args()

    ensure_cuda_home()
    cfg = load_config()
    train_cfg = cfg["training"]
    target_lang = cfg["project"]["target_lang"]
    val_batches = args.val_batches or train_cfg.get("limit_val_batches", 3000)

    dev_rows = read_nemo_manifest(args.manifest)
    rows = training_val_subset(dev_rows, val_batches=val_batches, batch_size=args.batch_size)
    if args.out_manifest:
        manifest_path = args.out_manifest
    else:
        manifest_path = repo_root() / "exp/hebrew_ft/dev_eval_punct/val_subset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    refs = [r["text"] for r in rows]
    hours = sum(float(r.get("duration", 0)) for r in rows) / 3600

    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    model = EncDecRNNTBPEModelWithPrompt.restore_from(str(args.model.resolve()), map_location=device)
    model.eval()

    out = model.transcribe(
        [str(manifest_path.resolve())],
        batch_size=args.infer_batch_size,
        verbose=True,
        target_lang=target_lang,
    )
    hyps = [item.text if hasattr(item, "text") else str(item) for item in out]

    raw = jiwer.wer(refs, hyps) * 100
    no_punct = jiwer.wer([strip_punct(r) for r in refs], [strip_punct(h) for h in hyps]) * 100

    print(f"val_batches={val_batches}  batch_size={args.batch_size}  clips={len(rows)}  hours={hours:.1f}")
    print(f"WER (with punctuation):     {raw:.2f}%")
    print(f"WER (no punctuation):       {no_punct:.2f}%")
    print(f"punctuation penalty:        {raw - no_punct:.2f} pp")


if __name__ == "__main__":
    main()
