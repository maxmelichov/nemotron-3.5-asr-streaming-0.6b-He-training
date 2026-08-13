#!/usr/bin/env python3
"""Grow the dev split until its WER confidence interval is tight enough to steer by.

A 461-clip dev set gives a 95% bootstrap CI of +/-3.0 WER points. Early stopping
compares against min_delta=0.001 (0.1 points) and checkpoint selection keeps the lowest
val_wer, so at that width both are reading noise. CI shrinks as 1/sqrt(clips), so ~2500
clips gets to roughly +/-1.4 points.

Sampling is stratified by source and drawn from the SAME pool the old dev came from.
New dev rows are removed from train, and any train row sharing a normalised transcript
with a dev row is dropped too -- the synthetic corpora have several voices reading one
sentence, so splitting on audio path alone still leaks the text.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def source_of(row: dict) -> str:
    parts = row["audio_filepath"].split("/")
    return parts[4] if len(parts) > 5 else "?"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, default=Path("/root/data/manifests"))
    ap.add_argument("--target", type=int, default=2500, help="total dev clips wanted")
    ap.add_argument("--seed", type=int, default=5613)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_path = args.manifest_dir / "train.json"
    dev_path = args.manifest_dir / "dev.json"
    train = load(train_path)
    dev = load(dev_path)
    print(f"before: train {len(train):,} clips, dev {len(dev):,} clips")

    need = args.target - len(dev)
    if need <= 0:
        print("dev already at target")
        return

    # Match the existing dev's source mix so the metric keeps measuring the same thing.
    dev_mix = defaultdict(int)
    for r in dev:
        dev_mix[source_of(r)] += 1
    total_dev = sum(dev_mix.values())
    quota = {src: max(1, round(need * n / total_dev)) for src, n in dev_mix.items()}
    print("per-source quota:", dict(quota))

    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        d = float(r.get("duration", 0))
        if 1.0 <= d <= 30.0 and len(r.get("text", "")) >= 10:
            by_source[source_of(r)].append(i)

    rng = random.Random(args.seed)
    picked: set[int] = set()
    for src, want in quota.items():
        pool = by_source.get(src, [])
        take = rng.sample(pool, min(want, len(pool)))
        picked.update(take)
        print(f"  {src:<18} took {len(take):,} of {len(pool):,} available")

    new_dev_rows = [train[i] for i in sorted(picked)]
    dev_out = dev + new_dev_rows

    # Exclude by PATH, not by index: train.json contains the same audio_filepath at more
    # than one index, so dropping only the sampled indices left 606 of the moved clips
    # still sitting in train -- a straight path leak that verify_dataset.py caught.
    dev_paths = {r["audio_filepath"] for r in dev_out}
    # Text leakage: the synthetic corpora have several voices reading one sentence, so
    # path-disjoint is not enough on its own.
    dev_texts = {norm(r.get("text", "")) for r in dev_out}
    dev_texts.discard("")
    train_out = [
        r for r in train
        if r["audio_filepath"] not in dev_paths and norm(r.get("text", "")) not in dev_texts
    ]

    removed = len(train) - len(train_out)
    hrs = lambda rows: sum(float(r.get("duration", 0)) for r in rows) / 3600
    print(f"\nafter: train {len(train_out):,} clips ({hrs(train_out):.1f} h), "
          f"dev {len(dev_out):,} clips ({hrs(dev_out):.1f} h)")
    print(f"  removed from train: {removed:,} "
          f"({len(picked):,} moved to dev, {removed - len(picked):,} text-leak duplicates)")

    if args.dry_run:
        print("dry run -- nothing written")
        return
    write(dev_path, dev_out)
    write(train_path, train_out)
    print(f"wrote {dev_path} and {train_path}")


if __name__ == "__main__":
    main()
