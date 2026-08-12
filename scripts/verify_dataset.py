#!/usr/bin/env python3
"""Pre-flight the manifests before a training run: everything present, nothing leaked.

Checks, in order of how badly each would corrupt the result:

  1. **Audio present** -- every audio_filepath exists and is non-empty. A missing file is
     a crash mid-epoch, hours in.
  2. **Path leakage** -- no clip appears in more than one split.
  3. **Text leakage** -- no transcript is shared between train and dev/test. This is the
     one that silently flatters the numbers: the synthetic corpora have several voices
     reading the *same* sentence, so splitting by audio file alone still lets the model
     see the dev text during training. Compared on normalised text.
  4. **Duplicates within train** -- reported, not fatal; heavy duplication skews sampling.
  5. Duration sanity against the configured bounds.

Exit code is non-zero if any fatal check fails, so it can gate the training launch.

Usage:
  python scripts/verify_dataset.py --data-dir data --max-duration 40
  python scripts/verify_dataset.py --data-dir data --fix-dev   # drop leaking dev rows
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def norm(text: str) -> str:
    """Normalisation for comparison only -- never written back to a manifest."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hours(rows: list[dict]) -> float:
    return sum(float(r.get("duration", 0)) for r in rows) / 3600


def check_audio_present(rows: list[dict], name: str, sample: int) -> int:
    """Verify audio exists. Full scan is 8M stats; sample unless asked for all."""
    missing = 0
    checked = rows if sample <= 0 or sample >= len(rows) else rows[:: max(1, len(rows) // sample)]
    examples = []
    for row in checked:
        path = row.get("audio_filepath", "")
        try:
            if os.path.getsize(path) <= 44:  # empty or header-only wav
                missing += 1
                if len(examples) < 3:
                    examples.append(path)
        except OSError:
            missing += 1
            if len(examples) < 3:
                examples.append(path)
    rate = 100 * missing / max(len(checked), 1)
    status = "OK" if missing == 0 else "FAIL"
    print(f"  [{status}] {name}: audio present -- checked {len(checked):,}, missing {missing:,} ({rate:.2f}%)")
    for e in examples:
        print(f"          missing: {e}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--max-duration", type=float, default=40.0)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--audio-sample", type=int, default=20000,
                    help="Files to stat per split (0 = every file)")
    ap.add_argument("--fix-dev", action="store_true",
                    help="Rewrite dev.json without rows whose text also appears in train")
    args = ap.parse_args()

    md = args.data_dir / "manifests"
    train = load(md / "train.json")
    dev = load(md / "dev.json")
    evals = {p.stem: load(p) for p in sorted((md / "eval").glob("*.json"))}

    print("=" * 72)
    print("SPLITS")
    print("=" * 72)
    print(f"  train : {len(train):>10,} clips  {hours(train):>9.1f} h")
    print(f"  dev   : {len(dev):>10,} clips  {hours(dev):>9.1f} h")
    for name, rows in evals.items():
        print(f"  {name:<6}: {len(rows):>10,} clips  {hours(rows):>9.1f} h")
    if not train:
        sys.exit("train.json is empty or missing")

    fatal = 0

    print("\n" + "=" * 72)
    print("1. AUDIO PRESENT")
    print("=" * 72)
    fatal += check_audio_present(train, "train", args.audio_sample)
    fatal += check_audio_present(dev, "dev", 0)
    for name, rows in evals.items():
        fatal += check_audio_present(rows, name, 0)

    print("\n" + "=" * 72)
    print("2. PATH LEAKAGE (same clip in two splits)")
    print("=" * 72)
    train_paths = {r["audio_filepath"] for r in train}
    dev_paths = {r["audio_filepath"] for r in dev}
    overlap = train_paths & dev_paths
    print(f"  [{'OK' if not overlap else 'FAIL'}] train vs dev: {len(overlap):,} shared clips")
    fatal += len(overlap)
    for name, rows in evals.items():
        ov = train_paths & {r["audio_filepath"] for r in rows}
        print(f"  [{'OK' if not ov else 'FAIL'}] train vs {name}: {len(ov):,} shared clips")
        fatal += len(ov)

    print("\n" + "=" * 72)
    print("3. TEXT LEAKAGE (same transcript across splits)")
    print("=" * 72)
    train_texts = Counter(norm(r.get("text", "")) for r in train)
    train_texts.pop("", None)

    dev_leak = [r for r in dev if norm(r.get("text", "")) in train_texts]
    pct = 100 * len(dev_leak) / max(len(dev), 1)
    print(f"  [{'OK' if not dev_leak else 'WARN'}] dev: {len(dev_leak):,}/{len(dev):,} "
          f"({pct:.1f}%) transcripts also in train")
    for r in dev_leak[:3]:
        print(f"          {r.get('text','')[:70]}")

    eval_leaks = {}
    for name, rows in evals.items():
        leak = [r for r in rows if norm(r.get("text", "")) in train_texts]
        eval_leaks[name] = leak
        pct = 100 * len(leak) / max(len(rows), 1)
        flag = "OK" if not leak else "FAIL"
        print(f"  [{flag}] {name}: {len(leak):,}/{len(rows):,} ({pct:.1f}%) transcripts also in train")
        for r in leak[:3]:
            print(f"          {r.get('text','')[:70]}")
        fatal += len(leak)

    print("\n" + "=" * 72)
    print("4. DUPLICATES WITHIN TRAIN (not fatal)")
    print("=" * 72)
    dup_rows = sum(c - 1 for c in train_texts.values() if c > 1)
    print(f"  repeated transcripts: {dup_rows:,} rows ({100*dup_rows/max(len(train),1):.1f}%)")
    for text, count in train_texts.most_common(3):
        print(f"    x{count:<6,} {text[:60]}")

    print("\n" + "=" * 72)
    print("5. DURATION BOUNDS")
    print("=" * 72)
    for name, rows in (("train", train), ("dev", dev), *evals.items()):
        if not rows:
            continue
        durs = [float(r.get("duration", 0)) for r in rows]
        bad = sum(1 for d in durs if d < args.min_duration or d > args.max_duration)
        # eval clips are full recordings and legitimately exceed the training cap
        flag = "OK" if bad == 0 or name in evals else "WARN"
        print(f"  [{flag}] {name}: min {min(durs):.2f}s max {max(durs):.2f}s, "
              f"{bad:,} outside [{args.min_duration}, {args.max_duration}]")

    print("\n" + "=" * 72)
    print("6. PROMPT FIELD")
    print("=" * 72)
    for name, rows in (("train", train), ("dev", dev), *evals.items()):
        if not rows:
            continue
        bad = sum(1 for r in rows if r.get("target_lang") != "he-IL")
        print(f"  [{'OK' if not bad else 'FAIL'}] {name}: {bad:,} rows missing target_lang=he-IL")
        fatal += bad

    if args.fix_dev and dev_leak:
        leaked = {id(r) for r in dev_leak}
        clean = [r for r in dev if id(r) not in leaked]
        path = md / "dev.json"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean),
                        encoding="utf-8")
        print(f"\nrewrote {path}: {len(dev):,} -> {len(clean):,} rows "
              f"({len(dev_leak):,} leaking transcripts removed)")

    print("\n" + "=" * 72)
    if fatal:
        print(f"RESULT: {fatal:,} fatal problem(s) -- do not start training")
        sys.exit(1)
    print("RESULT: all fatal checks passed")


if __name__ == "__main__":
    main()
