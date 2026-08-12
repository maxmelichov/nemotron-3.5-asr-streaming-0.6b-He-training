#!/usr/bin/env python3
"""Audit (and optionally filter) manifest rows whose text is not Hebrew/English.

Allowed: Hebrew letters + niqqud, Latin (English), digits, whitespace, and common
punctuation. Anything else -- Cyrillic, Arabic, CJK, Greek, emoji, private-use --
marks the row as foreign.

Hebrew/English code-switching is explicitly KEPT (config: project.hebrew_only=false);
this only removes text carrying a script the model has no business predicting.

  python scripts/audit_scripts.py data/manifests/train.json            # report only
  python scripts/audit_scripts.py data/manifests/train.json --filter   # rewrite in place
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def classify(ch: str) -> str:
    """Return a coarse script name for one character."""
    cp = ord(ch)
    if ch.isspace() or cp < 0x0080:
        return "ascii"  # Latin letters, digits, ASCII punctuation
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
        return "hebrew"
    # Punctuation/symbols/marks anywhere in Unicode are script-neutral (quotes,
    # dashes, NBSP, ellipsis) -- they are cleanup targets, not foreign-script signals.
    if unicodedata.category(ch)[0] in {"P", "S", "Z", "M"}:
        return "punct"
    if 0x0100 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return "latin"  # accented Latin
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "arabic"
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
        return "cjk"
    return f"other(U+{cp:04X})"


ALLOWED = {"ascii", "hebrew", "latin", "punct"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--filter", action="store_true", help="rewrite the manifest without foreign rows")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    script_rows: Counter[str] = Counter()
    bad_idx: list[int] = []
    examples: dict[str, str] = {}

    for i, r in enumerate(rows):
        text = r.get("text", "")
        foreign = {s for s in (classify(c) for c in text) if s not in ALLOWED}
        if foreign:
            bad_idx.append(i)
            for s in foreign:
                script_rows[s] += 1
                examples.setdefault(s, text[:70])

    print(f"{args.manifest}: {len(rows):,} rows")
    print(f"  foreign-script rows: {len(bad_idx):,} ({100*len(bad_idx)/max(len(rows),1):.4f}%)")
    for script, n in script_rows.most_common(12):
        print(f"    {script:<18} {n:>8,} rows   e.g. {examples[script]}")
    if not bad_idx:
        print("  clean -- Hebrew/English/punctuation only")

    if args.filter and bad_idx:
        keep = set(range(len(rows))) - set(bad_idx)
        hours_before = sum(float(r.get("duration", 0)) for r in rows) / 3600
        out = [rows[i] for i in sorted(keep)]
        hours_after = sum(float(r.get("duration", 0)) for r in out) / 3600
        with args.manifest.open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  rewrote: {len(rows):,} -> {len(out):,} rows "
              f"({hours_before:.1f} h -> {hours_after:.1f} h)")


if __name__ == "__main__":
    main()
