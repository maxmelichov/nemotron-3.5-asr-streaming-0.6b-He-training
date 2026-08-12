#!/usr/bin/env python3
"""Per-source audit of a manifest's reference text: nikud, length, sample rows."""
import json
import re
import sys
from collections import defaultdict

NIKUD = re.compile(r"[֑-ׇ]")  # cantillation + vowel points

path = sys.argv[1] if len(sys.argv) > 1 else "/root/data/manifests/dev.json"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by_src = defaultdict(lambda: {"n": 0, "nikud": 0, "dur": 0.0, "ex": None})
for r in rows:
    parts = r["audio_filepath"].split("/")
    src = parts[4] if len(parts) > 5 else "?"
    d = by_src[src]
    d["n"] += 1
    d["dur"] += float(r.get("duration", 0))
    if NIKUD.search(r.get("text", "")):
        d["nikud"] += 1
        if d["ex"] is None:
            d["ex"] = r["text"][:90]
print(f"total rows: {len(rows)}")
for src, d in sorted(by_src.items()):
    print(f"{src:>20}: {d['n']:>7,} clips {d['dur']/3600:8.1f} h  nikud rows: {d['nikud']:,}")
    if d["ex"]:
        print(f"{'':>22}nikud example: {d['ex']}")
