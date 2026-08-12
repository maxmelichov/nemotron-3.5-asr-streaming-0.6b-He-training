#!/usr/bin/env python3
"""Per-source audit of a manifest's reference text: nikud, length, sample rows."""
import json
import re
import sys
from collections import defaultdict

# Combining marks only: cantillation (U+0591-U+05AF), vowel points (U+05B0-U+05BD),
# rafe, shin/sin dots, and qamats qatan. The naive range U+0591-U+05C7 also swallows the
# Hebrew *punctuation* in that block -- maqaf U+05BE, paseq U+05C0, sof pasuq U+05C3,
# nun hafukha U+05C6 -- which are plain punctuation in unvocalized text and would flag
# nearly every row as nikud.
NIKUD = re.compile("[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]")

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
