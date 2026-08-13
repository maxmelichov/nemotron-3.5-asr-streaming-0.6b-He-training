#!/usr/bin/env python3
"""Drop training rows whose audio file is claimed by more than one transcript.

The staging pipeline derived local wav filenames from episode id + segment index, and
that name is not unique across sources/shards: 799,054 paths ended up written by more
than one segment. Only the last writer's audio survives on disk, so every manifest row
pointing at such a path is a coin flip -- 36.8% of train was audio/text mismatched, which
no learning rate can survive.

We cannot tell which row won the race (no provenance is recorded), so every row on a
conflicted path is dropped rather than guessed at. Paths whose rows all agree on the text
are kept: identical text means the collision was harmless.

Dev is unaffected (verified 0 conflicted paths) so evaluation history stays comparable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/root/data/manifests/train.json"))
    ap.add_argument("--out", type=Path, help="default: overwrite input")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = args.out or args.manifest

    texts: dict[str, set[str]] = defaultdict(set)
    total = 0
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            texts[r["audio_filepath"]].add(r.get("text", "").strip())
            total += 1

    bad = {p for p, t in texts.items() if len(t) > 1}
    print(f"{args.manifest}: {total:,} rows, {len(texts):,} unique paths")
    print(f"  conflicted paths: {len(bad):,}")

    kept: list[str] = []
    dropped = 0
    hours_kept = 0.0
    hours_dropped = 0.0
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            d = float(r.get("duration", 0))
            if r["audio_filepath"] in bad:
                dropped += 1
                hours_dropped += d
            else:
                kept.append(line if line.endswith("\n") else line + "\n")
                hours_kept += d

    print(f"  keep {len(kept):,} rows ({hours_kept/3600:,.0f} h)")
    print(f"  drop {dropped:,} rows ({hours_dropped/3600:,.0f} h, {100*dropped/total:.1f}%)")
    if args.dry_run:
        print("dry run -- nothing written")
        return
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(kept), encoding="utf-8")
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
