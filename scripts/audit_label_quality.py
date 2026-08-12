#!/usr/bin/env python3
"""Is the training data learnable, or are we fitting noise?

80% of train is ivrit30s, whose transcripts are Whisper pseudo-labels produced by our own
Step-1 pipeline. If those labels are misaligned or wrong, no learning rate will help --
the model is being asked to match a target that does not describe the audio.

Method: sample clips per source, transcribe with the untouched base model, and compare
against the manifest label. The base model is a decent multilingual ASR, so per-source
WER against our labels is a proxy for label quality. A source where the base scores far
worse than others is either genuinely harder audio or badly labelled -- printing the
worst disagreements makes it obvious which.

  python scripts/audit_label_quality.py --per-source 40
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_norm  # noqa: E402
from common import nemo_root  # noqa: E402


def source_of(row: dict) -> str:
    parts = row["audio_filepath"].split("/")
    return parts[4] if len(parts) > 5 else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/root/data/manifests/train.json"))
    ap.add_argument("--model", type=Path,
                    default=Path("/root/nemo-he/checkpoints/nemotron-3.5-asr-base.nemo"))
    ap.add_argument("--per-source", type=int, default=40)
    ap.add_argument("--scan", type=int, default=600_000, help="rows to scan for sampling")
    ap.add_argument("--att", default="[56,13]")
    ap.add_argument("--out", type=Path, default=Path("/root/label_audit"))
    args = ap.parse_args()

    buckets: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.scan:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if 3.0 <= float(r.get("duration", 0)) <= 20.0:
                b = buckets[source_of(r)]
                if len(b) < 4000:
                    b.append(r)

    rng = random.Random(5613)
    sample: list[dict] = []
    for src, rows in sorted(buckets.items()):
        take = rng.sample(rows, min(args.per_source, len(rows)))
        sample.extend(take)
        print(f"{src:<18} sampled {len(take)} of {len(rows)} scanned", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "sample.json"
    with manifest.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ntranscribing {len(sample)} clips with the BASE model ...", flush=True)

    infer = nemo_root() / "examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
    outdir = args.out / "hyp"
    outdir.mkdir(parents=True, exist_ok=True)
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(nemo_root()) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = (
        f"python {infer} model_path={args.model} dataset_manifest={manifest} "
        f"output_path={outdir} target_lang=he-IL att_context_size=\"{args.att}\" "
        f"decoder_type=rnnt pad_and_drop_preencoded=true batch_size=16 "
        f"strip_lang_tags=true cuda=0"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        sys.exit("inference failed")

    hyp_file = sorted(outdir.glob("*.json"), key=lambda p: p.stat().st_mtime)[-1]
    recs = [json.loads(l) for l in hyp_file.open(encoding="utf-8") if l.strip()]
    by_path = {r["audio_filepath"]: r for r in sample}

    per_src: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    for rec in recs:
        ref, hyp = rec.get("text", ""), rec.get("pred_text", "")
        row = by_path.get(rec.get("audio_filepath", ""))
        src = source_of(row) if row else "?"
        rate, _, _ = text_norm.wer([ref], [hyp], normalized=True)
        per_src[src].append((rate, ref, hyp))

    print("\n" + "=" * 78)
    print("BASE MODEL WER AGAINST OUR TRAINING LABELS (words only)")
    print("=" * 78)
    for src, items in sorted(per_src.items(), key=lambda kv: -sum(x[0] for x in kv[1]) / max(len(kv[1]), 1)):
        mean = 100 * sum(x[0] for x in items) / len(items)
        broken = sum(1 for r, _, _ in items if r > 0.9)
        print(f"  {src:<18} n={len(items):>3}  mean WER {mean:6.1f}%   "
              f"clips >90% WER: {broken} ({100*broken/len(items):.0f}%)")

    print("\nworst disagreements (label vs base-model hypothesis):")
    worst = sorted((x for items in per_src.values() for x in items), key=lambda x: -x[0])[:6]
    for rate, ref, hyp in worst:
        print(f"\n  WER {100*rate:.0f}%")
        print(f"    label: {ref[:110]}")
        print(f"    heard: {hyp[:110]}")


if __name__ == "__main__":
    main()
