#!/usr/bin/env python3
"""Show latest training / validation WER scores from finetune logs.

Reads:
  - exp/hebrew_ft/wer.jsonl  (live scores captured by finetune.py)
  - exp/hebrew_ft/*/events.out.tfevents.*  (tensorboard scalars)

Usage:
  uv run scripts/show_wer.py
  uv run scripts/show_wer.py --watch 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import latest_tensorboard_wer, load_config, read_wer_log, repo_root


def print_table(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{title}: (no data yet)")
        return
    print(f"\n{title}")
    print(f"| {'metric':<22} | {'step':>8} | {'WER %':>10} | {'raw':>10} |")
    print(f"|{'-' * 24}|{'-' * 10}|{'-' * 12}|{'-' * 12}|")
    for row in rows[-20:]:
        print(
            f"| {row.get('metric', '?'):<22} "
            f"| {row.get('step', ''):>8} "
            f"| {row.get('wer_pct', row.get('value', 0) * 100):>9.2f}% "
            f"| {row.get('value', 0):>10.4f} |"
        )


def collect(wer_log: Path, exp_dir: Path) -> tuple[list[dict], list[dict]]:
    json_rows = read_wer_log(wer_log)
    tb_rows = latest_tensorboard_wer(exp_dir)
    return json_rows, tb_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--watch", type=int, metavar="SEC", help="Refresh every N seconds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    wer_log = Path(train_cfg.get("wer_log", "exp/hebrew_ft/wer.jsonl"))
    exp_dir = Path(train_cfg.get("exp_dir", "exp")) / train_cfg.get("exp_name", "hebrew_ft")

    while True:
        json_rows, tb_rows = collect(wer_log, exp_dir)
        print_table("Live WER log (wer.jsonl)", json_rows)
        print_table("TensorBoard scalars (latest run)", tb_rows)

        val_rows = [r for r in json_rows + tb_rows if r.get("metric") == "val_wer"]
        if val_rows:
            best = min(val_rows, key=lambda r: r.get("value", 999))
            print(
                f"\nBest val_wer so far: {best.get('wer_pct', best['value'] * 100):.2f}% "
                f"@ step {best.get('step', '?')}"
            )
        elif json_rows or tb_rows:
            print(f"\nval_wer logged every {train_cfg.get('val_check_interval', 500)} steps — none yet.")
        else:
            print("\nNo WER logged yet. Training may still be starting or crashed before step 100.")

        if not args.watch:
            break
        print(f"\n--- refresh in {args.watch}s ---")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
