#!/usr/bin/env python3
"""Download and prepare notmax123/large-he-synthetic-tts-dataset for ASR training.

Source audio lives under AE slow_44K (config data.synthetic.audio_root) — not duplicated
on /mnt/windows_nvme. This script only manages metadata + 16 kHz resample cache under
data/synthetic/.

Steps:
  uv run scripts/prepare_synthetic.py download   # metadata archive from HF (optional)
  uv run scripts/prepare_synthetic.py resample   # AE 44.1 kHz -> wav_16k for NeMo
  uv run scripts/prepare_synthetic.py all        # download metadata + resample
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_config,
    repo_root,
    resolve_synthetic_source_wav,
    run,
    synthetic_cfg_root,
)


def cfg_paths(cfg: dict) -> dict[str, Path]:
    syn = cfg["data"]["synthetic"]
    root = synthetic_cfg_root(cfg)
    return {
        "root": root,
        "archive": root / syn["archive_name"],
        "wav_16k": root / syn["resampled_subdir"],
        "metadata": root / syn["metadata_file"],
    }


def cmd_download(cfg: dict) -> None:
    paths = cfg_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["metadata"].exists():
        print(f"Metadata already present: {paths['metadata']}")
        return
    if paths["archive"].exists() and paths["archive"].stat().st_size > 1_000_000_000:
        print(f"Extracting metadata from archive {paths['archive']} ...")
        run(f"zstd -dc {paths['archive']} | tar -x -C {paths['root']} --strip-components=1 '*.csv' README.md")
        return
    syn = cfg["data"]["synthetic"]
    from huggingface_hub import hf_hub_download

    print(f"Downloading metadata CSVs from {syn['hf_repo']} ...")
    for name in ("metadata_wer_025.csv", "metadata_wer_02.csv", "metadata_wer0.csv", "README.md"):
        hf_hub_download(
            repo_id=syn["hf_repo"],
            filename=name,
            repo_type="dataset",
            local_dir=str(paths["root"]),
            local_dir_use_symlinks=False,
        )
    print(f"Metadata -> {paths['root']}")


def _resample_one(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    src, dst, ffmpeg = args
    dst_path = Path(dst)
    if dst_path.exists():
        return src, True, "exists"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst,
    ]
    if subprocess.run(cmd).returncode != 0:
        return src, False, "ffmpeg failed"
    return src, True, "ok"


def cmd_resample(cfg: dict, jobs: int) -> None:
    paths = cfg_paths(cfg)
    dst_dir = paths["wav_16k"]
    metadata = paths["metadata"]
    if not metadata.exists():
        sys.exit(f"Metadata missing: {metadata}\nRun: uv run scripts/prepare_synthetic.py download")

    syn = cfg["data"]["synthetic"]
    delimiter = syn.get("delimiter", "|")
    id_col = (syn.get("columns") or {}).get("id", "id")

    todo: list[tuple[str, str, str]] = []
    missing_src = 0
    with metadata.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            clip_id = row.get(id_col) or ""
            if not clip_id:
                continue
            dst = dst_dir / f"{clip_id}.wav"
            if dst.exists():
                continue
            src_path = resolve_synthetic_source_wav(clip_id, cfg)
            if src_path is None:
                missing_src += 1
                continue
            todo.append((str(src_path), str(dst), "ffmpeg"))

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resampling {len(todo)} clips from AE slow_44K -> {dst_dir} ({jobs} workers) ...")
    if missing_src:
        print(f"  ({missing_src:,} metadata rows have no AE source wav — skipped)")

    ok = fail = 0
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_resample_one, item) for item in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                _, success, _ = fut.result()
                ok += int(success)
                fail += int(not success)
                if i % 5000 == 0 or i == len(futures):
                    print(f"  progress: {i}/{len(futures)}", flush=True)
    print(f"Resample done: ok={ok}, fail={fail}, out={dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("step", choices=("download", "resample", "all"))
    ap.add_argument("--jobs", type=int, default=8, help="Parallel ffmpeg workers for resample")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = ["download", "resample"] if args.step == "all" else [args.step]
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        if step == "download":
            cmd_download(cfg)
        elif step == "resample":
            cmd_resample(cfg, args.jobs)


if __name__ == "__main__":
    main()
