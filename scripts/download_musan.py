#!/usr/bin/env python3
"""Download MUSAN noise corpus for train-time augmentation (OpenSLR 17)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, repo_root, run


MUSAN_URL = "https://openslr.trmal.net/resources/17/musan.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--target-dir", type=Path, help="Extract root (default: asr_transcition/musan)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.target_dir or Path(cfg["data"]["asr_transcition"]["root"]) / "musan"
    archive = root.parent / "musan.tar.gz"

    root.parent.mkdir(parents=True, exist_ok=True)
    if (root / "noise").is_dir() and any((root / "noise").rglob("*.wav")):
        print(f"MUSAN already present: {root}")
        return

    if not archive.exists() or archive.stat().st_size < 1_000_000_000:
        print(f"Downloading MUSAN (~12 GB) -> {archive}")
        run(f"wget -c -O {archive} {MUSAN_URL}")

    print(f"Extracting {archive} -> {root.parent} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=root.parent)
    print(f"MUSAN ready: {root / 'noise'}")


if __name__ == "__main__":
    main()
