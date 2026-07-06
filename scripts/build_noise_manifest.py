#!/usr/bin/env python3
"""Build a NeMo noise manifest for online augmentation during training.

NeMo's `noise` augmentor expects JSONL entries:
  {"audio_filepath": "/path/to/noise.wav", "duration": 12.3, "text": ""}

Usage:
  python scripts/build_noise_manifest.py --noise-dir /path/to/noise/wavs
  python scripts/build_noise_manifest.py --musan /path/to/musan/noise
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, repo_root, write_nemo_manifest


def collect_wavs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def build_from_dirs(dirs: list[Path], out_path: Path, target_lang: str, max_clips: int | None) -> int:
    rows: list[dict] = []
    for root in dirs:
        if not root.is_dir():
            print(f"Skip missing dir: {root}")
            continue
        for wav in collect_wavs(root):
            try:
                info = sf.info(str(wav))
                duration = info.frames / info.samplerate
            except Exception as exc:
                print(f"Skip {wav}: {exc}")
                continue
            if duration < 0.1:
                continue
            rows.append({
                "audio_filepath": str(wav.resolve()),
                "duration": round(duration, 3),
                "text": "",
            })
            if max_clips and len(rows) >= max_clips:
                break
        if max_clips and len(rows) >= max_clips:
            break

    write_nemo_manifest(rows, out_path, target_lang)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--noise-dir", type=Path, action="append", help="Directory with noise wav files")
    ap.add_argument("--musan", type=Path, help="MUSAN root (uses musan/noise subdir)")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--max-clips", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    aug_cfg = cfg["training"]["augmentation"]
    out_path = args.output or Path(aug_cfg.get("noise_manifest", "./data/manifests/noise.json"))
    target_lang = cfg["project"]["target_lang"]

    dirs: list[Path] = list(args.noise_dir or [])
    if args.musan:
        dirs.append(args.musan / "noise")
    if not dirs:
        default = aug_cfg.get("noise_dirs") or []
        dirs = [Path(p) for p in default]

    if not dirs:
        sys.exit(
            "No noise directories specified.\n"
            "Use --noise-dir /path/to/wavs or set training.augmentation.noise_dirs in config.yaml"
        )

    n = build_from_dirs(dirs, out_path, target_lang, args.max_clips)
    print(f"Noise manifest: {out_path} ({n} clips)")


if __name__ == "__main__":
    main()
