#!/usr/bin/env python3
"""Apply training telephony augmentations to sample clips for listening tests.

Uses the same perturbation classes as NeMo train-time augmentation (config.yaml).

Usage:
  uv run scripts/demo_augmentation.py
  uv run scripts/demo_augmentation.py --audio path/to/clip.wav --out data/augment_demo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, register_custom_perturbations, repo_root, resolve_augmentor_cfg
from telephony_perturb import Phone8kResamplePerturbation


def load_segment(path: Path, target_sr: int = 16000):
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    return AudioSegment.from_file(str(path), target_sr=target_sr)


def save_segment(segment, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = segment.samples if hasattr(segment, "samples") else segment._samples
    sf.write(str(path), samples, int(segment.sample_rate))


def clone_segment(segment):
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    return AudioSegment(
        segment.samples.copy(),
        sample_rate=segment.sample_rate,
        target_sr=segment.sample_rate,
    )


def build_perturbations(augmentor: dict, noise_manifest: Path | None):
    from nemo.collections.asr.parts.preprocessing.perturb import (
        GainPerturbation,
        NoisePerturbation,
        WhiteNoisePerturbation,
    )

    phone_cfg = augmentor["phone_8k_resample"]
    perturbations = {
        "phone_8k_resample": Phone8kResamplePerturbation(
            sr=phone_cfg["sr"],
            phone_sr=phone_cfg["phone_sr"],
            resample_types=phone_cfg["resample_types"],
            same_method_up_down=phone_cfg.get("same_method_up_down", False),
            rng=42,
        ),
        "gain": GainPerturbation(
            min_gain_dbfs=augmentor["gain"]["min_gain_dbfs"],
            max_gain_dbfs=augmentor["gain"]["max_gain_dbfs"],
            rng=42,
        ),
        "white_noise": WhiteNoisePerturbation(
            min_level=augmentor["white_noise"]["min_level"],
            max_level=augmentor["white_noise"]["max_level"],
            rng=42,
        ),
    }
    if noise_manifest and noise_manifest.exists() and augmentor.get("noise"):
        noise_cfg = augmentor["noise"]
        perturbations["noise"] = NoisePerturbation(
            manifest_path=str(noise_manifest.resolve()),
            min_snr_db=noise_cfg["min_snr_db"],
            max_snr_db=noise_cfg["max_snr_db"],
            max_gain_db=noise_cfg.get("max_gain_db", 300.0),
            rng=42,
        )
    return perturbations


def apply_chain(segment, perturbations: dict, order: list[str]) -> None:
    for name in order:
        perturbations[name].perturb(segment)


def pick_clips(manifest: Path, n: int, seed: int) -> list[Path]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    random.seed(seed)
    chosen = random.sample(rows, min(n, len(rows)))
    return [Path(r["audio_filepath"]) for r in chosen]


def demo_clip(
    audio_path: Path,
    out_dir: Path,
    perturbations: dict,
    stack_order: list[str],
) -> list[Path]:
    stem = audio_path.stem
    clip_dir = out_dir / stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    original = load_segment(audio_path)
    written: list[Path] = []

    orig_path = clip_dir / "00_original.wav"
    save_segment(original, orig_path)
    written.append(orig_path)

    for idx, name in enumerate(stack_order, start=1):
        seg = clone_segment(original)
        perturbations[name].perturb(seg)
        out_path = clip_dir / f"{idx:02d}_{name}.wav"
        save_segment(seg, out_path)
        written.append(out_path)

    mixed = clone_segment(original)
    random.seed(42)
    apply_chain(mixed, perturbations, stack_order)
    mix_path = clip_dir / "99_training_stack.wav"
    save_segment(mixed, mix_path)
    written.append(mix_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/dev.json"))
    ap.add_argument("--audio", type=Path, action="append", help="Specific wav file(s) to augment")
    ap.add_argument("--num-clips", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("data/augment_demo"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    register_custom_perturbations(cfg)
    augmentor = resolve_augmentor_cfg(cfg)
    if not augmentor:
        sys.exit("Augmentation disabled in config (training.augmentation.enabled=false)")

    noise_manifest = Path(cfg["training"]["augmentation"].get("noise_manifest", ""))
    perturbations = build_perturbations(augmentor, noise_manifest)
    stack_order = [k for k in ("phone_8k_resample", "gain", "white_noise", "noise") if k in perturbations]

    if args.audio:
        clips = args.audio
    else:
        if not args.manifest.exists():
            sys.exit(f"Manifest not found: {args.manifest}")
        clips = pick_clips(args.manifest, args.num_clips, args.seed)

    print("Training augmentations (telephony preset):")
    for name, params in augmentor.items():
        prob = params.get("prob", "n/a")
        print(f"  - {name}: prob={prob}")
    print(f"\nWriting demos to {args.out.resolve()}/")

    all_written: list[Path] = []
    for clip in clips:
        if not clip.exists():
            print(f"SKIP missing: {clip}")
            continue
        paths = demo_clip(clip, args.out, perturbations, stack_order)
        all_written.extend(paths)
        print(f"  {clip.name} -> {paths[0].parent}/")

    if not all_written:
        sys.exit("No output files written.")
    print(f"\nDone — {len(all_written)} wav files. Listen to 99_training_stack.wav for the combined effect.")


if __name__ == "__main__":
    main()
