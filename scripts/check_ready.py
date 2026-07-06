#!/usr/bin/env python3
"""Verify training data, environment, and checkpoints before fine-tuning.

Usage:
  uv run scripts/check_ready.py
  uv run scripts/check_ready.py --fix   # build manifests if missing
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_config,
    manifest_hours,
    nemo_root,
    read_nemo_manifest,
    repo_root,
    resolve_synthetic_audio_dir,
    run,
    synthetic_cfg_root,
    validate_hebrew_manifest,
    hebrew_only_enabled,
)


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def synthetic_ready(cfg: dict) -> bool:
    syn = cfg["data"]["synthetic"]
    if not syn.get("enabled", True):
        return False
    metadata = synthetic_cfg_root(cfg) / syn["metadata_file"]
    if not metadata.exists():
        return False
    try:
        resolve_synthetic_audio_dir(cfg)
        return True
    except FileNotFoundError:
        return False


def voxknesset_ready(cfg: dict) -> bool:
    vk = cfg["data"]["asr_transcition"]
    root = Path(vk["root"])
    ds = vk["voxknesset_hebrew_ipa"]
    return (root / ds["manifest"]).exists() and (root / ds["audio_dir"]).is_dir()


def podcasts_ready(cfg: dict) -> bool:
    pc = cfg["data"]["podcasts"]
    if not pc.get("enabled", True):
        return False
    return Path(pc["transcripts_tsv"]).exists() and (Path(pc["root"]) / pc["audio_dir"]).is_dir()


def check_manifests(cfg: dict) -> tuple[bool, dict]:
    manifest_dir = Path(cfg["data"]["out_dir"]) / "manifests"
    train_path = manifest_dir / "train.json"
    dev_path = manifest_dir / "dev.json"
    issues: list[str] = []
    stats: dict = {}

    if not train_path.exists():
        issues.append(f"missing {train_path}")
        return False, {"issues": issues}

    train_rows = read_nemo_manifest(train_path)
    dev_rows = read_nemo_manifest(dev_path) if dev_path.exists() else []
    train_h = manifest_hours(train_rows)
    dev_h = manifest_hours(dev_rows)
    stats = {"train_clips": len(train_rows), "train_h": train_h, "dev_clips": len(dev_rows), "dev_h": dev_h}

    min_train_h = float(cfg.get("readiness", {}).get("min_train_hours", 100.0))
    if train_h < min_train_h:
        issues.append(f"train set only {train_h:.1f} h (need >= {min_train_h} h for full run)")
    if len(dev_rows) < 100:
        issues.append(f"dev set only {len(dev_rows)} clips")
    stats["issues"] = issues
    return len(issues) == 0, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--fix", action="store_true", help="Build mixed manifests if missing or too small")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    errors: list[str] = []

    print("=== Environment ===")
    if shutil.which("uv"):
        ok(f"uv {subprocess.check_output(['uv', '--version'], text=True).strip()}")
    else:
        fail("uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        errors.append("uv")

    for bin_name in ("ffmpeg", "sox"):
        if shutil.which(bin_name):
            ok(bin_name)
        else:
            fail(f"{bin_name} not found (run ./setup.sh)")
            errors.append(bin_name)

    lock = root / "uv.lock"
    venv = root / ".venv"
    if lock.exists() and venv.is_dir():
        ok(f"uv env ({lock.name})")
    else:
        warn("uv lock/venv missing — run: ./setup.sh")
        errors.append("uv sync")

    print("\n=== NeMo ===")
    nemo_dir = root / "NeMo"
    if nemo_dir.is_dir():
        ok(f"NeMo clone -> {nemo_dir}")
    else:
        warn("NeMo/ not cloned — run: ./setup.sh")
        errors.append("NeMo clone")

    try:
        nemo = nemo_root()
        ok(f"NEMO_ROOT={nemo}")
    except RuntimeError as exc:
        warn(str(exc))

    print("\n=== Training data sources ===")
    if voxknesset_ready(cfg):
        ok("VoxKnesset manifest + clips")
    else:
        fail("VoxKnesset not found")
        errors.append("voxknesset")

    if podcasts_ready(cfg):
        ok("Podcast segments + transcripts")
    else:
        warn("Podcasts disabled or missing")

    if synthetic_ready(cfg):
        ok("Synthetic TTS (local extract)")
    else:
        syn = cfg["data"]["synthetic"]
        if syn.get("enabled", True):
            warn(
                "Synthetic not extracted — mixed build skips it. "
                "Run: uv run scripts/prepare_synthetic.py all"
            )
        else:
            ok("Synthetic disabled in config (skipped)")

    print("\n=== Manifests ===")
    ready, stats = check_manifests(cfg)
    if ready:
        ok(f"train.json: {stats['train_clips']:,} clips, {stats['train_h']:.1f} h")
        ok(f"dev.json:   {stats['dev_clips']:,} clips, {stats['dev_h']:.1f} h  (VoxKnesset test)")
        test_path = Path(cfg["data"]["out_dir"]) / "manifests" / "test.json"
        if test_path.exists():
            test_rows = read_nemo_manifest(test_path)
            ok(f"test.json:  {len(test_rows):,} clips, {manifest_hours(test_rows):.1f} h  (ivrit eval suite)")
        else:
            warn("test.json missing — run: uv run scripts/build_eval_benchmarks.py && build_dataset.py")
        if hebrew_only_enabled(cfg):
            print("\n=== Hebrew-only transcripts ===")
            target = cfg["project"]["target_lang"]
            train_path = Path(cfg["data"]["out_dir"]) / "manifests" / "train.json"
            dev_path = Path(cfg["data"]["out_dir"]) / "manifests" / "dev.json"
            train_rows = read_nemo_manifest(train_path)
            dev_rows = read_nemo_manifest(dev_path) if dev_path.exists() else []
            try:
                validate_hebrew_manifest(train_rows, target, train_path)
                validate_hebrew_manifest(dev_rows, target, dev_path)
                ok(f"all train/dev text is Hebrew ({target}, langID prompt)")
            except SystemExit as exc:
                fail(str(exc.args[0] if exc.args else "Hebrew-only check failed"))
                errors.append("hebrew transcripts")
    else:
        for issue in stats.get("issues", []):
            warn(issue)
        if stats.get("train_clips"):
            warn(f"current train: {stats['train_clips']:,} clips, {stats['train_h']:.1f} h")
        errors.append("manifests")

    if args.fix and not ready:
        print("\n=== Building mixed manifests ===")
        run(f"uv run scripts/build_dataset.py --source mixed --config {args.config}")

    print("\n=== Checkpoints ===")
    base = root / "checkpoints" / "nemotron-3.5-asr-base.nemo"
    if base.exists():
        ok(f"base model ({base.stat().st_size / 1e9:.2f} GB)")
    else:
        warn("base model missing — run: uv run scripts/download_model.py")
        errors.append("base model")

    ft = root / "checkpoints" / "hebrew-finetuned.nemo"
    if ft.exists():
        ok(f"fine-tuned checkpoint linked")
    else:
        ok("fine-tuned checkpoint (will be created by training)")

    print("\n=== Noise augmentation (optional) ===")
    noise = Path(cfg["training"]["augmentation"].get("noise_manifest", "data/manifests/noise.json"))
    if noise.exists():
        ok(f"noise manifest ({len(read_nemo_manifest(noise))} clips)")
    else:
        warn("noise manifest missing — telephony training continues without noise aug")
        warn("  uv run scripts/build_noise_manifest.py --noise-dir /path/to/noise")

    print("\n" + "=" * 60)
    if errors:
        print("NOT READY — fix the items above, then re-run:")
        print("  uv run scripts/check_ready.py")
        if "manifests" in errors:
            print("  uv run scripts/build_eval_benchmarks.py")
            print("  uv run scripts/build_dataset.py --source mixed")
        if "base model" in errors:
            print("  uv run scripts/download_model.py")
        if "uv sync" in errors or "NeMo clone" in errors:
            print("  ./setup.sh")
        sys.exit(1)

    print("READY TO TRAIN")
    print("  uv run scripts/finetune.py")
    print("  # or: ./run.sh train")


if __name__ == "__main__":
    main()
