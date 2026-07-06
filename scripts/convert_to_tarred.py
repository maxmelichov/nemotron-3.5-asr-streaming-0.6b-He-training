#!/usr/bin/env python3
"""Convert NeMo manifests to tarred audio shards (blog Step 1 — scale data).

Uses NeMo's convert_to_tarred_audio_dataset.py for efficient Lhotse streaming.

Usage:
  python scripts/convert_to_tarred.py --manifest data/manifests/train.json
  python scripts/convert_to_tarred.py --manifest data/manifests/train.json --num-shards 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, nemo_root, repo_root, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--target-dir", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tar_cfg = cfg["data"]["tarred"]
    num_shards = args.num_shards or tar_cfg["num_shards"]
    target_dir = args.target_dir or Path(tar_cfg["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    nemo = nemo_root()
    script = nemo / "scripts" / "speech_recognition" / "convert_to_tarred_audio_dataset.py"
    if not script.exists():
        raise FileNotFoundError(f"NeMo tarred converter not found: {script}")

    manifest = args.manifest.resolve()
    shard_prefix = target_dir / manifest.stem
    run(
        f"python {script} "
        f"--manifest_path={manifest} "
        f"--target_dir={target_dir} "
        f"--num_shards={num_shards} "
        f"--shuffle "
        f"--shuffle_seed=42"
    )

    tar_pattern = str(target_dir / "audio_*.tar")
    tarred_manifest = target_dir / f"{manifest.stem}_tarred.json"
    print(f"\nTarred shards: {tar_pattern}")
    print(f"Tarred manifest: {tarred_manifest}")
    print("\nFor training, set in config or finetune CLI:")
    print(f"  model.train_ds.is_tarred=true")
    print(f"  model.train_ds.manifest_filepath={tarred_manifest}")
    print(f"  model.train_ds.tarred_audio_filepaths='{tar_pattern}'")


if __name__ == "__main__":
    main()
