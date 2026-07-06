#!/usr/bin/env python3
"""Download the base Nemotron 3.5 ASR checkpoint from Hugging Face."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, repo_root


def download_via_hub(model_name: str, output: Path) -> None:
    """Download .nemo weights without loading the model onto GPU."""
    from huggingface_hub import hf_hub_download

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} from Hugging Face Hub ...")
    path = hf_hub_download(
        repo_id=model_name,
        filename=f"{model_name.split('/')[-1]}.nemo",
        repo_type="model",
    )
    src = Path(path)
    if output.exists() or output.is_symlink():
        output.unlink()
    output.symlink_to(src.resolve())
    print(f"Saved -> {output.resolve()} ({src.stat().st_size / 1e9:.2f} GB)")


def download_via_nemo(model_name: str, output: Path) -> None:
    from common import ensure_cuda_home

    ensure_cuda_home()
    import nemo.collections.asr as nemo_asr

    print(f"Downloading {model_name} via NeMo ...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(output))
    print(f"Saved -> {output.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--output", type=Path, default=Path("checkpoints/nemotron-3.5-asr-base.nemo"))
    ap.add_argument("--hub-only", action="store_true", help="Download .nemo from HF Hub (no GPU load)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["project"]["base_model"]

    if args.hub_only:
        download_via_hub(model_name, args.output)
        return

    try:
        download_via_nemo(model_name, args.output)
    except Exception as exc:
        print(f"NeMo download failed ({exc}); falling back to Hugging Face Hub ...")
        download_via_hub(model_name, args.output)


if __name__ == "__main__":
    main()
