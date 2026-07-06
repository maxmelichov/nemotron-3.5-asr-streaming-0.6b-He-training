#!/usr/bin/env python3
"""Step 5 — Run streaming inference on audio files or a NeMo manifest.

Phone-call deployment: use att_context_size=[56,0] (80 ms, no lookahead) per NVIDIA blog.

Usage:
  python scripts/inference.py --audio call.wav
  python scripts/inference.py --audio call.wav --phone
  python scripts/inference.py --manifest data/manifests/eval/fleurs.json --output preds.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ensure_cuda_home, hebrew_only_enabled, load_config, nemo_root, repo_root, run


def make_single_manifest(audio: Path, target_lang: str) -> Path:
    import soundfile as sf

    info = sf.info(str(audio))
    duration = info.frames / info.samplerate
    manifest = Path(tempfile.mkdtemp()) / "infer_manifest.json"
    entry = {
        "audio_filepath": str(audio.resolve()),
        "duration": round(duration, 3),
        "text": "",
        "lang": target_lang,
        "target_lang": target_lang,
        "prompt_mode": "langID",
    }
    manifest.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def find_prediction_file(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("**/*pred*.json*")) + sorted(out_dir.glob("**/*.json*"))
    return candidates[-1] if candidates else None


def print_predictions(pred_path: Path, limit: int = 5) -> None:
    for line in pred_path.read_text(encoding="utf-8").strip().splitlines()[:limit]:
        row = json.loads(line)
        text = row.get("pred_text") or row.get("pred") or row.get("text") or ""
        print(" ", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--model", type=Path, default=Path("checkpoints/hebrew-finetuned.nemo"))
    ap.add_argument("--audio", type=Path, help="Single wav file (16 kHz mono recommended)")
    ap.add_argument("--manifest", type=Path, help="NeMo JSONL manifest")
    ap.add_argument("--output", type=Path, default=Path("preds.json"))
    ap.add_argument("--target-lang")
    ap.add_argument("--att-context", type=str, help='Override, e.g. "[56,0]"')
    ap.add_argument("--phone", action="store_true", help="Use 80 ms phone/voice-agent latency [56,0]")
    args = ap.parse_args()

    ensure_cuda_home()
    cfg = load_config(args.config)
    target_lang = cfg["project"]["target_lang"]
    if hebrew_only_enabled(cfg):
        if args.target_lang and args.target_lang != target_lang:
            sys.exit(
                f"This project is Hebrew-only (target_lang={target_lang}). "
                f"Remove --target-lang {args.target_lang}."
            )
    elif args.target_lang:
        target_lang = args.target_lang
    if args.phone:
        att = json.dumps([56, 0])
    else:
        att = args.att_context or json.dumps(cfg["inference"]["att_context_size"])
    strip_tags = cfg["inference"]["strip_lang_tags"]

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    if args.audio and args.manifest:
        sys.exit("Provide --audio OR --manifest, not both")
    if not args.audio and not args.manifest:
        sys.exit("Provide --audio or --manifest")

    manifest = args.manifest or make_single_manifest(args.audio, target_lang)
    out_dir = args.output.parent / "infer_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    nemo = nemo_root()
    infer_script = nemo / "examples" / "asr" / "asr_cache_aware_streaming" / "speech_to_text_cache_aware_streaming_infer.py"

    cmd = (
        f"python {infer_script} "
        f"model_path={args.model.resolve()} "
        f"dataset_manifest={manifest.resolve()} "
        f"output_path={out_dir.resolve()} "
        f"target_lang={target_lang} "
        f'att_context_size="{att}" '
        f"decoder_type=rnnt "
        f"pad_and_drop_preencoded=true "
        f"batch_size=8 "
        f"strip_lang_tags={'true' if strip_tags else 'false'} "
        f"cuda=0"
    )
    run(cmd)

    pred_file = find_prediction_file(out_dir)
    if pred_file:
        import shutil

        shutil.copy(pred_file, args.output)
        print(f"Predictions -> {args.output}")
        print_predictions(args.output)
    else:
        print(f"Check outputs in {out_dir}")


if __name__ == "__main__":
    main()
