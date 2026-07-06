#!/usr/bin/env python3
"""Build NeMo JSONL manifests for standard Hebrew ASR evaluation benchmarks.

Benchmarks (ivrit.ai-style suite):
  - ivrit-ai/eval-d1
  - ivrit-ai/eval-whatsapp (gated — requires HF access)
  - upai-inc/saspeech (SASpeech gold standard)
  - google/fleurs he_il test
  - mozilla-foundation/common_voice_17_0 he validated (HF or local cv_dir)
  - imvladikon/hebrew_speech_kan validation

Usage:
  python scripts/build_eval_benchmarks.py
  python scripts/build_eval_benchmarks.py --benchmark fleurs saspeech
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub.utils import build_hf_headers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    decode_and_save_wav,
    is_valid_text,
    load_config,
    normalize_text,
    repo_root,
    require_columns,
    transcode_mp3_to_wav,
    write_nemo_manifest,
)


def benchmark_specs(cfg: dict) -> dict[str, dict[str, Any]]:
    return cfg["evaluation"]["benchmarks"]


def request_gated_access(dataset: str, repo_type: str = "dataset") -> None:
    """Accept auto-gated HF repo terms (ivrit.ai contact-info gate, etc.)."""
    url = f"https://huggingface.co/{repo_type}s/{dataset}/ask-access"
    requests.post(url, headers=build_hf_headers(), timeout=30)


def load_hf_split(spec: dict[str, Any]):
    if spec.get("gated"):
        request_gated_access(spec["dataset"], "dataset")
    load_kwargs: dict[str, Any] = {"split": spec["split"]}
    if spec.get("config"):
        return load_dataset(spec["dataset"], spec["config"], **load_kwargs)
    return load_dataset(spec["dataset"], **load_kwargs)


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def export_rows(
    rows: list[dict],
    wav_root: Path,
    target_lang: str,
    manifest_path: Path,
) -> dict[str, Any]:
    write_nemo_manifest(rows, manifest_path, target_lang)
    return {
        "manifest": str(manifest_path),
        "clips": len(rows),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
        "wav_dir": str(wav_root),
    }


def require_hf_fields(examples: list[dict], fields: list[str], dataset: str) -> None:
    if not examples:
        return
    cols = set(examples[0].keys())
    missing = [field for field in fields if field not in cols]
    if missing:
        raise ValueError(
            f"{dataset}: missing field(s) {missing}\n"
            f"  expected: {fields}\n"
            f"  found:    {sorted(cols)}"
        )


def rows_from_hf_audio_text(
    examples: list[dict],
    *,
    text_field: str,
    wav_dir: Path,
    sample_rate: int,
    max_duration: float,
    id_field: str = "uuid",
    dataset: str = "hf",
) -> list[dict]:
    require_hf_fields(examples, [text_field, id_field, "audio"], dataset)
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, ex in enumerate(examples):
        text = normalize_text(ex.get(text_field) or "")
        if not is_valid_text(text):
            continue
        audio = ex.get("audio") or {}
        dst = wav_dir / f"{i:06d}.wav"
        dur = decode_and_save_wav(audio.get("bytes"), audio.get("path"), dst, sample_rate)
        if dur is None:
            continue
        if dur > max_duration:
            continue
        rows.append({
            "audio_filepath": str(dst.resolve()),
            "duration": round(dur, 3),
            "text": text,
            "speaker_id": str(ex.get(id_field) or i),
        })
    return rows


def build_simple_hf_benchmark(
    name: str,
    spec: dict[str, Any],
    cfg: dict,
    out_dir: Path,
) -> dict[str, Any]:
    target_lang = cfg["project"]["target_lang"]
    sr = cfg["data"]["sample_rate"]
    max_dur = float(cfg["evaluation"].get("max_eval_duration", 600.0))
    manifest_path = out_dir / f"{name}.json"
    wav_dir = out_dir / "wav" / name

    ds = load_hf_split(spec)
    ds = ds.cast_column("audio", Audio(decode=False))

    text_field = spec.get("text_field", "text")
    id_field = spec.get("id_field", "uuid")
    rows = rows_from_hf_audio_text(
        list(ds),
        text_field=text_field,
        wav_dir=wav_dir,
        sample_rate=sr,
        max_duration=max_dur,
        id_field=id_field,
        dataset=spec["dataset"],
    )
    if not rows:
        raise RuntimeError(f"{name}: no clips exported")
    meta = export_rows(rows, wav_dir, target_lang, manifest_path)
    meta["benchmark"] = name
    meta["dataset"] = spec["dataset"]
    return meta


def build_fleurs_benchmark(cfg: dict, out_dir: Path) -> dict[str, Any]:
    spec = benchmark_specs(cfg)["fleurs"]
    return build_simple_hf_benchmark("fleurs", spec, cfg, out_dir)


def build_saspeech_benchmark(cfg: dict, out_dir: Path) -> dict[str, Any]:
    spec = benchmark_specs(cfg)["saspeech"]
    return build_simple_hf_benchmark("saspeech", spec, cfg, out_dir)


def build_ivrit_benchmark(name: str, cfg: dict, out_dir: Path) -> dict[str, Any]:
    spec = benchmark_specs(cfg)[name]
    return build_simple_hf_benchmark(name, spec, cfg, out_dir)


def build_kan_benchmark(cfg: dict, out_dir: Path) -> dict[str, Any]:
    spec = benchmark_specs(cfg)["hebrew_speech_kan"]
    return build_simple_hf_benchmark("hebrew_speech_kan", spec, cfg, out_dir)


def build_common_voice_benchmark(cfg: dict, out_dir: Path) -> dict[str, Any]:
    target_lang = cfg["project"]["target_lang"]
    spec = benchmark_specs(cfg)["common_voice"]
    sr = cfg["data"]["sample_rate"]
    max_dur = float(cfg["evaluation"].get("max_eval_duration", 600.0))
    manifest_path = out_dir / "common_voice.json"
    wav_dir = out_dir / "wav" / "common_voice"

    rows: list[dict] = []
    hf_tried = False
    for dataset, config, split in [
        (spec["dataset"], spec["config"], spec["split"]),
        ("fsicoli/common_voice_17_0", "he", "validation"),
    ]:
        try:
            ds = load_dataset(dataset, config, split=split, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            rows = rows_from_hf_audio_text(
                list(ds),
                text_field=spec.get("text_field", "sentence"),
                wav_dir=wav_dir,
                sample_rate=sr,
                max_duration=max_dur,
                id_field="client_id",
                dataset=dataset,
            )
            if rows:
                print(f"Common Voice loaded from {dataset} ({split}): {len(rows)} clips")
                break
        except Exception as exc:
            hf_tried = True
            print(f"Common Voice HF load failed ({dataset}/{split}): {exc}")

    if not rows and hf_tried:
        try:
            ds_val = load_dataset("fsicoli/common_voice_17_0", "he", split="validation", trust_remote_code=True)
            ds_test = load_dataset("fsicoli/common_voice_17_0", "he", split="test", trust_remote_code=True)
            combined = list(ds_val.cast_column("audio", Audio(decode=False))) + list(
                ds_test.cast_column("audio", Audio(decode=False))
            )
            rows = rows_from_hf_audio_text(
                combined,
                text_field="sentence",
                wav_dir=wav_dir,
                sample_rate=sr,
                max_duration=max_dur,
                id_field="client_id",
                dataset="fsicoli/common_voice_17_0",
            )
            if rows:
                print(f"Common Voice loaded from fsicoli validation+test: {len(rows)} clips")
        except Exception as exc:
            print(f"Common Voice fsicoli fallback failed: {exc}")

    if not rows:
        print("Trying local cv_dir ...")
        cv_dir = Path(spec.get("cv_dir") or cfg["data"]["common_voice"]["cv_dir"])
        tsv_path = cv_dir / "validated.tsv"
        clips_dir = cv_dir / "clips"
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"Common Voice not on HF and local dir missing: {cv_dir}\n"
                "Download CV 17 Hebrew and set evaluation.benchmarks.common_voice.cv_dir"
            )
        wav_dir.mkdir(parents=True, exist_ok=True)
        cv_cols = cfg["data"]["common_voice"].get("columns") or {
            "text": "sentence",
            "audio_path": "path",
            "speaker_id": "client_id",
        }
        delimiter = cfg["data"]["common_voice"].get("delimiter", "\t")
        text_col = cv_cols["text"]
        audio_col = cv_cols["audio_path"]
        speaker_col = cv_cols["speaker_id"]
        df = pd.read_csv(tsv_path, sep=delimiter, low_memory=False)
        require_columns(list(df.columns), [text_col, audio_col, speaker_col], tsv_path)
        for i, rec in enumerate(df.itertuples(index=False)):
            text = normalize_text(getattr(rec, text_col))
            if not is_valid_text(text):
                continue
            dst = wav_dir / f"{i:06d}.wav"
            dur = transcode_mp3_to_wav(clips_dir / getattr(rec, audio_col), dst, sr)
            if dur is None or dur > max_dur:
                continue
            rows.append({
                "audio_filepath": str(dst.resolve()),
                "duration": round(dur, 3),
                "text": text,
                "speaker_id": str(getattr(rec, speaker_col)),
            })

    if not rows:
        raise RuntimeError("common_voice: no clips exported")
    meta = export_rows(rows, wav_dir, target_lang, manifest_path)
    meta["benchmark"] = "common_voice"
    meta["dataset"] = spec["dataset"]
    return meta


BUILDERS = {
    "ivrit_eval_d1": lambda cfg, out: build_ivrit_benchmark("ivrit_eval_d1", cfg, out),
    "ivrit_eval_whatsapp": lambda cfg, out: build_ivrit_benchmark("ivrit_eval_whatsapp", cfg, out),
    "saspeech": build_saspeech_benchmark,
    "fleurs": build_fleurs_benchmark,
    "common_voice": build_common_voice_benchmark,
    "hebrew_speech_kan": build_kan_benchmark,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--benchmark", nargs="*", help="Subset to build (default: all enabled)")
    ap.add_argument("--force", action="store_true", help="Rebuild even if manifest exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["evaluation"]["manifest_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = benchmark_specs(cfg)
    names = args.benchmark or [k for k, v in specs.items() if v.get("enabled", True)]
    summary: dict[str, Any] = {}

    for name in names:
        if name not in BUILDERS:
            sys.exit(f"Unknown benchmark: {name}")
        spec = specs[name]
        manifest_path = out_dir / f"{name}.json"
        if manifest_path.exists() and not args.force:
            rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
            hrs = sum(r["duration"] for r in rows) / 3600
            print(f"Skip {name}: {manifest_path} ({len(rows)} clips, {hrs:.2f} h)")
            summary[name] = {"manifest": str(manifest_path), "clips": len(rows), "hours": round(hrs, 2), "skipped": True}
            continue

        print(f"\n=== Building {name} ===", flush=True)
        try:
            summary[name] = BUILDERS[name](cfg, out_dir)
            print(f"OK {name}: {summary[name]['clips']} clips, {summary[name]['hours']} h")
        except Exception as exc:
            msg = str(exc)
            if spec.get("gated") or "gated dataset" in msg.lower():
                print(f"SKIP {name}: gated dataset — accept access on Hugging Face and run `hf auth login`")
                summary[name] = {"error": msg, "skipped": True, "gated": True}
            else:
                print(f"FAIL {name}: {exc}")
                summary[name] = {"error": msg, "skipped": True}

    meta_path = out_dir / "benchmark_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBenchmark summary -> {meta_path}")


if __name__ == "__main__":
    main()
