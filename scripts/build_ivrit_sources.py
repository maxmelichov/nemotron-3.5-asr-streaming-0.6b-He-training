#!/usr/bin/env python3
"""Materialize ivrit.ai HF training sources into NeMo manifest rows (with dedup).

Sources (config data.ivrit_sources):
  - ivrit-ai/crowd-transcribe-v5  (parquet, train split)
  - ivrit-ai/crowd-recital         (aligned session folders)
  - ivrit-ai/knesset-committees    (aligned session folders)
  - ivrit-ai/audio-v2              (umbrella corpus — processed last to avoid dupes)

Usage:
  uv run scripts/build_ivrit_sources.py
  uv run scripts/build_ivrit_sources.py --sources crowd-transcribe-v5 crowd-recital
  uv run scripts/build_ivrit_sources.py --download-missing
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import requests
import soundfile as sf
from huggingface_hub import snapshot_download
from huggingface_hub.utils import build_hf_headers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    accepts_asr_text,
    build_dedup_index,
    decode_and_save_wav,
    dedupe_rows,
    load_config,
    normalize_text,
    repo_root,
)


def request_gated_access(dataset: str) -> None:
    url = f"https://huggingface.co/datasets/{dataset}/ask-access"
    requests.post(url, headers=build_hf_headers(), timeout=30)


def ensure_hf_dataset(local_dir: Path, hf_repo: str, download: bool) -> bool:
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return True
    if not download:
        print(f"SKIP {hf_repo}: {local_dir} missing (use --download-missing)")
        return False
    request_gated_access(hf_repo)
    print(f"Downloading {hf_repo} -> {local_dir} (this may take hours for large repos) ...")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=str(local_dir),
        etag_timeout=60,
        max_workers=4,
        token=os.environ.get("HF_TOKEN"),
    )
    return local_dir.is_dir()


def segment_median_prob(segment: dict) -> float:
    probs = [
        float(w["probability"])
        for w in (segment.get("words") or [])
        if w.get("probability") is not None
    ]
    if not probs:
        return 1.0
    return float(statistics.median(probs))


def find_session_audio(session_dir: Path) -> Path | None:
    for pattern in ("audio.*", "*.mka", "*.m4a", "*.mp3", "*.wav", "*.mp4"):
        matches = sorted(session_dir.glob(pattern))
        for path in matches:
            if path.suffix.lower() in {".json", ".vtt", ".txt", ".csv", ".done"}:
                continue
            return path
    return None


def ffmpeg_extract_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    sample_rate: int,
) -> float | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        info = sf.info(str(dst))
        return info.frames / info.samplerate
    duration = max(0.0, end - start)
    if duration <= 0:
        return None
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    if subprocess.run(cmd).returncode != 0 or not dst.exists():
        return None
    info = sf.info(str(dst))
    return info.frames / info.samplerate


def load_session_metadata(session_dir: Path) -> dict:
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def iter_aligned_session_rows(
    root: Path,
    *,
    source_name: str,
    source_type: str,
    source_id: str,
    clip_cache: Path,
    sample_rate: int,
    min_dur: float,
    max_dur: float,
    min_segment_quality: float,
    min_session_quality: float,
    cfg: dict,
) -> Iterator[dict]:
    if not root.is_dir():
        return

    session_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    for session_dir in session_dirs:
        aligned_path = session_dir / "transcript.aligned.json"
        if not aligned_path.exists():
            continue
        meta = load_session_metadata(session_dir)
        session_quality = float(meta.get("quality_score", 1.0))
        if session_quality < min_session_quality:
            continue
        source_entry_id = str(
            meta.get("source_entry_id") or meta.get("session_id") or session_dir.name
        )
        speaker_id = str(
            meta.get("user_id") or meta.get("speaker_id") or f"{source_type}:{source_entry_id}"
        )

        audio_path = find_session_audio(session_dir)
        if audio_path is None:
            continue

        try:
            data = json.loads(aligned_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        segments = data.get("segments") or data.get("segements") or []

        for idx, segment in enumerate(segments):
            text = normalize_text(str(segment.get("text") or ""))
            if not accepts_asr_text(text, cfg):
                continue
            if segment_median_prob(segment) < min_segment_quality:
                continue
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            if end <= start:
                continue

            clip_id = f"{source_name}:{source_entry_id}:{idx:04d}"
            dst = clip_cache / source_name / f"{source_entry_id}_{idx:04d}.wav"
            dur = ffmpeg_extract_segment(audio_path, dst, start, end, sample_rate)
            if dur is None or not (min_dur <= dur <= max_dur):
                if dst.exists() and dur is not None and dur < min_dur:
                    dst.unlink(missing_ok=True)
                continue

            yield {
                "clip_id": clip_id,
                "audio_filepath": str(dst.resolve()),
                "duration": round(dur, 3),
                "text": text,
                "speaker_id": speaker_id,
                "source_type": source_type,
                "source_id": source_id,
                "source_entry_id": source_entry_id,
                "segment_start": round(start, 3),
                "segment_end": round(end, 3),
            }


def materialize_aligned_source(
    spec: dict,
    source_name: str,
    cfg: dict,
    clip_cache: Path,
    seen: set[str],
) -> tuple[list[dict], dict]:
    root = Path(spec["local_dir"])
    rows = list(
        iter_aligned_session_rows(
            root,
            source_name=source_name,
            source_type=spec.get("source_type", source_name),
            source_id=spec.get("source_id", source_name),
            clip_cache=clip_cache,
            sample_rate=int(cfg["data"]["sample_rate"]),
            min_dur=float(cfg["data"]["min_duration"]),
            max_dur=float(cfg["data"]["max_duration"]),
            min_segment_quality=float(spec.get("min_segment_quality", cfg["data"]["ivrit_sources"]["min_segment_quality"])),
            min_session_quality=float(spec.get("min_session_quality", cfg["data"]["ivrit_sources"]["min_session_quality"])),
            cfg=cfg,
        )
    )
    kept, skipped = dedupe_rows(rows, seen)
    stats = {"scanned": len(rows), "kept": len(kept), "dupes_skipped": skipped}
    return kept, stats


def materialize_crowd_transcribe_v5(
    spec: dict,
    cfg: dict,
    clip_cache: Path,
    seen: set[str],
) -> tuple[list[dict], dict]:
    data_dir = Path(spec["local_dir"]) / "data"
    if not data_dir.is_dir():
        return [], {"scanned": 0, "kept": 0, "dupes_skipped": 0, "error": "missing data dir"}

    sample_rate = int(cfg["data"]["sample_rate"])
    min_dur = float(cfg["data"]["min_duration"])
    max_dur = float(cfg["data"]["max_duration"])
    split = spec.get("split", "train")
    id_field = spec.get("id_field", "uuid")
    text_field = spec.get("text_field", "sentence")

    rows: list[dict] = []
    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=[id_field, text_field, "audio"])
        for row_id, text, audio in zip(
            table[id_field].to_pylist(),
            table[text_field].to_pylist(),
            table["audio"].to_pylist(),
        ):
            text = normalize_text(str(text or ""))
            if not accepts_asr_text(text, cfg):
                continue
            clip_id = f"crowd-transcribe-v5:{row_id}"
            dst = clip_cache / "crowd-transcribe-v5" / f"{row_id}.wav"
            if dst.exists() and dst.stat().st_size > 0:
                dur = sf.info(str(dst)).frames / sf.info(str(dst)).samplerate
            else:
                audio_obj = audio or {}
                dur = decode_and_save_wav(
                    audio_obj.get("bytes"),
                    audio_obj.get("path"),
                    dst,
                    sample_rate,
                )
            if dur is None or not (min_dur <= dur <= max_dur):
                continue
            rows.append({
                "clip_id": clip_id,
                "audio_filepath": str(dst.resolve()),
                "duration": round(float(dur), 3),
                "text": text,
                "speaker_id": f"crowd-transcribe:{row_id.split('/')[0] if '/' in str(row_id) else row_id[:8]}",
                "source_type": "crowd_transcribe",
                "source_entry_id": str(row_id),
            })

    kept, skipped = dedupe_rows(rows, seen)
    return kept, {"scanned": len(rows), "kept": len(kept), "dupes_skipped": skipped}


def materialize_audio_v2(
    spec: dict,
    cfg: dict,
    clip_cache: Path,
    seen: set[str],
) -> tuple[list[dict], dict]:
    """Walk audio-v2 tree; aligned sessions use the same extractor as other ivrit sources."""
    root = Path(spec["local_dir"])
    if not root.is_dir():
        return [], {"scanned": 0, "kept": 0, "dupes_skipped": 0}

    # audio-v2 nests sources — find any folder with transcript.aligned.json + audio
    rows: list[dict] = []
    for aligned_path in sorted(root.rglob("transcript.aligned.json")):
        session_dir = aligned_path.parent
        meta = load_session_metadata(session_dir)
        source_type = str(meta.get("source_type") or "audio_v2")
        source_id = str(meta.get("source_id") or session_dir.parent.name)
        source_entry_id = str(meta.get("source_entry_id") or meta.get("session_id") or session_dir.name)
        speaker_id = str(meta.get("user_id") or meta.get("speaker_id") or f"{source_type}:{source_entry_id}")

        session_quality = float(meta.get("quality_score", 1.0))
        if session_quality < float(cfg["data"]["ivrit_sources"]["min_session_quality"]):
            continue
        audio_path = find_session_audio(session_dir)
        if audio_path is None:
            continue
        try:
            data = json.loads(aligned_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        segments = data.get("segments") or data.get("segements") or []
        for idx, segment in enumerate(segments):
            text = normalize_text(str(segment.get("text") or ""))
            if not accepts_asr_text(text, cfg):
                continue
            if segment_median_prob(segment) < float(cfg["data"]["ivrit_sources"]["min_segment_quality"]):
                continue
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            if end <= start:
                continue
            clip_id = f"audio-v2:{source_type}:{source_entry_id}:{idx:04d}"
            dst = clip_cache / "audio-v2" / source_type / f"{source_entry_id}_{idx:04d}.wav"
            dur = ffmpeg_extract_segment(
                audio_path, dst, start, end, int(cfg["data"]["sample_rate"])
            )
            if dur is None or not (float(cfg["data"]["min_duration"]) <= dur <= float(cfg["data"]["max_duration"])):
                continue
            rows.append({
                "clip_id": clip_id,
                "audio_filepath": str(dst.resolve()),
                "duration": round(dur, 3),
                "text": text,
                "speaker_id": speaker_id,
                "source_type": source_type,
                "source_id": source_id,
                "source_entry_id": source_entry_id,
                "segment_start": round(start, 3),
                "segment_end": round(end, 3),
            })

    kept, skipped = dedupe_rows(rows, seen)
    return kept, {"scanned": len(rows), "kept": len(kept), "dupes_skipped": skipped}


def materialize_ivrit_sources(
    cfg: dict,
    seen: set[str] | None = None,
    *,
    sources: list[str] | None = None,
    download_missing: bool = False,
) -> tuple[list[dict], dict]:
    ivrit_cfg = cfg["data"].get("ivrit_sources") or {}
    if not ivrit_cfg.get("enabled", False):
        return [], {}

    clip_cache = Path(ivrit_cfg.get("clip_cache", "./data/ivrit_clips"))
    if not clip_cache.is_absolute():
        clip_cache = repo_root() / clip_cache

    source_specs: dict = ivrit_cfg.get("sources") or {}
    order = ivrit_cfg.get(
        "process_order",
        ["crowd-transcribe-v5", "crowd-recital", "knesset-committees", "audio-v2"],
    )
    chosen = [name for name in order if name in source_specs]
    if sources:
        chosen = [name for name in chosen if name in sources]

    seen = set(seen or [])
    all_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for name in chosen:
        spec = source_specs[name]
        if not spec.get("enabled", True):
            print(f"SKIP ivrit/{name}: disabled")
            continue
        local_dir = Path(spec["local_dir"])
        hf_repo = spec.get("hf_repo", f"ivrit-ai/{name}")
        allow_download = download_missing and spec.get("download", True)
        if not ensure_hf_dataset(local_dir, hf_repo, allow_download):
            stats[name] = {"kept": 0, "skipped": "missing local_dir"}
            continue

        print(f"\n=== ivrit/{name} ({local_dir}) ===")
        if name == "crowd-transcribe-v5":
            rows, st = materialize_crowd_transcribe_v5(spec, cfg, clip_cache, seen)
        elif name == "audio-v2":
            rows, st = materialize_audio_v2(spec, cfg, clip_cache, seen)
        else:
            rows, st = materialize_aligned_source(spec, name, cfg, clip_cache, seen)
        hrs = sum(r["duration"] for r in rows) / 3600
        print(
            f"  kept {st.get('kept', len(rows)):,} clips ({hrs:.1f} h), "
            f"dupes skipped {st.get('dupes_skipped', 0):,}"
        )
        stats[name] = st
        all_rows.extend(rows)

    return all_rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--sources", nargs="+", help="Subset of ivrit sources to process")
    ap.add_argument("--download-missing", action="store_true")
    ap.add_argument("--seed-seen-from", type=Path, help="Existing manifest JSONL to seed dedup index")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seen: set[str] = set()
    if args.seed_seen_from:
        seen = build_dedup_index([
            json.loads(line)
            for line in args.seed_seen_from.open(encoding="utf-8")
            if line.strip()
        ])
        print(f"Dedup seed: {len(seen):,} keys from {args.seed_seen_from}")

    rows, stats = materialize_ivrit_sources(
        cfg,
        seen=seen,
        sources=args.sources,
        download_missing=args.download_missing,
    )
    out = Path(cfg["data"]["out_dir"]) / "manifests" / "ivrit_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    target_lang = cfg["project"]["target_lang"]
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            entry = {
                "audio_filepath": row["audio_filepath"],
                "duration": row["duration"],
                "text": row["text"],
                "lang": target_lang,
                "target_lang": target_lang,
                "prompt_mode": "langID",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    hrs = sum(r["duration"] for r in rows) / 3600
    print(f"\nWrote {out}: {len(rows):,} clips, {hrs:.1f} h")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats -> {meta_path}")


if __name__ == "__main__":
    main()
