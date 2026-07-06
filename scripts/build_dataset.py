#!/usr/bin/env python3
"""Step 1 — Prepare Hebrew ASR data as NeMo JSONL manifests.

Split policy (config data.splits):
  train.json  — VoxKnesset train + synthetic train + podcasts train
  dev.json    — VoxKnesset test speakers only (Hebrew transcript, not IPA)
  test.json   — ivrit.ai eval benchmark suite (build_eval_benchmarks.py)

VoxKnesset uses field ``transcript`` (punctuated Hebrew). Never ``whisper_phonemes``.

Usage:
  python scripts/build_eval_benchmarks.py          # once: cache eval wav + test.json inputs
  python scripts/build_dataset.py --source mixed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

from common import (
    accepts_asr_text,
    build_dedup_index,
    decode_and_save_wav,
    dedupe_rows,
    json_field,
    load_config,
    normalize_text,
    prepare_hebrew_asr_text,
    repo_root,
    require_columns,
    resolve_synthetic_audio_dir,
    row_field,
    speaker_split,
    synthetic_cfg_root,
    synthetic_speaker_id,
    transcode_mp3_to_wav,
    write_nemo_manifest,
    filter_hebrew_rows,
)


def iter_voxknesset_records(
    manifest_path: Path,
    audio_dir: Path,
    ds_cfg: dict,
    min_dur: float,
    max_dur: float,
    min_whisper_confidence: float,
    min_vad_speech_ratio: float,
    cfg: dict | None = None,
) -> Iterator[dict]:
    text_field = ds_cfg["text_field"]
    forbidden = set(ds_cfg.get("forbidden_text_fields") or [])
    if text_field in forbidden:
        raise ValueError(f"{dataset_key}: text_field {text_field!r} is forbidden (use Hebrew transcript)")
    if text_field in ("whisper_phonemes", "phonemes", "ipa"):
        raise ValueError(f"{dataset_key}: text_field must be Hebrew (transcript), not {text_field!r}")
    wav_field = ds_cfg.get("wav_field", "wav")
    duration_field = ds_cfg.get("duration_field", "duration_sec")
    speaker_field = ds_cfg.get("speaker_field", "speaker_id")
    confidence_field = ds_cfg.get("confidence_field", "whisper_confidence")
    vad_field = ds_cfg.get("vad_field", "vad_speech_ratio")

    with manifest_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: bad JSON: {exc}") from exc

            wav_name = json_field(rec, wav_field) or json_field(rec, "audio_filepath")
            if not wav_name:
                continue

            raw_text = json_field(rec, text_field)
            if raw_text is None:
                continue
            text = normalize_text(raw_text)
            if not accepts_asr_text(text, cfg):
                continue

            duration = json_field(rec, duration_field) or json_field(rec, "duration")
            if duration is None:
                wav_path = audio_dir / wav_name
                if not wav_path.exists():
                    continue
                duration = sf.info(str(wav_path)).duration
            duration = float(duration)
            if not (min_dur <= duration <= max_dur):
                continue

            conf_raw = json_field(rec, confidence_field)
            if conf_raw is not None and float(conf_raw) < min_whisper_confidence:
                continue

            vad_raw = json_field(rec, vad_field)
            if vad_raw is not None and float(vad_raw) < min_vad_speech_ratio:
                continue

            wav_path = audio_dir / Path(wav_name).name
            if not wav_path.exists():
                continue

            speaker_id = json_field(rec, speaker_field) or wav_name
            yield {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(duration, 3),
                "text": text,
                "speaker_id": str(speaker_id),
            }


def materialize_voxknesset(
    cfg: dict,
    dataset_key: str,
    limit: Optional[int],
) -> dict[str, list[dict]]:
    vk_cfg = cfg["data"]["asr_transcition"]
    ds_cfg = vk_cfg[dataset_key]
    root = Path(vk_cfg["root"])
    manifest_path = root / ds_cfg["manifest"]
    audio_dir = root / ds_cfg["audio_dir"]

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")

    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    skipped = 0
    seen = 0

    for rec in iter_voxknesset_records(
        manifest_path=manifest_path,
        audio_dir=audio_dir,
        ds_cfg=ds_cfg,
        min_dur=cfg["data"]["min_duration"],
        max_dur=cfg["data"]["max_duration"],
        min_whisper_confidence=float(vk_cfg.get("min_whisper_confidence", 0.0)),
        min_vad_speech_ratio=float(vk_cfg.get("min_vad_speech_ratio", 0.0)),
        cfg=cfg,
    ):
        seen += 1
        split = speaker_split(
            rec["speaker_id"],
            float(vk_cfg.get("test_speaker_frac", 0.05)),
            float(vk_cfg.get("dev_speaker_frac", 0.05)),
        )
        rows_by_split[split].append(rec)
        if limit and sum(len(v) for v in rows_by_split.values()) >= limit:
            break

    print(f"VoxKnesset ({dataset_key}): scanned {seen}, kept {sum(len(v) for v in rows_by_split.values())}")
    for split, rows in rows_by_split.items():
        hrs = sum(r["duration"] for r in rows) / 3600
        speakers = len({r["speaker_id"] for r in rows})
        print(f"  {split:10s}: {len(rows):>8,d} clips  {hrs:>8.1f} h  {speakers:>4} speakers")
    if skipped:
        print(f"  skipped: {skipped}")
    return rows_by_split


def synthetic_data_available(cfg: dict) -> bool:
    syn = cfg["data"]["synthetic"]
    if not syn.get("enabled", True):
        return False
    try:
        resolve_synthetic_audio_dir(cfg)
        return (synthetic_cfg_root(cfg) / syn["metadata_file"]).exists()
    except FileNotFoundError:
        return False


def iter_synthetic_records(cfg: dict) -> Iterator[dict]:
    syn = cfg["data"]["synthetic"]
    root, audio_dir = resolve_synthetic_audio_dir(cfg)
    metadata_path = root / syn["metadata_file"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata missing: {metadata_path}")

    min_dur = cfg["data"]["min_duration"]
    max_dur = cfg["data"]["max_duration"]
    text_mode = syn.get("text_mode", "unvocalized")
    delimiter = syn.get("delimiter", "|")
    cols = syn.get("columns") or {"id": "id", "text": "text"}
    id_col = cols["id"]
    text_col = cols["text"]

    with metadata_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        require_columns(reader.fieldnames, [id_col, text_col], metadata_path)
        for row in reader:
            clip_id = row_field(row, id_col) or ""
            raw_text = row_field(row, text_col)
            if raw_text is None:
                continue
            text = prepare_hebrew_asr_text(raw_text, text_mode)
            if not clip_id or not accepts_asr_text(text, cfg):
                continue

            wav_path = audio_dir / f"{clip_id}.wav"
            if not wav_path.exists():
                continue

            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate
            if not (min_dur <= duration <= max_dur):
                continue

            yield {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(duration, 3),
                "text": text,
                "speaker_id": synthetic_speaker_id(clip_id),
            }


def materialize_synthetic(cfg: dict, limit: Optional[int]) -> dict[str, list[dict]]:
    syn = cfg["data"]["synthetic"]
    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    seen = 0

    for rec in iter_synthetic_records(cfg):
        seen += 1
        split = speaker_split(
            rec["speaker_id"],
            float(syn.get("test_speaker_frac", 0.05)),
            float(syn.get("dev_speaker_frac", 0.05)),
        )
        rows_by_split[split].append(rec)
        if limit and sum(len(v) for v in rows_by_split.values()) >= limit:
            break

    kept = sum(len(v) for v in rows_by_split.values())
    print(f"Synthetic ({syn['metadata_file']}): scanned {seen:,}, kept {kept:,}")
    for split, rows in rows_by_split.items():
        hrs = sum(r["duration"] for r in rows) / 3600
        speakers = len({r["speaker_id"] for r in rows})
        print(f"  {split:10s}: {len(rows):>8,d} clips  {hrs:>8.1f} h  {speakers:>4} speakers")
    return rows_by_split


def resolve_podcast_audio_path(root: Path, audio_dir: Path, segment_id: str, segment_path: str) -> Optional[Path]:
    if segment_path:
        p = Path(segment_path)
        if p.exists():
            return p
    candidate = audio_dir / f"{segment_id}.wav"
    if candidate.exists():
        return candidate
    candidate = root / "segments" / f"{segment_id}.wav"
    if candidate.exists():
        return candidate
    return None


def iter_podcast_records(cfg: dict) -> Iterator[dict]:
    pc = cfg["data"]["podcasts"]
    root = Path(pc["root"])
    audio_dir = root / pc.get("audio_dir", "segments")
    transcripts_path = Path(pc["transcripts_tsv"])
    min_conf = float(pc.get("min_token_confidence", 0.0))
    min_dur = cfg["data"]["min_duration"]
    max_dur = cfg["data"]["max_duration"]
    delimiter = pc.get("delimiter", "\t")
    cols = pc.get("columns") or {}
    segment_col = cols.get("segment_id", "segment_id")
    text_col = cols.get("text", "text")
    duration_col = cols.get("duration", "duration_sec")
    confidence_col = cols.get("confidence", "token_confidence")
    segment_path_col = cols.get("segment_path", "segment_path")
    speaker_col = cols.get("speaker_id", "youtube_id")

    if not transcripts_path.exists():
        raise FileNotFoundError(f"Podcast transcripts missing: {transcripts_path}")

    with transcripts_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        require_columns(
            reader.fieldnames,
            [segment_col, text_col, duration_col, segment_path_col, speaker_col],
            transcripts_path,
        )
        for row in reader:
            segment_id = row_field(row, segment_col) or ""
            raw_text = row_field(row, text_col)
            if raw_text is None:
                continue
            text = normalize_text(raw_text)
            if not segment_id or not accepts_asr_text(text, cfg):
                continue

            conf = row_field(row, confidence_col, optional=True)
            if conf is not None and float(conf) < min_conf:
                continue

            wav_path = resolve_podcast_audio_path(
                root, audio_dir, segment_id, row_field(row, segment_path_col, optional=True) or ""
            )
            if wav_path is None:
                continue

            duration = row_field(row, duration_col, optional=True)
            if duration is not None:
                duration = float(duration)
            else:
                duration = sf.info(str(wav_path)).duration
            if not (min_dur <= duration <= max_dur):
                continue

            speaker_id = row_field(row, speaker_col, optional=True) or segment_id.split("_")[0] or segment_id
            yield {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(float(duration), 3),
                "text": text,
                "speaker_id": str(speaker_id),
            }


def materialize_podcasts(cfg: dict, limit: Optional[int]) -> dict[str, list[dict]]:
    pc = cfg["data"]["podcasts"]
    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    seen = 0

    for rec in iter_podcast_records(cfg):
        seen += 1
        split = speaker_split(
            rec["speaker_id"],
            float(pc.get("test_speaker_frac", 0.05)),
            float(pc.get("dev_speaker_frac", 0.05)),
        )
        rows_by_split[split].append(rec)
        if limit and sum(len(v) for v in rows_by_split.values()) >= limit:
            break

    kept = sum(len(v) for v in rows_by_split.values())
    print(f"Podcasts ({pc['root']}): scanned {seen:,}, kept {kept:,}")
    for split, rows in rows_by_split.items():
        hrs = sum(r["duration"] for r in rows) / 3600
        speakers = len({r["speaker_id"] for r in rows})
        print(f"  {split:10s}: {len(rows):>8,d} clips  {hrs:>8.1f} h  {speakers:>4} speakers")
    return rows_by_split


def materialize_fleurs(cfg: dict, limit: Optional[int]) -> dict[str, list[dict]]:
    fleurs_cfg = cfg["data"]["fleurs"]
    sr = cfg["data"]["sample_rate"]
    out_dir = Path(cfg["data"]["out_dir"])
    min_dur = cfg["data"]["min_duration"]
    max_dur = cfg["data"]["max_duration"]

    rows_by_split: dict[str, list[dict]] = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset(fleurs_cfg["dataset"], fleurs_cfg["config"], split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        wav_dir = out_dir / "wav" / "fleurs" / split
        rows: list[dict] = []
        for i, ex in enumerate(ds):
            audio = ex["audio"]
            text = normalize_text(ex["transcription"])
            if not accepts_asr_text(text, cfg):
                continue
            dst = wav_dir / f"{i:05d}.wav"
            dur = decode_and_save_wav(audio.get("bytes"), audio.get("path"), dst, sr)
            if dur is None or not (min_dur <= dur <= max_dur):
                continue
            rows.append({
                "audio_filepath": str(dst.resolve()),
                "duration": round(dur, 3),
                "text": text,
                "speaker_id": str(ex.get("id", i)),
            })
        rows_by_split[split] = rows
        print(f"FLEURS {split}: {len(rows)} clips")

    return rows_by_split


def materialize_common_voice(cfg: dict, cv_dir: Path, limit: Optional[int]) -> dict[str, list[dict]]:
    sr = cfg["data"]["sample_rate"]
    out_dir = Path(cfg["data"]["out_dir"])
    min_dur = cfg["data"]["min_duration"]
    max_dur = cfg["data"]["max_duration"]
    clips_dir = cv_dir / "clips"
    cv_cfg = cfg["data"]["common_voice"]
    delimiter = cv_cfg.get("delimiter", "\t")
    cols = cv_cfg.get("columns") or {"text": "sentence", "audio_path": "path", "speaker_id": "client_id"}
    text_col = cols["text"]
    audio_col = cols["audio_path"]
    speaker_col = cols["speaker_id"]

    split_map = {
        "train": ["validated.tsv", "other.tsv"],
        "validation": ["dev.tsv"],
        "test": ["test.tsv"],
    }
    rows_by_split: dict[str, list[dict]] = {k: [] for k in split_map}

    for split, tsv_names in split_map.items():
        wav_dir = out_dir / "wav" / "common_voice" / split
        idx = 0
        for tsv_name in tsv_names:
            tsv_path = cv_dir / tsv_name
            if not tsv_path.exists():
                if split == "train" and tsv_name == "other.tsv":
                    continue
                raise FileNotFoundError(f"Missing {tsv_path}")
            df = pd.read_csv(tsv_path, sep=delimiter, low_memory=False)
            require_columns(list(df.columns), [text_col, audio_col, speaker_col], tsv_path)
            if limit and split == "train":
                df = df.head(limit)
            for rec in df.itertuples(index=False):
                text = normalize_text(getattr(rec, text_col))
                if not accepts_asr_text(text, cfg):
                    continue
                dst = wav_dir / f"{idx:06d}.wav"
                dur = transcode_mp3_to_wav(clips_dir / getattr(rec, audio_col), dst, sr)
                idx += 1
                if dur is None or not (min_dur <= dur <= max_dur):
                    continue
                rows_by_split[split].append({
                    "audio_filepath": str(dst.resolve()),
                    "duration": round(dur, 3),
                    "text": text,
                    "speaker_id": str(getattr(rec, speaker_col)),
                })
        print(f"Common Voice {split}: {len(rows_by_split[split])} clips")

    return rows_by_split


    return rows_by_split


def hours_and_clips(rows: list[dict]) -> tuple[int, float]:
    return len(rows), sum(r["duration"] for r in rows) / 3600


def materialize_combined_eval_test(cfg: dict) -> list[dict]:
    """Merge enabled ivrit.ai-style eval benchmarks into test.json."""
    eval_dir = Path(cfg["evaluation"]["manifest_dir"])
    specs = cfg["evaluation"]["benchmarks"]
    combined: list[dict] = []
    missing: list[str] = []

    for name, spec in specs.items():
        if not spec.get("enabled", True):
            continue
        manifest = eval_dir / f"{name}.json"
        if not manifest.exists():
            missing.append(name)
            continue
        n_before = len(combined)
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["benchmark"] = name
                combined.append(row)
        n = len(combined) - n_before
        hrs = sum(r["duration"] for r in combined[n_before:]) / 3600
        print(f"  eval {name:22s}: {n:>6,} clips  {hrs:>6.1f} h")

    if missing:
        print(
            "WARN: eval benchmarks not built yet: "
            + ", ".join(missing)
            + "\n  Run: uv run scripts/build_eval_benchmarks.py"
        )
    if not combined:
        raise FileNotFoundError(
            "No eval benchmarks found under "
            f"{eval_dir}. Run: uv run scripts/build_eval_benchmarks.py"
        )
    return combined


def validation_clip_cap(cfg: dict) -> int:
    """Clips actually seen per validation pass (limit_val_batches × validation batch_size)."""
    train_cfg = cfg.get("training") or {}
    batches = int(train_cfg.get("limit_val_batches", 3000))
    batch_size = int(train_cfg.get("validation_batch_size", 2))
    return batches * batch_size


def split_dev_for_training(dev_rows: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Keep in-training val subset on dev; move the rest to train."""
    cap = validation_clip_cap(cfg)
    if len(dev_rows) <= cap:
        return dev_rows, []
    return dev_rows[:cap], dev_rows[cap:]


def print_split_summary(label: str, rows: list[dict]) -> None:
    clips, hrs = hours_and_clips(rows)
    print(f"  {label:10s}: {clips:>8,} clips  {hrs:>8.1f} h")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument(
        "--source",
        choices=("voxknesset", "synthetic", "podcasts", "mixed", "ivrit", "fleurs", "common_voice"),
        default="mixed",
    )
    ap.add_argument(
        "--dataset",
        choices=("voxknesset_hebrew_ipa", "voxknesset_clips"),
        default="voxknesset_hebrew_ipa",
        help="Which asr_transcition manifest to use",
    )
    ap.add_argument("--cv-dir", type=Path, help="Common Voice release dir (he/)")
    ap.add_argument("--limit", type=int, help="Cap total clips kept (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target_lang = cfg["project"]["target_lang"]
    manifest_dir = Path(cfg["data"]["out_dir"]) / "manifests"
    if cfg["project"].get("hebrew_only", True):
        print(f"Hebrew-only mode: transcripts filtered, target_lang={target_lang}, prompt_mode=langID")

    combined: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    train_parts: dict[str, list[dict]] = {}
    vk_test: list[dict] = []
    ivrit_cfg = cfg["data"].get("ivrit_sources") or {}

    use_vk = args.source in ("voxknesset", "mixed")
    use_syn = args.source in ("synthetic", "mixed") and synthetic_data_available(cfg)
    use_pc = args.source in ("podcasts", "mixed") and cfg["data"]["podcasts"].get("enabled", True)

    if args.source in ("synthetic", "mixed") and not use_syn and args.source == "synthetic":
        sys.exit(
            "Synthetic data not available.\n"
            "Run: uv run scripts/prepare_synthetic.py all\n"
            "Or enable mixed without synthetic (already default: synthetic.enabled=false)."
        )
    if args.source == "mixed" and not use_syn:
        print("SKIP synthetic: not extracted or disabled (see config data.synthetic.enabled)")

    multi = sum(int(x) for x in (use_vk, use_syn, use_pc)) > 1

    if use_vk:
        vk_rows = materialize_voxknesset(cfg, args.dataset, args.limit if not multi else None)
        train_parts["voxknesset"] = vk_rows["train"]
        vk_test = vk_rows["test"]
        combined["train"].extend(vk_rows["train"])
        print(f"VoxKnesset -> dev (test split): {len(vk_test):,} clips")

    if use_syn:
        syn_limit = None
        if args.limit is not None:
            syn_limit = args.limit - sum(len(v) for v in combined.values())
            if syn_limit <= 0:
                syn_limit = 0
        if syn_limit != 0:
            syn_rows = materialize_synthetic(cfg, syn_limit)
            syn_train = syn_rows["train"] + syn_rows["validation"] + syn_rows["test"]
            train_parts["synthetic"] = syn_train
            combined["train"].extend(syn_train)
            if syn_rows["validation"] or syn_rows["test"]:
                print(
                    f"Synthetic val/test holdouts -> train: "
                    f"{len(syn_rows['validation']):,} val + {len(syn_rows['test']):,} test clips"
                )

    if use_pc:
        pc_limit = None
        if args.limit is not None:
            pc_limit = args.limit - sum(len(v) for v in combined.values())
            if pc_limit <= 0:
                pc_limit = 0
        if pc_limit != 0:
            pc_rows = materialize_podcasts(cfg, pc_limit)
            pc_train = pc_rows["train"] + pc_rows["validation"] + pc_rows["test"]
            train_parts["podcasts"] = pc_train
            combined["train"].extend(pc_train)
            if pc_rows["validation"] or pc_rows["test"]:
                print(
                    f"Podcast val/test holdouts -> train: "
                    f"{len(pc_rows['validation']):,} val + {len(pc_rows['test']):,} test clips"
                )

    if args.source in ("mixed", "ivrit") and ivrit_cfg.get("enabled", False):
        from build_ivrit_sources import materialize_ivrit_sources

        seen = build_dedup_index(combined["train"])
        ivrit_rows, ivrit_stats = materialize_ivrit_sources(cfg, seen=seen)
        if ivrit_rows:
            train_parts["ivrit"] = ivrit_rows
            combined["train"].extend(ivrit_rows)
            dupes = sum(int(v.get("dupes_skipped", 0)) for v in ivrit_stats.values())
            print(f"Ivrit sources: +{len(ivrit_rows):,} clips, {dupes:,} cross-source dupes skipped")

    if args.source == "ivrit":
        combined["train"] = train_parts.get("ivrit", [])

    if args.source == "fleurs":
        fleurs_rows = materialize_fleurs(cfg, args.limit)
        for split, rows in fleurs_rows.items():
            combined[split].extend(rows)

    if args.source == "common_voice":
        cv_dir = args.cv_dir or Path(cfg["data"]["common_voice"]["cv_dir"])
        cv_rows = materialize_common_voice(cfg, cv_dir, args.limit)
        for split, rows in cv_rows.items():
            combined[split].extend(rows)

    # dev = first N VoxKnesset test clips only (matches limit_val_batches × validation_batch_size)
    if use_vk and vk_test:
        dev_keep, dev_to_train = split_dev_for_training(vk_test, cfg)
        combined["validation"] = dev_keep
        if dev_to_train:
            combined["train"].extend(dev_to_train)
            print(
                f"VoxKnesset dev overflow -> train: {len(dev_to_train):,} clips "
                f"(dev capped at {len(dev_keep):,} for in-training val)"
            )
    elif use_vk:
        print("WARN: VoxKnesset test split empty — dev.json will be empty")

    # test = ivrit.ai eval suite (not VoxKnesset / synthetic / podcast held-out)
    if args.source in ("voxknesset", "mixed", "synthetic", "podcasts"):
        print("\nEval benchmarks -> test.json:")
        try:
            combined["test"] = materialize_combined_eval_test(cfg)
        except FileNotFoundError as exc:
            print(exc)
            test_path = manifest_dir / "test.json"
            if test_path.exists():
                print(f"Keeping existing {test_path}")
                combined["test"] = [
                    json.loads(line) for line in test_path.open(encoding="utf-8") if line.strip()
                ]
            else:
                combined["test"] = []

    print("\nFinal manifests:")
    if cfg["project"].get("hebrew_only", True):
        for split in ("train", "validation", "test"):
            filtered, dropped = filter_hebrew_rows(combined[split], cfg)
            if dropped:
                print(f"  Hebrew filter {split}: dropped {dropped:,} non-Hebrew rows")
            combined[split] = filtered
    print_split_summary("train", combined["train"])
    print_split_summary("dev", combined["validation"])
    print_split_summary("test", combined["test"])
    for name, rows in train_parts.items():
        print_split_summary(f"  train/{name}", rows)

    write_nemo_manifest(combined["train"], manifest_dir / "train.json", target_lang)
    write_nemo_manifest(combined["validation"], manifest_dir / "dev.json", target_lang)
    write_nemo_manifest(combined["test"], manifest_dir / "test.json", target_lang)

    meta = {
        "target_lang": target_lang,
        "source": args.source,
        "split_policy": cfg["data"].get("splits", {}),
        "dataset": args.dataset if use_vk else None,
        "synthetic_metadata": cfg["data"]["synthetic"]["metadata_file"] if use_syn else None,
        "podcasts_root": cfg["data"]["podcasts"]["root"] if use_pc else None,
        "ivrit_sources": ivrit_cfg.get("sources") if ivrit_cfg.get("enabled") else None,
        "train_sources": {k: {"clips": len(v), "hours": round(hours_and_clips(v)[1], 2)} for k, v in train_parts.items()},
        "splits": {k: len(v) for k, v in combined.items()},
        "hours": {k: round(sum(r["duration"] for r in v) / 3600, 2) for k, v in combined.items()},
        "speakers": {k: len({r.get("speaker_id") for r in v}) for k, v in combined.items()},
    }
    meta_path = manifest_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDataset summary -> {meta_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
