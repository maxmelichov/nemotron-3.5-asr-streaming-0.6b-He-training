#!/usr/bin/env python3
"""Build NeMo manifests for Hebrew fine-tuning straight from Hugging Face.

Replaces the /mnt/windows_nvme + /home/maxm layout the original build_dataset.py
assumed. Every source is pulled from the Hub, decoded to 16 kHz mono, and written as
NeMo rows carrying the he-IL prompt:

    {"audio_filepath": ..., "duration": ..., "text": ..., "target_lang": "he-IL"}

Sources (--source all, or name them individually):

  ivrit30s        notmax123/ivirits-audio-v2-30s      parquet, FLAC + whisper text
  voxknesset      notmax123/voxknesset-hebrew-ipa     manifest joined to the audio in
                                                      ivrit-ai/VoxKnesset by source_filename;
                                                      clips are cut at [start_sec, end_sec]
  crowd_transcribe ivrit-ai/crowd-transcribe-v5       parquet, human `sentence`
  crowd_recital   ivrit-ai/crowd-recital              audio.mka cut at the timestamps in
                                                      transcript.aligned.json
  podcast286      notmax123/audio_286h                podcast_clean.zip; 16 kHz wavs kept
                                                      as-is, text from the `hebrew` column
  lhs_synthetic   notmax123/large-he-synthetic-tts     131 GB tar.zst streamed and extracted
                                                      on the fly; text from hebrew_ipa_text
  saspeech        notmax123/SASPEECH_AUTO_clean       7z archive
  ranlevi         notmax123/RanLevi40h                7z archive
  synthetic       notmax123/SententicDataTTS          7z archive + metadata csv

Two sources the request listed are deliberately NOT here:
  * ivrit-ai/VoxKnesset on its own has no transcript column (speaker_id/age/gender/audio
    only) -- the transcripts live in notmax123/voxknesset-hebrew-ipa, which is what the
    `voxknesset` source joins against.
  * ivrit-ai/audio-transcripts has no audio column; it is the transcript layer for
    audio-v2 and is already represented by `ivrit30s`.

Eval is ivrit-ai/eval-whatsapp only (--source eval_whatsapp); dev is a small stratified
sample drawn from the training sources (--dev-hours).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGET_LANG = "he-IL"
SAMPLE_RATE = 16000
NIQQUD = re.compile(r"[֑-ׇ]")
HEBREW = re.compile(r"[֐-׿]")


class Skips:
    """Counts why rows were dropped.

    Every builder previously did `except Exception: continue`, so a missing ffmpeg made
    each mp3 row vanish without a word and produced a near-empty manifest that looked
    like a successful run. Failures are now counted and reported per source.
    """

    def __init__(self, source: str):
        self.source = source
        self.counts: dict[str, int] = {}
        self.first_error: str = ""

    def add(self, reason: str, detail: str = "") -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if reason == "decode" and not self.first_error:
            self.first_error = detail

    def report(self, kept: int) -> None:
        if not self.counts:
            return
        total = sum(self.counts.values())
        detail = ", ".join(f"{k}={v:,}" for k, v in sorted(self.counts.items()))
        print(f"[{self.source}] skipped {total:,} ({detail}); kept {kept:,}", flush=True)
        decoded = self.counts.get("decode", 0)
        if decoded and decoded > max(kept, 1):
            print(f"[{self.source}] WARNING: more rows failed to decode than were kept — "
                  f"first error: {self.first_error}", flush=True)


@dataclass
class Row:
    audio_filepath: str
    duration: float
    text: str
    source: str
    speaker: str = ""
    extra: dict = field(default_factory=dict)

    def to_nemo(self) -> dict:
        row = {
            "audio_filepath": self.audio_filepath,
            "duration": round(self.duration, 3),
            "text": self.text,
            "target_lang": TARGET_LANG,
        }
        row.update(self.extra)
        return row


# --------------------------------------------------------------------------- text


def safe_name(key: str, limit: int = 120) -> str:
    """Filesystem-safe, length-bounded stem.

    Source ids embed full episode titles in Hebrew; as UTF-8 those blow past the 255-byte
    filename limit and libsndfile fails with a bare "System error". Keep a readable head
    and append a hash so names stay unique.
    """
    import hashlib

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-") or "clip"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:limit]}_{digest}"


def normalize_text(text: str, *, strip_niqqud: bool = True) -> str:
    """ASR targets: unvocalized Hebrew, collapsed whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if strip_niqqud:
        text = NIQQUD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def acceptable(text: str, *, min_chars: int = 2) -> bool:
    return len(text) >= min_chars and HEBREW.search(text) is not None


# --------------------------------------------------------------------------- audio


def write_clip(audio: "object", path: Path, sample_rate: int = SAMPLE_RATE) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return len(audio) / sample_rate


def decode_bytes(payload: bytes, sample_rate: int = SAMPLE_RATE):
    """Decode arbitrary encoded audio bytes to mono float32 at `sample_rate`."""
    import numpy as np

    try:
        data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr == sample_rate:
            return data
    except Exception:  # noqa: BLE001 - fall through to ffmpeg for mka/m4a/opus
        pass
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        input=payload, capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"decode failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype("float32") / 32768.0


def decode_file(path: Path, sample_rate: int = SAMPLE_RATE):
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"decode failed for {path.name}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype("float32") / 32768.0


# --------------------------------------------------------------------------- hub


def decode_and_write(job: tuple) -> tuple:
    """(key, payload, path, text, speaker) -> (path, duration, text, speaker) or error.

    ffmpeg runs as a subprocess and soundfile releases the GIL, so a thread pool gives
    near-linear speed-up here; single-threaded this stage ran at ~11 clips/s.
    """
    key, payload, path, text, speaker = job
    try:
        audio = decode_bytes(payload)
    except Exception as error:  # noqa: BLE001
        return ("decode", f"{type(error).__name__}: {error}", None, None, None)
    try:
        dur = write_clip(audio, path)
    except Exception as error:  # noqa: BLE001
        return ("write", f"{type(error).__name__}: {error}", None, None, None)
    return ("ok", str(path.resolve()), dur, text, speaker)


def run_jobs(jobs: list, workers: int):
    from concurrent.futures import ThreadPoolExecutor

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(decode_and_write, jobs))


def pool_context():
    """Prefer fork on Linux.

    With spawn, every worker re-imports and gets its own copy of the caller's state --
    the VoxKnesset `wanted` map is ~865k entries, so 64 workers meant ~148 GB of
    duplicated dict and the run wedged. fork shares it copy-on-write.
    """
    import multiprocessing as mp

    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def _shard_worker(payload: tuple) -> tuple:
    """Process one parquet shard end to end, in its own process.

    Shard-level parallelism is what actually matters: with sequential shards the box sat
    ~82% idle waiting on a single HF download stream, so the in-shard thread pool had
    nothing to chew on. Each process now downloads and decodes its own shard.
    """
    kind, repo, shard, token, args, audio_dir = payload
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    first_error = ""

    def note(reason: str, detail: str = "") -> None:
        nonlocal first_error
        skipped[reason] = skipped.get(reason, 0) + 1
        if reason == "decode" and not first_error:
            first_error = detail

    if kind == "ivrit30s":
        cols = ["segment_id", "audio", "text", "duration_sec", "token_confidence",
                "vad_speech_ratio", "episode"]
    else:
        cols = ["uuid", "audio", "sentence"]

    try:
        batches = parquet_batches(repo, shard, token, cols, batch_size=args.workers * 2)
    except Exception as error:  # noqa: BLE001
        return ([], {"shard_open": 1}, f"{type(error).__name__}: {error}", 0.0)

    total = 0.0
    for batch in batches:
        jobs = []
        for r in batch:
            if "audio" not in r:
                note("no_audio_column")
                continue
            if kind == "ivrit30s":
                if r["token_confidence"] < args.min_confidence:
                    note("confidence"); continue
                if r["vad_speech_ratio"] < args.min_vad_ratio:
                    note("vad"); continue
                text = normalize_text(r["text"])
                if not acceptable(text):
                    note("text"); continue
                key, speaker = str(r["segment_id"]), str(r["episode"])
            else:
                text = normalize_text(r.get("sentence") or "")
                if not acceptable(text):
                    note("text"); continue
                key, speaker = str(r["uuid"]), ""
            jobs.append((key, r["audio"]["bytes"], audio_dir / f"{safe_name(key)}.wav",
                         text, speaker))
        for status, a, dur, text, speaker in run_jobs(jobs, args.workers):
            if status != "ok":
                note(status, a); continue
            if not (args.min_duration <= dur <= args.max_duration):
                note("duration")
                Path(a).unlink(missing_ok=True)
                continue
            total += dur
            rows.append({"path": a, "dur": dur, "text": text, "speaker": speaker})
    return (rows, skipped, first_error, total)


_VK_WANTED: dict = {}


def _vk_init(wanted: dict) -> None:
    global _VK_WANTED
    _VK_WANTED = wanted


def _vk_shard(payload: tuple) -> tuple:
    """Cut one VoxKnesset audio shard into the clips its manifest rows ask for."""
    repo, shard, token, args, audio_dir = payload
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    first_error = ""
    total = 0.0
    try:
        batches = parquet_batches(repo, shard, token, ["audio", "speaker_id"], batch_size=4)
    except Exception as error:  # noqa: BLE001
        return ([], {"shard_open": 1}, f"{type(error).__name__}: {error}", 0.0)

    for batch in batches:
        for r in batch:
            if "audio" not in r:
                # Repos carry stray parquet next to the audio shards (VoxKnesset ships a
                # transcripts.parquet of (filename, text)). Skip rather than kill the pool.
                skipped["no_audio_column"] = skipped.get("no_audio_column", 0) + 1
                continue
            clips = _VK_WANTED.get(r["audio"]["path"])
            if not clips:
                continue
            try:
                audio = decode_bytes(r["audio"]["bytes"])
            except Exception as error:  # noqa: BLE001
                skipped["decode"] = skipped.get("decode", 0) + 1
                if not first_error:
                    first_error = f"{type(error).__name__}: {error}"
                continue
            for clip in clips:
                a, b = int(clip["start"] * SAMPLE_RATE), int(clip["end"] * SAMPLE_RATE)
                piece = audio[a:b]
                dur = len(piece) / SAMPLE_RATE
                if not (args.min_duration <= dur <= args.max_duration):
                    skipped["duration"] = skipped.get("duration", 0) + 1
                    continue
                path = audio_dir / f"{safe_name(clip['wav'])}.wav"
                try:
                    write_clip(piece, path)
                except Exception as error:  # noqa: BLE001
                    skipped["write"] = skipped.get("write", 0) + 1
                    continue
                total += dur
                rows.append({"path": str(path.resolve()), "dur": dur,
                             "text": clip["text"], "speaker": clip["speaker"]})
    return (rows, skipped, first_error, total)


def build_parquet_source(kind: str, repo: str, shards: list[str], out: Path,
                         token: str, args) -> list[Row]:
    audio_dir = out / "audio" / kind
    audio_dir.mkdir(parents=True, exist_ok=True)
    skips = Skips(kind)
    rows: list[Row] = []
    total = 0.0
    budget = args.max_hours_per_source * 3600 if args.max_hours_per_source else float("inf")

    payloads = [(kind, repo, shard, token, args, audio_dir) for shard in shards]
    ctx = pool_context()
    done = 0
    with ctx.Pool(processes=args.shard_workers) as pool:
        for shard_rows, skipped, first_error, shard_total in pool.imap_unordered(
            _shard_worker, payloads
        ):
            done += 1
            for reason, count in skipped.items():
                for _ in range(count):
                    skips.add(reason, first_error)
            for r in shard_rows:
                rows.append(Row(r["path"], r["dur"], r["text"], kind, r["speaker"]))
            total += shard_total
            print(f"[{kind}] shard {done}/{len(shards)}: {len(rows):,} clips, "
                  f"{total/3600:.1f} h", flush=True)
            if total >= budget:
                print(f"[{kind}] hit {args.max_hours_per_source} h cap", flush=True)
                pool.terminate()
                break
            remaining = free_gb(out)
            if remaining < args.min_free_gb:
                # Filling the volume would leave no room for checkpoints, which are
                # multi-GB and written mid-training.
                print(f"[{kind}] stopping: {remaining:.0f} GB free < --min-free-gb "
                      f"{args.min_free_gb} ({total/3600:.0f} h staged)", flush=True)
                pool.terminate()
                break
    skips.report(len(rows))
    return rows


def shutil_copy(src, dst, chunk: int = 1 << 20) -> None:
    while True:
        buf = src.read(chunk)
        if not buf:
            return
        dst.write(buf)


def free_gb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1e9


# A dataset's audio and text rarely sit under the names you first guess, and they are
# not always in the same file: ivrit-ai/VoxKnesset keeps audio in data/*.parquet and its
# Hebrew text in a separate transcripts.parquet keyed by `filename`. Search by alias
# instead of assuming.
AUDIO_ALIASES = ("audio", "wav", "speech", "audio_bytes", "file", "path", "audio_filepath")
TEXT_ALIASES = ("text", "sentence", "transcript", "transcription", "normalized_text",
                "sentence_norm", "target", "label")
ID_ALIASES = ("filename", "file_name", "id", "uuid", "segment_id", "path", "audio_filepath")
# Never train ASR on these: they are phoneme/IPA renderings, not Hebrew orthography.
FORBIDDEN_TEXT = ("phonemes", "whisper_phonemes", "original_phonemes", "ipa")


def find_column(names, aliases, forbidden=FORBIDDEN_TEXT) -> str | None:
    lowered = {n.lower(): n for n in names}
    for alias in aliases:
        if alias in lowered and alias not in forbidden:
            return lowered[alias]
    return None


def describe_repo(repo: str, token: str) -> None:
    """Print each parquet schema in a repo, flagging audio/text/forbidden columns."""
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("hf", token=token)
    seen: dict[tuple, list[str]] = {}
    for f in hub_files(repo, token, ".parquet"):
        try:
            names = tuple(pq.read_schema(f"hf://datasets/{repo}/{f}", filesystem=fs).names)
        except Exception as error:  # noqa: BLE001
            print(f"  !! {f}: {type(error).__name__}")
            continue
        seen.setdefault(names, []).append(f)
    for names, files in seen.items():
        audio = find_column(names, AUDIO_ALIASES)
        text = find_column(names, TEXT_ALIASES)
        banned = [n for n in names if n.lower() in FORBIDDEN_TEXT]
        print(f"  {len(files)} file(s) e.g. {files[0]}")
        print(f"    columns : {list(names)}")
        print(f"    audio   -> {audio}")
        print(f"    text    -> {text}")
        if banned:
            print(f"    IGNORED (phoneme/IPA, not Hebrew): {banned}")


def hub_files(repo: str, token: str, suffix: str) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi(token=token).list_repo_files(repo, repo_type="dataset")
                  if f.endswith(suffix))


def parquet_batches(repo: str, path: str, token: str, columns: list[str] | None, batch_size: int = 64):
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("hf", token=token)
    handle = pq.ParquetFile(f"hf://datasets/{repo}/{path}", filesystem=fs)
    for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pylist()


def download(repo: str, filename: str, token: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                                local_dir=str(dest), token=token))


# --------------------------------------------------------------------------- sources


def build_ivrit30s(out: Path, token: str, args) -> list[Row]:
    """notmax123/ivirits-audio-v2-30s -- FLAC + whisper transcript in parquet."""
    repo = "notmax123/ivirits-audio-v2-30s"
    shards = hub_files(repo, token, ".parquet")
    print(f"[ivrit30s] {len(shards)} shards, {args.shard_workers} in parallel")
    return build_parquet_source("ivrit30s", repo, shards, out, token, args)


def build_voxknesset(out: Path, token: str, args) -> list[Row]:
    """notmax123/voxknesset-hebrew-ipa transcripts joined to ivrit-ai/VoxKnesset audio.

    The manifest carries `transcript` (Hebrew) plus clip offsets into `source_filename`;
    the audio itself lives in the ivrit-ai parquet keyed by audio.path == source_filename.
    Explicitly NOT whisper_phonemes / ipa / reference_text.
    """
    man_repo, audio_repo = "notmax123/voxknesset-hebrew-ipa", "ivrit-ai/VoxKnesset"
    cache = out / "cache"
    manifest_path = download(man_repo, "manifest.jsonl", token, cache)

    wanted: dict[str, list[dict]] = {}
    seen_clips: set[str] = set()
    duplicates = 0
    kept = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(m.get("whisper_confidence", 0)) < args.min_confidence:
                continue
            if float(m.get("vad_speech_ratio", 0)) < args.min_vad_ratio:
                continue
            text = normalize_text(m.get("transcript", ""))
            if not acceptable(text):
                continue
            # The manifest repeats ~4% of clips (some up to 6x). They all resolve to one
            # wav on disk, but each row would become a separate training example, so the
            # repeated clips would be oversampled.
            if m["wav"] in seen_clips:
                duplicates += 1
                continue
            seen_clips.add(m["wav"])
            wanted.setdefault(m["source_filename"], []).append(
                {"text": text, "start": float(m["start_sec"]), "end": float(m["end_sec"]),
                 "wav": m["wav"], "speaker": str(m.get("speaker_id", ""))}
            )
            kept += 1
    print(f"[voxknesset] {kept:,} clips wanted across {len(wanted):,} source files "
          f"({duplicates:,} duplicate rows dropped)")

    audio_dir = out / "audio" / "voxknesset"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    skips = Skips("voxknesset")
    budget = args.max_hours_per_source * 3600 if args.max_hours_per_source else float("inf")
    total = 0.0
    shards = [f for f in hub_files(audio_repo, token, ".parquet")
              if Path(f).name != "transcripts.parquet"]
    print(f"[voxknesset] {len(shards)} audio shards, {args.shard_workers} in parallel")

    payloads = [(audio_repo, shard, token, args, audio_dir) for shard in shards]
    ctx = pool_context()
    done = 0
    # With fork the workers inherit `wanted` copy-on-write; the initializer just binds it.
    with ctx.Pool(processes=args.shard_workers, initializer=_vk_init, initargs=(wanted,)) as pool:
        for shard_rows, skipped, first_error, shard_total in pool.imap_unordered(_vk_shard, payloads):
            done += 1
            for reason, count in skipped.items():
                for _ in range(count):
                    skips.add(reason, first_error)
            for r in shard_rows:
                rows.append(Row(r["path"], r["dur"], r["text"], "voxknesset", r["speaker"]))
            total += shard_total
            print(f"[voxknesset] shard {done}/{len(shards)}: {len(rows):,} clips, "
                  f"{total/3600:.1f} h", flush=True)
            if total >= budget:
                print(f"[voxknesset] hit {args.max_hours_per_source} h cap", flush=True)
                pool.terminate()
                break
            remaining = free_gb(out)
            if remaining < args.min_free_gb:
                print(f"[voxknesset] stopping: {remaining:.0f} GB free < --min-free-gb "
                      f"{args.min_free_gb} ({total/3600:.0f} h staged)", flush=True)
                pool.terminate()
                break
    skips.report(len(rows))
    return rows


def build_crowd_transcribe(out: Path, token: str, args) -> list[Row]:
    """ivrit-ai/crowd-transcribe-v5 -- human-corrected `sentence`."""
    repo = "ivrit-ai/crowd-transcribe-v5"
    shards = [f for f in hub_files(repo, token, ".parquet") if "/test-" not in f]
    print(f"[crowd_transcribe] {len(shards)} train shards, {args.shard_workers} in parallel")
    return build_parquet_source("crowd_transcribe", repo, shards, out, token, args)


def _recital_session(payload: tuple) -> tuple:
    """One crowd-recital session: mka audio cut at its aligned segment timestamps."""
    repo, session, token, args, audio_dir = payload
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    total = 0.0
    try:
        from huggingface_hub import hf_hub_download

        tpath = hf_hub_download(repo, f"{session}/transcript.aligned.json",
                                repo_type="dataset", token=token)
        aligned = json.loads(Path(tpath).read_text(encoding="utf-8"))
        segments = aligned.get("segments") or []
        if not segments:
            return ([], {"no_segments": 1}, "", 0.0)
        apath = hf_hub_download(repo, f"{session}/audio.mka", repo_type="dataset", token=token)
        audio = decode_file(Path(apath))
    except Exception as error:  # noqa: BLE001
        return ([], {"fetch": 1}, f"{type(error).__name__}: {error}", 0.0)

    for index, seg in enumerate(segments, 1):
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            skipped["bad_timestamps"] = skipped.get("bad_timestamps", 0) + 1
            continue
        text = normalize_text(seg.get("text") or "")
        if not acceptable(text):
            skipped["text"] = skipped.get("text", 0) + 1
            continue
        piece = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        dur = len(piece) / SAMPLE_RATE
        if not (args.min_duration <= dur <= args.max_duration):
            skipped["duration"] = skipped.get("duration", 0) + 1
            continue
        path = audio_dir / f"{safe_name(session)}_{index:05d}.wav"
        try:
            write_clip(piece, path)
        except Exception as error:  # noqa: BLE001
            skipped["write"] = skipped.get("write", 0) + 1
            continue
        total += dur
        rows.append({"path": str(path.resolve()), "dur": dur, "text": text, "speaker": session})
    return (rows, skipped, "", total)


def build_crowd_recital(out: Path, token: str, args) -> list[Row]:
    """ivrit-ai/crowd-recital -- read-aloud sessions: audio.mka + transcript.aligned.json.

    Not parquet like the other ivrit.ai sets, which is why it needs its own builder.
    """
    repo = "ivrit-ai/crowd-recital"
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    sessions = sorted({f.split("/")[0] for f in files if f.endswith("/audio.mka")})
    print(f"[crowd_recital] {len(sessions)} sessions, {args.shard_workers} in parallel")

    audio_dir = out / "audio" / "crowd_recital"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    skips = Skips("crowd_recital")
    total = 0.0
    budget = args.max_hours_per_source * 3600 if args.max_hours_per_source else float("inf")

    payloads = [(repo, sid, token, args, audio_dir) for sid in sessions]
    ctx = pool_context()
    done = 0
    with ctx.Pool(processes=args.shard_workers) as pool:
        for sess_rows, skipped, first_error, sess_total in pool.imap_unordered(
            _recital_session, payloads
        ):
            done += 1
            for reason, count in skipped.items():
                for _ in range(count):
                    skips.add(reason, first_error)
            for r in sess_rows:
                rows.append(Row(r["path"], r["dur"], r["text"], "crowd_recital", r["speaker"]))
            total += sess_total
            if done % 100 == 0 or done == len(sessions):
                print(f"[crowd_recital] {done}/{len(sessions)} sessions: {len(rows):,} clips, "
                      f"{total/3600:.1f} h", flush=True)
            if total >= budget or free_gb(out) < args.min_free_gb:
                pool.terminate()
                break
    skips.report(len(rows))
    return rows


def build_podcast286(out: Path, token: str, args) -> list[Row]:
    """notmax123/audio_286h -- podcast_clean.zip: 16 kHz mono segments + segments.tsv.

    The wavs are already 16 kHz mono PCM_16, so they are extracted as-is instead of being
    decoded and re-encoded. Text comes from the `hebrew` column; the sibling `phonemes`
    column is IPA and must never become an ASR target.
    """
    import csv
    import zipfile

    repo, archive = "notmax123/audio_286h", "podcast_clean.zip"
    staging = out / "raw" / "podcast286"
    staging.mkdir(parents=True, exist_ok=True)
    local = staging / archive
    if not local.exists():
        local = download(repo, archive, token, staging)
    size_gb = local.stat().st_size / 1e9

    audio_dir = out / "audio" / "podcast286"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    skips = Skips("podcast286")
    total = 0.0
    budget = args.max_hours_per_source * 3600 if args.max_hours_per_source else float("inf")

    with zipfile.ZipFile(local) as zf:
        meta = {}
        with zf.open("podcast_clean/segments.tsv") as handle:
            text_stream = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            for row in csv.DictReader(text_stream, delimiter="\t"):
                meta[row["segment_id"]] = row
        print(f"[podcast286] {len(meta):,} rows in segments.tsv", flush=True)

        members = [n for n in zf.namelist()
                   if n.startswith("podcast_clean/segments/") and n.endswith(".wav")]
        for index, name in enumerate(members, 1):
            if total >= budget:
                break
            seg_id = Path(name).stem
            row = meta.get(seg_id)
            if row is None:
                skips.add("no_metadata")
                continue
            try:
                if float(row.get("token_confidence") or 0) < args.min_confidence:
                    skips.add("confidence")
                    continue
                if float(row.get("vad_confidence_speech_ratio") or 0) < args.min_vad_ratio:
                    skips.add("vad")
                    continue
                dur = float(row["duration_sec"])
            except (TypeError, ValueError):
                skips.add("bad_numbers")
                continue
            if not (args.min_duration <= dur <= args.max_duration):
                skips.add("duration")
                continue
            text = normalize_text(row.get("hebrew") or "")
            if not acceptable(text):
                skips.add("text")
                continue
            target = audio_dir / f"{safe_name(seg_id)}.wav"
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil_copy(src, dst)
            total += dur
            rows.append(Row(str(target.resolve()), dur, text, "podcast286", row.get("youtube_id", "")))
            if index % 20000 == 0:
                print(f"[podcast286] {index:,}/{len(members):,}: {len(rows):,} clips, "
                      f"{total/3600:.1f} h", flush=True)

    local.unlink(missing_ok=True)
    print(f"[podcast286] removed {archive} (reclaimed ~{size_gb:.1f} GB)", flush=True)
    skips.report(len(rows))
    return rows


def build_synthetic(out: Path, token: str, args) -> list[Row]:
    """notmax123/SententicDataTTS -- Hebrew TTS audio, text via hebrew_text.parquet.

    The repo's own CSVs carry IPA only. hebrew_text.parquet (published alongside them)
    joins each clip id to its source line in Phonikud/phonikud-data knesset_nikud_v6.txt,
    which is what makes this corpus usable as an ASR target at all. `text` is the
    unvocalized Hebrew; `phonemes`/`text_nikud` are deliberately not used here.
    """
    import pyarrow.parquet as pq

    repo = "notmax123/SententicDataTTS"
    meta_path = download(repo, "hebrew_text.parquet", token, out / "cache")
    table = pq.read_table(meta_path, columns=["filename", "text"])
    lookup = {Path(f).stem: t for f, t in zip(table.column("filename").to_pylist(),
                                              table.column("text").to_pylist()) if t}
    print(f"[synthetic] {len(lookup):,} clips have Hebrew text", flush=True)

    root = extract_7z(repo, "chatterbox_44K.7z", token, out / "raw" / "synthetic")
    return build_from_audio_dir("synthetic", root, out, args, lookup)


def build_lhs_synthetic(out: Path, token: str, args) -> list[Row]:
    """notmax123/large-he-synthetic-tts-dataset -- 580k TTS clips.

    The audio ships as a single 131 GB tar.zst. It is streamed straight from the Hub
    through zstd|tar so the archive itself never lands on disk; only the wavs do.
    Text comes from hebrew_ipa_text.parquet (`text`, unvocalized) -- never `phonemes`.
    """
    import pyarrow.parquet as pq

    repo = "notmax123/large-he-synthetic-tts-dataset"
    archive = "large-he-synthetic-tts-dataset.tar.zst"
    meta_path = download(repo, "hebrew_ipa_text.parquet", token, out / "cache")
    table = pq.read_table(meta_path, columns=["id", "text"])
    lookup = {i: t for i, t in zip(table.column("id").to_pylist(),
                                   table.column("text").to_pylist()) if t}
    print(f"[lhs_synthetic] {len(lookup):,} clips have Hebrew text", flush=True)

    root = out / "raw" / "lhs_synthetic"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".extracted"
    if not marker.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{archive}"
        print(f"[lhs_synthetic] streaming {archive} (never stored whole)", flush=True)
        # curl | zstd -dc | tar -x keeps peak disk at the extracted wavs only.
        command = (
            f'curl -sL -H "Authorization: Bearer {token}" {url} '
            f'| zstd -dc | tar -x -C {root}'
        )
        result = subprocess.run(["bash", "-lc", command], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"stream extract failed rc={result.returncode}")
        marker.write_text("ok")

    return build_from_audio_dir("lhs_synthetic", root, out, args, lookup)


def build_eval_whatsapp(out: Path, token: str, args) -> list[Row]:
    """ivrit-ai/eval-whatsapp -- the only held-out benchmark for this run."""
    repo = "ivrit-ai/eval-whatsapp"
    rows: list[Row] = []
    audio_dir = out / "audio" / "eval_whatsapp"
    for shard in hub_files(repo, token, ".parquet"):
        for batch in parquet_batches(repo, shard, token, ["uuid", "audio", "text"], batch_size=8):
            for r in batch:
                text = normalize_text(r.get("text") or "")
                if not text:
                    continue
                try:
                    audio = decode_bytes(r["audio"]["bytes"])
                except Exception:  # noqa: BLE001
                    continue
                path = audio_dir / f"{safe_name(str(r['uuid']))}.wav"
                dur = write_clip(audio, path)
                rows.append(Row(str(path.resolve()), dur, text, "eval_whatsapp"))
    print(f"[eval_whatsapp] {len(rows):,} clips, {sum(r.duration for r in rows)/3600:.2f} h")
    return rows


def extract_7z(repo: str, archive: str, token: str, dest: Path) -> Path:
    """Fetch and unpack a .7z dataset archive (py7zr, falling back to the 7z binary)."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f".{archive}.done"
    if marker.exists():
        return dest
    staging = dest / "_dl"
    local = download(repo, archive, token, staging)
    size_gb = local.stat().st_size / 1e9
    print(f"[7z] extracting {archive} ({size_gb:.1f} GB)", flush=True)
    try:
        import py7zr

        with py7zr.SevenZipFile(local, mode="r") as zf:
            zf.extractall(path=dest)
    except ImportError:
        subprocess.run(["7z", "x", str(local), f"-o{dest}", "-y"], check=True)
    marker.write_text("ok")

    # Reclaim the archive immediately: these run to hundreds of GB and the extracted
    # audio is all we need. hf_hub_download also leaves a blob under _dl/.cache, which
    # is a second full copy.
    import shutil

    local.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[7z] removed {archive} + download cache (reclaimed ~{size_gb:.1f} GB)", flush=True)
    return dest


def build_from_audio_dir(name: str, root: Path, out: Path, args,
                         text_lookup: dict[str, str]) -> list[Row]:
    """Common tail for the archive datasets: walk audio, attach text by stem."""
    rows: list[Row] = []
    audio_dir = out / "audio" / name
    budget = args.max_hours_per_source * 3600 if args.max_hours_per_source else float("inf")
    total = 0.0
    exts = {".wav", ".mp3", ".flac", ".m4a", ".opus", ".ogg"}
    skips = Skips(name)
    jobs = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in exts):
        text = text_lookup.get(path.stem) or text_lookup.get(path.name)
        if text is None:
            skips.add("no_text")
            continue
        text = normalize_text(text)
        if not acceptable(text):
            skips.add("text")
            continue
        jobs.append((path, audio_dir / f"{safe_name(path.stem)}.wav", text))
    print(f"[{name}] {len(jobs):,} files to convert with {args.workers} threads", flush=True)

    def convert(job):
        src, dst, txt = job
        try:
            audio = decode_file(src)
        except Exception as error:  # noqa: BLE001
            return ("decode", f"{type(error).__name__}: {error}", None, None)
        dur = len(audio) / SAMPLE_RATE
        if not (args.min_duration <= dur <= args.max_duration):
            return ("duration", "", None, None)
        write_clip(audio, dst)
        return ("ok", str(dst.resolve()), dur, txt)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (status, a, dur, txt) in enumerate(pool.map(convert, jobs), 1):
            if status != "ok":
                skips.add(status, a)
                continue
            total += dur
            rows.append(Row(a, dur, txt, name))
            if index % 20000 == 0:
                print(f"[{name}] {index:,}/{len(jobs):,}: {len(rows):,} clips, "
                      f"{total/3600:.1f} h", flush=True)
            if total >= budget:
                break
    skips.report(len(rows))
    print(f"[{name}] {len(rows):,} clips, {total/3600:.1f} h")
    return rows


def load_text_table(paths: list[Path], id_col: str, text_col: str, delimiter: str) -> dict[str, str]:
    import csv

    lookup: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                key = (row.get(id_col) or "").strip()
                text = (row.get(text_col) or "").strip()
                if key and text:
                    lookup[Path(key).stem] = text
    return lookup


# --------------------------------------------------------------------------- driver


BUILDERS = {
    "ivrit30s": build_ivrit30s,
    "voxknesset": build_voxknesset,
    "crowd_transcribe": build_crowd_transcribe,
    "crowd_recital": build_crowd_recital,
    "podcast286": build_podcast286,
    "synthetic": build_synthetic,
    "lhs_synthetic": build_lhs_synthetic,
    "eval_whatsapp": build_eval_whatsapp,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append",
                    help="ivrit30s | voxknesset | crowd_transcribe | saspeech | ranlevi | "
                         "synthetic | eval_whatsapp | all")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--dev-hours", type=float, default=2.0,
                    help="Held-out dev sampled from training sources (kept small on purpose)")
    ap.add_argument("--max-hours-per-source", type=float, default=None)
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=40.0)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--min-vad-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 8)),
                    help="Decode/write threads inside one shard")
    ap.add_argument("--shard-workers", type=int, default=min(12, max(1, (os.cpu_count() or 8) // 8)),
                    help="Shards processed in parallel (each its own HF download stream)")
    ap.add_argument("--inspect", action="append",
                    help="Print the parquet schemas of a repo (audio/text column search) and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="Ignore per-source manifests from a previous run and redo them")
    ap.add_argument("--min-free-gb", type=float, default=60.0,
                    help="Stop staging a source when the volume drops below this, so "
                         "training still has room for checkpoints")
    return ap.parse_args()


def write_source_manifest(rows: list[Row], out: Path, name: str) -> None:
    """Persist a source the moment it completes.

    Manifests used to be written only after every source finished, so a crash in the
    last source discarded hours of completed work. These per-source files are merged at
    the end and let a rerun skip what is already done.
    """
    path = out / "manifests" / "sources" / f"{name}.json"
    write_manifest(rows, path)


def load_source_manifest(out: Path, name: str) -> list[Row] | None:
    path = out / "manifests" / "sources" / f"{name}.json"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(Row(d["audio_filepath"], d["duration"], d["text"], name))
    return rows


def write_manifest(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_nemo(), ensure_ascii=False) + "\n")
    hours = sum(r.duration for r in rows) / 3600
    print(f"wrote {path} — {len(rows):,} clips, {hours:.1f} h")


def require_ffmpeg() -> None:
    from shutil import which

    if which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Most source audio is mp3/m4a and cannot be decoded "
                 "without it (apt-get install -y ffmpeg).")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN")
    if args.inspect:
        for repo in args.inspect:
            print(f"=== {repo} ===")
            describe_repo(repo, token)
        return
    require_ffmpeg()
    sources = args.source or ["all"]
    if "all" in sources:
        sources = ["ivrit30s", "voxknesset", "crowd_transcribe", "crowd_recital",
                   "podcast286", "saspeech", "ranlevi", "synthetic", "eval_whatsapp"]

    out = args.out
    train_rows: list[Row] = []
    eval_rows: list[Row] = []

    for name in sources:
        if name == "eval_whatsapp":
            eval_rows = build_eval_whatsapp(out, token, args)
            continue
        if name in BUILDERS:
            cached = None if args.rebuild else load_source_manifest(out, name)
            if cached is not None:
                print(f"[{name}] reusing {len(cached):,} clips from a previous run "
                      f"({sum(r.duration for r in cached)/3600:.1f} h)")
                train_rows.extend(cached)
                continue
            built = BUILDERS[name](out, token, args)
            write_source_manifest(built, out, name)
            train_rows.extend(built)
        elif name == "saspeech":
            root = extract_7z("notmax123/SASPEECH_AUTO_clean", "saspeech_auto.7z", token,
                              out / "raw" / "saspeech")
            lookup = load_text_table(sorted(root.rglob("*.csv")), "id", "text", ",")
            train_rows.extend(build_from_audio_dir("saspeech", root, out, args, lookup))
        elif name == "ranlevi":
            root = extract_7z("notmax123/RanLevi40h", "ran_levi.7z", token, out / "raw" / "ranlevi")
            lookup = load_text_table(sorted(root.rglob("*.csv")), "id", "text", ",")
            train_rows.extend(build_from_audio_dir("ranlevi", root, out, args, lookup))
        else:
            sys.exit(f"unknown source: {name}")

    if train_rows:
        random.Random(args.seed).shuffle(train_rows)
        dev_budget = args.dev_hours * 3600
        dev: list[Row] = []
        by_source: dict[str, list[Row]] = {}
        for row in train_rows:
            by_source.setdefault(row.source, []).append(row)
        # Stratified: every training source is represented in dev, proportional to size.
        per_source = {k: dev_budget * len(v) / len(train_rows) for k, v in by_source.items()}
        taken: set[int] = set()
        for src, rows_ in by_source.items():
            budget = per_source[src]
            acc = 0.0
            for row in rows_:
                if acc >= budget:
                    break
                dev.append(row)
                taken.add(id(row))
                acc += row.duration
        train = [r for r in train_rows if id(r) not in taken]
        write_manifest(train, out / "manifests" / "train.json")
        write_manifest(dev, out / "manifests" / "dev.json")
        print("\nper-source hours:")
        for src, rows_ in sorted(by_source.items()):
            print(f"  {src:18s} {sum(r.duration for r in rows_)/3600:8.1f} h  {len(rows_):>9,} clips")
    if eval_rows:
        write_manifest(eval_rows, out / "manifests" / "eval" / "eval_whatsapp.json")


if __name__ == "__main__":
    main()
