"""Shared helpers for Nemotron 3.5 ASR fine-tuning scripts."""
from __future__ import annotations

import glob
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import librosa
import soundfile as sf
import yaml

HEBREW_LETTER_RE = re.compile(r"[\u05D0-\u05EA]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
# IPA / phoneme transcripts (e.g. VoxKnesset whisper_phonemes) — must never be training targets
IPA_LETTER_RE = re.compile(r"[\u0250-\u02AF\u02B0-\u02FF\u0290-\u02FF]")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def nemo_root() -> Path:
    root = os.environ.get("NEMO_ROOT")
    if root and Path(root).is_dir():
        return Path(root)
    candidate = repo_root() / "NeMo"
    if candidate.is_dir():
        return candidate
    raise RuntimeError(
        "NeMo not found. Run ./setup.sh or set NEMO_ROOT to your NeMo clone."
    )


def ensure_cuda_home() -> None:
    """Point numba at libnvvm from the nvidia-cuda-nvcc wheel (RNNT loss/decode)."""
    for d in sys.path:
        hits = glob.glob(os.path.join(d, "nvidia", "cuda_nvcc", "nvvm", "lib64", "libnvvm.so*"))
        if not hits:
            continue
        home = os.path.join(d, "nvidia", "cuda_nvcc")
        libdir = os.path.join(home, "nvvm", "lib64")
        os.environ["CUDA_HOME"] = home
        os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        unversioned = os.path.join(libdir, "libnvvm.so")
        if not os.path.exists(unversioned):
            try:
                os.symlink(hits[0], unversioned)
            except OSError:
                pass
        return


def run(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True)


_WER_METRIC_RE = re.compile(
    r"(?P<name>val_wer|training_batch_wer|train_wer|test_wer)"
    r"(?:['\"]|\\)?\s*[:=]\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_STEP_RE = re.compile(r"(\d+)/\d+\s+\[")


def _append_wer_log(wer_log: Path, record: dict[str, Any]) -> None:
    wer_log.parent.mkdir(parents=True, exist_ok=True)
    with wer_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emit_wer(name: str, value: float, step: int | None, wer_log: Path | None) -> None:
    pct = value * 100.0 if value <= 1.0 else value
    step_s = f"step {step}" if step is not None else "step ?"
    banner = f"\n>>> WER [{name}] {step_s}: {pct:.2f}% (raw={value:.4f})\n"
    print(banner, flush=True)
    if wer_log is not None:
        _append_wer_log(
            wer_log,
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "metric": name,
                "value": value,
                "wer_pct": round(pct, 4),
                "step": step,
            },
        )


_EPOCH_PROGRESS_RE = re.compile(r"^Epoch \d+:\s+\d+%\|")

# Defer multi-GB best.* copies until DDP ranks finish validation/checkpoint I/O.
_DEFAULT_CKPT_SYNC_DELAY_SEC = int(os.environ.get("NEMO_CKPT_SYNC_DELAY_SEC", "180"))


def _iter_descendant_pids(root_pid: int) -> list[int]:
    by_ppid: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        by_ppid.setdefault(ppid, []).append(pid)

    out: list[int] = []
    stack = list(by_ppid.get(root_pid, []))
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(by_ppid.get(pid, []))
    return out


def _kill_process_tree(root_pid: int, *, sig: signal.Signals = signal.SIGTERM) -> None:
    """Kill root_pid and every descendant (covers DDP workers in other sessions)."""
    for pid in reversed(_iter_descendant_pids(root_pid) + [root_pid]):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _kill_stray_nemo_workers() -> None:
    """Last resort: DDP children reparented to init after the shell exits."""
    markers = (b"nemo_finetune_entry.py", b"speech_to_text_finetune")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid <= 1:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if any(m in cmdline for m in markers):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _terminate_process_tree(proc: subprocess.Popen[bytes] | subprocess.Popen[str], *, grace_sec: float = 3) -> None:
    """Stop shell + all DDP training ranks."""
    if proc.poll() is not None:
        return
    _kill_process_tree(proc.pid, sig=signal.SIGTERM)
    if grace_sec <= 0:
        _kill_process_tree(proc.pid, sig=signal.SIGKILL)
        _kill_stray_nemo_workers()
    else:
        try:
            proc.wait(timeout=grace_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        _kill_process_tree(proc.pid, sig=signal.SIGKILL)
        _kill_stray_nemo_workers()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_training_proc(
    proc: subprocess.Popen[bytes] | subprocess.Popen[str],
    stop_count_fn: Any,
) -> int:
    while proc.poll() is None:
        stop_count = stop_count_fn()
        if stop_count > 0:
            sig = signal.SIGKILL if stop_count >= 2 else signal.SIGTERM
            _kill_process_tree(proc.pid, sig=sig)
            if stop_count >= 2:
                _kill_stray_nemo_workers()
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            continue
    return proc.returncode if proc.returncode is not None else 1


def run_training(
    cmd: str,
    *,
    wer_log: Path | None = None,
    exp_dir: Path | None = None,
    checkpoint_top_k: int = 0,
    checkpoint_sync_delay_sec: int | None = None,
    check: bool = True,
    stream_output: bool = True,
) -> int:
    """Run NeMo training and highlight WER metrics as they appear.

    When stream_output is True (default), NeMo writes directly to the terminal so
    Lightning/tqdm can refresh a single progress line. Piping stdout breaks TTY
    detection and prints one line per step in tmux.
    """
    print("+", cmd, flush=True)
    last_step: int | None = None
    seen: set[tuple[str, int | None, float]] = set()
    stop_poll = threading.Event()
    sync_lock = threading.Lock()
    sync_timer: threading.Timer | None = None
    proc: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None
    stop_count = 0
    sync_delay = (
        checkpoint_sync_delay_sec
        if checkpoint_sync_delay_sec is not None
        else _DEFAULT_CKPT_SYNC_DELAY_SEC
    )

    def sync_checkpoints_now(*, reason: str) -> None:
        if checkpoint_top_k <= 0 or exp_dir is None:
            return
        with sync_lock:
            try:
                sync_top_k_checkpoints(exp_dir.parent, exp_dir.name, k=checkpoint_top_k)
            except FileNotFoundError:
                pass
            except (OSError, TypeError, RuntimeError) as exc:
                print(f"Checkpoint sync skipped ({reason}): {exc}", flush=True)

    def schedule_checkpoint_sync() -> None:
        nonlocal sync_timer
        if checkpoint_top_k <= 0 or exp_dir is None:
            return
        delay = sync_delay

        def _run() -> None:
            sync_checkpoints_now(reason=f"{delay}s after val_wer")

        with sync_lock:
            if sync_timer is not None:
                sync_timer.cancel()
            sync_timer = threading.Timer(delay, _run)
            sync_timer.daemon = True
            sync_timer.start()
        print(
            f"Checkpoint sync scheduled in {delay}s "
            "(avoids disk I/O during DDP validation/checkpoint writes)",
            flush=True,
        )

    def on_wer(name: str, value: float, step: int | None) -> None:
        _emit_wer(name, value, step, wer_log)
        if name == "val_wer":
            schedule_checkpoint_sync()

    def _stop_training(_signum: int, _frame: object | None) -> None:
        nonlocal stop_count
        stop_count += 1
        if stop_count == 1:
            print("\nStopping training and all DDP ranks (Ctrl-C again to force)...", flush=True)
        elif stop_count >= 2:
            print("\nForce killing...", flush=True)
            raise KeyboardInterrupt

    def poll_tb() -> None:
        if exp_dir is None:
            return
        while not stop_poll.is_set():
            try:
                for row in latest_tensorboard_wer(exp_dir):
                    key = (row["metric"], row.get("step"), round(float(row["value"]), 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    if row["metric"] != "val_wer":
                        continue
                    on_wer(row["metric"], float(row["value"]), row.get("step"))
            except OSError as exc:
                print(f"TensorBoard poll skipped: {exc}", flush=True)
            stop_poll.wait(20)

    tb_thread = threading.Thread(target=poll_tb, daemon=True)
    tb_thread.start()

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _stop_training)
    signal.signal(signal.SIGTERM, _stop_training)
    rc = 1
    try:
        popen_kw: dict[str, Any] = {"shell": True, "start_new_session": True}
        if stream_output:
            proc = subprocess.Popen(cmd, **popen_kw)
            rc = _wait_for_training_proc(proc, lambda: stop_count)
        else:
            proc = subprocess.Popen(
                cmd,
                **popen_kw,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    if _EPOCH_PROGRESS_RE.match(line):
                        continue
                    print(line, end="", flush=True)
                    step_match = _STEP_RE.search(line)
                    if step_match:
                        last_step = int(step_match.group(1))
                    for match in _WER_METRIC_RE.finditer(line):
                        name = match.group("name")
                        value = float(match.group("value"))
                        key = (name, last_step, round(value, 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        on_wer(name, value, last_step)
            finally:
                rc = _wait_for_training_proc(proc, lambda: stop_count)
    except KeyboardInterrupt:
        rc = 130
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc, grace_sec=0)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc)
        stop_poll.set()
        tb_thread.join(timeout=1)
        with sync_lock:
            if sync_timer is not None:
                sync_timer.cancel()
                sync_timer = None

    if exp_dir is not None:
        for row in latest_tensorboard_wer(exp_dir):
            key = (row["metric"], row.get("step"), round(float(row["value"]), 6))
            if key in seen:
                continue
            seen.add(key)
            if row["metric"] != "val_wer":
                continue
            on_wer(row["metric"], float(row["value"]), row.get("step"))
    sync_checkpoints_now(reason="training finished")
    if check and rc not in (0, -signal.SIGTERM, -signal.SIGINT, 130):
        raise subprocess.CalledProcessError(rc, cmd)
    return rc if rc is not None else 1


def read_wer_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def latest_tensorboard_wer(exp_dir: Path) -> list[dict[str, Any]]:
    """Read val_wer from the newest tfevents file under exp_dir."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return []

    run_dirs = [
        p
        for p in exp_dir.glob("*")
        if p.is_dir() and _RUN_DIR_RE.match(p.name)
    ]
    run_dirs.sort(key=_safe_mtime, reverse=True)
    for run_dir in run_dirs:
        ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        out: list[dict[str, Any]] = []
        for tag in ("val_wer",):
            if tag not in tags:
                continue
            for ev in ea.Scalars(tag):
                pct = ev.value * 100.0 if ev.value <= 1.0 else ev.value
                out.append(
                    {
                        "metric": tag,
                        "step": int(ev.step),
                        "value": ev.value,
                        "wer_pct": round(pct, 4),
                        "source": str(run_dir.name),
                    }
                )
        if out:
            return out
    return []


def read_nemo_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_hours(rows: list[dict]) -> float:
    return sum(float(r.get("duration", 0)) for r in rows) / 3600


def filter_manifest_by_duration(rows: list[dict], max_duration: float) -> tuple[list[dict], int]:
    """Drop clips longer than max_duration seconds (hard filter for RNNT memory)."""
    kept = [r for r in rows if float(r.get("duration", 0)) <= max_duration]
    return kept, len(rows) - len(kept)


def compute_train_steps_per_epoch(
    train_rows: list[dict],
    batch_duration: int,
    grad_accum: int,
    devices: int,
    num_nodes: int = 1,
) -> int:
    """Optimizer steps for one full pass over the train manifest (Lightning progress bar total)."""
    import math

    seconds = sum(float(r.get("duration", 0)) for r in train_rows)
    global_secs_per_step = batch_duration * grad_accum * devices * num_nodes
    return max(1, math.ceil(seconds / global_secs_per_step))


def decode_and_save_wav(
    audio_bytes: bytes | None,
    audio_path: str | None,
    dst: Path,
    sample_rate: int,
) -> Optional[float]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes is not None:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    elif audio_path:
        arr, sr = sf.read(audio_path, dtype="float32")
    else:
        return None
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != sample_rate:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=sample_rate)
    sf.write(dst, arr, sample_rate, subtype="PCM_16")
    return len(arr) / sample_rate


def transcode_mp3_to_wav(src: Path, dst: Path, sample_rate: int) -> Optional[float]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dst),
    ]
    if subprocess.run(cmd).returncode != 0 or not dst.exists():
        return None
    info = sf.info(str(dst))
    return info.frames / info.samplerate


def speaker_split(speaker_id: str, test_frac: float, dev_frac: float) -> str:
    """Deterministic speaker-disjoint split (stable across re-runs)."""
    import hashlib

    bucket = int(hashlib.md5(str(speaker_id).encode()).hexdigest(), 16) % 10_000
    test_cut = int(test_frac * 10_000)
    dev_cut = int((test_frac + dev_frac) * 10_000)
    if bucket < test_cut:
        return "test"
    if bucket < dev_cut:
        return "validation"
    return "train"


def require_columns(fieldnames: list[str] | None, required: list[str], path: Path) -> None:
    """Fail fast when a CSV/TSV is missing expected header columns."""
    cols = list(fieldnames or [])
    missing = [name for name in required if name not in cols]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {missing}\n"
            f"  expected: {required}\n"
            f"  found:    {cols}"
        )


def row_field(row: dict, column: str, *, optional: bool = False) -> str | None:
    """Read one CSV/TSV cell using the configured column name."""
    if column not in row:
        if optional:
            return None
        raise KeyError(f"column {column!r} missing from row keys {list(row)}")
    val = row.get(column)
    if val is None or str(val).strip() == "":
        return None if optional else ""
    return str(val).strip()


def json_field(rec: dict, field: str) -> str | None:
    """Read one JSONL field using the configured key; blank/missing -> None."""
    if field not in rec:
        return None
    val = rec.get(field)
    if val is None or str(val).strip() == "":
        return None
    return str(val).strip()


def normalize_text(text: str) -> str:
    """Preserve casing/punctuation (Nemotron outputs punctuated, cased text)."""
    import unicodedata

    s = unicodedata.normalize("NFC", str(text)).strip()
    for bad, good in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), (" ", " ")):
        s = s.replace(bad, good)
    return " ".join(s.split())


def is_valid_text(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def is_hebrew_asr_text(text: str) -> bool:
    """True when the transcript is Hebrew (minor Latin allowed, e.g. acronyms)."""
    if is_ipa_transcript(text):
        return False
    hebrew_letters = len(HEBREW_LETTER_RE.findall(text))
    if hebrew_letters == 0:
        return False
    latin_letters = len(LATIN_LETTER_RE.findall(text))
    return latin_letters <= hebrew_letters


def is_ipa_transcript(text: str) -> bool:
    """True when text looks like IPA/phonemes rather than Hebrew orthography."""
    if not text:
        return False
    ipa_chars = len(IPA_LETTER_RE.findall(text))
    hebrew_letters = len(HEBREW_LETTER_RE.findall(text))
    if ipa_chars >= 3 and ipa_chars > hebrew_letters:
        return True
    # Common IPA-only clips with no Hebrew letters
    if hebrew_letters == 0 and ipa_chars > 0:
        return True
    return False


def hebrew_only_enabled(cfg: dict) -> bool:
    return bool(cfg.get("project", {}).get("hebrew_only", True))


def accepts_asr_text(text: str, cfg: dict | None = None) -> bool:
    if not is_valid_text(text):
        return False
    if cfg is not None and hebrew_only_enabled(cfg) and not is_hebrew_asr_text(text):
        return False
    return True


def clip_dedup_key(row: dict) -> str:
    """Stable key for cross-source deduplication during manifest merges."""
    clip_id = row.get("clip_id")
    if clip_id:
        return f"id:{clip_id}"
    source_type = row.get("source_type")
    source_entry_id = row.get("source_entry_id")
    if source_type is not None and source_entry_id is not None:
        start = row.get("segment_start", row.get("start"))
        end = row.get("segment_end", row.get("end"))
        if start is not None and end is not None:
            return f"seg:{source_type}:{source_entry_id}:{float(start):.2f}:{float(end):.2f}"
    return f"path:{Path(row['audio_filepath']).resolve()}"


def build_dedup_index(rows: list[dict]) -> set[str]:
    return {clip_dedup_key(r) for r in rows}


def dedupe_rows(rows: list[dict], seen: set[str]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    skipped = 0
    for row in rows:
        key = clip_dedup_key(row)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, skipped


def filter_hebrew_rows(rows: list[dict], cfg: dict) -> tuple[list[dict], int]:
    """Drop rows with wrong lang tag or non-Hebrew transcript when hebrew_only is enabled."""
    if not hebrew_only_enabled(cfg):
        return rows, 0
    target = cfg["project"]["target_lang"]
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        lang = row.get("target_lang") or row.get("lang")
        text = row.get("text", "")
        if lang != target or not accepts_asr_text(text, cfg) or is_ipa_transcript(text):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def validate_hebrew_manifest(rows: list[dict], target_lang: str, path: Path) -> None:
    """Fail fast if manifests contain non-Hebrew language tags or transcripts."""
    bad_lang = 0
    bad_text = 0
    samples: list[str] = []
    for row in rows:
        lang = row.get("target_lang") or row.get("lang")
        if lang != target_lang:
            bad_lang += 1
        elif not is_hebrew_asr_text(row.get("text", "")) or is_ipa_transcript(row.get("text", "")):
            bad_text += 1
            if len(samples) < 3:
                samples.append(str(row.get("text", ""))[:100])
    if bad_lang or bad_text:
        msg = [f"{path}: Hebrew-only check failed ({bad_lang} wrong lang, {bad_text} non-Hebrew text)."]
        msg.extend(f"  example: {s!r}" for s in samples)
        msg.append("Rebuild: uv run scripts/build_dataset.py --source mixed")
        sys.exit("\n".join(msg))


def unvocalize_hebrew(text: str) -> str:
    """Strip niqqud and PhoniKud markers from vocalized Hebrew (synthetic TTS corpus)."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFD", str(text))
    s = re.sub(r"[\u0590-\u05cf|]", "", s)
    return unicodedata.normalize("NFC", s)


def prepare_hebrew_asr_text(text: str, mode: str = "unvocalized") -> str:
    """Normalize synthetic or real Hebrew transcripts for Nemotron ASR training."""
    if mode == "unvocalized":
        text = unvocalize_hebrew(text)
    elif mode == "vocalized_strip_markers":
        text = text.replace("|", "").replace("\u05af", "").replace("\u05bd", "").replace("\u05ab", "")
    return normalize_text(text)


def synthetic_speaker_id(clip_id: str) -> str:
    """e.g. female1_000002 -> female1, yoav_028326 -> yoav."""
    if "_" not in clip_id:
        return clip_id
    return clip_id.rsplit("_", 1)[0]


# AE slow_44K layout: {speaker}_slow/utt_{index}.wav (44.1 kHz source for HF synthetic corpus)
_SYNTHETIC_SPEAKER_DIRS: dict[str, str] = {
    "female1": "female1_slow",
    "female2": "female2_slow",
    "female3": "female3_slow",
    "female4": "female4_slow",
    "female5": "female5_slow",
    "male1": "male1_slow",
    "male2": "male2_slow",
    "male3": "male3_slow",
    "male4": "male4_slow",
    "male5": "male5_slow",
    "yoav": "yoav_slow_gen",
}
_SYNTHETIC_UTT_OFFSETS: dict[str, int] = {
    "female1": 0,
    "female2": 1_000_000,
    "female3": 1_090_000,
    "female4": 1_090_000,
    "female5": 1_180_000,
    "male1": 0,
    "male2": 200_000,
    "male3": 1_270_000,
    "male4": 1_270_000,
    "male5": 0,
    "yoav": 601_000,
}


def synthetic_cfg_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["synthetic"]["local_dir"])
    if not root.is_absolute():
        root = repo_root() / root
    return root


def synthetic_audio_root(cfg: dict) -> Path:
    syn = cfg["data"]["synthetic"]
    root = syn.get("audio_root") or syn["local_dir"]
    return Path(root)


def resolve_synthetic_source_wav(clip_id: str, cfg: dict) -> Path | None:
    """Map HF clip id (female1_000001) to AE slow_44K wav path, if present."""
    if "_" not in clip_id:
        return None
    speaker, num_s = clip_id.rsplit("_", 1)
    try:
        num = int(num_s)
    except ValueError:
        return None
    speaker_dirs = cfg["data"]["synthetic"].get("speaker_dirs") or _SYNTHETIC_SPEAKER_DIRS
    utt_offsets = cfg["data"]["synthetic"].get("utt_offsets") or _SYNTHETIC_UTT_OFFSETS
    subdir = speaker_dirs.get(speaker)
    if subdir is None:
        return None
    idx = num + int(utt_offsets.get(speaker, 0))
    base = synthetic_audio_root(cfg) / subdir
    for name in (f"utt_{idx}.wav", f"utt_{idx:06d}.wav", f"utt_{idx:07d}.wav"):
        path = base / name
        if path.exists():
            return path
    return None


def resolve_synthetic_audio_dir(cfg: dict) -> tuple[Path, Path]:
    """Return (metadata_root, audio_dir). Prefer 16 kHz cache; else flat 24 kHz under local_dir."""
    syn = cfg["data"]["synthetic"]
    root = synthetic_cfg_root(cfg)
    resampled = root / syn["resampled_subdir"]
    raw = root / syn.get("wav_subdir", "wav")
    if resampled.is_dir() and any(resampled.glob("*.wav")):
        return root, resampled
    if raw.is_dir() and any(raw.glob("*.wav")):
        print("WARNING: using 24 kHz wav/ — run: uv run scripts/prepare_synthetic.py resample")
        return root, raw
    raise FileNotFoundError(
        f"Synthetic 16 kHz cache missing under {resampled}.\n"
        "Run: uv run scripts/prepare_synthetic.py resample"
    )


def write_nemo_manifest(rows: list[dict], path: Path, target_lang: str, *, force_langid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            entry = {
                "audio_filepath": row["audio_filepath"],
                "duration": row["duration"],
                "text": row["text"],
                "lang": target_lang,
                "target_lang": target_lang,
            }
            if force_langid:
                entry["prompt_mode"] = "langID"
            elif "prompt_mode" in row:
                entry["prompt_mode"] = row["prompt_mode"]
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"Wrote {path}: {len(rows)} clips, {hours:.2f} h")


def find_latest_nemo(exp_dir: Path) -> Path:
    cands = sorted(exp_dir.glob("**/*.nemo"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .nemo checkpoint under {exp_dir}")
    return cands[-1]


_VAL_WER_CKPT_RE = re.compile(r"val_wer=([0-9.]+)")
CHECKPOINT_TOP_K_SLOTS = ("best", "best_top2", "best_top3")
_CANONICAL_CKPT_KEEP = frozenset(
    {"wer.jsonl"}
    | {f"{slot}{ext}" for slot in CHECKPOINT_TOP_K_SLOTS for ext in (".ckpt", ".nemo", ".json")}
    | {f"best_val{ext}" for ext in (".ckpt", ".nemo", ".json")}
)


def _checkpoint_slot_name(rank: int) -> str:
    if rank == 1:
        return "best"
    return f"best_top{rank}"


def _scan_all_val_wer_ckpts(root: Path) -> list[tuple[Path, float]]:
    """Unique checkpoints under root, sorted by val_wer ascending."""
    by_ckpt: dict[str, tuple[Path, float]] = {}
    for path in root.glob("**/*"):
        if not path.is_file() or "-last." in path.name:
            continue
        match = _VAL_WER_CKPT_RE.search(path.name)
        if match is None or path.suffix not in {".ckpt", ".nemo"}:
            continue
        ckpt = _resolve_ckpt_path(path)
        if ckpt.suffix != ".ckpt" or not ckpt.is_file():
            continue
        key = str(ckpt.resolve())
        wer = float(match.group(1))
        prev = by_ckpt.get(key)
        if prev is None or wer < prev[1]:
            by_ckpt[key] = (ckpt, wer)
    return sorted(by_ckpt.values(), key=lambda item: item[1])


def _scan_top_k_val_wer_ckpts(root: Path, k: int) -> list[tuple[Path, float]]:
    return _scan_all_val_wer_ckpts(root)[: max(1, k)]


def _scan_best_val_wer_ckpt(root: Path) -> tuple[Path, float] | None:
    ranked = _scan_all_val_wer_ckpts(root)
    return ranked[0] if ranked else None


def read_checkpoint_resume_info(ckpt_path: Path) -> dict:
    """Lightning ckpt metadata for resume logging, plus the LR schedule it will restore.

    On resume Lightning calls load_state_dict on the scheduler, and NoamAnnealing
    inherits _LRScheduler's dict-based state_dict/load_state_dict — so warmup_steps and
    base_lrs come back from the checkpoint and silently override whatever the config or
    CLI asked for. Callers need to see those values to detect the clobber.

    sched_warmup_steps / sched_base_lrs are None when the checkpoint carries no
    recognizable scheduler state; that is reported, never assumed to be a match.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = {
        "global_step": int(ckpt.get("global_step") or 0),
        "epoch": int(ckpt.get("epoch") or 0),
        "sched_warmup_steps": None,
        "sched_base_lrs": None,
    }
    scheds = ckpt.get("lr_schedulers") or []
    if scheds and isinstance(scheds[0], dict):
        state = scheds[0]
        warmup = state.get("warmup_steps")
        if isinstance(warmup, (int, float)) and not isinstance(warmup, bool):
            info["sched_warmup_steps"] = int(warmup)
        base_lrs = state.get("base_lrs")
        if isinstance(base_lrs, (list, tuple)) and base_lrs:
            try:
                info["sched_base_lrs"] = [float(x) for x in base_lrs]
            except (TypeError, ValueError):
                pass
    return info


def find_best_checkpoint(exp_dir: Path, exp_name: str = "hebrew_ft") -> tuple[Path, float] | None:
    """Return (path, val_wer) for the lowest val_wer checkpoint under exp_dir."""
    link_dir = exp_dir / exp_name
    scanned = _scan_best_val_wer_ckpt(link_dir)
    if scanned is not None:
        return scanned
    for slot in ("best", "best_val"):
        meta = link_dir / f"{slot}.json"
        canonical_ckpt = link_dir / f"{slot}.ckpt"
        if canonical_ckpt.is_file() and meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return canonical_ckpt, float(data["val_wer"])
    return None


def _resolve_ckpt_path(path: Path) -> Path:
    if path.suffix == ".ckpt":
        return path
    match = _VAL_WER_CKPT_RE.search(path.name)
    if match is None:
        return path
    wer_tag = match.group(0)
    for candidate in path.parent.glob("*.ckpt"):
        if wer_tag in candidate.name:
            return candidate
    return path


def _slot_needs_update(link_dir: Path, slot: str, ckpt: Path, wer: float) -> bool:
    meta_path = link_dir / f"{slot}.json"
    dest_ckpt = link_dir / f"{slot}.ckpt"
    if not meta_path.is_file() or not dest_ckpt.is_file():
        return True
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if data.get("source") != str(ckpt.resolve()):
        return True
    return abs(float(data["val_wer"]) - wer) > 1e-6


def _companion_nemo(ckpt: Path) -> Path | None:
    """Run-dir .nemo written alongside Lightning checkpoints (always_save_nemo=true)."""
    preferred = ckpt.parent / "hebrew_ft.nemo"
    if preferred.is_file():
        return preferred
    nemos = [p for p in ckpt.parent.glob("*.nemo") if "-last." not in p.name]
    if not nemos:
        return None
    return max(nemos, key=lambda p: p.stat().st_mtime)


def _install_checkpoint_slot(link_dir: Path, slot: str, ckpt: Path, wer: float, *, rank: int) -> None:
    if not _slot_needs_update(link_dir, slot, ckpt, wer):
        return
    dest_ckpt = link_dir / f"{slot}.ckpt"
    dest_nemo = link_dir / f"{slot}.nemo"
    meta_path = link_dir / f"{slot}.json"
    print(f"Updating {slot}: val_wer={wer:.4f} <- {ckpt}")
    if ckpt.resolve() != dest_ckpt.resolve():
        tmp_ckpt = dest_ckpt.with_name(dest_ckpt.name + ".tmp")
        shutil.copy2(ckpt, tmp_ckpt)
        tmp_ckpt.replace(dest_ckpt)
    src_nemo = _companion_nemo(ckpt)
    if src_nemo is not None:
        tmp_nemo = dest_nemo.with_name(dest_nemo.name + ".tmp")
        shutil.copy2(src_nemo, tmp_nemo)
        tmp_nemo.replace(dest_nemo)
    elif dest_nemo.is_file():
        print(f"  {slot}: kept existing .nemo (no sibling next to source ckpt)")
    else:
        print(f"  {slot}: ckpt only (no .nemo next to source yet)")
    meta_path.write_text(
        json.dumps(
            {"slot": slot, "rank": rank, "val_wer": wer, "source": str(ckpt.resolve())},
            indent=2,
        ),
        encoding="utf-8",
    )


def _link_legacy_best_val(link_dir: Path) -> None:
    """Symlink best_val.* -> best.* for older scripts."""
    for ext in (".ckpt", ".nemo", ".json"):
        src = link_dir / f"best{ext}"
        dst = link_dir / f"best_val{ext}"
        if not src.is_file():
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink(missing_ok=True)
        dst.symlink_to(f"best{ext}")


def sync_top_k_checkpoints(exp_dir: Path, exp_name: str = "hebrew_ft", *, k: int = 3) -> Path:
    """Refresh exp/<name>/best, best_top2, best_top3 from global top-k val_wer."""
    link_dir = exp_dir / exp_name
    link_dir.mkdir(parents=True, exist_ok=True)
    ranked = _scan_top_k_val_wer_ckpts(link_dir, k)
    if not ranked:
        raise FileNotFoundError(f"No val_wer checkpoint under {exp_dir / exp_name}")
    for rank, (ckpt, wer) in enumerate(ranked, start=1):
        _install_checkpoint_slot(link_dir, _checkpoint_slot_name(rank), ckpt, wer, rank=rank)
    _link_legacy_best_val(link_dir)
    return link_dir / "best.nemo"


def sync_best_val_checkpoint(exp_dir: Path, exp_name: str = "hebrew_ft") -> Path:
    return sync_top_k_checkpoints(exp_dir, exp_name, k=1)


def consolidate_exp_checkpoints(exp_dir: Path, exp_name: str = "hebrew_ft", *, k: int = 3) -> Path:
    """Copy global top-k checkpoints to canonical slots and delete old run dirs."""
    link_dir = exp_dir / exp_name
    link_dir.mkdir(parents=True, exist_ok=True)

    if _scan_best_val_wer_ckpt(link_dir) is None and not (link_dir / "best.ckpt").is_file():
        raise FileNotFoundError(f"No val_wer checkpoint under {exp_dir / exp_name}")

    sync_top_k_checkpoints(exp_dir, exp_name, k=k)
    best_meta = link_dir / "best.json"
    best_wer = float(json.loads(best_meta.read_text(encoding="utf-8"))["val_wer"])

    removed = 0
    for path in list(link_dir.iterdir()):
        if path.name in _CANONICAL_CKPT_KEEP:
            continue
        if path.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_", path.name):
            print(f"Removing old run: {path}")
            shutil.rmtree(path)
            removed += 1
        elif path.is_dir():
            print(f"Removing: {path}")
            shutil.rmtree(path)
            removed += 1
        elif path.suffix in {".ckpt", ".nemo"}:
            path.unlink(missing_ok=True)

    print(
        f"Kept top-{k} at {link_dir} (best val_wer={best_wer:.4f}, removed {removed} run dirs)"
    )
    return link_dir / "best.nemo"


def export_ckpt_to_nemo(ckpt_path: Path, nemo_path: Path) -> Path:
    """Export a Lightning .ckpt to .nemo (cached; skipped if nemo is up to date)."""
    if nemo_path.exists() and nemo_path.stat().st_mtime >= ckpt_path.stat().st_mtime:
        return nemo_path
    nemo_path.parent.mkdir(parents=True, exist_ok=True)
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    model = EncDecRNNTBPEModelWithPrompt.load_from_checkpoint(
        str(ckpt_path.resolve()),
        map_location="cpu",
    )
    model.save_to(str(nemo_path.resolve()))
    return nemo_path


def update_best_symlinks(
    link_dir: Path,
    best_ckpt: Path,
    *,
    nemo_path: Path | None = None,
) -> None:
    """No-op: canonical best_val.* files are maintained by consolidate_exp_checkpoints."""
    _ = (link_dir, best_ckpt, nemo_path)


def find_best_nemo(
    exp_dir: Path,
    exp_name: str = "hebrew_ft",
    *,
    k: int = 3,
    dry_run: bool = False,
) -> Path:
    """Best val_wer checkpoint as .nemo — scans all runs, refreshes top-k slots if stale."""
    link_dir = exp_dir / exp_name
    found = find_best_checkpoint(exp_dir, exp_name)
    if found is None:
        raise FileNotFoundError(f"No val_wer checkpoint under {exp_dir / exp_name}")
    best_path, best_wer = found
    dest_nemo = link_dir / "best.nemo"
    if dry_run:
        stale = ""
        meta = link_dir / "best.json"
        if not meta.is_file():
            meta = link_dir / "best_val.json"
        if meta.is_file():
            canonical_wer = float(json.loads(meta.read_text(encoding="utf-8"))["val_wer"])
            if best_wer < canonical_wer - 1e-6:
                stale = f" (canonical {canonical_wer:.4f} is stale)"
        print(f"Best checkpoint: val_wer={best_wer:.4f} <- {best_path}{stale}")
        return dest_nemo if dest_nemo.exists() else best_path
    dest_nemo = sync_top_k_checkpoints(exp_dir, exp_name, k=k)
    best_wer = float(json.loads((link_dir / "best.json").read_text(encoding="utf-8"))["val_wer"])
    print(f"Best checkpoint: {dest_nemo} (val_wer={best_wer:.4f}, top-{k} synced)")
    return dest_nemo


def setup_training_env(*, devices: int) -> None:
    """Env vars for stable CUDA/NCCL training (avoids DDP watchdog kills on GeForce)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Non-deprecated names (old NCCL_* aliases now warn on recent torch).
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    if devices > 1:
        # The killer is the ProcessGroup COLLECTIVE timeout (default 30 min), not
        # the heartbeat. Rank 1's ALLREDUCE timed out while rank 0 wrote a 7 GB
        # checkpoint + validated. patch_ddp_timeout() reads TORCH_NCCL_PG_TIMEOUT_SEC
        # and forces init_process_group(timeout=...) to this on every rank.
        os.environ.setdefault("TORCH_NCCL_PG_TIMEOUT_SEC", str(4 * 3600))
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(8 * 3600))
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")
        print(
            "DDP: collective timeout "
            f"{int(os.environ['TORCH_NCCL_PG_TIMEOUT_SEC']) // 3600}h "
            "(covers slow checkpoint/validation phases)"
        )


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        inner = ",".join(_hydra_value(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, str):
        if any(ch in value for ch in " \t,[]{}:"):
            return json.dumps(value)
        return value
    return str(value)


def augmentor_hydra_overrides(augmentor: dict[str, Any], prefix: str = "model.train_ds.augmentor") -> list[str]:
    """Flatten NeMo train_ds.augmentor dict into Hydra CLI overrides."""
    overrides: list[str] = []

    def walk(base: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                walk(f"{base}.{key}", val)
        else:
            overrides.append(f"+{base}={_hydra_value(node)}")

    for name, params in augmentor.items():
        if isinstance(params, dict):
            walk(f"{prefix}.{name}", params)
        else:
            overrides.append(f"+{prefix}.{name}={_hydra_value(params)}")
    return overrides


def resolve_augmentor_cfg(cfg: dict) -> dict[str, Any] | None:
    """Return NeMo augmentor block from config (telephony preset for phone-call ASR)."""
    aug_cfg = cfg.get("training", {}).get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return None
    preset = aug_cfg.get("preset", "telephony")
    if preset == "none":
        return None
    if preset in aug_cfg:
        return aug_cfg[preset]
    return aug_cfg.get("telephony")


def register_custom_perturbations(cfg: dict) -> None:
    """Register optional custom NeMo perturbations (telephony bandpass)."""
    aug_cfg = cfg.get("training", {}).get("augmentation", {})
    if not aug_cfg.get("register_custom_perturbations", True):
        return
    try:
        from telephony_perturb import register_telephony_perturbations

        register_telephony_perturbations()
    except ImportError:
        pass

