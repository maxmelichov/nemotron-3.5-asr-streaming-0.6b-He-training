#!/usr/bin/env python3
"""Resume ivrit-ai/knesset-committees download until disk free space hits min_free_gb."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.utils import build_hf_headers

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, repo_root


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def request_gated_access(repo: str) -> None:
    url = f"https://huggingface.co/datasets/{repo}/ask-access"
    requests.post(url, headers=build_hf_headers(), timeout=30)


def main() -> None:
    cfg = load_config()
    spec = cfg["data"]["ivrit_sources"]["sources"]["knesset-committees"]
    dest = Path(spec["local_dir"])
    repo = spec["hf_repo"]
    min_free = float(spec.get("min_free_gb", 120))
    mount = Path("/mnt/windows_nvme")

    dest.mkdir(parents=True, exist_ok=True)
    request_gated_access(repo)

    free = free_gb(mount)
    print(f"knesset-committees -> {dest}")
    print(f"Disk free: {free:.1f} GB  (stop below {min_free:.0f} GB)")
    print(f"Current local: {sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / 1024**3:.1f} GB")

    if free <= min_free + 5:
        sys.exit(f"Not enough free space to download (need > {min_free + 5:.0f} GB free)")

    log = repo_root() / "exp" / "knesset-committees-download.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os; from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={repo!r}, repo_type='dataset', "
                f"local_dir={str(dest)!r}, etag_timeout=60, max_workers=4, "
                "token=os.environ.get('HF_TOKEN'))"
            ),
        ],
        stdout=log.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )

    try:
        last_sessions = 0
        while proc.poll() is None:
            free = free_gb(mount)
            print(f"  free={free:.1f} GB  pid={proc.pid}", flush=True)
            if free < min_free:
                print(f"Stopping download: free {free:.1f} GB < {min_free:.0f} GB reserve", flush=True)
                proc.terminate()
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                break
            time.sleep(120)
    except KeyboardInterrupt:
        proc.terminate()
        raise

    rc = proc.wait()
    free = free_gb(mount)
    sessions = sum(1 for _ in dest.rglob("transcript.aligned.json")) if dest.is_dir() else 0
    local_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024**3)
    print(f"\nDone rc={rc}  local={local_gb:.1f} GB  sessions={sessions}  free={free:.1f} GB")
    print(f"Log: {log}")


if __name__ == "__main__":
    main()
