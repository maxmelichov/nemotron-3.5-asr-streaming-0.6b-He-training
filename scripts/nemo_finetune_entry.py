#!/usr/bin/env python3
"""Run NeMo speech_to_text_finetune.py after applying quiet-training log patches.

The target script path comes from NEMO_FINETUNE_SCRIPT (env) so that Lightning's
DDP subprocess relaunch (which re-executes `python <sys.argv>`) goes through this
wrapper on every rank and the patches apply everywhere.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nemo_train_logging  # noqa: E402

if __name__ == "__main__":
    script = os.environ.get("NEMO_FINETUNE_SCRIPT")
    if not script:
        sys.exit("NEMO_FINETUNE_SCRIPT env var must point to speech_to_text_finetune.py")
    nemo_train_logging.patch_ddp_timeout()
    nemo_train_logging.apply()
    runpy.run_path(script, run_name="__main__")
