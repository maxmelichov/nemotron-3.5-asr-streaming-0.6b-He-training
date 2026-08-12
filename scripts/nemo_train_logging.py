"""NeMo training log filters — applied before speech_to_text_finetune imports models.

Goals:
  - no per-step WER reference/prediction spam during training
  - at most N (default 2) reference/prediction example pairs per validation pass,
    printed only on global rank 0
"""
from __future__ import annotations

import inspect
import os

_APPLIED = False
_val_examples_logged = 0
_val_example_limit = 2


def _is_rank_zero() -> bool:
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return val == "0"
    return True


def patch_ddp_timeout() -> None:
    """Force a long DDP collective timeout on every rank.

    PyTorch's ProcessGroup default collective timeout is 30 min. When rank 0
    stalls on a multi-GB checkpoint write + validation, rank 1's ALLREDUCE hits
    that 30-min wall and NCCL's watchdog tears the whole job down. Neither
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC nor the heartbeat monitor governs this —
    only the `timeout` passed to init_process_group does. We inject a large
    timeout (default 4h, override via TORCH_NCCL_PG_TIMEOUT_SEC) so slow
    checkpoint/validation phases can't trip the watchdog.
    """
    from datetime import timedelta

    import torch.distributed as dist

    seconds = int(os.environ.get("TORCH_NCCL_PG_TIMEOUT_SEC", str(4 * 3600)))
    big_timeout = timedelta(seconds=seconds)

    _orig_init = dist.init_process_group

    def _patched_init(*args, **kwargs):
        current = kwargs.get("timeout")
        if current is None or current < big_timeout:
            kwargs["timeout"] = big_timeout
        return _orig_init(*args, **kwargs)

    dist.init_process_group = _patched_init  # type: ignore[assignment]

    # Lightning may build a new default timeout via its own constant; also bump
    # the DDPStrategy default so the strategy path honours the long timeout.
    try:
        from lightning.pytorch.strategies import DDPStrategy

        _orig_ddp_init = DDPStrategy.__init__

        def _patched_ddp_init(self, *args, **kwargs):
            if kwargs.get("timeout") is None or kwargs["timeout"] < big_timeout:
                kwargs["timeout"] = big_timeout
            return _orig_ddp_init(self, *args, **kwargs)

        DDPStrategy.__init__ = _patched_ddp_init  # type: ignore[method-assign]
    except Exception:
        pass


def apply(*, val_examples: int | None = None) -> None:
    global _APPLIED, _val_example_limit
    if _APPLIED:
        return
    _APPLIED = True
    if val_examples is None:
        val_examples = int(os.environ.get("NEMO_VAL_LOG_EXAMPLES", "2"))
    _val_example_limit = val_examples if _is_rank_zero() else 0

    from nemo.collections.asr.metrics import wer as wer_mod

    _orig_update = wer_mod.WER.update

    def _patched_update(self, *args, **kwargs):
        global _val_examples_logged
        prev = self.log_prediction
        try:
            frames = {f.function for f in inspect.stack()}
            if "validation_pass" in frames and _val_examples_logged < _val_example_limit:
                self.log_prediction = True
                result = _orig_update(self, *args, **kwargs)
                _val_examples_logged += 1
                return result
            self.log_prediction = False
            return _orig_update(self, *args, **kwargs)
        finally:
            self.log_prediction = prev

    wer_mod.WER.update = _patched_update  # type: ignore[method-assign]

    if os.environ.get("NEMO_WER_NORMALIZE", "0") == "1":
        patch_wer_normalize(wer_mod)

    from nemo.collections.asr.models import rnnt_bpe_models_prompt as prompt_mod

    _orig_val_pass = prompt_mod.EncDecRNNTBPEModelWithPrompt.validation_pass

    def _patched_val_pass(self, batch, batch_idx, dataloader_idx=0):
        global _val_examples_logged
        if batch_idx == 0:
            _val_examples_logged = 0
        return _orig_val_pass(self, batch, batch_idx, dataloader_idx)

    prompt_mod.EncDecRNNTBPEModelWithPrompt.validation_pass = _patched_val_pass  # type: ignore[method-assign]

    patch_lhotse_sox_quiet()


def patch_wer_normalize(wer_mod) -> None:
    """Score val_wer on words only, ignoring punctuation and niqqud.

    WER counts a whitespace token wrong if ANY character differs, so predicting `לומר`
    where the reference has `לומר,` is penalised exactly like the wrong word. 20.9% of
    our dev reference words carry punctuation, and on the base model this inflated WER
    by 8.1 points (50.55% -> 42.45%). Since val_wer drives both checkpoint selection and
    early stopping, leaving it raw means those decisions are partly about comma
    placement.

    Patching `edit_distance` rather than `WER.update` keeps us off NeMo's internals: the
    surrounding word count is unchanged because normalising a token never splits it
    (only standalone punctuation tokens vanish, which are vanishingly rare). Training
    targets are untouched -- the model still learns to emit punctuation.
    """
    import text_norm

    _orig_edit = wer_mod.edit_distance

    def _normalized_edit(r_list, h_list, *args, **kwargs):
        r2 = [t for t in (text_norm.normalize(t) for t in r_list) if t]
        h2 = [t for t in (text_norm.normalize(t) for t in h_list) if t]
        return _orig_edit(r2, h2, *args, **kwargs)

    wer_mod.edit_distance = _normalized_edit  # type: ignore[assignment]
    print(">>> val_wer scored WITHOUT punctuation/niqqud (NEMO_WER_NORMALIZE=1)", flush=True)


def patch_lhotse_sox_quiet() -> None:
    """Suppress libsox stderr clipping warnings and pre-normalize hot clips."""
    try:
        from lhotse.tools import libsox as libsox_mod
    except ImportError:
        return

    _orig = libsox_mod.libsox_rate

    def _quiet_rate(audio, sample_rate, target_rate, **kwargs):
        import os
        import sys

        import numpy as np

        peak = float(np.max(np.abs(audio)))
        if peak > 0.99:
            audio = audio * (0.95 / peak)

        stderr_fd = sys.stderr.fileno()
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(stderr_fd)
        try:
            os.dup2(devnull, stderr_fd)
            return _orig(audio, sample_rate, target_rate, **kwargs)
        finally:
            os.dup2(saved, stderr_fd)
            os.close(saved)
            os.close(devnull)

    libsox_mod.libsox_rate = _quiet_rate  # type: ignore[assignment]
