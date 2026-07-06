"""Custom NeMo audio perturbations for telephony / phone-call ASR fine-tuning."""
from __future__ import annotations

import random
from typing import Sequence

import librosa
import numpy as np
from scipy.signal import resample as scipy_resample
from scipy.signal import resample_poly

try:
    from nemo.collections.asr.parts.preprocessing.perturb import Perturbation, register_perturbation
except ImportError:
    Perturbation = object  # type: ignore[misc, assignment]
    register_perturbation = None  # type: ignore[assignment]

# librosa res_type values + polyphase (exact rational ratio via scipy)
DEFAULT_RESAMPLE_METHODS: tuple[str, ...] = (
    "kaiser_fast",
    "kaiser_best",
    "scipy",
    "fft",
    "polyphase",
    "soxr_hq",
    "soxr_mq",
)


def _mono_samples(data) -> np.ndarray:
    samples = np.asarray(data._samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int, method: str) -> np.ndarray:
    if orig_sr == target_sr:
        return samples

    if method == "polyphase":
        g = np.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(samples, up, down).astype(np.float32)

    if method in ("scipy", "fft"):
        n = int(round(len(samples) * target_sr / orig_sr))
        if n < 1:
            return samples
        return scipy_resample(samples, n).astype(np.float32)

    try:
        return librosa.core.resample(
            samples,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type=method,
        ).astype(np.float32)
    except Exception:
        # Last-resort: rational polyphase (exact ratio for 16k<->8k)
        g = np.gcd(orig_sr, target_sr)
        return resample_poly(samples, target_sr // g, orig_sr // g).astype(np.float32)


def _match_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    if samples.shape[0] == target_len:
        return samples
    if samples.shape[0] > target_len:
        return samples[:target_len]
    return np.pad(samples, (0, target_len - samples.shape[0]))


class Phone8kResamplePerturbation(Perturbation):
    """Simulate narrowband phone audio: 16 kHz -> 8 kHz -> 16 kHz with random resamplers.

    Each application picks independent downsample and upsample methods (e.g. kaiser_fast
    down + scipy up), mimicking codec/transcode loss on phone calls while keeping the
    model's native 16 kHz input format.
    """

    def __init__(
        self,
        sr: int = 16000,
        phone_sr: int = 8000,
        resample_types: Sequence[str] | None = None,
        same_method_up_down: bool = False,
        rng: int | None = None,
    ):
        if phone_sr <= 0 or sr <= 0:
            raise ValueError("sr and phone_sr must be positive")
        if phone_sr >= sr:
            raise ValueError("phone_sr must be lower than sr (e.g. 8000 vs 16000)")

        self._sr = int(sr)
        self._phone_sr = int(phone_sr)
        self._methods = list(resample_types or DEFAULT_RESAMPLE_METHODS)
        self._same_method = bool(same_method_up_down)
        random.seed(rng)

    def max_augmentation_length(self, length: float) -> float:
        return length

    def perturb(self, data) -> None:
        original = _mono_samples(data)
        n = original.shape[0]

        down_method = random.choice(self._methods)
        up_method = down_method if self._same_method else random.choice(self._methods)

        narrowband = _resample(original, self._sr, self._phone_sr, down_method)
        restored = _resample(narrowband, self._phone_sr, self._sr, up_method)
        data._samples = _match_length(restored, n)


class TelephonyBandpassPerturbation(Perturbation):
    """Legacy band-limit augmentor (kept for optional use)."""

    def __init__(
        self,
        sr: int = 16000,
        bands: Sequence[tuple[int, int]] | None = None,
        order: int = 4,
        rng: int | None = None,
    ):
        from scipy.signal import butter, sosfilt

        self._sr = sr
        self._bands = list(bands or [(300, 3400), (200, 3800)])
        self._order = order
        self._sosfilt = sosfilt
        from scipy.signal import butter as _butter

        self._butter = _butter
        random.seed(rng)

    def max_augmentation_length(self, length: float) -> float:
        return length

    def perturb(self, data) -> None:
        low_hz, high_hz = random.choice(self._bands)
        nyq = 0.5 * self._sr
        low = max(low_hz / nyq, 1e-4)
        high = min(high_hz / nyq, 0.999)
        if low >= high:
            return
        sos = self._butter(self._order, [low, high], btype="band", output="sos")
        samples = _mono_samples(data)
        data._samples = self._sosfilt(sos, samples).astype(np.float32)


def register_telephony_perturbations() -> None:
    if register_perturbation is None:
        return
    register_perturbation("phone_8k_resample", Phone8kResamplePerturbation)
    register_perturbation("telephony_bandpass", TelephonyBandpassPerturbation)
