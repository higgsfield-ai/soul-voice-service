"""Reading reference recordings for `clone` and `direction`.

The training loader band-limited and cropped references at random to widen the
distribution. None of that belongs here; what does carry over is the high-pass,
because the checkpoint was trained on filtered references and handing it
unfiltered ones costs identity quietly. The cutoff is in the bundle manifest.

The crop start is still drawn from the request's seed, so two renders of one
request read the same seconds of audio.
"""

import math
import random
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1920


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal
    from scipy import signal as scipy_signal

    divisor = math.gcd(int(source_rate), int(target_rate))
    return np.asarray(
        scipy_signal.resample_poly(
            signal, target_rate // divisor, int(source_rate) // divisor,
            window=("kaiser", 14.769656459379492)),
        dtype=np.float32)


def _highpass(signal: np.ndarray, cutoff: float, rate: int) -> np.ndarray:
    from scipy import signal as scipy_signal

    sos = scipy_signal.butter(4, cutoff, btype="high", fs=rate, output="sos")
    return np.asarray(scipy_signal.sosfiltfilt(sos, signal), dtype=np.float32)


def read_reference(path: str | Path, *, highpass_hz: float = 0.0,
                   seconds: float = 8.0, seed: int = 0) -> np.ndarray:
    """One recording as mono float32 at 24 kHz, cropped and filtered."""
    import soundfile as sf

    rng = random.Random(seed)
    with sf.SoundFile(str(path)) as handle:
        source_rate = handle.samplerate
        total = len(handle)
        wanted = min(total, max(1, round(seconds * source_rate)))
        handle.seek(rng.randint(0, total - wanted) if total > wanted else 0)
        block = handle.read(frames=wanted, dtype="float32", always_2d=True)

    mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
    mono = _resample(np.asarray(mono, dtype=np.float32), source_rate, SAMPLE_RATE)
    if highpass_hz > 0 and mono.size > 64:
        mono = _highpass(mono, highpass_hz, SAMPLE_RATE)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return np.ascontiguousarray(mono, dtype=np.float32)


def pad_batch(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad to whole codec frames, keeping the frame count aligned."""
    lengths = np.array([len(item) for item in waveforms], dtype=np.int64)
    width = int(math.ceil(lengths.max() / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME)
    padded = np.zeros((len(waveforms), width), dtype=np.float32)
    for row, item in enumerate(waveforms):
        padded[row, :len(item)] = item
    return padded, lengths
