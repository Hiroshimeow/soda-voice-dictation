"""Deterministic PCM gain stage plus tonal-safe Vietnamese DSP front-end.

Backward compatible with the original validate_gain / gain_pcm16 API while adding
StreamingAudioFrontend (80 Hz HPF + F0-preserving Wiener with Observation Addition
+ noise-gated AGC + soft-knee limiter) for higher Vietnamese ASR accuracy.
"""
from __future__ import annotations

from array import array
import math
import sys

from vi_enhance.audio_frontend import (
    AudioFrontendConfig,
    StreamingAudioFrontend,
    enhance_pcm16,
    pad_utterance_pcm,
)

__all__ = [
    "validate_gain",
    "gain_pcm16",
    "AudioFrontendConfig",
    "StreamingAudioFrontend",
    "enhance_pcm16",
    "pad_utterance_pcm",
]


def validate_gain(value: float) -> float:
    gain = float(value)
    if not math.isfinite(gain) or not -12 <= gain <= 30:
        raise ValueError("Gain must be between -12 and 30 dB")
    return gain


def gain_pcm16(pcm: bytes, gain_db: float = 0) -> tuple[bytes, dict]:
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected non-empty signed 16-bit PCM")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    factor = 10 ** (validate_gain(gain_db) / 20)
    raw_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    output = array("h")
    clipped = 0
    for value in samples:
        scaled = round(value * factor)
        clipped += int(scaled < -32768 or scaled > 32767)
        output.append(max(-32768, min(32767, scaled)))
    rms = math.sqrt(sum(value * value for value in output) / len(output))
    peak = max(abs(value) for value in output)

    def dbfs(value: float) -> float:
        return max(-96.0, 20 * math.log10(max(value, .00001) / 32768))

    levels = {"input_dbfs": dbfs(raw_rms), "rms_dbfs": dbfs(rms),
              "peak_dbfs": dbfs(peak), "clipped_samples": clipped}
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes(), levels
