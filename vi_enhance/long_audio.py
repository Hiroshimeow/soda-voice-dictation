"""VAD pause-aware audio segmentation for long Vietnamese recordings.

Grounded in COLING 2025 ("Evaluating Open-Source ASR Systems") and this repo's
development findings (docs/VI_DEVELOPMENT_RESULTS.md): long continuous streams
are prone to transducer state drift and dropped segments. Splitting at natural
energy/VAD silence valleys into 15-30s segments with small safety overlap keeps
each utterance inside SODA's highest-accuracy window.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SegmentPlan:
    start_sample: int
    end_sample: int

    @property
    def start_byte(self) -> int:
        return self.start_sample * 2

    @property
    def end_byte(self) -> int:
        return self.end_sample * 2

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / SAMPLE_RATE


def plan_vad_segments(
    pcm: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
    target_seconds: float = 22.0,
    max_seconds: float = 30.0,
    min_seconds: float = 8.0,
    overlap_seconds: float = 0.35,
    frame_ms: int = 50,
) -> list[SegmentPlan]:
    """Plan pause-aware segment boundaries on 16-bit mono PCM."""
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected non-empty signed 16-bit PCM")
    total_samples = len(pcm) // 2
    max_samples = int(max_seconds * sample_rate)
    if total_samples <= max_samples:
        return [SegmentPlan(0, total_samples)]

    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = (total_samples + frame_len - 1) // frame_len
    frame_rms = []
    for f in range(n_frames):
        s0 = f * frame_len
        s1 = min(total_samples, s0 + frame_len)
        seg = samples[s0:s1]
        rms = math.sqrt(sum(v * v for v in seg) / max(1, len(seg))) / 32768.0
        frame_rms.append(rms)

    target_frames = max(1, int(target_seconds * 1000 / frame_ms))
    max_frames = max(target_frames + 1, int(max_seconds * 1000 / frame_ms))
    min_frames = max(1, int(min_seconds * 1000 / frame_ms))
    overlap_samples = int(overlap_seconds * sample_rate)

    plans: list[SegmentPlan] = []
    cursor_frame = 0
    while cursor_frame < n_frames:
        remaining = n_frames - cursor_frame
        if remaining <= max_frames:
            start_s = max(0, cursor_frame * frame_len - (overlap_samples if plans else 0))
            plans.append(SegmentPlan(start_s, total_samples))
            break

        # Search for the lowest-energy valley in [cursor + min_frames, cursor + max_frames]
        search_lo = cursor_frame + min_frames
        search_hi = min(n_frames - 1, cursor_frame + max_frames)
        # Prefer valleys close to target_frames using energy + mild distance penalty
        best_frame = min(
            range(search_lo, search_hi + 1),
            key=lambda idx: (
                frame_rms[idx]
                + 0.002 * abs(idx - (cursor_frame + target_frames)) / max(1, target_frames)
            ),
        )
        start_s = max(0, cursor_frame * frame_len - (overlap_samples if plans else 0))
        end_s = min(total_samples, (best_frame + 1) * frame_len)
        plans.append(SegmentPlan(start_s, end_s))
        cursor_frame = best_frame + 1

    return plans
