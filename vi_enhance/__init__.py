"""Model-agnostic Vietnamese accuracy & robustness upgrades for Google SODA STT.

All modules operate strictly outside Google SODA weights:
- audio_frontend: Tonal-safe (F0-preserving) DSP front-end with DC/80 Hz HPF,
  phase-preserving spectral Wiener + SNR-adaptive Observation Addition (OA),
  energy/ZCR VAD, adaptive noise-gated AGC + soft-knee limiter, and HAT
  trailing silence padding.
- phonetics: Vietnamese syllable decomposition, Unicode NFC + modern tone
  placement normalization, multi-dialect (North/Central/South) phonetic
  confusion distance, and English-in-Vietnamese code-switch phonetic matching.
- postprocessor: Zero-hallucination Vietnamese post-processing pipeline with
  streaming boundary-overlap deduplication, phonetic-gated personal vocabulary
  & code-switch rescue, deterministic Vietnamese ITN (numbers, currency,
  percent, date, time, phone, versions), punctuation/capitalization polish,
  and optional acoustic-gated local LLM corrector hook.
- dynamic_bias: Information-theoretic top-40 phrase ranker and calibrated
  boost-weight assigner for SODA's 40-phrase contextual biasing cap.
- long_audio: VAD pause-aware audio chunker for long recordings to prevent
  SODA HAT state drift and dropped segments.
"""
from __future__ import annotations

from vi_enhance.audio_frontend import (
    AudioFrontendConfig,
    StreamingAudioFrontend,
    enhance_pcm16,
    pad_utterance_pcm,
)
from vi_enhance.dynamic_bias import build_smart_bias_profile, score_bias_phrase
from vi_enhance.long_audio import SegmentPlan, plan_vad_segments
from vi_enhance.phonetics import (
    normalize_vietnamese_tone_placement,
    parse_vietnamese_syllable,
    phrase_phonetic_similarity,
    vietnamese_phonetic_distance,
)
from vi_enhance.postprocessor import (
    PostProcessConfig,
    VietnamesePostProcessor,
    apply_vietnamese_itn,
    stitch_segments,
)

__all__ = [
    "AudioFrontendConfig",
    "StreamingAudioFrontend",
    "enhance_pcm16",
    "pad_utterance_pcm",
    "build_smart_bias_profile",
    "score_bias_phrase",
    "SegmentPlan",
    "plan_vad_segments",
    "normalize_vietnamese_tone_placement",
    "parse_vietnamese_syllable",
    "phrase_phonetic_similarity",
    "vietnamese_phonetic_distance",
    "PostProcessConfig",
    "VietnamesePostProcessor",
    "apply_vietnamese_itn",
    "stitch_segments",
]
