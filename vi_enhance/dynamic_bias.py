"""Smart dynamic contextual biasing optimizer for Google SODA's 40-phrase cap.

Empirical testing in this repository established that Chrome SODA fails on large
bias profiles and enforces a practical 40-phrase cap (SODA_CONTEXT_LIMIT = 40).
Instead of assigning a flat boost=1.0 in insertion order, this module scores and
ranks phrases using:
1. OOV / code-switching / acronym rarity (uppercase acronyms, digits, Latin terms)
2. Multi-syllable specificity (2-4 syllables bias reliably with low false-positive rate)
3. Source priority (user personal vocabulary > mined error terms > curated seeds)
4. Common-word ambiguity penalty
and assigns calibrated per-term boost weights within [0.8, 2.5] (<= MAX_BOOST=3.5).
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Iterable

log = logging.getLogger(__name__)

from training.context_profile import MAX_BOOST, profile_phrase_safety_error, validate_profile
from training.personal_vocabulary import SODA_CONTEXT_LIMIT

_COMMON_SHORT_VI = frozenset({
    "là", "và", "của", "cho", "với", "trong", "này", "đó", "khi", "thì",
    "mô hình", "dữ liệu", "tác vụ", "giọng nói", "người dùng", "hệ thống",
})

_VI_DIACRITIC_RE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)


def score_bias_phrase(
    phrase: str,
    *,
    source_weight: float = 1.0,
    error_frequency: int = 0,
) -> tuple[float, float]:
    """Return (priority_score, recommended_boost) for a candidate SODA bias phrase."""
    clean = unicodedata.normalize("NFC", phrase).strip()
    words = clean.split()
    n_words = len(words)
    low = clean.casefold()

    # 1. Rarity / OOV / code-switch signals
    is_acronym = bool(re.search(r"\b[A-Z0-9]{2,}\b", clean))
    has_digits = any(ch.isdigit() for ch in clean)
    is_camel_or_tech = bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]|[-_/]", clean))
    is_non_vi_latin = not bool(_VI_DIACRITIC_RE.search(clean)) and bool(re.search(r"[a-zA-Z]{2,}", clean))

    priority = source_weight * 10.0
    if is_acronym or has_digits:
        priority += 6.0
    if is_camel_or_tech:
        priority += 5.0
    elif is_non_vi_latin:
        priority += 4.0

    # 2. Multi-syllable specificity sweet spot (2-4 words)
    if 2 <= n_words <= 4:
        priority += 3.0
    elif n_words == 1 and not (is_acronym or has_digits or is_camel_or_tech):
        priority -= 2.5

    # 3. Empirical error frequency boost (from confusion miner)
    if error_frequency > 0:
        priority += min(8.0, 2.5 * error_frequency)

    # 4. Penalize overly generic Vietnamese phrases that cause false biasing
    if low in _COMMON_SHORT_VI:
        priority -= 6.0

    # Calibrated SODA boost weight (bounded safely inside [0.8, 2.5] <= MAX_BOOST=3.5)
    if is_acronym or has_digits or is_camel_or_tech:
        boost = 2.0 if n_words == 1 else 1.8
    elif is_non_vi_latin:
        boost = 1.6
    elif 2 <= n_words <= 4:
        boost = 1.35
    else:
        boost = 1.0

    if error_frequency >= 2:
        boost = min(2.5, boost + 0.35)
    if low in _COMMON_SHORT_VI:
        boost = 0.8

    boost = round(min(MAX_BOOST, max(0.5, boost)), 2)
    return round(priority, 3), boost


def build_smart_bias_profile(
    *,
    personal_phrases: Iterable[str] = (),
    curated_profile: dict | None = None,
    mined_terms: Iterable[dict] = (),
    limit: int = SODA_CONTEXT_LIMIT,
) -> dict | None:
    """Merge and rank phrases from all sources into one calibrated SODA profile."""
    candidates: dict[str, tuple[float, float, str]] = {}
    rejected_reasons: Counter[str] = Counter()

    def _consider(phrase: str, source_weight: float, error_freq: int = 0, explicit_boost: float | None = None) -> None:
        normalized = unicodedata.normalize("NFC", phrase)
        safety_error = profile_phrase_safety_error(normalized)
        clean = normalized.strip()
        if safety_error is not None:
            # Keep diagnostics bounded and deliberately content-free. Raw
            # vocabulary is personal data and must not reach logs, profiles, or
            # the profile digest.
            rejected_reasons[safety_error] += 1
            return
        key = clean.casefold()
        prio, rec_boost = score_bias_phrase(
            clean, source_weight=source_weight, error_frequency=error_freq
        )
        # Curated values are part of the calibration/runtime contract. In
        # particular, a deliberate 0.0 must remain disabled rather than being
        # silently promoted to a 0.5 boost.
        if explicit_boost is None:
            boost = rec_boost
        else:
            try:
                numeric_boost = float(explicit_boost)
            except (TypeError, ValueError, OverflowError):
                rejected_reasons["invalid boost"] += 1
                return
            if not math.isfinite(numeric_boost):
                rejected_reasons["invalid boost"] += 1
                return
            boost = min(MAX_BOOST, max(0.0, round(numeric_boost, 2)))
        prev = candidates.get(key)
        if prev is None or prio > prev[0]:
            candidates[key] = (prio, boost, clean)

    # Priority order: user personal vocabulary > mined error terms > curated profile
    for p in personal_phrases:
        if isinstance(p, str):
            _consider(p, source_weight=2.0)

    for item in mined_terms:
        if isinstance(item, dict) and isinstance(item.get("phrase"), str):
            _consider(
                item["phrase"],
                source_weight=1.6,
                error_freq=int(item.get("count", 1)),
                explicit_boost=item.get("boost"),
            )

    if curated_profile and isinstance(curated_profile.get("terms"), list):
        for item in curated_profile["terms"]:
            if isinstance(item, dict) and isinstance(item.get("phrase"), str):
                _consider(
                    item["phrase"],
                    source_weight=1.0,
                    explicit_boost=item.get("boost"),
                )

    if not candidates:
        if rejected_reasons:
            log.warning(
                "No validator-safe Vietnamese bias candidates; rejected_count=%d reasons=%s",
                sum(rejected_reasons.values()), dict(sorted(rejected_reasons.items())),
            )
        return None

    ranked = sorted(candidates.values(), key=lambda x: (-x[0], x[2].casefold()))[:limit]
    terms = [{"phrase": phrase, "boost": boost} for _, boost, phrase in ranked]
    result = validate_profile({"schema_version": 1, "language": "vi-VN", "terms": terms})
    # Diagnostics are bounded aggregate facts only. Rejected vocabulary is
    # personal data and must not reach logs, snapshots, or profile digests.
    if rejected_reasons:
        result["diagnostics"] = {
            "rejected_count": sum(rejected_reasons.values()),
            "rejected_reasons": dict(sorted(rejected_reasons.items())),
        }
    return result
