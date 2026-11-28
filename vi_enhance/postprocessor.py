"""Zero-hallucination Vietnamese ASR post-processing pipeline.

Stages (all deterministic & model-agnostic by default):
1. Unicode NFC + modern Vietnamese tone-mark placement normalization.
2. Streaming segment stitcher with boundary overlap deduplication (1-4 words).
3. Deterministic user heard -> write replacements.
4. Built-in + extensible Vietnamese tech code-switching phonetic rescue
   (e.g., "đì ploy" -> "deploy", "sô đa" -> "SODA", "gít hắp" -> "GitHub").
5. Phonetic-gated personal vocabulary rescue:
   Only edits a span if Vietnamese phonetic similarity >= threshold AND
   the top match beats the runner-up by a safety margin (prevents over-correction
   in the low-WER regime, per arXiv:2405.15216).
6. Deterministic Vietnamese ITN (Inverse Text Normalization):
   Converts spoken Vietnamese numbers, percentages, currency, dates, times,
   phone numbers, and version numbers (modes: "off", "safe", "full").
7. Punctuation spacing & Vietnamese sentence capitalization polish.
8. Optional acoustic-gated local LLM/SLM corrector interface (disabled by default)
   that rejects any LLM output whose edit/phonetic divergence exceeds a strict bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import calendar
import json
import re
import unicodedata
from typing import Callable, Iterable
import urllib.request

from vi_enhance.phonetics import (
    normalize_vietnamese_tone_placement,
    phrase_phonetic_similarity,
    strip_vietnamese_diacritics,
)

# High-frequency Vietnamese function words that must NEVER be overwritten by fuzzy rescue
_PROTECTED_VI_WORDS = frozenset({
    "là", "và", "của", "cho", "với", "trong", "này", "đó", "khi", "thì",
    "mà", "nhưng", "vì", "nên", "nếu", "để", "từ", "đến", "về", "trên",
    "dưới", "sau", "trước", "cũng", "đã", "đang", "sẽ", "không", "chưa",
    "rất", "quá", "lắm", "được", "bị", "có", "làm", "đi", "ra", "vào",
    "lên", "xuống", "lại", "như", "hay", "hoặc", "một", "hai", "ba", "bốn",
    "năm", "sáu", "bảy", "tám", "chín", "mười", "tôi", "bạn", "anh", "chị",
    "em", "chúng", "ta", "họ", "nó", "mình", "người", "cái", "con", "chiếc",
})

# High-precision Vietnamese code-switching phonetic confusions for tech/dev speech
# (grounded in CoSE-E ACL SALMA 2026: matrix-language Vietnamese + English domain terms)
DEFAULT_CODE_SWITCH_MAP: dict[str, str] = {
    "sô đa": "SODA",
    "xô đa": "SODA",
    "gồ gồ": "Google",
    "gu gồ": "Google",
    "ây pi ai": "API",
    "a pi ai": "API",
    "pái thần": "Python",
    "pai thơn": "Python",
    "đốc cơ": "Docker",
    "đóc cơ": "Docker",
    "gít hắp": "GitHub",
    "ghít hắp": "GitHub",
    "gít lắp": "GitLab",
    "đì ploy": "deploy",
    "đề ploy": "deploy",
    "đì ploi": "deploy",
    "cơm mít": "commit",
    "căm mít": "commit",
    "pô lít": "pull request",
    "rì viu": "review",
    "rì víu": "review",
    "bách en": "backend",
    "bắc en": "backend",
    "phờ rôn en": "frontend",
    "phờ răn en": "frontend",
    "oép sóc két": "WebSocket",
    "quép sóc két": "WebSocket",
    "phát ây pi ai": "FastAPI",
    "phát a pi ai": "FastAPI",
    "tê leo": "Tailscale",
    "eo eo em": "LLM",
    "ép ép em pếch": "FFmpeg",
    "vê ét cốt": "VS Code",
    "lai rít": "LiteRT",
    "tên sơ phờ lâu": "TensorFlow",
    "pai tót": "PyTorch",
    "cu bơ nét tít": "Kubernetes",
    "pốt gờ rét": "PostgreSQL",
    "rê đít": "Redis",
    "clao phe": "Cloudflare",
    "cờ lao phe": "Cloudflare",
}


@dataclass(frozen=True)
class PostProcessConfig:
    enable_tone_norm: bool = True
    enable_code_switch: bool = True
    enable_phonetic_rescue: bool = True
    phonetic_threshold: float = 0.80
    phonetic_margin: float = 0.08
    itn_mode: str = "safe"  # "off" | "safe" | "full"
    enable_punctuation_polish: bool = True
    remove_filler_stutters: bool = False
    extra_code_switch_map: dict[str, str] = field(default_factory=dict)


def stitch_segments(
    segments: Iterable[str],
    *,
    overlap_windows: bool = False,
    max_overlap_words: int = 4,
) -> str:
    """Join finals, optionally deduplicating explicitly overlapping windows.

    Ordinary continuous recognizer finals are independent committed output and
    must not be treated as overlapping windows.  Callers may opt in only when
    their input segmentation contract documents overlap.
    """
    stitched_words: list[str] = []
    stitched_norm: list[str] = []

    for seg in segments:
        cleaned = unicodedata.normalize("NFC", seg).strip()
        if not cleaned:
            continue
        words = cleaned.split()
        norm_words = [
            re.sub(r"^[^\w]+|[^\w]+$", "", w, flags=re.UNICODE).casefold()
            for w in words
        ]
        overlap = 0
        if overlap_windows and stitched_norm:
            limit = min(max_overlap_words, len(stitched_norm), len(norm_words))
            for k in range(limit, 0, -1):
                if stitched_norm[-k:] == norm_words[:k] and any(norm_words[:k]):
                    overlap = k
                    break
        stitched_words.extend(words[overlap:])
        stitched_norm.extend(norm_words[overlap:])

    return " ".join(stitched_words)


# --- Deterministic Vietnamese ITN (Spoken -> Written) ---

_VI_DIGIT_MAP: dict[str, int] = {
    "không": 0, "linh": 0, "lẻ": 0,
    "một": 1, "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "nhăm": 5,
    "sáu": 6,
    "bảy": 7, "bẩy": 7,
    "tám": 8,
    "chín": 9,
}

_VI_NUM_TOKENS = set(_VI_DIGIT_MAP) | {
    "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "rưỡi",
}


def parse_vietnamese_number(phrase: str) -> int | None:
    """Parse spoken Vietnamese integer phrase (0 .. 999,999,999,999) into int."""
    tokens = [t for t in unicodedata.normalize("NFC", phrase).casefold().split() if t]
    if not tokens or any(t not in _VI_NUM_TOKENS for t in tokens):
        return None
    # Reject bare "linh"/"lẻ"/"mốt"/"lăm"/"mươi" without valid context
    if len(tokens) == 1 and tokens[0] in {"linh", "lẻ", "mốt", "lăm", "nhăm", "mươi", "rưỡi"}:
        return None

    # Special digit-by-digit check is handled separately for phone numbers; here parse cardinal
    def _parse_under_1000(sub: list[str]) -> int | None:
        if not sub:
            return 0
        val = 0
        i = 0
        n = len(sub)
        # Hundreds
        if i + 1 < n and sub[i + 1] == "trăm":
            if sub[i] not in _VI_DIGIT_MAP:
                return None
            val += _VI_DIGIT_MAP[sub[i]] * 100
            i += 2
            if i < n and sub[i] in {"linh", "lẻ"}:
                i += 1
                if i < n and sub[i] in _VI_DIGIT_MAP:
                    val += _VI_DIGIT_MAP[sub[i]]
                    i += 1
                    return val if i == n else None
                return None
            if i < n and sub[i] == "rưỡi":
                val += 50
                i += 1
                return val if i == n else None
        # Tens
        if i < n and sub[i] == "mười":
            val += 10
            i += 1
            if i < n and sub[i] in _VI_DIGIT_MAP:
                val += _VI_DIGIT_MAP[sub[i]]
                i += 1
            return val if i == n else None
        if i + 1 < n and sub[i + 1] == "mươi":
            if sub[i] not in _VI_DIGIT_MAP:
                return None
            val += _VI_DIGIT_MAP[sub[i]] * 10
            i += 2
            if i < n and sub[i] in _VI_DIGIT_MAP:
                val += _VI_DIGIT_MAP[sub[i]]
                i += 1
            return val if i == n else None
        # Shorthand "hai lăm" / "ba mốt" / "hai tư" / "bốn lăm" when tens >= 2
        if i + 1 < n and sub[i] in _VI_DIGIT_MAP and sub[i + 1] in {"mốt", "lăm", "nhăm", "tư"}:
            tens = _VI_DIGIT_MAP[sub[i]]
            if tens >= 2:
                val += tens * 10 + _VI_DIGIT_MAP[sub[i + 1]]
                i += 2
                return val if i == n else None
        # Single digit
        if i < n and sub[i] in _VI_DIGIT_MAP:
            val += _VI_DIGIT_MAP[sub[i]]
            i += 1
            return val if i == n else None
        return val if i == n else None

    total = 0
    current: list[str] = []
    multipliers = {"tỷ": 1_000_000_000, "tỉ": 1_000_000_000, "triệu": 1_000_000, "nghìn": 1_000, "ngàn": 1_000}
    last_mult = 10**15
    for idx, tok in enumerate(tokens):
        if tok in multipliers:
            mult = multipliers[tok]
            if mult >= last_mult:
                return None
            part = _parse_under_1000(current) if current else 1
            if part is None:
                return None
            total += part * mult
            last_mult = mult
            current = []
        elif tok == "rưỡi" and idx == len(tokens) - 1 and last_mult >= 1000 and not current:
            total += last_mult // 2
            last_mult = 1
        else:
            current.append(tok)

    if current:
        # Handle "một triệu năm" shorthand (= 1,500,000) only when preceded by scale & single digit?
        # Keep strict/safe: only standard under-1000 parsing.
        rem = _parse_under_1000(current)
        if rem is None:
            return None
        total += rem
    return total


_NUM_WORD_PAT = r"(?:không|linh|lẻ|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín|mười|mươi|trăm|nghìn|ngàn|triệu|tỷ|tỉ|rưỡi)"
_NUM_SEQ_RE = re.compile(rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)(?!\w)", re.IGNORECASE | re.UNICODE)


def _format_vi_thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def apply_vietnamese_itn(text: str, *, mode: str = "safe") -> str:
    """Deterministic Vietnamese spoken-to-written Inverse Text Normalization."""
    if not text or mode == "off":
        return text
    if mode not in ("safe", "full"):
        raise ValueError(f"Unsupported ITN mode: {mode}")

    out = text
    # Twelve or more spoken digits are deliberately not guessed: the 10/11
    # digit phone rule below must never partially rewrite a longer identifier.
    spoken_digit = r"(?:không|một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín)"
    long_digit_re = re.compile(
        rf"(?<!\w)(?:{spoken_digit}\s+){{11,}}{spoken_digit}(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    # Every protected span gets a token selected after checking the complete
    # current text.  This is deliberately registry-based rather than a later
    # global replacement: a user may legitimately dictate a token-looking
    # private-use string, and it must remain byte-for-byte unchanged.
    registry = _TokenRegistry(out)
    out = long_digit_re.sub(lambda m: registry.hide(m.group(0), "LONGDIGIT", out), out)
    # Ambiguous number/currency prose is immutable. This prevents decimal and
    # currency stages from corrupting nouns (for example ``ba đồng nghiệp`` or
    # ``âm hai phẩy năm đồng``).

    ambiguous_re = re.compile(
        rf"(?<!\w)(?:âm\s+)?({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*\s+phẩy\s+{_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*\s+(?:đồng|việt nam đồng|đô la|đô|vnđ))(?!\w)|"
        rf"(?<!\w)(?:âm\s+)?({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*\s+(?:đồng|việt nam đồng|đô la|đô|vnđ))(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    out = ambiguous_re.sub(lambda m: registry.hide(m.group(0), "AMBIGUOUS", out), out)
    # Half-hour clock expressions are intentionally ambiguous in safe mode.
    clock_re = re.compile(
        rf"(?<!\w){_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*\s+giờ\s+rưỡi(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    out = clock_re.sub(lambda m: registry.hide(m.group(0), "CLOCK", out), out)

    # 1. Phone numbers: 10-11 spoken digits starting with "không"
    single_digit = r"(?:không|một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín)"
    phone_re = re.compile(
        rf"(?<!\w)(không(?:\s+{single_digit}){{9,10}})(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    out = phone_re.sub(
        lambda m: "".join(str(_VI_DIGIT_MAP[t]) for t in m.group(1).casefold().split()),
        out,
    )

    # 2. Dates: "ngày <N> tháng <M> năm <Y>" or "ngày <N> tháng <M>"
    date_full_re = re.compile(
        rf"(?<!\w)(ngày)\s+({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*?)\s+tháng\s+({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*?)\s+năm\s+({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_date_full(m: re.Match[str]) -> str:
        d = parse_vietnamese_number(m.group(2))
        mo = parse_vietnamese_number(m.group(3))
        y = parse_vietnamese_number(m.group(4))
        if (d is not None and mo is not None and y is not None and
                1900 <= y <= 2100 and 1 <= mo <= 12 and
                1 <= d <= calendar.monthrange(y, mo)[1]):
            return f"{m.group(1)} {d:02d}/{mo:02d}/{y}"
        return registry.hide(m.group(0), "DATE", out)

    out = date_full_re.sub(_sub_date_full, out)

    # 3. Time. In safe mode a bare "giờ <number>" is ordinary prose ("sáu
    # giờ ba người"), so only an explicit minute marker or "rưỡi" qualifies.
    time_re = re.compile(
        rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*?)\s+giờ\s+(rưỡi|({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*?)\s+phút)(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_time(m: re.Match[str]) -> str:
        # ``giờ ... phút`` is also a duration.  A trailing ``nữa/sau`` or a
        # preceding duration marker makes the clock interpretation unsafe.
        before = m.string[max(0, m.start() - 16):m.start()].casefold()
        after = m.string[m.end():m.end() + 16].casefold()
        if (re.search(r"(?:^|\s)(?:trong|sau|cách|kéo\s+dài)\s*$", before) or
                re.match(r"\s+(?:nữa|sau|tới|nữa rồi)\b", after)):
            return m.group(0)
        h = parse_vietnamese_number(m.group(1))
        if h is None or not 0 <= h < 24:
            return m.group(0)
        # ``giờ rưỡi`` is semantically ambiguous (duration vs clock time),
        # so safe ITN leaves it faithful; explicit minutes are unambiguous.
        if m.group(2).casefold() == "rưỡi":
            return m.group(0)
        mins = parse_vietnamese_number(m.group(3))
        if mins is not None and 0 <= mins <= 59:
            return f"{h}:{mins:02d}"
        return m.group(0)

    out = time_re.sub(_sub_time, out)

    # 4. Version numbers: "phiên bản <A> chấm <B> (chấm <C>)"
    # Do not normalize a prefix of a longer or mixed spoken version. In
    # particular, ``hai chấm ba mươi``, a fourth component, and a mixed suffix
    # remain faithful instead of being partially rewritten.
    ver_re = re.compile(
        rf"(?<!\w)(phiên bản|bản|version|v)\s+({single_digit})(?:\s+chấm\s+({single_digit}))(?:\s+chấm\s+({single_digit}))?(?!\w)(?!\s+\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_ver(m: re.Match[str]) -> str:
        parts = [str(_VI_DIGIT_MAP[m.group(2).casefold()]), str(_VI_DIGIT_MAP[m.group(3).casefold()])]
        if m.group(4):
            parts.append(str(_VI_DIGIT_MAP[m.group(4).casefold()]))
        return f"{m.group(1)} {'.'.join(parts)}"

    out = ver_re.sub(_sub_ver, out)

    # 5. Percentages: "<N> (phẩy <D>) phần trăm"
    pct_re = re.compile(
        rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)(?:\s+phẩy\s+({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*))?\s+phần\s+trăm(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_pct(m: re.Match[str]) -> str:
        whole = parse_vietnamese_number(m.group(1))
        if whole is None:
            return m.group(0)
        if m.group(2):
            frac = parse_vietnamese_number(m.group(2))
            if frac is None:
                return m.group(0)
            return f"{whole},{frac}%"
        return f"{whole}%"

    out = pct_re.sub(_sub_pct, out)

    # 6. Currency is deliberately not normalized. Safe ITN cannot distinguish
    # ``ba đồng nghiệp`` from a VND amount without semantic context, so leave
    # all numeric/currency-looking prose unchanged.
    vnd_re = re.compile(
        rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)\s+(?:đồng|việt nam đồng|vnđ)(?!\w)(?!\s+hồ\b)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_vnd(m: re.Match[str]) -> str:
        val = parse_vietnamese_number(m.group(1))
        if val is None or val <= 0:
            return m.group(0)
        return f"{_format_vi_thousands(val)}đ"

    # Ambiguous currency input is faithful by design; no VND rewrite.

    usd_re = re.compile(
        rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)\s+(?:đô la|đô)(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_usd(m: re.Match[str]) -> str:
        val = parse_vietnamese_number(m.group(1))
        if val is None or val <= 0:
            return m.group(0)
        return f"{val} USD"

    # Ambiguous currency input is faithful by design; no USD rewrite.

    # 7. Decimal numbers: "<N> phẩy <D>"
    dec_re = re.compile(
        rf"(?<!\w)({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)\s+phẩy\s+({_NUM_WORD_PAT}(?:\s+{_NUM_WORD_PAT})*)(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )

    def _sub_dec(m: re.Match[str]) -> str:
        whole = parse_vietnamese_number(m.group(1))
        frac = parse_vietnamese_number(m.group(2))
        if whole is None or frac is None:
            return m.group(0)
        return f"{whole},{frac}"

    out = dec_re.sub(_sub_dec, out)

    # 8. Standalone cardinals are intentionally a full-mode feature.  Without
    # an explicit semantic marker, Vietnamese number words are also ordinary
    # nouns/adjectives ("hai triệu chứng", "hai tư vấn viên", "hai tỷ lệ").
    # The preceding stages retain high-confidence forms such as phones, dates,
    # explicit clock minutes, percentages, and versions in safe mode.
    if mode == "full":
        def _sub_num(m: re.Match[str]) -> str:
            raw_span = m.group(1)
            val = parse_vietnamese_number(raw_span)
            return str(val) if val is not None else raw_span

        out = _NUM_SEQ_RE.sub(_sub_num, out)
    return registry.restore(out)


def _apply_whole_term_map(text: str, mapping: dict[str, str], *, allow_soda: bool = True) -> str:
    if not text or not mapping:
        return text
    lookup = {k.casefold(): v for k, v in mapping.items() if k and v}
    if not lookup:
        return text
    keys = sorted(lookup.keys(), key=len, reverse=True)
    rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE | re.UNICODE)
    def replace(match: re.Match[str]) -> str:
        # Ordinary ``sô đa`` means the drink. Only rewrite it in an explicit
        # technical context; user replacement rules are applied separately.
        key = unicodedata.normalize("NFC", match.group(0)).casefold()
        if key in {"sô đa", "xô đa"} and not allow_soda:
            return match.group(0)
        # Unicode IGNORECASE can match a spelling whose casefold differs from
        # the compiled key.  A missing key must never abort recognition.
        return lookup.get(key, match.group(0))
    return rx.sub(replace, text)


def phonetic_vocab_rescue(
    text: str,
    target_phrases: Iterable[str],
    *,
    threshold: float = 0.80,
    margin: float = 0.08,
) -> tuple[str, list[dict]]:
    """Phonetic-similarity-gated vocabulary rescue (zero-hallucination guardrail).

    Only replaces a transcript n-gram with a target vocabulary phrase when:
    - The n-gram is not already an exact match and not made purely of protected
      common Vietnamese function words.
    - Its Vietnamese phonetic similarity to the target phrase is >= threshold.
    - The best target beats the runner-up by >= margin (avoids ambiguous swaps).
    """
    candidates = [
        unicodedata.normalize("NFC", p).strip()
        for p in target_phrases
        if isinstance(p, str) and len(p.strip()) >= 2
    ]
    if not text or not candidates:
        return text, []

    # Group candidates by syllable/word count (1..5 words)
    by_len: dict[int, list[str]] = {}
    for cand in candidates:
        n_words = len(re.findall(r"[^\W_]+", cand, re.UNICODE))
        if 1 <= n_words <= 5:
            by_len.setdefault(n_words, []).append(cand)

    # Tokenize preserving exact punctuation/whitespace around word tokens
    parts = re.split(r"(\s+)", text)
    # Build word token indices
    word_indices = [idx for idx, p in enumerate(parts) if p and not p.isspace()]
    edits: list[dict] = []

    # Scan longest phrases first so multi-word phrases take precedence
    max_n = max(by_len.keys(), default=0)
    locked: set[int] = set()

    for n in range(max_n, 0, -1):
        cands = by_len.get(n)
        if not cands:
            continue
        i = 0
        while i <= len(word_indices) - n:
            span_slots = word_indices[i : i + n]
            if any(slot in locked for slot in span_slots):
                i += 1
                continue
            # Strip outer punctuation from first/last word for comparison
            first_raw = parts[span_slots[0]]
            last_raw = parts[span_slots[-1]]
            m_lead = re.match(r"^([^\w]*)", first_raw, re.UNICODE)
            m_trail = re.search(r"([^\w]*)$", last_raw, re.UNICODE)
            lead_p = m_lead.group(1) if m_lead else ""
            trail_p = m_trail.group(1) if m_trail else ""

            core_words = []
            for k, slot in enumerate(span_slots):
                w = parts[slot]
                if k == 0 and lead_p:
                    w = w[len(lead_p) :]
                if k == n - 1 and trail_p:
                    w = w[: len(w) - len(trail_p)]
                core_words.append(w)

            span_text = " ".join(core_words)
            span_low = span_text.casefold()
            # Never touch single protected common Vietnamese words
            if n == 1 and span_low in _PROTECTED_VI_WORDS:
                i += 1
                continue
            # Exact spelling or casefold-equivalent vocabulary is authoritative:
            # a weaker fuzzy candidate must never rewrite an already-known term.
            if any(span_text == c or span_low == c.casefold() for c in cands):
                locked.update(span_slots)
                i += n
                continue

            # Score against candidates
            scored: list[tuple[float, str]] = []
            for cand in cands:
                # Exact spelling/case is already correct. In particular, do not
                # turn literal ``AI``/``ai`` into another acronym merely because
                # a case-folded fuzzy candidate scores perfectly.
                if span_text == cand:
                    locked.update(span_slots)
                    break
                # Case-only normalization is not an acoustic correction. User
                # replacement rules remain the explicit way to request it.
                if span_low == cand.casefold():
                    continue
                # Short isolated syllables and abbreviations have too little
                # acoustic/context evidence for a fuzzy rewrite (e.g. ``di``
                # must not become ``Git``). Multi-word phrases remain eligible.
                if n == 1 and (
                    len(strip_vietnamese_diacritics(span_text)) < 3
                    or len(strip_vietnamese_diacritics(cand)) < 3
                ):
                    continue
                # Fast pre-filter on ASCII-folded length ratio
                if abs(len(strip_vietnamese_diacritics(span_text)) - len(strip_vietnamese_diacritics(cand))) > max(2, n * 2):
                    continue
                sim = phrase_phonetic_similarity(span_text, cand)
                if sim >= threshold:
                    scored.append((sim, cand))

            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                best_sim, best_cand = scored[0]
                second_sim = scored[1][0] if len(scored) > 1 else 0.0
                # For 1-word rescue, require higher threshold (0.85) unless exact case-fold match
                effective_thresh = threshold if n >= 2 or best_sim == 1.0 else max(threshold, 0.85)
                # Even perfect matches need a runner-up margin: tied targets are
                # ambiguous and must remain faithful to the native transcript.
                if best_sim >= effective_thresh and best_sim - second_sim >= margin:
                    # Preserve sentence-initial capitalization when target is lowercase
                    replacement = best_cand
                    if i == 0 and span_text[:1].isupper() and replacement[:1].islower():
                        replacement = replacement[0].upper() + replacement[1:]
                    # Replace span_slots in parts
                    parts[span_slots[0]] = lead_p + replacement + trail_p
                    for extra_slot in span_slots[1:]:
                        parts[extra_slot] = ""
                        # Also remove the whitespace immediately preceding extra_slot
                        if extra_slot - 1 >= 0 and parts[extra_slot - 1].isspace():
                            parts[extra_slot - 1] = ""
                    locked.update(span_slots)
                    edits.append({
                        "from": span_text,
                        "to": replacement,
                        "similarity": best_sim,
                    })
                    i += n
                    continue
            i += 1

    return "".join(parts), edits


_LITERAL_SPAN_RE = re.compile(
    # A Windows path may contain spaces.  Stop at the first recognized file
    # extension rather than using ``[^\\s]+`` (which protects only the first
    # path component) or consuming the rest of the sentence.
    r"(?:https?://[^\s]+|[A-Za-z]:\\[^\r\n]*?\.(?:py|js|ts|json|yaml|yml|toml|cfg|ini|sh|ps1|exe|txt)\b|"
    r"\\\\[^\s]+|/(?=[^\s/])[^\s]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}|"
    r"\b[A-Za-z0-9_-]+\.(?:py|js|ts|json|yaml|yml|toml|cfg|ini|sh|ps1|exe|txt)\b)"
)


class _TokenRegistry:
    """Collision-free exact span registry used across rewrite stages."""
    def __init__(self, source: str = "") -> None:
        self._source = source
        self._values: dict[str, str] = {}
        self._counter = 0

    def hide(self, value: str, kind: str, current: str | None = None) -> str:
        while True:
            token = f"\ue000SODA_{kind}_{self._counter}\ue001"
            self._counter += 1
            if token not in self._source and token not in (current or "") and token not in self._values:
                self._values[token] = value
                return token

    def restore(self, text: str) -> str:
        if not self._values:
            return text
        pattern = re.compile("|".join(re.escape(token) for token in self._values), re.UNICODE)
        return pattern.sub(lambda match: self._values[match.group(0)], text)

    def protect_values(self, text: str, values: Iterable[str], kind: str) -> str:
        """Protect values only in text outside already-registered spans."""
        if not self._values:
            token_pattern = None
        else:
            token_pattern = re.compile("|".join(re.escape(token) for token in self._values), re.UNICODE)
        value_pattern = re.compile(
            "|".join(re.escape(value) for value in sorted(set(values), key=len, reverse=True)),
            re.UNICODE,
        ) if values else None
        if value_pattern is None:
            return text

        def rewrite(segment: str) -> str:
            return value_pattern.sub(lambda match: self.hide(match.group(0), kind, text), segment)

        if token_pattern is None:
            return rewrite(text)
        return token_pattern.sub(lambda match: match.group(0), text) if not value_pattern.search(text) else "".join(
            rewrite(part) if not is_token else part
            for is_token, part in self._split_tokens(text, token_pattern)
        )

    @staticmethod
    def _split_tokens(text: str, token_pattern: re.Pattern[str]) -> Iterable[tuple[bool, str]]:
        end = 0
        for match in token_pattern.finditer(text):
            if match.start() > end:
                yield False, text[end:match.start()]
            yield True, match.group(0)
            end = match.end()
        if end < len(text):
            yield False, text[end:]


class _ImmutableRegistry(_TokenRegistry):
    """Registry for syntax and explicit user outputs that must remain exact."""


def _protect_literal_spans(text: str, values: Iterable[str] = ()) -> tuple[str, _ImmutableRegistry]:
    """Hide immutable syntax/explicit outputs from every semantic stage."""
    registry = _ImmutableRegistry(text)
    patterns = [_LITERAL_SPAN_RE]
    explicit = [v for v in values if isinstance(v, str) and v]
    if explicit:
        patterns.append(re.compile("|".join(re.escape(v) for v in sorted(set(explicit), key=len, reverse=True))))
    combined = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in patterns), re.IGNORECASE | re.UNICODE)
    return combined.sub(lambda match: registry.hide(match.group(0), "LITERAL"), text), registry


def polish_vietnamese_punctuation(text: str) -> str:
    """Clean punctuation without modifying URLs, paths, domains, or filenames."""
    if not text:
        return text
    # Use the same source-aware registry as ITN.  Punctuation used to protect
    # spans with predictable NUL strings and restore them with global replace,
    # allowing user input to impersonate a placeholder.
    registry = _TokenRegistry(text)
    literal_re = re.compile(
        r"(?:https?://[^\s]+|[A-Za-z]:\\[^\r\n]*?\.(?:py|js|ts|json|yaml|yml|toml|cfg|ini|sh|ps1|exe|txt)\b|"
        r"\\\\[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}|"
        r"\b[A-Za-z0-9_-]+\.(?:py|js|ts|json|yaml|yml|toml|cfg|ini|sh|ps1|exe|txt)\b)"
    )
    s = literal_re.sub(lambda match: registry.hide(match.group(0), "PUNCT", text), text)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove space before punctuation, ensure single space after punctuation
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    s = re.sub(r"([,.;:?!])([^\s\d.,;:?!\"')\]])", r"\1 \2", s)
    # Collapse duplicate punctuation like ",," or ".." (keep "...")
    s = re.sub(r",+", ",", s)
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)
    # Capitalize first character and character after . ? !
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    s = re.sub(
        r"([.?!]\s+)([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])",
        lambda m: m.group(1) + m.group(2).upper(),
        s,
    )
    return registry.restore(s)


class GatedLLMCorrector:
    """Optional local OpenAI-compatible/Ollama SLM corrector with strict phonetic guardrails.

    Disabled by default. When enabled, any rewrite whose phonetic similarity to the
    original SODA transcript drops below `min_phonetic_fidelity` or whose word-count
    changes beyond `max_word_delta_ratio` is rejected to guarantee zero hallucination.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 4.0,
        min_phonetic_fidelity: float = 0.85,
        max_word_delta_ratio: float = 0.15,
        requester: Callable[[str, dict, float], dict] | None = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.min_phonetic_fidelity = min_phonetic_fidelity
        self.max_word_delta_ratio = max_word_delta_ratio
        self._requester = requester or self._default_post

    @staticmethod
    def _default_post(url: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def correct(self, text: str, vocabulary_phrases: Iterable[str] = ()) -> tuple[str, bool]:
        if not text.strip():
            return text, False
        vocab_hint = ", ".join(list(vocabulary_phrases)[:30])
        prompt = (
            "Sửa lỗi chính tả, dấu thanh và thuật ngữ cho bản ghi giọng nói tiếng Việt sau. "
            "CHỈ sửa lỗi nghe nhầm âm thanh; KHÔNG thêm, bớt hay diễn giải lại nội dung. "
            "Chỉ trả về đúng câu đã sửa."
        )
        if vocab_hint:
            prompt += f"\nTừ vựng ưu tiên: {vocab_hint}"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        }
        try:
            data = self._requester(self.endpoint, payload, self.timeout)
            candidate = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception:
            return text, False

        if not candidate:
            return text, False
        orig_words = text.split()
        cand_words = candidate.split()
        if abs(len(cand_words) - len(orig_words)) > max(2, int(len(orig_words) * self.max_word_delta_ratio)):
            return text, False
        fidelity = phrase_phonetic_similarity(text, candidate)
        if fidelity < self.min_phonetic_fidelity:
            return text, False
        return candidate, (candidate != text)


class VietnamesePostProcessor:
    """Composable Vietnamese ASR post-processing pipeline."""

    def __init__(
        self,
        config: PostProcessConfig | None = None,
        *,
        llm_corrector: GatedLLMCorrector | None = None,
    ):
        self.cfg = config or PostProcessConfig()
        self.llm_corrector = llm_corrector

    def process(
        self,
        text: str,
        *,
        vocabulary: dict | None = None,
        extra_phrases: Iterable[str] = (),
    ) -> tuple[str, dict]:
        if not text or not text.strip():
            return "", {"edits": [], "itn_applied": False, "llm_applied": False}

        # Protect URLs, email addresses, paths, code filenames, and explicit
        # rule outputs before *every* rewrite stage, not merely punctuation.
        # Protect syntax already present in recognition output.  Explicit
        # replacement outputs are protected only after their heard-side rule
        # runs; protecting them here would make case-insensitive ``github`` ->
        # ``GitHub`` rules collide with the output and silently block them.
        out, immutable_registry = _protect_literal_spans(text)
        if self.cfg.enable_tone_norm:
            out = normalize_vietnamese_tone_placement(out)

        # Deterministic user heard -> write rules first. Replacement output is
        # authoritative and must not be reinterpreted by any later stage. Use
        # printable sentinels: NUL placeholders were vulnerable to downstream
        # string/regex code that assumed text contained no control characters.
        if vocabulary and vocabulary.get("replacements"):
            user_map = {
                entry["heard"]: entry["write"]
                for entry in vocabulary["replacements"]
                if isinstance(entry, dict) and entry.get("heard") and entry.get("write")
            }
            out = _apply_whole_term_map(out, user_map)
            # Hide replacement outputs only in unprotected segments. A global
            # replace would be able to rewrite text inside an immutable token
            # (and makes literal/replacement overlap dependent on ordering).
            out = immutable_registry.protect_values(
                out,
                (str(entry["write"]) for entry in vocabulary["replacements"]
                 if isinstance(entry, dict) and entry.get("write")),
                "USER",
            )

        # Code-switching map (Vietnamese phonetic approximation -> English tech term)
        if self.cfg.enable_code_switch:
            cs_map = dict(DEFAULT_CODE_SWITCH_MAP)
            cs_map.update(self.cfg.extra_code_switch_map)
            # SODA is a product name, not a general rewrite of the beverage
            # phrase. Permit it only with an explicit vocabulary/context signal
            # or an already technical term nearby.
            technical_context = bool(
                vocabulary and (
                    vocabulary.get("phrases") or vocabulary.get("replacements")
                )
            ) or bool(re.search(
                r"(?i)\b(?:api|backend|frontend|github|google|docker|deploy|server|code|kỹ thuật)\b",
                out,
            ))
            out = _apply_whole_term_map(out, cs_map, allow_soda=technical_context)

        # Phonetic-gated personal vocabulary rescue
        edits: list[dict] = []
        if self.cfg.enable_phonetic_rescue:
            phrases = list(extra_phrases)
            if vocabulary and vocabulary.get("phrases"):
                phrases.extend(vocabulary["phrases"])
            if phrases:
                out, edits = phonetic_vocab_rescue(
                    out,
                    phrases,
                    threshold=self.cfg.phonetic_threshold,
                    margin=self.cfg.phonetic_margin,
                )

        # Vietnamese ITN
        before_itn = out
        if self.cfg.itn_mode != "off":
            out = apply_vietnamese_itn(out, mode=self.cfg.itn_mode)
        itn_applied = out != before_itn

        # Optional disfluency cleanup (only when explicitly opted in)
        if self.cfg.remove_filler_stutters:
            out = re.sub(r"(?<!\w)(?:à|ừm|ờm|ờ)(?:\s+(?:à|ừm|ờm|ờ))+(?!\w)", "", out, flags=re.IGNORECASE)

        if self.cfg.enable_punctuation_polish:
            out = polish_vietnamese_punctuation(out)

        llm_applied = False
        if self.llm_corrector is not None:
            phrases = list(vocabulary.get("phrases", [])) if vocabulary else []
            out, llm_applied = self.llm_corrector.correct(out, phrases)

        # Do not let punctuation/ITN rewrite an explicit user replacement or
        # a literal URL/path/code span.
        out = immutable_registry.restore(out)
        return out, {
            "edits": edits,
            "itn_applied": itn_applied,
            "llm_applied": llm_applied,
        }
