"""Vietnamese syllabic phonetics, tone normalization, and multi-dialect confusion model.

Supports:
1. Canonical Unicode NFC + modern Vietnamese tone-mark placement normalization
   (e.g. oà -> òa, uý -> úy, oè -> òe, uỷ -> ủy).
2. Full Vietnamese syllable decomposition: (onset, rime_base, coda, tone).
3. Multi-dialect acoustic confusion distance covering Northern, Central, and
   Southern Vietnamese merges (l/n, s/x, ch/tr, r/d/gi, v/d/gi, -n/-ng, -t/-c,
   hoi/nga tone merge) plus English-in-Vietnamese code-switch phonetic matching.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

# Tone IDs: 0=ngang, 1=huyen, 2=sac, 3=hoi, 4=nga, 5=nang
TONE_NAMES = ("ngang", "huyen", "sac", "hoi", "nga", "nang")

# Map every Vietnamese accented vowel to (base_vowel_with_quality, tone_id)
_VOWEL_TONE_TABLE: dict[str, tuple[str, int]] = {
    "a": ("a", 0), "à": ("a", 1), "á": ("a", 2), "ả": ("a", 3), "ã": ("a", 4), "ạ": ("a", 5),
    "ă": ("ă", 0), "ằ": ("ă", 1), "ắ": ("ă", 2), "ẳ": ("ă", 3), "ẵ": ("ă", 4), "ặ": ("ă", 5),
    "â": ("â", 0), "ầ": ("â", 1), "ấ": ("â", 2), "ẩ": ("â", 3), "ẫ": ("â", 4), "ậ": ("â", 5),
    "e": ("e", 0), "è": ("e", 1), "é": ("e", 2), "ẻ": ("e", 3), "ẽ": ("e", 4), "ẹ": ("e", 5),
    "ê": ("ê", 0), "ề": ("ê", 1), "ế": ("ê", 2), "ể": ("ê", 3), "ễ": ("ê", 4), "ệ": ("ê", 5),
    "i": ("i", 0), "ì": ("i", 1), "í": ("i", 2), "ỉ": ("i", 3), "ĩ": ("i", 4), "ị": ("i", 5),
    "o": ("o", 0), "ò": ("o", 1), "ó": ("o", 2), "ỏ": ("o", 3), "õ": ("o", 4), "ọ": ("o", 5),
    "ô": ("ô", 0), "ồ": ("ô", 1), "ố": ("ô", 2), "ổ": ("ô", 3), "ỗ": ("ô", 4), "ộ": ("ô", 5),
    "ơ": ("ơ", 0), "ờ": ("ơ", 1), "ớ": ("ơ", 2), "ở": ("ơ", 3), "ỡ": ("ơ", 4), "ợ": ("ơ", 5),
    "u": ("u", 0), "ù": ("u", 1), "ú": ("u", 2), "ủ": ("u", 3), "ũ": ("u", 4), "ụ": ("u", 5),
    "ư": ("ư", 0), "ừ": ("ư", 1), "ứ": ("ư", 2), "ử": ("ư", 3), "ữ": ("ư", 4), "ự": ("ư", 5),
    "y": ("y", 0), "ỳ": ("y", 1), "ý": ("y", 2), "ỷ": ("y", 3), "ỹ": ("y", 4), "ỵ": ("y", 5),
}

_BASE_TO_TONED: dict[tuple[str, int], str] = {v: k for k, v in _VOWEL_TONE_TABLE.items()}

# Canonicalize older orthography (hòa/hoà, thúy/thuý, khỏe/khoẻ) to modern standard
_OLD_TO_NEW_TONE_CLUSTERS: dict[str, str] = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}

_VI_ONSETS = (
    "ngh", "ng", "nh", "ch", "tr", "th", "ph", "kh", "gh", "gi", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x",
)
_VI_CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t")

# Regional acoustic confusion equivalence / similarity groups
_ONSET_CONFUSION_PAIRS: dict[frozenset[str], float] = {
    frozenset({"l", "n"}): 0.20,          # Northern Delta merge
    frozenset({"s", "x"}): 0.15,          # Northern merge
    frozenset({"ch", "tr"}): 0.15,        # Northern merge
    frozenset({"r", "d"}): 0.15,          # Northern merge
    frozenset({"r", "gi"}): 0.15,         # Northern merge
    frozenset({"d", "gi"}): 0.10,         # Pan-Vietnamese merge
    frozenset({"v", "d"}): 0.20,          # Southern merge
    frozenset({"v", "gi"}): 0.20,         # Southern merge
    frozenset({"c", "k"}): 0.05,          # Allographic
    frozenset({"g", "gh"}): 0.05,         # Allographic
    frozenset({"ng", "ngh"}): 0.05,       # Allographic
    frozenset({"t", "th"}): 0.35,         # Aspiration confusion
    frozenset({"c", "kh"}): 0.35,
    frozenset({"k", "kh"}): 0.35,
    frozenset({"p", "b"}): 0.30,          # Code-switch voicing
    frozenset({"t", "đ"}): 0.35,
}

_CODA_CONFUSION_PAIRS: dict[frozenset[str], float] = {
    frozenset({"n", "ng"}): 0.20,         # Southern coda merge
    frozenset({"t", "c"}): 0.20,          # Southern coda merge
    frozenset({"nh", "n"}): 0.25,         # Southern palatal coda merge
    frozenset({"ch", "t"}): 0.25,         # Southern palatal coda merge
    frozenset({"nh", "ng"}): 0.25,
    frozenset({"ch", "c"}): 0.25,
    frozenset({"m", "n"}): 0.35,
    frozenset({"p", "t"}): 0.35,
    frozenset({"", "t"}): 0.45,           # Unreleased final stop drop
    frozenset({"", "c"}): 0.45,
    frozenset({"", "p"}): 0.45,
}

_TONE_CONFUSION_PAIRS: dict[frozenset[int], float] = {
    frozenset({3, 4}): 0.18,              # hoi <-> nga (Southern/Central merge)
    frozenset({2, 5}): 0.35,              # sac <-> nang (checked syllables)
    frozenset({1, 5}): 0.35,              # huyen <-> nang (low register)
    frozenset({0, 1}): 0.40,              # ngang <-> huyen
    frozenset({0, 2}): 0.40,              # ngang <-> sac
    frozenset({1, 3}): 0.35,              # huyen <-> hoi
}

_VOWEL_CONFUSION_PAIRS: dict[frozenset[str], float] = {
    frozenset({"a", "ă"}): 0.20,
    frozenset({"a", "â"}): 0.25,
    frozenset({"ă", "â"}): 0.25,
    frozenset({"e", "ê"}): 0.25,
    frozenset({"o", "ô"}): 0.25,
    frozenset({"o", "ơ"}): 0.25,
    frozenset({"ô", "ơ"}): 0.25,
    frozenset({"u", "ư"}): 0.25,
    frozenset({"i", "y"}): 0.05,
    frozenset({"iê", "yê"}): 0.05,
    frozenset({"ia", "ya"}): 0.05,
    frozenset({"iê", "ia"}): 0.20,
    frozenset({"uô", "ua"}): 0.20,
    frozenset({"ươ", "ưa"}): 0.20,
}

_ASCII_FOLD_MAP = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
)


def strip_vietnamese_diacritics(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().translate(_ASCII_FOLD_MAP)


def normalize_vietnamese_tone_placement(text: str) -> str:
    """Normalize Unicode to NFC and unify Vietnamese tone placement on open oa/oe/uy."""
    if not text:
        return text
    nfc = unicodedata.normalize("NFC", text)

    def _fix_word(match: re.Match[str]) -> str:
        word = match.group(0)
        low = word.casefold()
        # Only normalize when the cluster is at the end of the syllable (no final coda)
        # e.g. hoà -> hòa, thuý -> thúy, khoẻ -> khỏe; keep hoàn, toán, khuyến intact!
        for old, new in _OLD_TO_NEW_TONE_CLUSTERS.items():
            if low.endswith(old) and not low.startswith("qu"):
                prefix = word[:-2]
                tail = word[-2:]
                if tail.isupper():
                    return prefix + new.upper()
                if tail[0].isupper():
                    return prefix + new[0].upper() + new[1]
                return prefix + new
        return word

    return re.sub(r"[^\W\d_]+", _fix_word, nfc, flags=re.UNICODE)


@dataclass(frozen=True)
class VietnameseSyllable:
    raw: str
    onset: str
    nucleus: str
    coda: str
    tone: int


def parse_vietnamese_syllable(token: str) -> VietnameseSyllable:
    """Decompose a Vietnamese token into (onset, nucleus, coda, tone)."""
    low = unicodedata.normalize("NFC", token).strip().casefold()
    if not low:
        return VietnameseSyllable(raw="", onset="", nucleus="", coda="", tone=0)

    # Extract tone and strip tone marks (keeping vowel quality like ă, â, ê, ô, ơ, ư)
    tone = 0
    chars: list[str] = []
    for ch in low:
        if ch in _VOWEL_TONE_TABLE:
            base, t = _VOWEL_TONE_TABLE[ch]
            if t != 0:
                tone = t
            chars.append(base)
        else:
            chars.append(ch)
    toneless = "".join(chars)

    # Extract onset
    onset = ""
    rest = toneless
    for cand in _VI_ONSETS:
        if rest.startswith(cand):
            # Special case: "gi" followed by nothing or consonant -> onset "g", vowel "i"
            if cand == "gi" and (len(rest) == 2 or rest[2] not in _VOWEL_TONE_TABLE):
                onset = "gi"
                rest = "i" + rest[2:]
            else:
                onset = cand
                rest = rest[len(cand) :]
            break

    # Extract coda
    coda = ""
    for cand in _VI_CODAS:
        if rest.endswith(cand) and len(rest) > len(cand):
            coda = cand
            rest = rest[: -len(cand)]
            break

    return VietnameseSyllable(raw=low, onset=onset, nucleus=rest, coda=coda, tone=tone)


def _char_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = cur
    return row[-1]


def vietnamese_phonetic_distance(syl_a: str, syl_b: str) -> float:
    """Weighted acoustic confusion distance in [0.0, 1.0] between two Vietnamese syllables."""
    a = parse_vietnamese_syllable(syl_a)
    b = parse_vietnamese_syllable(syl_b)
    if a.raw == b.raw:
        return 0.0
    if not a.raw or not b.raw:
        return 1.0

    # Onset cost (weight 0.30)
    if a.onset == b.onset:
        onset_cost = 0.0
    else:
        onset_cost = _ONSET_CONFUSION_PAIRS.get(frozenset({a.onset, b.onset}), 1.0)

    # Nucleus cost (weight 0.35)
    if a.nucleus == b.nucleus:
        nuc_cost = 0.0
    elif frozenset({a.nucleus, b.nucleus}) in _VOWEL_CONFUSION_PAIRS:
        nuc_cost = _VOWEL_CONFUSION_PAIRS[frozenset({a.nucleus, b.nucleus})]
    else:
        max_len = max(len(a.nucleus), len(b.nucleus), 1)
        nuc_cost = min(1.0, _char_edit_distance(a.nucleus, b.nucleus) / max_len)

    # Coda cost (weight 0.20)
    if a.coda == b.coda:
        coda_cost = 0.0
    else:
        coda_cost = _CODA_CONFUSION_PAIRS.get(frozenset({a.coda, b.coda}), 1.0)

    # Tone cost (weight 0.15)
    if a.tone == b.tone:
        tone_cost = 0.0
    else:
        tone_cost = _TONE_CONFUSION_PAIRS.get(frozenset({a.tone, b.tone}), 0.65)

    return round(0.30 * onset_cost + 0.35 * nuc_cost + 0.20 * coda_cost + 0.15 * tone_cost, 4)


def phrase_phonetic_similarity(phrase_a: str, phrase_b: str) -> float:
    """Syllable-aligned phonetic similarity in [0.0, 1.0] for multi-word Vietnamese phrases."""
    toks_a = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", phrase_a).casefold(), re.UNICODE)
    toks_b = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", phrase_b).casefold(), re.UNICODE)
    if not toks_a or not toks_b:
        return 0.0
    if toks_a == toks_b:
        return 1.0

    # Dynamic programming alignment with phonetic substitution cost
    m, n = len(toks_a), len(toks_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = float(i)
    for j in range(1, n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = vietnamese_phonetic_distance(toks_a[i - 1], toks_b[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 0.9, dp[i][j - 1] + 0.9, dp[i - 1][j - 1] + sub)
    norm_dist = dp[m][n] / max(m, n)
    return round(max(0.0, 1.0 - norm_dist), 4)
