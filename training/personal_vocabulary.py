"""Editable, local Vietnamese terminology and deterministic corrections.

The terminology list biases recognition at the beginning of each *new* session.
The correction rules apply only to completed Windows dictation, not raw audio.
No model weights change, and no recordings are saved by this module.
"""
from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from pathlib import Path

from training.context_profile import DATA, MAX_TERMS, validate_profile
from training.storage import atomic_bytes

VOCAB_PATH = DATA / "user_vocabulary.json"
SODA_CONTEXT_LIMIT = 40
DEFAULT = {"schema_version": 1, "language": "vi-VN",
           "phrases": ["CDPA", "G8", "G6"], "replacements": []}
REJECT = re.compile(r"(?i)(https?://|[/\\]|@|\b(?:sk-|ghp_|token|password|secret|api[_ -]?key)\b)")


def _valid_term(value: str, *, max_length: int = 75) -> str:
    if not isinstance(value, str):
        raise ValueError("Vocabulary entries must be strings")
    if not 2 <= len(value) <= max_length or value.strip() != value:
        raise ValueError("Phrases must be 2-75 characters without surrounding whitespace")
    if REJECT.search(value) or any(unicodedata.category(ch).startswith("C") for ch in value):
        raise ValueError("Paths, URLs, tokens, and control characters are not allowed")
    return value


def validate_vocabulary(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("language") != "vi-VN":
        raise ValueError("Invalid vocabulary schema")
    phrases, rules = raw.get("phrases"), raw.get("replacements")
    if not isinstance(phrases, list) or len(phrases) > MAX_TERMS:
        raise ValueError("At most 64 priority phrases are allowed")
    if not isinstance(rules, list) or len(rules) > 64:
        raise ValueError("At most 64 correction rules are allowed")
    seen, valid_phrases = set(), []
    for value in phrases:
        value = _valid_term(value)
        if value.casefold() in seen:
            raise ValueError("Duplicate priority phrase")
        seen.add(value.casefold())
        valid_phrases.append(value)
    seen, valid_rules = set(), []
    for entry in rules:
        if not isinstance(entry, dict):
            raise ValueError("Each correction rule must contain heard and write")
        heard, write = _valid_term(entry.get("heard")), _valid_term(entry.get("write"))
        if heard.casefold() in seen:
            raise ValueError("Duplicate correction source")
        seen.add(heard.casefold())
        valid_rules.append({"heard": heard, "write": write})
    return {"schema_version": 1, "language": "vi-VN",
            "phrases": valid_phrases, "replacements": valid_rules}


def vocabulary_fingerprint(path: Path = VOCAB_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def load_vocabulary(path: Path = VOCAB_PATH) -> dict:
    if not path.is_file():
        return validate_vocabulary({"schema_version": 1, "language": "vi-VN",
                                    "phrases": [], "replacements": []})
    return validate_vocabulary(json.loads(path.read_text(encoding="utf-8")))


def save_vocabulary(raw: dict, path: Path = VOCAB_PATH) -> None:
    clean = validate_vocabulary(raw)
    atomic_bytes(path, (json.dumps(clean, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def to_soda_profile(vocabulary: dict) -> dict | None:
    vocab = validate_vocabulary(vocabulary)
    if not vocab["phrases"]:
        return None
    return validate_profile({"schema_version": 1, "language": "vi-VN",
                             "terms": [{"phrase": term, "boost": 1.0}
                                       for term in vocab["phrases"][:SODA_CONTEXT_LIMIT]]})


def merge_profile(curated: dict | None, personal: dict | None) -> dict | None:
    if curated is None and personal is None:
        return None
    terms, seen = [], set()
    # Prefer explicit user terminology if the old curated profile is already full.
    for source in (personal, curated):
        if source is None:
            continue
        for item in source["terms"]:
            if item["phrase"].casefold() not in seen and len(terms) < SODA_CONTEXT_LIMIT:
                seen.add(item["phrase"].casefold())
                terms.append(item)
    return validate_profile({"schema_version": 1, "language": "vi-VN", "terms": terms})


def apply_replacements(text: str, vocabulary: dict) -> str:
    """Case-insensitive, whole-term replacements, each match only once."""
    rules = validate_vocabulary(vocabulary)["replacements"]
    if not rules or not text:
        return text
    lookup = {entry["heard"].casefold(): entry["write"] for entry in rules}
    keys = sorted((entry["heard"] for entry in rules), key=len, reverse=True)
    rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in keys) + r")(?!\w)", re.I)
    return rx.sub(lambda match: lookup[match.group(0).casefold()], text)
