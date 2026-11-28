"""Validated contextual bias profiles and release-gate utilities.

Profiles are local configuration, not changes to Google SODA weights.
No private Palace excerpts are persisted.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROFILE_PATH = DATA / "palace_vi_bias.json"
ACTIVE_PATH = DATA / "active_bias.json"
MAX_TERMS = 64
MAX_BOOST = 3.5
# Limit to human reviewable, curated phrases. Never accept arbitrary urls/paths.
# Native SODA context is stricter than vocabulary storage: technical tokens
# containing digits can be retained locally but must not enter the protobuf
# profile (for example S20, WSL2, Qwen3.5, and RTX 2080 Ti).
REJECT = re.compile(r"(?i)(https?://|[/\\]|@|\b(?:sk-|ghp_|token|password|secret|api[_ -]?key)\b|\d)")


def profile_phrase_safety_error(phrase: object) -> str | None:
    """Return a bounded, non-content safety reason for a profile candidate."""
    if not isinstance(phrase, str) or not 2 <= len(phrase) <= 75:
        return "invalid phrase length"
    # Reject lone surrogates and all control/format characters before any
    # normalization or protobuf assignment.  This also makes malformed UTF-8
    # decoded with surrogateescape fail closed without exposing its contents.
    try:
        phrase.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return "invalid unicode"
    if any(unicodedata.category(ch).startswith("C") for ch in phrase):
        return "control character"
    if phrase != phrase.strip() or REJECT.search(phrase):
        return "unsafe phrase format"
    return None


def _storage_phrase_error(phrase: object) -> str | None:
    """Validate persisted terms without applying the native protobuf policy.

    Older curated profiles legitimately contain technical identifiers such as
    ``WSL2``.  They remain readable for rollback/editing; ``apply_to_soda`` is
    the mandatory native boundary where the stricter digit policy is applied.
    """
    if not isinstance(phrase, str) or not 2 <= len(phrase) <= 75:
        return "invalid phrase length"
    try:
        phrase.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return "invalid unicode"
    if any(unicodedata.category(ch).startswith("C") for ch in phrase):
        return "control character"
    if phrase != phrase.strip() or re.search(
        r"(?i)(https?://|[/\\]|@|\b(?:sk-|ghp_|token|password|secret|api[_ -]?key)\b)", phrase
    ):
        return "unsafe phrase format"
    return None


def validate_profile(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("language") != "vi-VN":
        raise ValueError("Invalid schema or language")
    terms = raw.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= MAX_TERMS:
        raise ValueError("Terms must have between 1 and 64 entries")
    seen: set[str] = set()
    validated = []
    for item in terms:
        if not isinstance(item, dict):
            raise ValueError("Invalid term entry")
        phrase, boost = item.get("phrase"), item.get("boost")
        storage_error = _storage_phrase_error(phrase)
        if storage_error == "invalid phrase length":
            raise ValueError("Invalid phrase length")
        if storage_error is not None:
            raise ValueError("Unsafe phrase format")
        folded = phrase.casefold()
        if folded in seen:
            raise ValueError("Duplicate phrase")
        seen.add(folded)
        if not isinstance(boost,(int,float)) or isinstance(boost,bool) or not math.isfinite(boost) or not 0.0 <= boost <= MAX_BOOST:
            raise ValueError("Boost out of bounds")
        validated.append({"phrase":phrase,"boost":round(float(boost),2)})
    return {"schema_version":1, "language":"vi-VN", "terms":validated,
            "type":"context_biasing_not_weight_finetune"}


def native_profile(profile: dict | None) -> dict | None:
    """Return the deterministic protobuf-safe subset of a stored profile."""
    if profile is None:
        return None
    validated = validate_profile(profile)
    terms = [term for term in validated["terms"]
             if profile_phrase_safety_error(term["phrase"]) is None]
    if not terms:
        return None
    return {**validated, "terms": terms}


def load_profile(path: Path = ACTIVE_PATH) -> dict | None:
    if not path.is_file():
        return None
    data=json.loads(path.read_text(encoding="utf-8"))
    return validate_profile(data)


def prepare_candidate(source: Path, target: Path = PROFILE_PATH) -> dict:
    raw=json.loads(source.read_text(encoding="utf-8"))
    prepared={"schema_version":1,"language":"vi-VN","terms":raw["terms"][:MAX_TERMS]}
    result=validate_profile(prepared)
    target.parent.mkdir(parents=True,exist_ok=True)
    write_atomic(target, result)
    return result


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    from training.storage import atomic_bytes
    atomic_bytes(path,json.dumps(data,ensure_ascii=False,indent=2).encode("utf-8"))


def _digest_payload(profile: dict | None) -> dict | None:
    if profile is None:
        return None
    effective = native_profile(profile)
    if effective is None:
        return None
    return {
        "schema_version": effective["schema_version"],
        "language": effective["language"],
        "terms": effective["terms"],
        "type": effective.get("type", "context_biasing_not_weight_finetune"),
    }


def profile_sha256(profile: dict | None) -> str:
    """Hash the deterministic, validator-safe effective profile."""
    effective = _digest_payload(profile)
    if effective is None:
        return "baseline"
    return hashlib.sha256(json.dumps(
        effective, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def legacy_profile_sha256(profile: dict | None) -> str:
    """Return the pre-release schema-v2 digest for rollback verification."""
    if profile is None:
        return "baseline"
    return hashlib.sha256(json.dumps(
        profile, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def profile_digest_matches(profile: dict | None, expected: object) -> bool:
    """Accept only the current digest or the exact historical serializer."""
    return expected in {profile_sha256(profile), legacy_profile_sha256(profile)}


def apply_to_soda(recognizer, profile: dict | None, *, context_name: str = "palace-vi-safe-context-v1") -> None:
    if profile is None:
        return
    # Import the binary protobuf interface only at server runtime under WSL/Linux.
    from soda_stt import soda_api_pb2
    profile=native_profile(profile)
    if profile is None:
        return
    conf=soda_api_pb2.ExtendedSodaConfigMsg.FromString(recognizer._config_bytes)
    ctx=conf.recognition_context.context.add()
    if context_name not in ("palace-vi-safe-context-v1", "android-speech-api-generic-phrases"):
        raise ValueError("Unknown context name")
    ctx.name=context_name
    for term in profile["terms"]:
        phrase=ctx.phrases.phrase.add()
        phrase.phrase=term["phrase"]
        phrase.boost=term["boost"]
    recognizer._config_bytes=conf.SerializeToString()
