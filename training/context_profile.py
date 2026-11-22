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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROFILE_PATH = DATA / "palace_vi_bias.json"
ACTIVE_PATH = DATA / "active_bias.json"
MAX_TERMS = 64
MAX_BOOST = 3.5
# Limit to human reviewable, curated phrases. Never accept arbitrary urls/paths.
REJECT = re.compile(r"(?i)(https?://|[/\\]|@|\b(?:sk-|ghp_|token|password|secret|api[_ -]?key)\b|\b\d{4,}\b)")


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
        if not isinstance(phrase,str) or not 2 <= len(phrase) <= 75:
            raise ValueError("Invalid phrase length")
        if phrase != phrase.strip() or REJECT.search(phrase):
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


def profile_sha256(profile: dict | None) -> str:
    if profile is None:
        return "baseline"
    return hashlib.sha256(json.dumps(profile,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def apply_to_soda(recognizer, profile: dict | None, *, context_name: str = "palace-vi-safe-context-v1") -> None:
    if profile is None:
        return
    # Import the binary protobuf interface only at server runtime under WSL/Linux.
    from soda_stt import soda_api_pb2
    profile=validate_profile(profile)
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
