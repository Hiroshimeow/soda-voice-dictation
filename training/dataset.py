"""Explicit local dataset intake with persistent, content-based partitions.

Recognition itself stores no audio. A separate request must contain consent
and a user-corrected transcript before this module saves a recording.
"""
from __future__ import annotations
import hashlib, io, json, threading, wave
from datetime import datetime, timezone
from pathlib import Path
from .context_profile import DATA
from .storage import exclusive_lock, stable_split, atomic_bytes

_LOCK = threading.Lock()
MAX_SECONDS = 30
RATE = 16000
MAX_PCM_BYTES = RATE * 2 * MAX_SECONDS
MANIFEST = DATA / "eval_manifest.jsonl"
AUDIO = DATA / "audio"

def validate_sample(item: dict, *, pcm_bytes: bytes) -> str:
    if not isinstance(item, dict) or item.get("consent") is not True:
        raise ValueError("Explicit consent required")
    if item.get("language") != "vi-VN":
        raise ValueError("Only vi-VN labeled recordings allowed")
    reference = item.get("reference")
    if not isinstance(reference, str) or not 2 <= len(reference.strip()) <= 3000:
        raise ValueError("A corrected transcript is required")
    if len(pcm_bytes) % 2 or not RATE <= len(pcm_bytes) <= MAX_PCM_BYTES:
        raise ValueError("Expected 0.5-30s of 16kHz mono signed PCM16")
    return reference.strip()

def save_sample(item: dict, pcm_bytes: bytes, *, data_dir: Path = DATA) -> dict:
    ref = validate_sample(item, pcm_bytes=pcm_bytes)
    digest = hashlib.sha256(pcm_bytes).hexdigest()
    name = digest + ".wav"
    audio_dir = data_dir / "audio"
    manifest = data_dir / "eval_manifest.jsonl"
    with _LOCK, exclusive_lock(data_dir):
        existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
        if len(existing) > 20_000_000:
            raise ValueError("Dataset maximum size reached")
        rows = [json.loads(line) for line in existing.splitlines() if line.strip()]
        if any(row.get("audio") == name or row.get("pcm_sha256") == digest for row in rows):
            raise ValueError("Duplicate audio recording")
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = audio_dir / name
        if path.exists():
            # Recover a crash after the WAV rename but before the manifest rename.
            with wave.open(str(path), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
                    raise ValueError("Orphan WAV format mismatch")
                if w.readframes(w.getnframes()) != pcm_bytes:
                    raise ValueError("Orphan WAV content mismatch")
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm_bytes)
            atomic_bytes(path, buf.getvalue())
        row = {
            "audio": name, "reference": ref, "schema_version": 2,
            "pcm_sha256": digest, "split": stable_split(digest),
            "purpose": "personalization", "language": "vi-VN", "consent": True,
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        separator = "" if not existing or existing.endswith("\n") else "\n"
        atomic_bytes(manifest, (existing + separator + json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        return {"stored": True, "sample_id": digest[:12], "count": len(rows) + 1}

def dataset_count(*, data_dir: Path = DATA) -> int:
    path = data_dir / "eval_manifest.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
