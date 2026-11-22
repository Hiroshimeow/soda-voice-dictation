"""Local dataset governance: status, delete one sample, or purge all.

No transcript or audio is printed. Mutations share the intake process lock.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from training.context_profile import DATA
from training.storage import atomic_bytes, exclusive_lock

MANIFEST = DATA / "eval_manifest.jsonl"
AUDIO = DATA / "audio"


def read_rows(*, data_dir: pathlib.Path = DATA) -> list[dict]:
    manifest = data_dir / "eval_manifest.jsonl"
    if not manifest.exists():
        return []
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_rows(rows: list[dict], *, data_dir: pathlib.Path) -> None:
    """Caller holds the shared dataset lock across read, write and audio removal."""
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_bytes(data_dir / "eval_manifest.jsonl", content.encode("utf-8"))


def delete_sample(sample_id: str, *, data_dir: pathlib.Path = DATA) -> None:
    if len(sample_id) != 12 or any(ch not in "0123456789abcdef" for ch in sample_id):
        raise ValueError("--id must be a 12-character sample ID")
    with exclusive_lock(data_dir):
        # Read only after acquiring the same lock used by save_sample.
        rows = read_rows(data_dir=data_dir)
        matched = [row for row in rows if row["audio"].startswith(sample_id)]
        if not matched:
            raise ValueError("Sample not found")
        if len(matched) != 1:
            raise ValueError("Ambiguous prefix")
        audio_dir = data_dir / "audio"
        target = audio_dir / matched[0]["audio"]
        if not target.resolve().is_relative_to(audio_dir.resolve()):
            raise ValueError("Invalid stored path")
        keep = [row for row in rows if not row["audio"].startswith(sample_id)]
        _write_rows(keep, data_dir=data_dir)
        target.unlink(missing_ok=True)


def purge_dataset(*, data_dir: pathlib.Path = DATA) -> None:
    with exclusive_lock(data_dir):
        # Keep the lock file and directory intact. Intake may proceed after release.
        # Never purge source Palace; only this STT dataset is owned here.
        (data_dir / "eval_manifest.jsonl").unlink(missing_ok=True)
        audio_dir = data_dir / "audio"
        if audio_dir.exists():
            shutil.rmtree(audio_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "delete", "purge"])
    parser.add_argument("--id", default="")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        with exclusive_lock(DATA):
            count = len(read_rows())
        print(json.dumps({"saved": count, "local_storage": str(DATA)}, ensure_ascii=False))
        return
    if not args.confirm:
        raise SystemExit("Explicit --confirm required")
    try:
        if args.action == "delete":
            delete_sample(args.id)
            print("Deleted exactly one sample")
        else:
            purge_dataset()
            print("Purged local STT labeled dataset")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
