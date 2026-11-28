"""Prepare the SODA engine and all configured language packs for the backend.

The repository copy of the Vietnamese pack is optional: when it is absent,
``ensure_language_pack("vi-VN")`` downloads it just like every other pack.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from soda_languages import LANGUAGES

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_VI_VERSION = "1.3071.0"
BUNDLED_VI = (
    REPO_ROOT / "third_party" / "google-soda" / "vi-VN" / BUNDLED_VI_VERSION
)


def _cache_root() -> Path:
    configured = os.environ.get("SODA_STT_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "soda-stt"


def seed_bundled_vietnamese() -> Path | None:
    """Optionally seed the cache from the source checkout's Vietnamese pack.

    Returning ``None`` is intentional.  The caller always invokes
    ``ensure_language_pack`` for every configured locale, so a lightweight
    checkout without model payloads still downloads Vietnamese on first run.
    """
    if not (BUNDLED_VI / "SODAModels").is_dir():
        return None

    dest = _cache_root() / "langpacks" / "vi-VN" / BUNDLED_VI_VERSION
    if (dest / "SODAModels").is_dir():
        return dest / "SODAModels"

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(temp, ignore_errors=True)
    print(f"Seeding bundled Vietnamese model {BUNDLED_VI_VERSION} into local SODA cache...")
    shutil.copytree(BUNDLED_VI, temp)
    if not (temp / "SODAModels" / "endtoendmodel" / "medium-encoder.tflite").is_file():
        shutil.rmtree(temp, ignore_errors=True)
        raise RuntimeError("Bundled Vietnamese model is incomplete")
    temp.replace(dest)
    return dest / "SODAModels"


def main() -> None:
    from soda_stt import _config
    from soda_stt.download import ensure_engine, ensure_language_pack

    _config.LANGUAGE_PACK_IDS.update(
        {locale: cfg["component_id"] for locale, cfg in LANGUAGES.items()}
    )

    seeded = seed_bundled_vietnamese()
    if seeded:
        print(f"Bundled Vietnamese model ready: {seeded}")

    print("Resolving SODA engine...")
    print(ensure_engine())
    for locale, cfg in LANGUAGES.items():
        print(f"Resolving {cfg['label']} ({locale})...")
        print(ensure_language_pack(locale))
    print("SODA engine and all supported language packs are ready.")


if __name__ == "__main__":
    main()
