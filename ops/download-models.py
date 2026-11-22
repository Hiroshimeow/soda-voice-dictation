"""Download/cache the SODA engine and all desktop-supported language packs."""
from soda_languages import LANGUAGES

def main() -> None:
    from soda_stt import _config
    from soda_stt.download import ensure_engine, ensure_language_pack
    _config.LANGUAGE_PACK_IDS.update(
        {locale: cfg["component_id"] for locale, cfg in LANGUAGES.items()}
    )
    print("Resolving SODA engine...")
    print(ensure_engine())
    for locale, cfg in LANGUAGES.items():
        print(f"Resolving {cfg['label']} ({locale})...")
        print(ensure_language_pack(locale))
    print("SODA engine and all supported language packs are ready.")

if __name__ == "__main__":
    main()
