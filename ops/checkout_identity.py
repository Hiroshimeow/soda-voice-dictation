"""Deterministic identity for source that can affect the local SODA launch.

This is deliberately an allowlist rather than a checkout-wide hash.  Runtime
caches, models, data, and generated files must not change the identity of a
healthy service.  The launcher calls this file directly with the repository
root, so it also works in a Git-free portable extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import re
import stat
from pathlib import Path

# Keep this list aligned with the source/configuration needed by server.py,
# desktop/app.py, and their local launchers.  A new launch-relevant directory
# must be added explicitly instead of silently becoming part of the identity.
_SOURCE_DIRECTORIES = ("desktop", "ops", "training", "vi_enhance", "web")
_ROOT_FILES = {
    "native_session.py",
    "requirements.txt",
    "requirements-desktop.txt",
    "requirements-model-research.txt",
    "run-desktop-windows.bat",
    "run-server-linux.sh",
    "run-server-windows.bat",
    "server.py",
    "soda_languages.py",
}
_SOURCE_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".conf",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".plan",
    ".cache",
    "__pycache__",
    "artifacts",
    "cache",
    "data",
    "engine",
    "langpacks",
    "models",
    "research",
    "samples",
    "soda-cache",
    "SODA_STT_HOME",
    "SODAFiles",
    "SODAModels",
}
_EXCLUDED_SUFFIXES = {
    ".aac",
    ".bin",
    ".crx3",
    ".dll",
    ".flac",
    ".log",
    ".m4a",
    ".mfar",
    ".mp3",
    ".ogg",
    ".pb",
    ".pyc",
    ".pyo",
    ".so",
    ".tflite",
    ".tmp",
    ".temp",
    ".bak",
    ".wav",
    ".zip",
}


def _is_excluded(relative: Path) -> bool:
    for part in relative.parts:
        lowered = part.casefold()
        if lowered in {name.casefold() for name in _EXCLUDED_DIRECTORY_NAMES}:
            return True
        if lowered.startswith(".venv"):
            return True
        if lowered.endswith((".tmp", ".temp", ".bak")):
            return True
    return relative.suffix.casefold() in _EXCLUDED_SUFFIXES


def _is_relevant(relative: Path) -> bool:
    if _is_excluded(relative):
        return False
    if len(relative.parts) == 1:
        return relative.name.casefold() in {name.casefold() for name in _ROOT_FILES}
    return (
        relative.parts[0].casefold()
        in {name.casefold() for name in _SOURCE_DIRECTORIES}
        and relative.suffix.casefold() in _SOURCE_EXTENSIONS
    )


def _entry_stat(entry: os.DirEntry[str]) -> os.stat_result:
    """Read metadata without following links, propagating access failures."""
    return entry.stat(follow_symlinks=False)


def _is_link_or_reparse(entry: os.DirEntry[str], metadata: os.stat_result) -> bool:
    """Return whether *entry* must not be traversed or hashed."""
    # is_symlink() does not follow the link.  On Windows, directory junctions
    # are reparse points but are not necessarily reported as symlinks.
    if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
        return True
    if os.name == "nt":
        return bool(
            metadata.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    return False


def iter_manifest(root: Path) -> list[tuple[str, Path]]:
    """Return safe, deterministic ``(POSIX relative name, file)`` entries.

    This intentionally uses ``scandir`` rather than ``Path.rglob``.  Excluded
    directories are rejected by name before any filesystem query, and links or
    reparse points are rejected before deciding whether to recurse.  An error
    elsewhere is allowed to propagate: silently omitting a launch-relevant
    source file would make the identity unverifiable.
    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Identity root is not a directory: {root}")
    entries: list[tuple[str, Path]] = []
    source_directories = {name.casefold() for name in _SOURCE_DIRECTORIES}
    root_files = {name.casefold() for name in _ROOT_FILES}

    def walk(directory: Path, parent: Path) -> None:
        # A failure here is deliberately not suppressed.  The caller must not
        # certify a checkout when an in-scope source directory is unreadable.
        with os.scandir(directory) as children:
            for child in children:
                relative = parent / child.name
                if _is_excluded(relative):
                    # In particular, do not stat an inaccessible .venv or
                    # runtime directory merely to discover that it is one.
                    continue

                # The allowlist makes unrelated root directories irrelevant;
                # skipping them also avoids probing arbitrary runtime mounts.
                if len(relative.parts) == 1:
                    if relative.name.casefold() not in source_directories and (
                        relative.name.casefold() not in root_files
                    ):
                        continue

                # Fetch metadata only after exclusion and allowlist pruning.
                # An in-scope access failure must not turn into an incomplete
                # manifest that could still pass the health identity gate.
                metadata = _entry_stat(child)
                if _is_link_or_reparse(child, metadata):
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    walk(Path(child.path), relative)
                    continue
                if stat.S_ISREG(metadata.st_mode) and _is_relevant(relative):
                    entries.append((relative.as_posix(), Path(child.path)))

    walk(root, Path())
    entries.sort(key=lambda entry: entry[0].casefold())
    return entries


def canonical_identity_root(root: Path | str) -> str:
    """Return the checkout root in a stable form shared by Windows and WSL.

    Windows drive paths are represented by the corresponding WSL mount path so
    ``E:\\checkout`` and ``/mnt/e/checkout`` bind the same checkout identity.
    The absolute, normalized path remains part of the identity; a copied
    checkout therefore cannot certify a service rooted elsewhere.
    """
    value = str(root).replace("\\", "/")
    drive_match = re.fullmatch(r"([A-Za-z]):(?:/(.*))?", value)
    if drive_match:
        rest = drive_match.group(2) or ""
        value = f"/mnt/{drive_match.group(1).lower()}"
        if rest:
            value += f"/{rest}"
    return posixpath.normpath(value)


def checkout_identity(root: Path) -> str:
    """Hash the canonical absolute root plus names and bytes of the manifest."""
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(b"soda-launch-identity-v1\0")
    digest.update(canonical_identity_root(root).encode("utf-8") + b"\0")
    for relative, path in iter_manifest(root):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args()
    entries = iter_manifest(args.root)
    if args.print_manifest:
        for relative, _ in entries:
            print(relative)
    print(checkout_identity(args.root))


if __name__ == "__main__":
    main()
