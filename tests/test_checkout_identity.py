import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.checkout_identity import canonical_identity_root, checkout_identity, iter_manifest


class CheckoutIdentityTests(unittest.TestCase):
    def make_checkout(self, root: Path) -> None:
        (root / "server.py").write_text("server = 1\n", encoding="utf-8")
        (root / "desktop" / "app.py").parent.mkdir()
        (root / "desktop" / "app.py").write_text("app = 1\n", encoding="utf-8")
        (root / "ops" / "start-backend.ps1").parent.mkdir()
        (root / "ops" / "start-backend.ps1").write_text("# launcher\n", encoding="utf-8")

    def test_manifest_is_stable_and_excludes_runtime_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="soda identity with spaces ") as name:
            root = Path(name)
            self.make_checkout(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "server.cpython-312.pyc").write_bytes(b"cache")
            (root / ".venv-wsl" / "bin").mkdir(parents=True)
            (root / ".venv-wsl" / "bin" / "python").write_bytes(b"runtime")
            (root / "data").mkdir()
            (root / "data" / "generated.json").write_text("data", encoding="utf-8")
            (root / "server.py.tmp").write_text("temp", encoding="utf-8")
            before = checkout_identity(root)
            names = [relative for relative, _ in iter_manifest(root)]
            self.assertEqual(names, ["desktop/app.py", "ops/start-backend.ps1", "server.py"])

            (root / "__pycache__" / "server.cpython-312.pyc").write_bytes(b"different cache")
            (root / "data" / "generated.json").write_text("different data", encoding="utf-8")
            self.assertEqual(checkout_identity(root), before)

            (root / "server.py").write_text("server = 2\n", encoding="utf-8")
            self.assertNotEqual(checkout_identity(root), before)

    def test_excluded_runtime_directory_is_not_probed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_checkout(root)
            runtime = root / ".venv-wsl"
            runtime.mkdir()

            real_scandir = os.scandir

            def scandir_without_runtime(path):
                if Path(path) == runtime:
                    raise AssertionError("excluded runtime directory was probed")
                return real_scandir(path)

            with patch("ops.checkout_identity.os.scandir", side_effect=scandir_without_runtime):
                self.assertEqual(
                    [relative for relative, _ in iter_manifest(root)],
                    ["desktop/app.py", "ops/start-backend.ps1", "server.py"],
                )

    def test_directory_symlink_is_not_traversed(self):
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as target_name:
            root, target = Path(name), Path(target_name)
            self.make_checkout(root)
            (target / "linked.py").write_text("must not be hashed\n", encoding="utf-8")
            link = root / "desktop" / "linked-runtime"
            try:
                os.symlink(target, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            self.assertEqual(
                [relative for relative, _ in iter_manifest(root)],
                ["desktop/app.py", "ops/start-backend.ps1", "server.py"],
            )

    def test_relevant_source_directory_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_checkout(root)
            real_scandir = os.scandir
            source_directory = root / "desktop"

            def scandir_with_source_failure(path):
                if Path(path) == source_directory:
                    raise OSError("source directory is inaccessible")
                return real_scandir(path)

            with patch("ops.checkout_identity.os.scandir", side_effect=scandir_with_source_failure):
                with self.assertRaisesRegex(OSError, "source directory is inaccessible"):
                    iter_manifest(root)

    def test_relevant_file_read_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.make_checkout(root)
            relevant = root / "server.py"
            real_read_bytes = Path.read_bytes

            def read_bytes_with_failure(path):
                if path == relevant:
                    raise OSError("relevant source file is inaccessible")
                return real_read_bytes(path)

            with patch.object(Path, "read_bytes", read_bytes_with_failure):
                with self.assertRaisesRegex(OSError, "relevant source file is inaccessible"):
                    checkout_identity(root)

    def test_windows_and_wsl_roots_have_same_canonical_identity_root(self):
        self.assertEqual(
            canonical_identity_root(r"E:\git-project\google-soda-native-stt"),
            canonical_identity_root("/mnt/e/git-project/google-soda-native-stt"),
        )
        self.assertEqual(
            canonical_identity_root("e:\\git-project\\google-soda-native-stt\\"),
            "/mnt/e/git-project/google-soda-native-stt",
        )

    def test_absolute_root_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root, second_root = Path(first), Path(second)
            self.make_checkout(first_root)
            self.make_checkout(second_root)
            self.assertNotEqual(checkout_identity(first_root), checkout_identity(second_root))


if __name__ == "__main__":
    unittest.main()
