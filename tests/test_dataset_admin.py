"""A concurrent intake must survive deletion/purge of earlier recordings."""
from __future__ import annotations
import hashlib
import multiprocessing
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from training import data_admin
from training.dataset import save_sample

ITEM = {"reference": "xin chào Việt Nam", "language": "vi-VN", "consent": True}


def pcm(index):
    return index.to_bytes(2, "little") * 8000


def paused_admin(root, action, sample_id, entered, release):
    root = pathlib.Path(root)
    if action == "delete":
        original = data_admin.atomic_bytes

        def paused_write(path, content):
            entered.set()
            if not release.wait(8):
                raise TimeoutError("Test did not release the delete transaction")
            original(path, content)

        with patch.object(data_admin, "atomic_bytes", side_effect=paused_write):
            data_admin.delete_sample(sample_id, data_dir=root)
    else:
        original = data_admin.shutil.rmtree

        def paused_remove(path):
            entered.set()
            if not release.wait(8):
                raise TimeoutError("Test did not release the purge transaction")
            original(path)

        with patch.object(data_admin.shutil, "rmtree", side_effect=paused_remove):
            data_admin.purge_dataset(data_dir=root)


def append_sample(root, started, completed):
    started.set()
    save_sample(ITEM, pcm(3), data_dir=pathlib.Path(root))
    completed.set()


class DatasetAdminConcurrencyTests(unittest.TestCase):
    def test_append_survives_admin_transaction(self):
        ctx = multiprocessing.get_context("spawn")
        for action in ("delete", "purge"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                first = save_sample(ITEM, pcm(1), data_dir=root)
                save_sample(ITEM, pcm(2), data_dir=root)
                entered, release, started, completed = [ctx.Event() for _ in range(4)]
                admin = ctx.Process(
                    target=paused_admin,
                    args=(td, action, first["sample_id"], entered, release),
                )
                writer = ctx.Process(target=append_sample, args=(td, started, completed))
                try:
                    admin.start()
                    self.assertTrue(entered.wait(5), "Admin did not reach its mutation")
                    writer.start()
                    self.assertTrue(started.wait(5), "Intake process did not start")
                    self.assertFalse(
                        completed.wait(.25),
                        "Intake crossed the lock while admin held an older dataset snapshot",
                    )
                    release.set()
                    admin.join(5)
                    writer.join(5)
                    self.assertEqual(admin.exitcode, 0)
                    self.assertEqual(writer.exitcode, 0)
                    self.assertTrue(completed.is_set())
                finally:
                    release.set()
                    for process in (admin, writer):
                        if process.pid is not None:
                            process.join(1)
                            if process.is_alive():
                                process.terminate()
                                process.join(1)
                expected_indices = (2, 3) if action == "delete" else (3,)
                expected = {hashlib.sha256(pcm(i)).hexdigest() for i in expected_indices}
                rows = data_admin.read_rows(data_dir=root)
                self.assertEqual({row["pcm_sha256"] for row in rows}, expected)
                self.assertEqual({path.stem for path in (root / "audio").glob("*.wav")}, expected)
                self.assertTrue((root / ".dataset.lock").exists())


if __name__ == "__main__":
    unittest.main()
