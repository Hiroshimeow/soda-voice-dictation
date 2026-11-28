"""Privacy, schema, and release-gate regression tests. No live Palace needed."""
import io
import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.context_profile import load_profile, native_profile, validate_profile, write_atomic
from training.dataset import save_sample, dataset_count
from training.calibrate import normalized,rate,distance


class Tests(unittest.TestCase):
    def test_legacy_digit_profile_loads_and_native_subset_is_deterministic(self):
        raw = {"schema_version": 1, "language": "vi-VN",
               "terms": [{"phrase": "WSL2", "boost": 2.0},
                         {"phrase": "FastAPI", "boost": 1.5}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "active.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_profile(path)["terms"], raw["terms"])
        self.assertEqual([term["phrase"] for term in native_profile(raw)["terms"]], ["FastAPI"])

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI runtime dependency unavailable")
    def test_status_accepts_legacy_profile_and_reports_effective_terms(self):
        import server
        raw = {"schema_version": 1, "language": "vi-VN",
               "terms": [{"phrase": "WSL2", "boost": 2.0},
                         {"phrase": "FastAPI", "boost": 1.5}]}
        vocabulary = {"schema_version": 1, "language": "vi-VN",
                      "phrases": [], "replacements": []}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "active.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with patch.object(server, "BIAS_FILE", path), patch.object(server, "load_vocabulary", return_value=vocabulary):
                status = server.adaptation_status()
        self.assertEqual(status["phrase_count"], 1)
        self.assertTrue(status["enabled"])

    def test_profile_rejects_secrets(self):
        base={"schema_version":1,"language":"vi-VN","terms":[{"phrase":"MCP Device","boost":2.0}]}
        self.assertEqual(len(validate_profile(base)["terms"]),1)
        for phrase in ("http://secret","user@example.com","C:\\Users\\foo","api_key"):
            item={**base,"terms":[{"phrase":phrase,"boost":2.0}]}
            with self.assertRaises(ValueError):
                validate_profile(item)
        for bad in (-1,4.0,float("nan")):
            with self.assertRaises(ValueError):
                validate_profile({**base,"terms":[{"phrase":"Kubernetes","boost":bad}]})

    def test_profile_requires_unique(self):
        with self.assertRaises(ValueError):
            validate_profile({"schema_version":1,"language":"vi-VN",
                             "terms":[{"phrase":"MCP","boost":2},
                                      {"phrase":"mcp","boost":2}]})

    def test_private_consent_and_duplicate(self):
        pcm=b"\0\0"*8000
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            item={"reference":"Xin chào Việt Nam", "language":"vi-VN", "consent":False}
            with self.assertRaises(ValueError):
                save_sample(item,pcm,data_dir=folder)
            self.assertEqual(dataset_count(data_dir=folder),0)
            item["consent"]=True
            stored=save_sample(item,pcm,data_dir=folder)
            self.assertTrue(stored["stored"])
            self.assertEqual(stored["count"],1)
            with self.assertRaises(ValueError):
                save_sample(item,pcm,data_dir=folder)
            self.assertEqual(dataset_count(data_dir=folder),1)
            path=next((folder/"audio").glob("*.wav"))
            with wave.open(str(path),"rb") as w:
                self.assertEqual((w.getframerate(),w.getnchannels(),w.getsampwidth()),(16000,1,2))

    def test_wer_and_cer(self):
        self.assertEqual(rate("Xin chào", "xin chào","word"),(0,2))
        self.assertEqual(rate("Xin chào","xin","word"),(1,2))
        self.assertEqual(rate("Xin chào","xin","char"),(4,7))
        self.assertEqual(distance(list("abc"),list("ac")) ,1)

    def test_atomic_file(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"foo.json"
            write_atomic(target,{"a":1})
            self.assertEqual(json.loads(target.read_text())["a"],1)


if __name__=="__main__":
    unittest.main()
