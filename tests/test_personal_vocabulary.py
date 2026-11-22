"""Terminology and hold shortcut behavior for SODA Dictation."""
import sys
import tempfile
import unittest
from pathlib import Path

from training.personal_vocabulary import (
    validate_vocabulary, save_vocabulary, load_vocabulary,
    to_soda_profile, apply_replacements, merge_profile,
)

class VocabularyTests(unittest.TestCase):
    def test_default_technical_terms_and_soda_profile(self):
        raw = {"schema_version": 1, "language": "vi-VN",
               "phrases": ["CDPA", "G8", "G6"], "replacements": []}
        profile = to_soda_profile(raw)
        self.assertEqual([t["phrase"] for t in profile["terms"]], ["CDPA", "G8", "G6"])
        self.assertTrue(all(t["boost"] == 1 for t in profile["terms"]))

    def test_save_load_and_whole_term_replacement(self):
        data = {"schema_version": 1, "language": "vi-VN", "phrases": ["CDPA"],
                "replacements": [{"heard": "xi đi pi ây", "write": "CDPA"},
                                 {"heard": "gờ tám", "write": "G8"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            save_vocabulary(data, path)
            self.assertEqual(load_vocabulary(path), data)
        self.assertEqual(apply_replacements("Tôi dùng xi đi pi ây và gờ tám.", data),
                         "Tôi dùng CDPA và G8.")
        self.assertEqual(apply_replacements("prefixi đi pi ây và gờ tám lần đầu.", data),
                         "prefixi đi pi ây và G8 lần đầu.")

    def test_invalid_terms_rejected(self):
        raw = {"schema_version":1, "language":"vi-VN", "phrases":["CDPA", "cdpa"],"replacements":[]}
        with self.assertRaises(ValueError):
            validate_vocabulary(raw)
        raw["phrases"]=["https://example.com"]
        with self.assertRaises(ValueError):
            validate_vocabulary(raw)
        raw["phrases"]=list(map(str, range(100)))
        with self.assertRaises(ValueError):
            validate_vocabulary(raw)

    def test_merge_prefers_personal_and_caps_at_soda_safe_limit(self):
        curated = {"schema_version":1,"language":"vi-VN",
                   "terms":[{"phrase":f"cụm {i}", "boost":1.1} for i in range(64)]}
        user = to_soda_profile({"schema_version":1,"language":"vi-VN",
                               "phrases":["CDPA","G8","G6"],"replacements":[]})
        result=merge_profile(curated,user)
        self.assertEqual(len(result["terms"]),40)
        self.assertEqual([x["phrase"] for x in result["terms"][:3]],["CDPA","G8","G6"])

@unittest.skipUnless(sys.platform == "win32", "Windows-only keyboard tests")
class ShortcutTests(unittest.TestCase):
    def test_ctrl_space_hold_exactly_once(self):
        from pynput import keyboard
        from desktop.app import App
        app=App.__new__(App)
        app.hotkey_key="space"
        app.hotkey_modifiers={"ctrl"}
        app.key_pressed=set()
        app.hotkey_latched=False
        app.running=False
        app._post=lambda e: events.append(e["type"])
        events=[]
        app._key_down(keyboard.Key.ctrl_l)
        app._key_down(keyboard.Key.space)
        app._key_down(keyboard.Key.space)
        self.assertEqual(events,["hotkey_down"])
        app._key_up(keyboard.Key.space)
        self.assertEqual(events,["hotkey_down","hotkey_up"])
        app._key_up(keyboard.Key.ctrl_l)
        self.assertEqual(events,["hotkey_down","hotkey_up"])

if __name__ == "__main__":
    unittest.main()
