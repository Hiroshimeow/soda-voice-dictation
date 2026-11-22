"""Behavioral tests for the Windows Handy-like client (no real microphone required)."""

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")
import asyncio
import json
import sys
import unittest
from pathlib import Path
from threading import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.core import DictationBuffer, AudioSample, DictationSession
from desktop.windows import utf16_units, can_inject


class TranscriptTests(unittest.TestCase):
    def test_partial_is_preview_only(self):
        b = DictationBuffer()
        b.partial("Xin chào")
        self.assertEqual(b.preview, "Xin chào")
        self.assertEqual(b.committed, "")
        b.final("Xin chào.")
        b.partial("Các bạn")
        self.assertEqual(b.preview, "Xin chào. Các bạn")
        self.assertEqual(b.committed, "Xin chào.")
        b.final("Các bạn!")
        self.assertEqual(b.committed, "Xin chào. Các bạn!")

    def test_unicode_survives_utf16_input(self):
        s = "Tiếng Việt: xin chào 👋"
        units = utf16_units(s)
        self.assertEqual(b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le"), s)

    def test_insertion_must_match_foreground_and_control(self):
        self.assertTrue(can_inject((100, 200), (100, 200)))
        self.assertFalse(can_inject((100, 200), (101, 200)))
        self.assertFalse(can_inject((100, 200), (100, 201)))
        self.assertFalse(can_inject((0, 0), (0, 0)))
        self.assertFalse(can_inject((100, 0), (100, 999)))

    def test_sample_never_pairs_trailing_audio_to_long_session(self):
        s = AudioSample()
        for _ in range(300):
            s.add(b"\x00" * 3200)
        self.assertEqual(len(s.get_if_eligible()), 960000)
        s.add(b"\x00" * 3200)
        self.assertIsNone(s.get_if_eligible())
        self.assertIsNone(s.get_if_eligible())

    def test_sample_too_short(self):
        s = AudioSample()
        for _ in range(4):
            s.add(b"\x00" * 3200)
        self.assertIsNone(s.get_if_eligible())


class HotkeyTests(unittest.TestCase):
    def test_ctrl_modified_letter_uses_physical_virtual_key(self):
        from pynput import keyboard
        from desktop.app import App
        app = App.__new__(App)
        self.assertEqual(app._key_name(keyboard.KeyCode.from_vk(0x44)), 'd')

    def test_chord_once_and_release_any_modifier(self):
        from pynput import keyboard
        from desktop.app import App
        app = App.__new__(App)
        app.hotkey_key = "space"
        app.key_pressed = set()
        app.hotkey_latched = False
        app.running = False
        events = []
        app._post = events.append
        app._key_down(keyboard.Key.ctrl_l)
        app._key_down(keyboard.Key.alt_l)
        app._key_down(keyboard.Key.space)
        app._key_down(keyboard.Key.space)
        self.assertEqual([e["type"] for e in events], ["hotkey_down"])
        app._key_up(keyboard.Key.ctrl_l)
        self.assertEqual([e["type"] for e in events], ["hotkey_down", "hotkey_up"])

    def test_escape_requests_cancel_without_synthesizing_keys(self):
        from pynput import keyboard
        from desktop.app import App
        app = App.__new__(App)
        app.hotkey_key = "space"
        app.key_pressed = set()
        app.hotkey_latched = False
        app.running = True
        events = []
        app._post = events.append
        app._key_down(keyboard.Key.esc)
        self.assertEqual(events, [{"type": "cancel"}])


class FakeMic:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.opened = False
        self.aborted = False

    def __enter__(self):
        self.opened = True
        return self

    def __exit__(self, *_):
        self.closed = True

    def abort(self):
        self.aborted = True

    def read(self, frames):
        import time
        time.sleep(frames / 16000)
        return b"\x10\x00" * frames, False


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_unavailable_is_actionable_and_no_result(self):
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            unused_port = probe.getsockname()[1]
        events = []
        sess = DictationSession(url=f"ws://127.0.0.1:{unused_port}/v1/stt?lang=vi-VN",
                                microphone=FakeMic, on_event=events.append)
        await sess.run()
        finished = [ev for ev in events if ev["type"] == "complete"]
        self.assertEqual(len(finished), 1)
        self.assertIsNotNone(finished[0]["error"])
        self.assertEqual(finished[0]["text"], "")

    async def test_ready_pcm_stop_final_closed(self):
        from websockets.asyncio.server import serve
        received = []

        async def handler(ws):
            await ws.send(json.dumps({"type": "ready", "locale": "vi-VN"}))
            async for packet in ws:
                received.append(packet)
                if packet == "stop":
                    await ws.send(json.dumps({"type": "final", "text": "Xin chào Việt Nam."}))
                    await ws.send(json.dumps({"type": "closed"}))
                    break

        events = []
        mic = FakeMic()
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            sess = DictationSession(
                url=f"ws://127.0.0.1:{port}/v1/stt?lang=vi-VN",
                microphone=lambda **kwargs: mic,
                on_event=events.append,
            )
            async def stop_soon():
                await asyncio.sleep(.16)
                sess.stop()
            task = asyncio.create_task(stop_soon())
            await sess.run()
            await task
        self.assertIn("stop", received)
        self.assertTrue(any(isinstance(p, bytes) and len(p) == 3200 for p in received))
        complete = [e for e in events if e["type"] == "complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["text"], "Xin chào Việt Nam.")
        self.assertFalse(complete[0]["cancelled"])
        self.assertIsNone(complete[0]["error"])
        self.assertTrue(mic.closed)

    async def test_backend_error_prevents_insertion_result(self):
        from websockets.asyncio.server import serve

        async def handler(ws):
            await ws.send(json.dumps({"type": "ready", "locale": "vi-VN"}))
            await ws.send(json.dumps({"type": "error", "code": "recognition_failed"}))
            await ws.send(json.dumps({"type": "closed"}))

        events = []
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            sess = DictationSession(
                url=f"ws://127.0.0.1:{port}/v1/stt?lang=vi-VN",
                microphone=lambda **kwargs: FakeMic(),
                on_event=events.append,
            )
            await sess.run()
        complete = [e for e in events if e["type"] == "complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["text"], "")
        self.assertIn("recognition_failed", complete[0]["error"])

    async def test_cancel_never_yields_text(self):
        from websockets.asyncio.server import serve
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            try:
                async for packet in ws:
                    if packet == "stop":
                        await ws.send(json.dumps({"type": "final", "text": "must not insert"}))
                        await ws.send(json.dumps({"type": "closed"}))
                        break
            except Exception:
                pass
        events = []
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            sess = DictationSession(
                url=f"ws://127.0.0.1:{port}/v1/stt?lang=vi-VN",
                microphone=lambda **kwargs: FakeMic(),
                on_event=events.append,
            )
            async def cancel_soon():
                await asyncio.sleep(.09)
                sess.cancel()
            task = asyncio.create_task(cancel_soon())
            await sess.run()
            await task
        complete = [e for e in events if e["type"] == "complete"][0]
        self.assertTrue(complete["cancelled"])
        self.assertEqual(complete["text"], "")
        self.assertIsNone(complete["pcm"])


if __name__ == "__main__":
    unittest.main()
