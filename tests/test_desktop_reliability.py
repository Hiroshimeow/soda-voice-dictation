"""Regression cases for interruption, quiet periods, gain, and safe insertion."""

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")
import asyncio
from array import array
import json
import time
import unittest
from threading import Event
from unittest.mock import Mock, patch

from websockets.asyncio.server import serve

from desktop.audio import gain_pcm16
from desktop.core import DictationSession
from desktop.windows import dictation_text, insert_after_release
from tests.test_desktop import FakeMic


class GainAndInsertionTests(unittest.TestCase):
    def test_zero_gain_is_exact_and_clipping_is_detected(self):
        pcm = array("h", [0, 20, -20, 32000, -32000]).tobytes()
        self.assertEqual(gain_pcm16(pcm)[0], pcm)
        raised, levels = gain_pcm16(pcm, 12)
        values = array("h")
        values.frombytes(raised)
        self.assertEqual(values[-2:], array("h", [32767, -32768]))
        self.assertEqual(levels["clipped_samples"], 2)
        with self.assertRaises(ValueError):
            gain_pcm16(pcm, float("nan"))

    def test_automatic_insertion_never_contains_control_characters(self):
        self.assertEqual(dictation_text("xin\nchào\r\t\x1b\x85\u2028Việt 👋"),
                         "xin chào     Việt 👋")

    def test_pending_insertion_can_cancel_before_touching_windows(self):
        cancelled = Event()
        cancelled.set()
        with patch("desktop.windows.u32.SendInput") as send:
            with self.assertRaises(RuntimeError):
                insert_after_release("must not type", (0, 0), cancel=cancelled)
            send.assert_not_called()

    def test_new_session_cannot_start_during_insertion(self):
        from desktop.app import App
        app = App.__new__(App)
        app.running = False
        app.injecting = True
        app.status = Mock()
        with patch("desktop.app.DictationSession") as session:
            app.start((1, 2), from_hotkey=True)
            session.assert_not_called()

    def test_escape_cancels_pending_insertion(self):
        from desktop.app import App
        app = App.__new__(App)
        app.running = False
        app.injecting = True
        app.session = None
        app.status = Mock()
        app.insertion_cancel = Event()
        app.cancel()
        self.assertTrue(app.insertion_cancel.is_set())


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_silence_does_not_use_finalization_deadline(self):
        packets = []
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            async for packet in ws:
                if packet == "stop":
                    await ws.send(json.dumps({"type": "final", "text": "Kết quả."}))
                    await ws.send(json.dumps({"type": "closed"}))
                    break
                packets.append(packet)
        events = []
        mic = FakeMic()
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = DictationSession(url=f"ws://127.0.0.1:{port}", microphone=lambda **_: mic,
                                       on_event=events.append, final_timeout=.04)
            task = asyncio.create_task(session.run())
            await asyncio.sleep(.20)
            self.assertFalse(task.done(), "Quiet recording must remain active")
            session.stop()
            await asyncio.wait_for(task, 1)
        self.assertTrue(mic.closed)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertEqual(finished["text"], "Kết quả.")
        self.assertIsNone(finished["error"])

    async def test_gain_stream_and_saved_pcm_are_identical(self):
        packets = []
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            async for packet in ws:
                if packet == "stop":
                    await ws.send(json.dumps({"type": "final", "text": "Tiếng Việt"}))
                    await ws.send(json.dumps({"type": "closed"}))
                    break
                packets.append(packet)
        events = []
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=FakeMic, on_event=events.append, gain_db=6.020599913)
            task = asyncio.create_task(session.run())
            await asyncio.sleep(.55)
            session.stop()
            await task
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertEqual(finished["pcm"], b"".join(packets))
        self.assertEqual(packets[0][:2], b"\x20\x00")
        levels = [e for e in events if e["type"] == "level"]
        self.assertAlmostEqual(levels[0]["rms_dbfs"] - levels[0]["input_dbfs"], 6.02, places=2)

    async def test_cancel_is_prompt_while_waiting_ready(self):
        async def handler(ws):
            await ws.wait_closed()
        events = []
        mic = FakeMic()
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=lambda **_: mic, on_event=events.append)
            task = asyncio.create_task(session.run())
            await asyncio.sleep(.08)
            self.assertTrue(mic.opened, "Capture must start before backend ready")
            start = time.monotonic()
            session.cancel()
            await asyncio.wait_for(task, .8)
            self.assertLess(time.monotonic() - start, .5)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertTrue(finished["cancelled"])
        self.assertIsNone(finished["pcm"])
        self.assertTrue(mic.closed)
        self.assertTrue(mic.aborted)

    async def test_cancel_is_prompt_while_waiting_final(self):
        stopped = asyncio.Event()
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            async for packet in ws:
                if packet == "stop":
                    stopped.set()
                    await ws.wait_closed()
                    break
        events = []
        mic = FakeMic()
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=lambda **_: mic, on_event=events.append)
            task = asyncio.create_task(session.run())
            await asyncio.sleep(.08)
            session.stop()
            await asyncio.wait_for(stopped.wait(), 1)
            session.cancel()
            await asyncio.wait_for(task, .8)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertTrue(finished["cancelled"])
        self.assertEqual(finished["text"], "")
        self.assertTrue(mic.closed)

    async def test_final_timeout_keeps_preview_without_insertable_result(self):
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            await ws.send(json.dumps({"type": "final", "text": "Bản chép chưa hoàn tất"}))
            async for packet in ws:
                if packet == "stop":
                    await ws.wait_closed()
                    break
        events = []
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=FakeMic, on_event=events.append, final_timeout=.05)
            task = asyncio.create_task(session.run())
            await asyncio.sleep(.08)
            session.stop()
            await asyncio.wait_for(task, 1)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertEqual(finished["text"], "")
        self.assertEqual(finished["preview"], "Bản chép chưa hoàn tất")
        self.assertIsNotNone(finished["error"])
        self.assertIsNone(finished["pcm"])

    async def test_microphone_failure_never_announces_listening(self):
        def unavailable(**_):
            raise RuntimeError("device disconnected")
        async def handler(ws):
            await ws.send(json.dumps({"type": "ready"}))
            await ws.wait_closed()
        events = []
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=unavailable, on_event=events.append)
            await session.run()
        self.assertFalse(any("Đang nghe" in e.get("text", "") for e in events))
        self.assertIsNotNone([e for e in events if e["type"] == "complete"][0]["error"])


class StartupBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_ready_keeps_every_frame_and_stop_closes_mic_first(self):
        loop = asyncio.get_running_loop()
        closed = asyncio.Event()
        packets, received_at, events = [], [], []
        holder = {}

        class SequenceMic(FakeMic):
            def __init__(self):
                super().__init__()
                self.frames = []

            def read(self, frames):
                time.sleep(frames / 16000)
                chunk = (len(self.frames) + 1).to_bytes(2, "little") * frames
                self.frames.append(chunk)
                if len(self.frames) == 5:
                    holder["session"].stop()
                return chunk, False

            def __exit__(self, *args):
                super().__exit__(*args)
                loop.call_soon_threadsafe(closed.set)

        mic = SequenceMic()
        async def handler(ws):
            # The server cannot become ready until Stop has closed capture.
            # All five distinct frames must survive this startup delay.
            await asyncio.wait_for(closed.wait(), 2)
            await ws.send(json.dumps({"type": "ready"}))
            async for packet in ws:
                if packet == "stop":
                    await ws.send(json.dumps({"type": "final", "text": "Giữ nguyên đầu câu."}))
                    await ws.send(json.dumps({"type": "closed"}))
                    break
                received_at.append(time.perf_counter())
                packets.append(packet)

        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=lambda **_: mic, on_event=events.append)
            holder["session"] = session
            await asyncio.wait_for(session.run(), 3)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertIsNone(finished["error"])
        self.assertTrue(mic.closed)
        self.assertEqual(len(mic.frames), 5)
        self.assertEqual(packets, mic.frames, "No initial frame may be lost, duplicated or reordered")
        self.assertEqual(finished["pcm"], b"".join(mic.frames))
        self.assertEqual(finished["text"], "Giữ nguyên đầu câu.")
        self.assertTrue(all(b - a >= .045 for a, b in zip(received_at, received_at[1:])),
                        "Buffered audio must not exceed the 2x catch-up cadence")
        capture_index = next(i for i, e in enumerate(events) if e.get("capture_ready"))
        backend_index = next(i for i, e in enumerate(events) if e.get("backend_ready"))
        self.assertLess(capture_index, backend_index)

    async def test_full_startup_buffer_fails_explicitly_without_partial_audio_sample(self):
        class FastMic(FakeMic):
            def read(self, frames):
                return bytes((16, 0)) * frames, False
        async def handler(ws):
            await ws.wait_closed()
        events = []
        mic = FastMic()
        async with serve(handler, "127.0.0.1", 0) as server:
            session = DictationSession(url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                                       microphone=lambda **_: mic, on_event=events.append)
            await asyncio.wait_for(session.run(), 2)
        finished = [e for e in events if e["type"] == "complete"][0]
        self.assertIn("audio buffer filled up", finished["error"])
        self.assertEqual(finished["text"], "")
        self.assertIsNone(finished["pcm"])
        self.assertFalse(finished["cancelled"])
        self.assertTrue(mic.closed)


if __name__ == "__main__":
    unittest.main()
