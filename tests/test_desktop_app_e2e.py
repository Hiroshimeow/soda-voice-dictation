"""Windows acceptance: real WAV -> client session -> SODA -> app -> owned edit.

Only the microphone boundary is supplied by a private WAV. Recognition, app
completion, cancellation, and Windows Unicode insertion are the production paths.
No user-facing app is used as an injection target.
"""
from __future__ import annotations

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")

import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import threading
import time
import tkinter as tk
import wave
from unittest.mock import patch

from PIL import ImageGrab

from desktop.app import App
from desktop.core import DictationSession
from desktop.windows import (INPUT, MOUSEINPUT, RECT, can_inject, dictation_text,
                             focused_target, u32, window_process_id)

WAV = Path(os.environ.get("SODA_TEST_WAV", str(Path(__file__).resolve().parents[1] / "samples" / "vi_technology_02.wav")))
GAIN = float(os.environ.get("SODA_TEST_GAIN", "22"))
UI_ONLY = os.environ.get("SODA_UI_ONLY") == "1"
OUT = Path("data/desktop_e2e")
OUT.mkdir(parents=True, exist_ok=True)


class FileMicrophone:
    def __init__(self):
        self.eof = threading.Event()
        self.closed = False
        self.aborted = False
        self.blocks = 0

    def __enter__(self):
        self.source = wave.open(str(WAV), "rb")
        assert (self.source.getframerate(), self.source.getnchannels(),
                self.source.getsampwidth()) == (16000, 1, 2)
        self.original_seconds = self.source.getnframes() / 16000
        self.next_at = time.perf_counter()
        return self

    def read(self, frames):
        self.next_at += frames / 16000
        time.sleep(max(0, self.next_at - time.perf_counter()))
        if self.aborted:
            raise RuntimeError("file microphone aborted")
        data = self.source.readframes(frames)
        if len(data) < frames * 2:
            self.eof.set()
            data += b"\0" * (frames * 2 - len(data))
        self.blocks += 1
        return data, False

    def abort(self):
        self.aborted = True

    def __exit__(self, *_):
        self.source.close()
        self.closed = True


def main():
    app = App()
    configured_gain = app.gain_var.get()
    microphone = FileMicrophone()
    root = app.root
    root.title("SODA Dictation · Kiểm tra cửa sổ riêng")
    target_window = tk.Toplevel(root)
    target_window.title("SODA — cửa sổ nhập thử thuộc bài kiểm tra")
    target_window.geometry("680x320+30+100")
    target_window.attributes("-topmost", True)
    field = tk.Text(target_window, height=6, wrap="word")
    field.pack(fill="both", expand=True, padx=12, pady=12)
    other_window = tk.Toplevel(root)
    other_window.title("SODA — cửa sổ kiểm tra đổi focus")
    other_window.geometry("680x150+30+470")
    other_window.attributes("-topmost", True)
    second = tk.Text(other_window, height=3)
    second.pack(fill="both", expand=True, padx=12, pady=12)
    result = {"passed": False, "mode": "ui_only" if UI_ONLY else "real_soda",
              "real_soda_used": not UI_ONLY}
    phase = "initial"
    started = time.perf_counter()
    stop_sent = False
    modifiers_patch = None
    original = None
    captured_text = ""
    generation = 0
    settle_started = None
    completion = {}
    last_transcripts = []
    timings = {}
    session_started = [None]

    def owned_focus(widget):
        from ctypes import wintypes
        window = widget.winfo_toplevel()
        window.lift()
        root.update()
        u32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        u32.GetAncestor.restype = wintypes.HWND
        u32.WindowFromPoint.argtypes = [wintypes.POINT]
        u32.WindowFromPoint.restype = wintypes.HWND
        u32.SetForegroundWindow.argtypes = [wintypes.HWND]
        u32.SetForegroundWindow.restype = wintypes.BOOL
        hwnd = u32.GetAncestor(window.winfo_id(), 2)
        assert window_process_id(hwnd) == os.getpid()
        u32.SetForegroundWindow(hwnd)
        root.update()
        old_cursor = None
        click_point = None
        if focused_target()[0] != hwnd:
            # Windows may deny a background process foreground activation.
            # Click only our own visible titlebar as an ordinary interactive action.
            # No foreground-lock policy, synthetic modifiers, or foreign windows.
            assert not (u32.GetAsyncKeyState(1) & 0x8000), "Mouse button is held"
            rect = RECT()
            u32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
            click_point = wintypes.POINT(rect.left + 70, rect.top + 15)
            under_point = u32.WindowFromPoint(click_point)
            assert window_process_id(under_point) == os.getpid(), "Owned titlebar is obscured"
            old_cursor = wintypes.POINT()
            u32.GetCursorPos(ctypes.byref(old_cursor))
            u32.SetCursorPos(click_point.x, click_point.y)
            inputs = (INPUT * 2)()
            inputs[0].type = inputs[1].type = 0
            inputs[0].payload.mi = MOUSEINPUT(0, 0, 0, 0x0002, 0, 0)
            inputs[1].payload.mi = MOUSEINPUT(0, 0, 0, 0x0004, 0, 0)
            assert u32.SendInput(2, inputs, ctypes.sizeof(INPUT)) == 2
        for _ in range(15):
            root.update()
            if focused_target()[0] == hwnd:
                break
            time.sleep(.02)
        assert focused_target()[0] == hwnd, "Windows did not activate our owned window"
        if old_cursor is not None:
            current_cursor = wintypes.POINT()
            u32.GetCursorPos(ctypes.byref(current_cursor))
            if (current_cursor.x, current_cursor.y) == (click_point.x, click_point.y):
                u32.SetCursorPos(old_cursor.x, old_cursor.y)
        widget.focus_force()
        root.update()
        target = focused_target()
        assert target[0] and target[1], "No usable interactive focus"
        assert window_process_id(target[0]) == os.getpid(), "Refuse foreign-process target"
        assert window_process_id(target[1]) == os.getpid(), "Refuse foreign focused control"
        assert root.focus_get() == widget, "The owned Tk edit field did not receive focus"
        return target

    def finish(error=None):
        nonlocal modifiers_patch
        if modifiers_patch:
            modifiers_patch.stop()
            modifiers_patch = None
        if error is not None:
            result["error"] = str(error)
        if result.get("passed"):
            app.gain_var.set(configured_gain)
            app._set_text("")
            app.status.set(f"Sẵn sàng • {app.hotkey_var.get()}")
            root.update()
            from ctypes import wintypes
            u32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            u32.GetAncestor.restype = ctypes.c_void_p
            hwnd = u32.GetAncestor(root.winfo_id(), 2)
            rect, origin = RECT(), wintypes.POINT(0, 0)
            u32.GetClientRect(ctypes.c_void_p(hwnd), ctypes.byref(rect))
            u32.ClientToScreen(ctypes.c_void_p(hwnd), ctypes.byref(origin))
            ImageGrab.grab(bbox=(origin.x, origin.y, origin.x + rect.right,
                                origin.y + rect.bottom)).save(OUT / "ui.png")
        (OUT / ("result-ui.json" if UI_ONLY else "result.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        app.quit()

    def begin():
        nonlocal original, generation, phase, stop_sent
        try:
            original = owned_focus(field)
            app.gain_var.set(f"{GAIN:g}")  # Test-only choice; does not persist user gain.
            def factory(**kwargs):
                original_event = kwargs["on_event"]
                def observe(event):
                    elapsed_ms = round((time.perf_counter() - session_started[0]) * 1000, 2)
                    if event.get("capture_ready"):
                        timings["wav_capture_ready_ms"] = elapsed_ms
                    if event.get("backend_ready"):
                        timings["backend_ready_ms"] = elapsed_ms
                    if event["type"] == "transcript":
                        key = "first_final_ms" if event["final"] else "first_partial_ms"
                        timings.setdefault(key, elapsed_ms)
                        last_transcripts.append({"final": event["final"], "text": event["text"]})
                        del last_transcripts[:-6]
                    if event["type"] == "complete":
                        timings["complete_ms"] = elapsed_ms
                        completion.update(event)
                        completion["pending_tail"] = session.buffer.pending
                    original_event(event)
                kwargs["on_event"] = observe
                session = DictationSession(microphone=lambda **_: microphone, **kwargs)
                return session
            session_started[0] = time.perf_counter()
            if UI_ONLY:
                from types import SimpleNamespace
                fixture = "Xin chào Việt Nam. Đây là ô nhập thử của SODA. " * 5
                app.generation += 1
                app.running = True
                app.target = original
                app.overlay.show(app.hotkey_var.get())
                result["overlay_preserves_focus_immediate"] = can_inject(original, focused_target())
                assert result["overlay_preserves_focus_immediate"], "Overlay stole focus in UI fixture"
                app.session = SimpleNamespace(error=None, cancel=lambda: None)
                microphone.closed = True
                microphone.original_seconds = 0
                stop_sent = True
                completion.update({"cancelled": False, "error": None, "text": fixture,
                                   "preview": fixture, "pcm": None})
                app.handle_complete(completion)
            else:
                with patch("desktop.app.DictationSession", side_effect=factory):
                    app.start(original, from_hotkey=True)
                result["overlay_preserves_focus_immediate"] = can_inject(original, focused_target())
                assert result["overlay_preserves_focus_immediate"], "Overlay stole keyboard focus at start"
            generation = app.generation
            phase = "recognizing"
            root.after(40, tick)
        except Exception as exc:
            finish(exc)

    def tick():
        nonlocal phase, stop_sent, captured_text, modifiers_patch, settle_started
        try:
            if time.perf_counter() - started > 80:
                raise RuntimeError("Owned-window acceptance timed out")
            if original and phase == "recognizing":
                current = focused_target()
                if not can_inject(original, current) and len(result.get("focus_drift", [])) < 3:
                    title_buffer = ctypes.create_unicode_buffer(250)
                    u32.GetWindowTextW(ctypes.c_void_p(current[0]), title_buffer, 250)
                    from desktop.overlay import toplevel_hwnd
                    result.setdefault("focus_drift", []).append({
                        "elapsed": round(time.perf_counter() - started, 3),
                        "original": original, "current": current,
                        "new_window_pid": window_process_id(current[0]),
                        "new_window_title": title_buffer.value,
                        "overlay_hwnd": toplevel_hwnd(app.overlay.top),
                        "main_hwnd": toplevel_hwnd(app.root),
                    })
            if phase == "recognizing":
                if not app.running and not stop_sent:
                    raise RuntimeError(f"Session ended before WAV finished: {app.session.error}")
                if microphone.eof.is_set() and not stop_sent:
                    timings["stop_ms"] = round((time.perf_counter() - session_started[0]) * 1000, 2)
                    app.stop()
                    stop_sent = True
                if stop_sent and not app.running and not app.injecting:
                    assert microphone.closed, "Microphone adapter was not closed"
                    assert app.session.error is None, "SODA session failed"
                    captured_text = completion.get("text", "")
                    pending_tail = completion.get("pending_tail", "")
                    result.update({
                        "preview_differs_from_final": app.result_text != captured_text,
                        "capture_adapter": "ui_fixture" if UI_ONLY else "wav_file",
                        "timings_ms": timings,
                        "committed_characters": len(captured_text),
                        "preview_characters": len(app.result_text),
                        "pending_tail_characters": len(pending_tail),
                        "pending_tail_word_count": len(re.findall(r"\w+", pending_tail)),
                        "pending_tail": pending_tail,
                        "last_transcripts": last_transcripts,
                    })
                    assert len(captured_text) > 50, "No useful real recognition result"
                    actual = field.get("1.0", "end-1c")
                    if actual != dictation_text(captured_text):
                        if settle_started is None:
                            settle_started = time.perf_counter()
                        if time.perf_counter() - settle_started < 1.0:
                            root.after(40, tick)
                            return
                        result.update({"raw_text": captured_text, "preview_text": app.result_text, "inserted_text": actual,
                                       "expected_characters": len(dictation_text(captured_text)),
                                       "actual_characters": len(actual), "ui_status": app.status.get(),
                                       "foreground_matches": can_inject(original, focused_target())})
                        raise AssertionError("Recognized text was not inserted exactly")
                    result.update({"audio_seconds": microphone.original_seconds,
                                   "gain_db": GAIN, "final_characters": len(actual),
                                   "sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
                                   "raw_text": captured_text,
                                   "preview_text": app.result_text,
                                   "recognition_to_edit_pass": not UI_ONLY,
                                   "automatic_insertion_pass": True,
                                   "has_eligible_pcm": bool(app.last_pcm),
                                   "elapsed_s": round(time.perf_counter() - started, 3)})
                    other = owned_focus(second)
                    assert not can_inject(original, other)
                    app.target = original
                    app.handle_complete({"cancelled": False, "error": None, "text": captured_text,
                                         "preview": captured_text, "pcm": None})
                    phase = "focus_guard"
            elif phase == "focus_guard" and not app.injecting:
                assert second.get("1.0", "end-1c") == ""
                assert field.get("1.0", "end-1c") == dictation_text(captured_text)
                assert app.result_text == captured_text
                result["changed_focus_retains_text_pass"] = True
                app.target = owned_focus(second)
                modifiers_patch = patch("desktop.windows.modifiers_released", return_value=False)
                modifiers_patch.start()
                app.handle_complete({"cancelled": False, "error": None, "text": captured_text,
                                     "preview": captured_text, "pcm": None})
                app.start(app.target, from_hotkey=True)
                assert app.generation == generation, "New recording started during pending insertion"
                root.after(60, app.cancel)
                phase = "cancel_guard"
            elif phase == "cancel_guard" and not app.injecting:
                assert second.get("1.0", "end-1c") == "", "Cancelled insertion typed text"
                assert app.result_text == captured_text
                result["cancel_and_new_session_guard_pass"] = True
                result["passed"] = True
                finish()
                return
            root.after(40, tick)
        except Exception as exc:
            finish(exc)

    root.after(700, begin)
    root.mainloop()
    summary = {key: value for key, value in result.items() if key not in ("raw_text", "preview_text", "inserted_text", "pending_tail", "last_transcripts")}
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    assert result["passed"], "Desktop acceptance failed; inspect ignored data/desktop_e2e/"


if __name__ == "__main__":
    main()