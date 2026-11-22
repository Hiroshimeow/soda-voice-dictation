"""Local multi-language Windows dictation backed by Chrome SODA."""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw
from pynput import keyboard
import pystray
import sounddevice as sd

from desktop.audio import validate_gain
from desktop.overlay import ListeningOverlay
from training.personal_vocabulary import (load_vocabulary, save_vocabulary, apply_replacements)
from desktop.core import DictationSession
from desktop.file_transcribe import SUPPORTED_EXTENSIONS, transcribe_file
from desktop.windows import focused_target, insert_after_release, window_process_id
from soda_languages import LANGUAGES, DEFAULT_LANGUAGE

APP_TITLE = "SODA Voice Dictation"
LOCAL_DATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SodaDictation"
CONFIG_PATH = LOCAL_DATA / "config.json"
HISTORY_PATH = LOCAL_DATA / "history.json"
API_BASE = "http://127.0.0.1:18765"
MODE_LABELS = {"toggle": "Toggle", "hold": "Hold to talk"}
LANG_LABELS = {cfg["label"]: locale for locale, cfg in LANGUAGES.items()}


def _config() -> dict:
    try:
        value = json.loads(CONFIG_PATH.read_text("utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError):
        pass
    return {}


def _save_config(value: dict) -> None:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 3, 54, 61), radius=17, fill="#2364c8")
    draw.rounded_rectangle((27, 12, 37, 36), radius=5, fill="white")
    draw.arc((20, 24, 44, 46), 0, 180, fill="white", width=3)
    draw.line((32, 45, 32, 52), fill="white", width=3)
    draw.line((23, 52, 41, 52), fill="white", width=3)
    return image


class SingleInstance:
    """Advisory OS lock auto-released on process exit, no stale PID files."""
    def __init__(self):
        import msvcrt
        self._msvcrt = msvcrt
        LOCAL_DATA.mkdir(parents=True, exist_ok=True)
        self.handle = (LOCAL_DATA / "desktop.lock").open("a+b")
        if not self.handle.seek(0, 2):
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("SODA Voice Dictation is already running. Open it from the system tray.") from exc

    def close(self):
        try:
            self.handle.seek(0)
            self._msvcrt.locking(self.handle.fileno(), self._msvcrt.LK_UNLCK, 1)
        finally:
            self.handle.close()


class App:
    def __init__(self):
        self.lock = SingleInstance()
        self.settings = _config()
        # One-time migration to the explicitly requested Ctrl+Space hold shortcut.
        # Preserve preferences after the first migration.
        if not self.settings.get("shortcut_v2"):
            self.settings["hotkey"] = "Ctrl+Space"
            self.settings["mode"] = "hold"
            self.settings["shortcut_v2"] = True
            _save_config(self.settings)
        self.bus: queue.Queue[dict] = queue.Queue(maxsize=500)
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("580x500+820+120")
        self.root.minsize(520, 470)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.root.configure(bg="#f3f6fc")
        self.session: DictationSession | None = None
        self.worker: threading.Thread | None = None
        self.target: tuple[int, int] = (0, 0)
        self.result_text = ""
        self.last_pcm: bytes | None = None
        self.running = False
        self.injecting = False
        self.file_running = False
        self.insertion_cancel = threading.Event()
        self.generation = 0
        self.saving_generation = None
        self.closing = False
        self.hold_started = False
        self.hotkey_latched = False
        self.key_pressed: set[str] = set()
        self.mode_var = tk.StringVar(value=MODE_LABELS.get(self.settings.get("mode"), MODE_LABELS["toggle"]))
        self.hotkey_var = tk.StringVar(value=self.settings.get("hotkey", "Ctrl+Space"))
        saved_locale = self.settings.get("language", DEFAULT_LANGUAGE)
        self.language_var = tk.StringVar(value=LANGUAGES.get(saved_locale, LANGUAGES[DEFAULT_LANGUAGE])["label"])
        self.hotkey_modifiers = {"ctrl"}
        self.hotkey_key = "space"
        self._assign_hotkey()
        self.device_var = tk.StringVar()
        self.gain_var = tk.StringVar(value="0")
        self.level_var = tk.StringVar(value="Microphone is off")
        self.level_value = tk.DoubleVar(value=0)
        self.backend_status = tk.StringVar(value="SODA: checking backend...")
        self.status = tk.StringVar(value=f"Ready • {self.hotkey_var.get()}")
        self.consent = tk.BooleanVar(value=False)
        self.auto_insert_var = tk.BooleanVar(value=bool(self.settings.get("auto_insert", True)))
        self.make_ui()
        self.overlay = ListeningOverlay(self.root)
        self.refresh_devices()
        # Hide only once the Tk window and tray icon have been created.
        self.root.after(250, self.hide_window)
        self.tray = pystray.Icon(
            "soda-dictation", _icon(), APP_TITLE,
            menu=pystray.Menu(
                pystray.MenuItem("Show window", lambda *_: self._post({"type": "show"}), default=True),
                pystray.MenuItem("Start / stop", lambda *_: self._post({"type": "toggle_hotkey"})),
                pystray.MenuItem("Exit", lambda *_: self._post({"type": "quit"})),
            ),
        )
        self.tray.run_detached()
        self.listener = keyboard.Listener(on_press=self._key_down, on_release=self._key_up)
        self.listener.start()
        self.root.after(40, self.process_bus)
        threading.Thread(target=self.check_backend, daemon=True).start()

    def make_ui(self) -> None:
        pad = ttk.Frame(self.root, padding=14)
        pad.pack(fill="both", expand=True)
        ttk.Label(pad, text="SODA Dictation", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(pad, text="Global hotkey • Esc cancels • never sends Enter", foreground="#596b84").pack(anchor="w", pady=(0, 7))
        row = ttk.Frame(pad)
        row.pack(fill="x", pady=(0, 7))
        ttk.Label(row, text="Mode").pack(side="left")
        mode = ttk.Combobox(row, state="readonly", width=13, textvariable=self.mode_var, values=tuple(MODE_LABELS.values()))
        mode.pack(side="left", padx=8)
        mode.bind("<<ComboboxSelected>>", lambda _ev: self._update_settings())
        ttk.Label(row, text="Language").pack(side="left", padx=(6, 0))
        language = ttk.Combobox(row, state="readonly", width=13, textvariable=self.language_var,
                                values=tuple(LANG_LABELS))
        language.pack(side="left", padx=(6, 8))
        language.bind("<<ComboboxSelected>>", lambda _ev: self._update_settings())
        ttk.Label(row, text="Microphone").pack(side="left", padx=(6, 0))
        self.device_select = ttk.Combobox(row, state="readonly", width=24, textvariable=self.device_var)
        self.device_select.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.device_select.bind("<<ComboboxSelected>>", lambda _ev: self._device_changed())
        shortcut_row = ttk.Frame(pad)
        shortcut_row.pack(fill="x", pady=(0, 7))
        ttk.Label(shortcut_row, text="Hotkey").pack(side="left")
        hotkey_combo = ttk.Combobox(shortcut_row, state="readonly", width=21, textvariable=self.hotkey_var,
                                    values=("Ctrl+Space", "Ctrl+Alt+Space", "Ctrl+Alt+D", "Ctrl+Alt+F9"))
        hotkey_combo.pack(side="left", padx=8)
        hotkey_combo.bind("<<ComboboxSelected>>", lambda _ev: self._update_settings())
        gain_row = ttk.Frame(pad)
        gain_row.pack(fill="x", pady=(0, 6))
        ttk.Label(gain_row, text="Microphone gain").pack(side="left")
        self.gain_spin = ttk.Spinbox(gain_row, from_=-12, to=30, increment=3,
                                     width=6, textvariable=self.gain_var, command=self._gain_changed)
        self.gain_spin.pack(side="left", padx=8)
        self.gain_spin.bind("<Return>", lambda _ev: self._gain_changed())
        self.gain_spin.bind("<FocusOut>", lambda _ev: self._gain_changed())
        ttk.Label(gain_row, text="dB  -  saved per input device").pack(side="left")
        behavior = ttk.Frame(pad)
        behavior.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(behavior, text="Auto-insert into the focused field", variable=self.auto_insert_var, command=self._update_settings).pack(side="left")
        ttk.Progressbar(pad, maximum=60, variable=self.level_value).pack(fill="x", pady=(0, 2))
        ttk.Label(pad, textvariable=self.level_var, foreground="#596b84", wraplength=530).pack(anchor="w")
        btn = ttk.Frame(pad)
        btn.pack(fill="x")
        self.start_btn = ttk.Button(btn, text="Start", command=self.toggle_from_ui)
        self.start_btn.pack(side="left")
        ttk.Button(btn, text="Cancel", command=self.cancel).pack(side="left", padx=6)
        ttk.Button(btn, text="Copy", command=self.copy_text).pack(side="left")
        self.file_btn = ttk.Button(btn, text="Transcribe file...", command=self.choose_audio_file)
        self.file_btn.pack(side="left", padx=6)
        ttk.Button(btn, text="Hide", command=self.hide_window).pack(side="right")
        ttk.Label(pad, textvariable=self.status, foreground="#214d94", wraplength=530).pack(anchor="w", pady=(8, 3))
        self.text = tk.Text(pad, height=5, wrap="word", font=("Segoe UI", 12), relief="flat",
                            borderwidth=1, highlightthickness=1, highlightbackground="#c6d7e9")
        self.text.pack(fill="both", expand=True)
        bottom = ttk.Frame(pad)
        bottom.pack(fill="x", pady=(7, 0))
        ttk.Checkbutton(bottom, text="Consent to save corrected sample locally", variable=self.consent).pack(side="left")
        self.save_btn = ttk.Button(bottom, text="Save sample", command=self.save_sample)
        self.save_btn.pack(side="right")
        ttk.Label(pad, text="Review and correct the transcript before saving; sessions over 30 seconds are transcription-only.",
                  foreground="#677991", wraplength=530).pack(anchor="w", pady=(4, 0))
        ttk.Label(pad, textvariable=self.backend_status, foreground="#677991").pack(anchor="w", pady=(2, 0))
        vocab_row = ttk.Frame(pad)
        vocab_row.pack(fill="x", pady=(5, 0))
        ttk.Button(vocab_row, text="Personal vocabulary", command=self.open_vocabulary).pack(side="left")
        ttk.Label(vocab_row, text="Vietnamese priority phrases and deterministic corrections",
                  foreground="#677991").pack(side="left", padx=8)

    def _post(self, event: dict) -> None:
        if self.closing:
            return
        try:
            self.bus.put_nowait(event)
        except queue.Full:
            if self.session:
                self.session.cancel()
            self.insertion_cancel.set()
            try:
                self.bus.get_nowait()
            except queue.Empty:
                pass
            try:
                self.bus.put_nowait(event if event.get("type") == "complete" else
                                   {"type": "status", "text": "UI event queue overflow; session cancelled"})
            except queue.Full:
                pass

    def _key_name(self, key) -> str:
        try:
            if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                return "ctrl"
            if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                return "alt"
            if key == keyboard.Key.space:
                return "space"
            if key == keyboard.Key.f9:
                return "f9"
            if getattr(key, "vk", None) == 0x44:
                return "d"
            if getattr(key, "char", None) and key.char.lower() == "d":
                return "d"
            if key == keyboard.Key.esc:
                return "esc"
        except Exception:
            pass
        return ""

    def _assign_hotkey(self) -> None:
        valid = {
            "Ctrl+Space": ({"ctrl"}, "space"),
            "Ctrl+Alt+Space": ({"ctrl", "alt"}, "space"),
            "Ctrl+Alt+D": ({"ctrl", "alt"}, "d"),
            "Ctrl+Alt+F9": ({"ctrl", "alt"}, "f9"),
        }
        if self.hotkey_var.get() not in valid:
            self.hotkey_var.set("Ctrl+Space")
        self.hotkey_modifiers, self.hotkey_key = valid[self.hotkey_var.get()]

    def _hotkey_down(self) -> bool:
        return getattr(self, "hotkey_modifiers", {"ctrl", "alt"}).union({self.hotkey_key}) <= self.key_pressed

    def _key_down(self, key) -> None:
        name = self._key_name(key)
        if not name:
            return
        self.key_pressed.add(name)
        if name == "esc" and (self.running or getattr(self, "injecting", False)):
            if getattr(self, "injecting", False):
                self.insertion_cancel.set()
            self._post({"type": "cancel"})
        hotkey_key = self.hotkey_key
        if self._hotkey_down() and not self.hotkey_latched:
            self.hotkey_latched = True
            self._post({"type": "hotkey_down", "target": focused_target()})

    def _key_up(self, key) -> None:
        name = self._key_name(key)
        if name:
            self.key_pressed.discard(name)
        hotkey_key = self.hotkey_key
        if name in getattr(self, "hotkey_modifiers", {"ctrl", "alt"}).union({hotkey_key}) and self.hotkey_latched:
            self.hotkey_latched = False
            self._post({"type": "hotkey_up"})
        if not self._hotkey_down():
            self.hotkey_latched = False

    def _mode(self) -> str:
        return next((key for key, label in MODE_LABELS.items()
                     if label == self.mode_var.get()), self.mode_var.get())

    def _device_key(self) -> str:
        return self.device_keys.get(self.device_var.get(), "default")

    def _read_gain(self) -> float:
        try:
            return validate_gain(self.gain_var.get())
        except (TypeError, ValueError):
            self.gain_var.set("0")
            self.status.set("Invalid gain; reset to 0 dB")
            return 0

    def _gain_changed(self) -> None:
        gain = self._read_gain()
        gains = self.settings.setdefault("device_gains_db", {})
        if not isinstance(gains, dict):
            gains = self.settings["device_gains_db"] = {}
        gains[self._device_key()] = gain
        _save_config(self.settings)

    def _device_changed(self) -> None:
        gains = self.settings.get("device_gains_db", {})
        stored = gains.get(self._device_key(), 0) if isinstance(gains, dict) else 0
        try:
            gain = validate_gain(stored)
        except (TypeError, ValueError):
            gain = 0
        self.gain_var.set(f"{gain:g}")
        self._update_settings()

    def _locale(self) -> str:
        return LANG_LABELS.get(self.language_var.get(), DEFAULT_LANGUAGE)

    def _ws_url(self, locale: str | None = None) -> str:
        return f"ws://127.0.0.1:18765/v1/stt?lang={locale or self._locale()}"

    def _update_settings(self) -> None:
        self.settings.update({"mode": self._mode(),
                              "device": self.devices.get(self.device_var.get()),
                              "hotkey": self.hotkey_var.get(),
                              "language": self._locale(),
                              "auto_insert": bool(self.auto_insert_var.get())})
        self._assign_hotkey()
        self._gain_changed()

    def check_backend(self) -> None:
        try:
            with urllib.request.urlopen(f"{API_BASE}/health", timeout=3) as response:
                body = json.loads(response.read(4096))
            if not body.get("ok"):
                raise RuntimeError("backend is not ready")
            self._post({"type": "backend", "text": "SODA: local on-device recognition ready"})
        except Exception:
            self._post({"type": "backend", "text": "SODA is offline - run run-desktop-windows.bat again"})

    def refresh_devices(self) -> None:
        devices = ["Default"]
        self.devices: dict[str, int | None] = {"Default": None}
        self.device_keys = {"Default": "default"}
        try:
            available = sd.query_devices()
            default_index = int(sd.default.device[0])
            if 0 <= default_index < len(available):
                current = available[default_index]
                self.device_keys["Default"] = f"{current['hostapi']}:{current['name']}"
            for i, dev in enumerate(available):
                if dev["max_input_channels"] > 0:
                    label = f"{i} • {dev['name'][:42]}"
                    devices.append(label)
                    self.devices[label] = i
                    self.device_keys[label] = f"{dev['hostapi']}:{dev['name']}"
        except Exception as exc:
            self.status.set(f"Could not enumerate microphones: {exc}")
        self.device_select["values"] = devices
        preferred = self.settings.get("device")
        self.device_var.set(next((label for label in devices if preferred is not None and label.startswith(f"{preferred} • ")), "Default"))
        self._device_changed()

    def _set_text(self, value: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)

    def _get_text(self) -> str:
        return self.text.get("1.0", "end-1c").strip()

    def _remember_transcript(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        history = []
        try:
            value = json.loads(HISTORY_PATH.read_text("utf-8"))
            if isinstance(value, list):
                history = value
        except (OSError, ValueError):
            pass
        import datetime
        history.insert(0, {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                           "text": text})
        LOCAL_DATA.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history[:20], ensure_ascii=False, indent=2), encoding="utf-8")

    def start(self, target: tuple[int, int], *, from_hotkey: bool) -> None:
        if self.running or self.injecting or self.file_running:
            self.status.set("Another task is finishing; press Esc to cancel it")
            return
        self.generation += 1
        generation = self.generation
        self.saving_generation = None
        self.save_btn.configure(state="normal")
        self.target = target if from_hotkey else (0, 0)
        if from_hotkey and hasattr(self, "overlay"):
            self.overlay.show(self.hotkey_var.get())
        self.running = True
        self.hold_started = from_hotkey and self._mode() == "hold"
        self.last_pcm = None
        self.result_text = ""
        self._set_text("")
        self.consent.set(False)
        self.insertion_cancel.clear()
        self.start_btn.configure(text="Stop")
        self.gain_spin.configure(state="disabled")
        self.device_select.configure(state="disabled")
        self.status.set("Opening microphone...")
        self.active_locale = self._locale()
        self.session = DictationSession(
            on_event=lambda event: self._post({**event, "generation": generation}),
            url=self._ws_url(self.active_locale),
            device=self.devices.get(self.device_var.get()), gain_db=self._read_gain())
        session = self.session
        self.worker = threading.Thread(target=lambda: __import__("asyncio").run(session.run()), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        if self.session and self.running:
            self.session.stop()
            self.status.set("Finalizing...")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.processing()

    def cancel(self) -> None:
        self.insertion_cancel.set()
        if self.session and self.running:
            self.session.cancel()
        if self.running or self.injecting:
            self.status.set("Cancelling; the current transcript is kept for review")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.finish(False, "Cancelled")

    def toggle_from_ui(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start((0, 0), from_hotkey=False)

    def on_hotkey(self, target: tuple[int, int]) -> None:
        if self.injecting:
            self.status.set("Inserting transcript; press Esc to cancel")
            return
        if self.running:
            if self._mode() == "toggle":
                self.stop()
        else:
            if window_process_id(target[0]) == os.getpid():
                target = (0, 0)
            self.start(target, from_hotkey=True)

    def handle_complete(self, event: dict) -> None:
        self.running = False
        self.hold_started = False
        self.start_btn.configure(text="Start")
        self.gain_spin.configure(state="normal")
        self.device_select.configure(state="readonly")
        self.level_value.set(0)
        self.level_var.set("Microphone is off")
        if hasattr(self, "overlay") and self.overlay.visible:
            self.overlay.processing()
        self.result_text = event.get("preview") or ""
        text = event.get("text") or ""
        try:
            if getattr(self, "active_locale", DEFAULT_LANGUAGE) == "vi-VN":
                text = apply_replacements(text, load_vocabulary())
        except (ValueError, OSError) as error:
            self.status.set(f"Vocabulary error: {error}")
            text = ""  # Fail closed, not a silent unreviewed edit.
        self.last_pcm = event.get("pcm")
        if not event.get("error") and not event["cancelled"] and text:
            self.result_text = text
            self._remember_transcript(text)
        self._set_text(self.result_text)
        if event["cancelled"]:
            self.last_pcm = None
            self.status.set("Cancelled - text was not inserted")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.finish(False, "Cancelled")
            return
        if event.get("error"):
            self.last_pcm = None
            self.status.set(f"Error: {event['error']}. Transcript kept for copying.")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.finish(False, "Recognition error")
            return
        if not text:
            self.status.set("No speech was recognized")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.finish(False, "No speech recognized")
            return
        if not self.target[0] or not self.auto_insert_var.get():
            self.status.set("Transcribed - copied to clipboard; press Ctrl+V anywhere")
            if hasattr(self, "overlay") and self.overlay.visible:
                self.overlay.finish(False, "Transcript ready")
            return
        self.injecting = True
        self.start_btn.configure(state="disabled")
        self.status.set("Validating the original target...")
        threading.Thread(target=self._inject_worker,
                         args=(text, self.target, self.generation), daemon=True).start()

    def _inject_worker(self, text: str, target: tuple[int, int], generation: int) -> None:
        try:
            insert_after_release(text, target, cancel=self.insertion_cancel)
            message = "Transcript inserted - also kept in the clipboard"
        except Exception as exc:
            message = f"Insert failed: {exc}. The transcript is still available for copying."
        self._post({"type": "insertion_complete", "text": message,
                    "success": message.startswith("Transcript inserted"), "generation": generation})

    def process_bus(self) -> None:
        try:
            for _ in range(60):
                ev = self.bus.get_nowait()
                kind = ev["type"]
                if ev.get("generation", self.generation) != self.generation:
                    continue
                if kind == "hotkey_down":
                    self.on_hotkey(ev["target"])
                elif kind == "hotkey_up" and self.hold_started:
                    self.stop()
                elif kind == "toggle_hotkey":
                    self.toggle_from_ui()
                elif kind == "cancel":
                    self.cancel()
                elif kind == "transcript" and self.running:
                    self._set_text(ev["text"])
                    if hasattr(self, "overlay") and self.overlay.visible:
                        self.overlay.update_transcript(ev["text"])
                elif kind == "level" and self.running:
                    self.level_value.set(max(0, min(60, ev["rms_dbfs"] + 60)))
                    note = ""
                    if ev["clipped_samples"] or ev["peak_dbfs"] > -.5:
                        note = " - clipping detected: reduce gain"
                    elif ev["rms_dbfs"] < -40 and ev["input_dbfs"] > -80:
                        note = " - low level: move closer or raise gain slightly"
                    self.level_var.set(f"Input {ev['input_dbfs']:.0f} dBFS -> output {ev['rms_dbfs']:.0f} dBFS{note}")
                    if hasattr(self, "overlay") and self.overlay.visible:
                        self.overlay.update_level(ev["rms_dbfs"])
                elif kind == "complete":
                    self.handle_complete(ev)
                elif kind == "insertion_complete":
                    self.injecting = False
                    self.start_btn.configure(state="normal")
                    self.status.set(ev["text"])
                    if hasattr(self, "overlay") and self.overlay.visible:
                        self.overlay.finish(ev.get("success", False),
                                            "Inserted into target" if ev.get("success", False)
                                            else "Insert failed - open the window to copy")
                elif kind == "status":
                    self.status.set(ev["text"])
                elif kind == "file_transcript":
                    self._set_text(ev["text"])
                    self.status.set("Transcribing file..." if ev.get("partial") else "Received another finalized segment")
                elif kind == "file_progress":
                    duration=max(.001,float(ev.get("duration") or 0))
                    done=max(0.0,float(ev.get("seconds") or 0))
                    self.level_value.set(min(60,60*done/duration))
                    self.level_var.set(f"File: {done:.1f}/{duration:.1f} seconds")
                elif kind == "file_complete":
                    self.file_running=False
                    self.file_btn.configure(state="normal")
                    self.start_btn.configure(state="normal")
                    self.level_value.set(0)
                    self.level_var.set("Microphone is off")
                    text=ev.get("text") or ""
                    try:
                        if ev.get("locale") == "vi-VN":
                            text=apply_replacements(text,load_vocabulary())
                    except (ValueError,OSError) as error:
                        self.status.set(f"Vocabulary error: {error}")
                        text=""
                    if text:
                        self.result_text=text
                        self._set_text(text)
                        self._remember_transcript(text)
                        self.status.set(f"File transcribed - {ev.get('name','audio')} - copied to clipboard")
                    else:
                        self.status.set("No speech was recognized in the file")
                elif kind == "file_error":
                    self.file_running=False
                    self.file_btn.configure(state="normal")
                    self.start_btn.configure(state="normal")
                    self.level_value.set(0)
                    self.level_var.set("Microphone is off")
                    self.status.set(f"File transcription error: {ev.get('text','unknown')}")
                elif kind == "backend":
                    self.backend_status.set(ev["text"])
                elif kind in ("saved", "save_error"):
                    self.saving_generation = None
                    self.save_btn.configure(state="normal")
                    if kind == "saved":
                        self.last_pcm = None
                        self.consent.set(False)
                        self.status.set(f"Saved sample {ev['sample_id']} locally")
                    else:
                        self.status.set(ev["text"])
                elif kind == "show":
                    self.show_window()
                elif kind == "quit":
                    self.quit()
                    return
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(40, self.process_bus)

    def choose_audio_file(self) -> None:
        if self.running or self.injecting or self.file_running:
            self.status.set("Finish the current task first")
            return
        patterns=" ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        path=filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio",patterns),("All files","*.*")]
        )
        if not path:
            return
        self.file_running=True
        self.last_pcm=None
        self.file_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self._set_text("")
        self.status.set(f"Preparing file transcription: {Path(path).name}")
        locale=self._locale()
        threading.Thread(target=self._file_worker,args=(path,locale),daemon=True).start()

    def _file_worker(self,path:str,locale:str) -> None:
        import asyncio
        name=Path(path).name
        try:
            result=asyncio.run(transcribe_file(path,self._ws_url(locale),lambda ev:self._post(ev)))
            self._post({"type":"file_complete","text":result["text"],"name":name,"locale":locale})
        except Exception as exc:
            self._post({"type":"file_error","text":str(exc)[:300]})

    def copy_text(self) -> None:
        value = self._get_text()
        if not value:
            self.status.set("There is no text to copy")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update()
        self.status.set("Copied transcript to clipboard")

    def save_sample(self) -> None:
        if getattr(self, "active_locale", DEFAULT_LANGUAGE) != "vi-VN":
            self.status.set("Labeled-sample training is currently available only for Vietnamese.")
            return
        if self.running or self.injecting or self.file_running:
            self.status.set("Finish the current task before saving")
            return
        if not self.consent.get():
            self.status.set("Consent is required before saving a sample")
            return
        if not self.last_pcm:
            self.status.set("Only complete 0.5-30 second microphone sessions can be saved as labeled samples.")
            return
        reference = self._get_text()
        if not 2 <= len(reference) <= 3000:
            self.status.set("Enter an exact corrected transcript between 2 and 3000 characters")
            return
        if self.saving_generation == self.generation:
            return
        pcm = self.last_pcm
        self.saving_generation = self.generation
        self.save_btn.configure(state="disabled")
        self.status.set("Saving labeled sample locally...")
        threading.Thread(target=self._save_worker,
                         args=(pcm, reference, self.generation), daemon=True).start()

    def _save_worker(self, pcm: bytes, reference: str, generation: int) -> None:
        body = json.dumps({
            "language": "vi-VN", "consent": True, "reference": reference,
            "pcm_base64": base64.b64encode(pcm).decode("ascii"),
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{API_BASE}/v1/dataset/ingest", data=body, method="POST",
            headers={"Content-Type": "application/json", "Origin": API_BASE},
        )
        try:
            with urllib.request.urlopen(request, timeout=12) as resp:
                result = json.loads(resp.read(4096))
            self._post({"type": "saved", "sample_id": result["sample_id"], "generation": generation})
        except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
            self._post({"type": "save_error", "text": f"Could not save sample: {exc}", "generation": generation})

    def open_vocabulary(self) -> None:
        if getattr(self, "vocabulary_window", None):
            try:
                if self.vocabulary_window.winfo_exists():
                    self.vocabulary_window.lift()
                    return
            except tk.TclError:
                pass
        top = tk.Toplevel(self.root)
        self.vocabulary_window = top
        top.title("Vietnamese personal vocabulary")
        top.geometry("590x510")
        frame = ttk.Frame(top, padding=15)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Priority phrases (one per line)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Up to 64 phrases; the first 40 are sent to SODA for each new Vietnamese session.",
                  foreground="#58687a").pack(anchor="w")
        phrases = tk.Text(frame, height=7, font=("Segoe UI", 10))
        phrases.pack(fill="x", pady=(4, 10))
        ttk.Label(frame, text="Deterministic corrections (one per line: heard => write)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Example: xi di pi ay => CDPA. Corrections are applied after final recognition.",
                  foreground="#58687a").pack(anchor="w")
        rules = tk.Text(frame, height=8, font=("Segoe UI", 10))
        rules.pack(fill="both", expand=True, pady=(4, 10))
        status = tk.StringVar(value="")
        ttk.Label(frame, textvariable=status, foreground="#1e5483",
                  wraplength=550).pack(anchor="w")
        try:
            vocab = load_vocabulary()
            phrases.insert("1.0", "\n".join(vocab["phrases"]))
            rules.insert("1.0", "\n".join(
                f"{r['heard']} => {r['write']}" for r in vocab["replacements"]))
        except Exception as exc:
            status.set(f"Could not load vocabulary: {exc}")

        def save() -> None:
            try:
                raw = [line.strip() for line in phrases.get("1.0", "end-1c").splitlines()
                       if line.strip()]
                mapping = []
                for line in rules.get("1.0", "end-1c").splitlines():
                    if not line.strip():
                        continue
                    if "=>" not in line:
                        raise ValueError("Use => in each correction rule")
                    heard, write = (part.strip() for part in line.split("=>", 1))
                    mapping.append({"heard": heard, "write": write})
                save_vocabulary({"schema_version": 1, "language": "vi-VN",
                                 "phrases": raw, "replacements": mapping})
                status.set("Saved. Priority phrases apply to new Vietnamese sessions; corrections apply immediately.")
            except (ValueError, OSError) as exc:
                status.set(f"Not saved: {exc}")

        ttk.Button(frame, text="Save vocabulary", command=save).pack(anchor="e")

    def hide_window(self) -> None:
        self.root.withdraw()

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def quit(self) -> None:
        self.closing = True
        self.insertion_cancel.set()
        if self.session:
            self.session.cancel()
        self.listener.stop()
        self.tray.stop()
        if hasattr(self, "overlay"):
            self.overlay.destroy()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)
        self.lock.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        App().run()
    except RuntimeError as exc:
        messagebox.showwarning(APP_TITLE, str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
