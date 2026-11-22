"""Windows-only foreground validation and Unicode text injection (clipboard unchanged)."""
from __future__ import annotations

import ctypes
import sys
import time
import unicodedata
from threading import Event
from ctypes import wintypes

if sys.platform != "win32":
    raise RuntimeError("Windows desktop client requires Windows")

u32 = ctypes.WinDLL("user32", use_last_error=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
ULONG_PTR = wintypes.WPARAM
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1
VK_MODIFIERS = (0x10, 0x11, 0x12, 0x5B, 0x5C)
VK_SPACE = 0x20


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class INPUT_UNION(ctypes.Union):
    # Windows INPUT's union has the size of MOUSEINPUT, even for keyboard input.
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("payload", INPUT_UNION)]


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", RECT),
    ]


u32.GetForegroundWindow.restype = wintypes.HWND
u32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
u32.GetGUIThreadInfo.restype = wintypes.BOOL
u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
u32.SendInput.restype = wintypes.UINT
u32.GetAsyncKeyState.argtypes = [ctypes.c_int]
u32.GetAsyncKeyState.restype = wintypes.SHORT
u32.IsWindow.argtypes = [wintypes.HWND]
u32.IsWindow.restype = wintypes.BOOL


def focused_target() -> tuple[int, int]:
    hwnd = int(u32.GetForegroundWindow() or 0)
    if not hwnd:
        return (0, 0)
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(info)
    focused = int(info.hwndFocus or 0) if u32.GetGUIThreadInfo(0, ctypes.byref(info)) else 0
    return hwnd, focused


def can_inject(original: tuple[int, int], current: tuple[int, int]) -> bool:
    return bool(
        original[0]
        and original[0] == current[0]
        and bool(original[1])
        and original[1] == current[1]
    )


def utf16_units(text: str) -> tuple[int, ...]:
    raw = text.encode("utf-16-le")
    return tuple(int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2))


def modifiers_released() -> bool:
    return not any(u32.GetAsyncKeyState(key) & 0x8000 for key in (*VK_MODIFIERS, VK_SPACE, 0x44, 0x78))


def dictation_text(text: str) -> str:
    """Never type control characters into a terminal/chat; retain original in the UI."""
    return "".join(" " if (unicodedata.category(char).startswith("C") or
                           unicodedata.category(char) in ("Zl", "Zp")) else char
                   for char in text)


def window_process_id(hwnd: int) -> int:
    pid = wintypes.DWORD()
    u32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


def send_unicode(text: str, target: tuple[int, int] | None = None,
                 cancel: Event | None = None) -> None:
    """Inject bounded Unicode batches without Enter or changing the clipboard."""
    clean = dictation_text(text)
    for offset in range(0, len(clean), 32):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Insertion was cancelled; if partial text appeared, review it before copying")
        if target is not None and not can_inject(target, focused_target()):
            raise RuntimeError("The target window or input position changed during insertion")
        group = utf16_units(clean[offset:offset + 32])
        inputs = (INPUT * (len(group) * 2))()
        for i, unit in enumerate(group):
            inputs[2 * i].type = INPUT_KEYBOARD
            inputs[2 * i].payload.ki = KEYBDINPUT(0, unit, KEYEVENTF_UNICODE, 0, 0)
            inputs[2 * i + 1].type = INPUT_KEYBOARD
            inputs[2 * i + 1].payload.ki = KEYBDINPUT(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
        sent = u32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))
        if sent != len(inputs):
            raise OSError("Windows blocked text injection; only part of the transcript may have been inserted")


def insert_after_release(text: str, target: tuple[int, int], timeout: float = 3.0,
                         cancel: Event | None = None) -> None:
    until = time.monotonic() + timeout
    while time.monotonic() < until and not modifiers_released():
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Text insertion was cancelled")
        time.sleep(.02)
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Text insertion was cancelled")
    if not modifiers_released():
        raise RuntimeError("The hotkey is still pressed; the transcript remains available for copying")
    if not u32.IsWindow(target[0]) or not can_inject(target, focused_target()):
        raise RuntimeError("The target input changed; the transcript remains available for copying")
    send_unicode(text, target, cancel)
