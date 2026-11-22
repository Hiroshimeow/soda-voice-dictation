"""Explicit Windows interactive check: typed UTF-16 must appear in *our* Tk window."""

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")
import tkinter as tk
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop.windows import focused_target, insert_after_release, window_process_id

EXPECTED = "Xin chào Việt Nam — thử gõ chữ có dấu 👋"


def main():
    root = tk.Tk()
    root.title("SODA injection E2E neutral test (owned)")
    root.geometry("540x150+140+140")
    field = tk.Text(root, width=60, height=3)
    field.pack(padx=20, pady=20)
    outcome = {"checked": False}

    def verify():
        root.lift()
        root.focus_force()
        field.focus_force()
        root.update()
        target = focused_target()
        if target[0] == 0 or window_process_id(target[0]) != os.getpid():
            print("SKIP_NO_INTERACTIVE_FOREGROUND", flush=True)
            root.destroy()
            return
        try:
            insert_after_release(EXPECTED, target, timeout=1)
            root.update()
            actual = field.get("1.0", "end-1c")
            assert actual == EXPECTED, f"Actual Unicode mismatch: {actual!r}"
            outcome["checked"] = True
            print("PASS_ACTUAL_UNICODE_TEXT", len(actual), "target_hwnd", target[0], flush=True)
        finally:
            root.destroy()

    root.after(400, verify)
    root.after(5000, lambda: root.destroy() if root.winfo_exists() else None)
    root.mainloop()
    assert outcome["checked"], "Interactive injection into owned Tk window not verified"


if __name__ == "__main__":
    main()
