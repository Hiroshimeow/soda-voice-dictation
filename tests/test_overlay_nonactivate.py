"""Win32 overlay must never activate itself over another focused app."""
import ctypes
import tkinter as tk
import unittest

from desktop.overlay import ListeningOverlay, toplevel_hwnd, WS_EX_NOACTIVATE, GWL_EXSTYLE


class OverlayTests(unittest.TestCase):
    def test_popup_has_nonactivating_style_and_keeps_foreground(self):
        u32=ctypes.WinDLL("user32", use_last_error=True)
        u32.GetForegroundWindow.restype=ctypes.c_void_p
        u32.GetWindowLongPtrW.argtypes=(ctypes.c_void_p,ctypes.c_int)
        u32.GetWindowLongPtrW.restype=ctypes.c_ssize_t
        before=u32.GetForegroundWindow()
        root=tk.Tk()
        root.withdraw()
        overlay=ListeningOverlay(root)
        # Creating Tk itself may briefly activate a window; isolate overlay.show.
        before=u32.GetForegroundWindow()
        try:
            overlay.show("Ctrl+Space")
            root.update()
            hwnd=toplevel_hwnd(overlay.top)
            style=u32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
            self.assertTrue(style & WS_EX_NOACTIVATE)
            self.assertEqual(u32.GetForegroundWindow(),before)
            overlay.update_level(-20)
            overlay.update_transcript("đây là transcript đang nhận theo thời gian thực")
            root.update()
            self.assertEqual(u32.GetForegroundWindow(),before)
            overlay.processing()
            root.update()
            self.assertEqual(u32.GetForegroundWindow(),before)
            overlay.finish(True)
            root.update()
            self.assertEqual(u32.GetForegroundWindow(),before)
        finally:
            overlay.destroy()
            root.destroy()


if __name__=="__main__":
    unittest.main()
