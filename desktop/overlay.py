"""Thin, non-activating live transcript + waveform overlay for Windows dictation."""
from __future__ import annotations
import ctypes, math, time, tkinter as tk

W,H=430,62
BG="#102038"; ACCENT="#5ad9ff"; PALE="#7f91a8"; TEXT="#edf7ff"
WS_EX_TOOLWINDOW=0x80; WS_EX_NOACTIVATE=0x08000000; GWL_EXSTYLE=-20
SWP_NOSIZE=1; SWP_NOMOVE=2; SWP_NOACTIVATE=0x10; SWP_SHOWWINDOW=0x40

def no_activate(hwnd:int)->None:
    u32=ctypes.WinDLL("user32",use_last_error=True)
    get=u32.GetWindowLongPtrW; get.restype=ctypes.c_ssize_t; get.argtypes=(ctypes.c_void_p,ctypes.c_int)
    set_=u32.SetWindowLongPtrW; set_.restype=ctypes.c_ssize_t; set_.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_ssize_t)
    pos=u32.SetWindowPos
    ex=get(ctypes.c_void_p(hwnd),GWL_EXSTYLE)
    set_(ctypes.c_void_p(hwnd),GWL_EXSTYLE,ex|WS_EX_TOOLWINDOW|WS_EX_NOACTIVATE)
    pos(ctypes.c_void_p(hwnd),ctypes.c_void_p(-1),0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE|SWP_SHOWWINDOW)

def toplevel_hwnd(widget)->int:
    u32=ctypes.WinDLL("user32",use_last_error=True); u32.GetAncestor.restype=ctypes.c_void_p
    return int(u32.GetAncestor(ctypes.c_void_p(widget.winfo_id()),2) or widget.winfo_id())

class ListeningOverlay:
    def __init__(self,parent:tk.Tk):
        self.root=parent; self.top=tk.Toplevel(parent); self.top.withdraw(); self.top.overrideredirect(True)
        self.top.configure(bg=BG); self.top.attributes("-topmost",True); self.top.attributes("-disabled",True)
        try:self.top.attributes("-alpha",0.70)
        except tk.TclError:pass
        self.canvas=tk.Canvas(self.top,width=W,height=H,bg=BG,bd=0,highlightthickness=0); self.canvas.pack()
        self.level=0.; self.active=False; self.visible=False; self.transcript=""; self._tick_id=None; self._hide_id=None; self._paint()
    def show(self,shortcut:str)->None:
        self.cancel_hide(); self.active=True; self.level=0.; self.transcript=""
        x=max(0,(self.root.winfo_screenwidth()-W)//2); y=max(0,self.root.winfo_screenheight()-H-28)
        u32=ctypes.windll.user32; previous=int(u32.GetForegroundWindow() or 0)
        self.top.geometry(f"{W}x{H}+{x}+{y}"); self.top.update_idletasks()
        hwnd=toplevel_hwnd(self.top); no_activate(hwnd)
        # Tk can still activate a freshly mapped popup on some Windows builds.
        # Restore the previously focused application immediately after mapping.
        u32.ShowWindow(ctypes.c_void_p(hwnd), 4)  # SW_SHOWNOACTIVATE
        no_activate(hwnd)
        if previous and int(u32.GetForegroundWindow() or 0) != previous:
            u32.SetForegroundWindow(ctypes.c_void_p(previous))
        self.visible=True; self._tick()
    def update_level(self,rms_dbfs:float)->None:self.level=max(0.,min(1.,(float(rms_dbfs)+48)/34))
    def update_transcript(self,text:str)->None:
        text=" ".join(str(text).split())
        self.transcript=("…"+text[-86:]) if len(text)>86 else text
        self._paint()
    def processing(self)->None:self.active=False; self.level=.10; self._paint()
    def finish(self,success:bool,text:str="")->None:
        self.active=False; self.level=.02
        if text and not self.transcript:self.transcript=text
        self._paint(); self.cancel_hide(); self._hide_id=self.top.after(650 if success else 1100,self.hide)
    def cancel_hide(self):
        if self._hide_id is not None:
            try:self.top.after_cancel(self._hide_id)
            except tk.TclError:pass
            self._hide_id=None
    def hide(self):
        self.cancel_hide(); self.active=False; self.visible=False; self.level=0; self.transcript=""
        try:self.top.withdraw()
        except tk.TclError:pass
        if self._tick_id is not None:
            try:self.top.after_cancel(self._tick_id)
            except tk.TclError:pass
            self._tick_id=None
    def destroy(self):self.hide();self.top.destroy()
    def _tick(self):
        if not self.visible:return
        self._paint(); self._tick_id=self.top.after(55,self._tick)
    def _paint(self):
        c=self.canvas;c.delete("all");t=time.monotonic()
        label=self.transcript or ("Listening..." if self.active else "Processing...")
        c.create_text(16,15,anchor="w",text=label,fill=TEXT,font=("Segoe UI",9),width=W-30)
        c.create_oval(14,43,21,50,fill=ACCENT if self.active else "#f1b14c",outline="")
        level=self.level if self.active else .08
        for i in range(42):
            x=32+i*8.8; env=max(.18,1-abs(i-20.5)/25); mod=.3+.7*abs(math.sin(i*.72+t*8))
            h=max(2,min(15,2+14*level*env*mod))
            c.create_line(x,47-h/2,x,47+h/2,width=2.5,fill=ACCENT if level>.12 else PALE,capstyle=tk.ROUND)
