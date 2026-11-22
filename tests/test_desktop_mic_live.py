"""Windows microphone → live SODA integration, no audio/transcript persisted or printed."""

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop.core import DictationSession


async def main():
    types = []
    started = asyncio.Event()
    def on_event(ev):
        types.append(ev["type"])
        if ev["type"] == "status" and "Đang nghe" in ev.get("text", ""):
            started.set()
        if ev["type"] == "complete":
            final = ev
            print("MIC_BACKEND_COMPLETE", "cancelled", final["cancelled"],
                  "error", final["error"], "text_len", len(final["text"]),
                  "captured_pcm_bytes", len(final["pcm"] or b""), flush=True)
    session = DictationSession(on_event=on_event)
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(started.wait(), timeout=30)
        await asyncio.sleep(1.2)
        session.stop()
        await asyncio.wait_for(task, timeout=55)
        assert "complete" in types
        assert session.error is None, session.error
        print("MIC_TO_NATIVE_SODA_PASS")
    finally:
        session.cancel()
        if not task.done():
            await task


if __name__ == "__main__":
    asyncio.run(main())
