"""Opt-in local integration: stream an existing private user WAV to the running SODA."""

import sys as _platform_sys
import unittest as _platform_unittest
if _platform_sys.platform != "win32":
    raise _platform_unittest.SkipTest("Windows desktop client test")
import asyncio
import hashlib
import json
import os
import time
import wave
from pathlib import Path

from websockets.asyncio.client import connect

WAV = Path(os.environ.get("SODA_TEST_WAV", str(Path(__file__).resolve().parents[1] / "samples" / "processed" / "vi_technology_02.wav")))
URL = "ws://127.0.0.1:18765/v1/stt?lang=vi-VN"


async def main() -> None:
    with wave.open(str(WAV), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
        pcm = source.readframes(source.getnframes())
    events = []
    start = time.monotonic()
    async with connect(URL, open_timeout=6) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 35))
        assert ready["type"] == "ready", ready
        async def receive():
            while True:
                obj = json.loads(await asyncio.wait_for(ws.recv(), 50))
                events.append(obj)
                if obj["type"] == "error":
                    raise AssertionError(f"Error from backend: {obj.get('code')}")
                if obj["type"] == "closed":
                    return
        task = asyncio.create_task(receive())
        for offset in range(0, len(pcm), 3200):
            await ws.send(pcm[offset: offset + 3200])
            await asyncio.sleep(.1)
        await ws.send("stop")
        await asyncio.wait_for(task, timeout=55)
    final = " ".join(obj.get("text", "") for obj in events if obj["type"] == "final")
    print("SODA_NATIVE_LIVE_PASS", json.dumps({
        "audio_seconds": round(len(pcm) / 32000, 3),
        "partial_count": sum(obj["type"] == "partial" for obj in events),
        "final_count": sum(obj["type"] == "final" for obj in events),
        "final_length": len(final),
        "final_sha256_prefix": hashlib.sha256(final.encode("utf-8")).hexdigest()[:12],
        "elapsed_s": round(time.monotonic() - start, 2),
    }), flush=True)
    assert any(obj["type"] == "partial" for obj in events)
    assert final.strip()


if __name__ == "__main__":
    asyncio.run(main())
