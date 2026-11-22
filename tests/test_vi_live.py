"""Real WebSocket acceptance test using a user-independent sample WAV.

Usage in WSL:
    .venv-wsl/bin/python tests/test_vi_live.py
"""
import asyncio
import json
import os
import time
import wave
from pathlib import Path

import websockets

WAV = Path("/mnt/e/git-project/live-transcribe-stt-lab/models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/test_wavs/0.wav")
URL = os.environ.get("STT_TEST_URL", "ws://127.0.0.1:18765/v1/stt?lang=vi-VN")


async def main():
    with wave.open(str(WAV), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())

    start = time.monotonic()
    partial = []
    final = []
    async with websockets.connect(URL) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
        assert ready["type"] == "ready", ready

        async def receiver():
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "partial":
                    partial.append(msg["text"])
                    if len(partial) == 1:
                        print("FIRST_PARTIAL_S", round(time.monotonic() - start, 3))
                elif msg["type"] == "final":
                    final.append(msg["text"])
                    print("FINAL_S", round(time.monotonic() - start, 3))
                    print("TEXT", msg["text"])
                elif msg["type"] == "error":
                    raise AssertionError(msg)
                elif msg["type"] == "closed":
                    break

        read_task = asyncio.create_task(receiver())
        for offset in range(0, len(pcm), 3200):
            await ws.send(pcm[offset:offset + 3200])
            await asyncio.sleep(0.10)
        await asyncio.sleep(1.0)
        await ws.send("stop")
        await asyncio.wait_for(read_task, timeout=35)
    assert partial, "No partial transcription"
    assert final and "hỗ trợ" in final[-1].lower(), f"Unexpected final: {final}"
    print("PASS", len(partial), "partial events", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
