"""Isolated native-server check: active profile changes apply to new sessions.

Uses a temporary profile and loopback port. Never writes live data/active_bias.json.
Run inside WSL using .venv-wsl/bin/python tests/test_profile_reload.py.
"""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.context_profile import profile_sha256, validate_profile, write_atomic


def status(base, path="/v1/adaptation/status"):
    try:
        with urllib.request.urlopen(base + path, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


async def session(url, expected):
    ws = await websockets.connect(url, open_timeout=5)
    ready = json.loads(await asyncio.wait_for(ws.recv(), 20))
    assert ready["type"] == "ready", ready
    assert ready["bias_profile_sha256"] == expected, ready
    return ws


async def finish(ws):
    # Exercise native config initialization and clean end-of-audio without
    # recording any microphone or borrowing a personal test transcript.
    await ws.send(bytes(3200))
    await ws.send("stop")
    async def drain():
        async for raw in ws:
            event = json.loads(raw)
            assert event["type"] != "error", event
            if event["type"] == "closed":
                return
        raise AssertionError("Missing closed event")
    await asyncio.wait_for(drain(), 20)
    await ws.close()


async def exercise(base, profile_path):
    url = base.replace("http:", "ws:") + "/v1/stt?lang=vi-VN"
    for _ in range(100):
        try:
            if (await asyncio.to_thread(status, base, "/health"))[0] == 200:
                break
        except (OSError, TimeoutError):
            pass
        await asyncio.sleep(.1)
    else:
        raise AssertionError("Staging server did not become ready")

    code, initial = await asyncio.to_thread(status, base)
    assert code == 200 and not initial["enabled"]
    assert initial["profile_sha256"] == "baseline"
    old_session = await session(url, "baseline")

    profile = validate_profile({"schema_version": 1, "language": "vi-VN",
                                "terms": [{"phrase": "xin chào", "boost": 0.0}]})
    expected = profile_sha256(profile)
    write_atomic(profile_path, profile)
    code, active = await asyncio.to_thread(status, base)
    assert code == 200 and active["enabled"] and active["phrase_count"] == 1
    assert active["profile_sha256"] == expected
    new_session = await session(url, expected)
    # The already-open baseline session remains alive while the file changes.
    await finish(old_session)
    await finish(new_session)
    print("BASELINE_TO_PROFILE", expected, flush=True)

    profile_path.unlink()
    code, reverted = await asyncio.to_thread(status, base)
    assert code == 200 and not reverted["enabled"]
    assert reverted["profile_sha256"] == "baseline"
    await finish(await session(url, "baseline"))
    print("ROLLBACK_NEW_SESSION baseline", flush=True)

    write_atomic(profile_path, {"schema_version": 1, "language": "vi-VN", "terms": []})
    code, invalid = await asyncio.to_thread(status, base)
    assert (code, invalid) == (503, {"detail": "invalid_bias_profile"})
    async with websockets.connect(url) as ws:
        error = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert error == {"type": "error", "code": "invalid_bias_profile"}, error
    profile_path.unlink()
    await finish(await session(url, "baseline"))
    print("INVALID_PROFILE refused; baseline recovered", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="soda-profile-reload-") as temporary:
        folder = Path(temporary)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "STT_BIAS_PROFILE": str(folder / "active.json")}
        with (folder / "server.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app", "--host",
                 "127.0.0.1", "--port", str(port), "--no-access-log"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                asyncio.run(exercise(f"http://127.0.0.1:{port}", folder / "active.json"))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        assert not (folder / "active.json").exists()
    print("PASS temporary staging config removed; live profile untouched", flush=True)


if __name__ == "__main__":
    main()
