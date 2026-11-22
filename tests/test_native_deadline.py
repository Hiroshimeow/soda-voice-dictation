"""Inject an owned hung native group into an isolated server, never live SODA."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def hung_worker(incoming, outgoing, aborted, locale, profile, digest):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (Path(tempfile.gettempdir()) / "owned-test.cfg").write_text("test-only")
    child = subprocess.Popen([sys.executable, "-u", "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(300)"],
        stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "ready"
    outgoing.send({"type": "ready", "locale": locale, "engine": "test_owned_hang",
                   "bias_profile_sha256": digest, "worker_pid": os.getpid(),
                   "helper_pid": child.pid, "temp_dir": tempfile.gettempdir()})
    while True:
        time.sleep(1)


def serve(port):
    import uvicorn
    import server
    from native_session import NativeSession

    class HungSession(NativeSession):
        def __init__(self, max_chunks=100):
            super().__init__(max_chunks, target=hung_worker)

    server.NativeSession = HungSession
    server.NATIVE_FINAL_TIMEOUT_SECONDS = .4
    uvicorn.run(server.app, host="127.0.0.1", port=port, access_log=False)


def health(base):
    with urllib.request.urlopen(base + "/health", timeout=3) as response:
        return json.load(response)


async def zero_sessions(base):
    for _ in range(100):
        if (await asyncio.to_thread(health, base))["active_sessions"] == 0:
            return
        await asyncio.sleep(.1)
    raise AssertionError("Capacity was not recovered")


async def gone(pids):
    for _ in range(50):
        if all(not Path(f"/proc/{pid}").exists() for pid in pids):
            return
        await asyncio.sleep(.1)
    raise AssertionError(("Owned process survived", pids))


async def open_session(url):
    ws = await websockets.connect(url)
    ready = json.loads(await asyncio.wait_for(ws.recv(), 10))
    assert ready["type"] == "ready", ready
    assert os.getpgid(ready["worker_pid"]) == ready["worker_pid"]
    assert os.getpgid(ready["helper_pid"]) == ready["worker_pid"]
    return ws, ready


async def exercise(base):
    for _ in range(100):
        try:
            if (await asyncio.to_thread(health, base))["ok"]:
                break
        except OSError:
            pass
        await asyncio.sleep(.1)
    else:
        raise AssertionError("Staging server not ready")
    url = base.replace("http:", "ws:") + "/v1/stt?lang=vi-VN"
    ws, ready = await open_session(url)
    # Valid open-session silence must outlive the short test finalization limit.
    for _ in range(12):
        await ws.send(bytes(3200))
        await asyncio.sleep(.1)
    assert (await asyncio.to_thread(health, base))["active_sessions"] == 1
    began = time.monotonic()
    await ws.send("stop")
    events = []
    async def drain():
        async for raw in ws:
            event = json.loads(raw)
            events.append(event)
            if event["type"] == "closed":
                return
    await asyncio.wait_for(drain(), 8)
    await ws.close()
    await zero_sessions(base)
    await gone([ready["worker_pid"], ready["helper_pid"]])
    assert not Path(ready["temp_dir"]).exists(), "Owned config directory leaked"
    assert any(event.get("code") == "native_finalization_timeout" for event in events), events
    print("HUNG_FINALIZATION_RECOVERED_SECONDS", round(time.monotonic() - began, 3), flush=True)

    ws, ready = await open_session(url)
    began = time.monotonic()
    await ws.close()
    await zero_sessions(base)
    await gone([ready["worker_pid"], ready["helper_pid"]])
    assert not Path(ready["temp_dir"]).exists(), "Owned config directory leaked"
    print("HUNG_DISCONNECT_RECOVERED_SECONDS", round(time.monotonic() - began, 3), flush=True)

    # Admission after both failures demonstrates slot recovery, not just PID exit.
    ws, ready = await open_session(url)
    await ws.close()
    await zero_sessions(base)
    await gone([ready["worker_pid"], ready["helper_pid"]])
    assert not Path(ready["temp_dir"]).exists(), "Owned config directory leaked"
    print("PASS valid silence preserved; hung parent+helper reaped; capacity recovered", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="soda-deadline-") as temporary:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "STT_BIAS_PROFILE": str(Path(temporary) / "never-active.json")}
        with (Path(temporary) / "server.log").open("w") as log:
            process = subprocess.Popen([sys.executable, __file__, "--serve", str(port)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                asyncio.run(exercise(f"http://127.0.0.1:{port}"))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(int(sys.argv[2]))
    else:
        main()
