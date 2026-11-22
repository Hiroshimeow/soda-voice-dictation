"""Native Google SODA service: 16 kHz, mono, signed little-endian PCM."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import queue
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from training.context_profile import ACTIVE_PATH, load_profile, profile_sha256
from training.personal_vocabulary import load_vocabulary, to_soda_profile, merge_profile
from training.dataset import dataset_count, save_sample
from native_session import NativeSession
from soda_languages import LANGUAGES

LANGUAGES = frozenset(LANGUAGES)
AUDIO_RATE = 16000
MAX_FRAME = 65536
MAX_BUFFERED_CHUNKS = 100
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))
if not 1 <= MAX_CONCURRENT <= 8:
    raise ValueError("MAX_CONCURRENT must be between 1 and 8")
IDLE_TIMEOUT_SECONDS = 300
NATIVE_READY_TIMEOUT_SECONDS = 30
NATIVE_FINAL_TIMEOUT_SECONDS = 15
NETWORK_CONFIG = Path(__file__).parent / "data" / "network.json"


def allowed_origins() -> frozenset[str]:
    origins = {"http://127.0.0.1:18765", "http://localhost:18765"}
    if NETWORK_CONFIG.is_file():
        config = json.loads(NETWORK_CONFIG.read_text(encoding="utf-8-sig"))
        extra = config.get("allowed_origins", [])
        if not isinstance(extra, list):
            raise ValueError("allowed_origins must be a list")
        origins.update(extra)
    origins.update(value.strip().rstrip("/") for value in
                   os.getenv("STT_ALLOWED_ORIGINS", "").split(",") if value.strip())
    for origin in origins:
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or
                parsed.path or parsed.query or parsed.fragment or parsed.username):
            raise ValueError("Invalid configured origin")
    return frozenset(origins)


ALLOWED_ORIGINS = allowed_origins()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("soda.server")
app = FastAPI(title="Google SODA Native STT", version="0.2.0")
slots = asyncio.Semaphore(MAX_CONCURRENT)
active_workers = 0
BIAS_FILE = Path(os.environ.get("STT_BIAS_PROFILE", str(ACTIVE_PATH)))


def bias_snapshot() -> tuple[dict | None, str]:
    """Read one validated active profile. Each session keeps its own snapshot."""
    try:
        profile = load_profile(BIAS_FILE)
    except FileNotFoundError:
        # Rollback may remove the file between is_file() and read_text().
        profile = None
    personal = to_soda_profile(load_vocabulary())
    effective = merge_profile(profile, personal)
    return effective, profile_sha256(effective)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "backend": "chrome_soda_native",
        "runtime": "linux_x86_64",
        "audio_format": "pcm_s16le",
        "sample_rate": AUDIO_RATE,
        "channels": 1,
        "languages": sorted(LANGUAGES),
        "active_sessions": active_workers,
        "max_sessions": MAX_CONCURRENT,
        "model_status": "inspect_per_session",
        "note": "This is Chrome SODA, not verified as identical to Android ML Kit.",
    }


@app.get("/v1/adaptation/status")
def adaptation_status() -> dict:
    try:
        profile, digest = bias_snapshot()
    except (OSError, ValueError) as exc:
        raise HTTPException(503, detail="invalid_bias_profile") from exc
    vocabulary = load_vocabulary()
    return {"language": "vi-VN", "enabled": profile is not None,
            "profile_sha256": digest,
            "phrase_count": len(profile["terms"]) if profile else 0,
            "user_phrase_count": len(vocabulary["phrases"]),
            "type": "context_biasing_not_acoustic_finetuning"}


@app.get("/v1/dataset/status")
def dataset_status() -> dict:
    return {"language": "vi-VN", "saved_samples": dataset_count(),
            "minimum_for_calibration": 30,
            "privacy": "audio saved locally only after explicit consent"}


@app.post("/v1/dataset/ingest")
async def dataset_ingest(request: Request) -> dict:
    if request.headers.get("origin") not in ALLOWED_ORIGINS:
        raise HTTPException(403, detail="same_origin_required")
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, detail="application_json_required")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > 1_500_000:
            raise HTTPException(413, detail="body_too_large")
        chunks.append(chunk)
    try:
        obj = json.loads(b"".join(chunks))
        if not isinstance(obj, dict):
            raise ValueError("JSON object required")
        pcm = base64.b64decode(obj.get("pcm_base64", ""), validate=True)
        return save_sample(obj, pcm)
    except (ValueError, KeyError, TypeError, OverflowError, binascii.Error) as exc:
        raise HTTPException(422, detail=str(exc)[:160]) from exc


@app.websocket("/v1/stt")
async def stt(ws: WebSocket) -> None:
    """One bounded worker per connection. A normal stop drains all accepted PCM."""
    global active_workers
    origin = ws.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008, reason="invalid_origin")
        return
    await ws.accept()
    locale = ws.query_params.get("lang", "vi-VN")
    if locale not in LANGUAGES:
        await ws.send_json({"type": "error", "code": "unsupported_language"})
        await ws.close(code=1008)
        return
    if slots.locked():
        await ws.send_json({"type": "error", "code": "capacity_exceeded"})
        await ws.close(code=1013)
        return
    await slots.acquire()
    active_workers += 1
    native = NativeSession(max_chunks=MAX_BUFFERED_CHUNKS)
    audio_queue = native.audio
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_BUFFERED_CHUNKS)
    loop = asyncio.get_running_loop()
    aborted = native.aborted
    input_finished = native.input_finished
    worker_finished = threading.Event()
    sender_finished = False
    event_overflow = False

    def emit(event: dict[str, Any]) -> None:
        def add() -> None:
            nonlocal event_overflow
            if sender_finished or event_overflow:
                return
            if events.full():
                # A slow recipient cannot retain an unbounded result backlog.
                # Wake the sender with a terminal error instead of leaving it blocked.
                event_overflow = True
                aborted.set()
                while not events.empty():
                    events.get_nowait()
                events.put_nowait({"type": "error", "code": "result_backpressure"})
                events.put_nowait({"type": "closed"})
                return
            events.put_nowait(event)
        if not loop.is_closed():
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(add)

    def worker() -> None:
        try:
            try:
                profile, digest = bias_snapshot() if locale == "vi-VN" else (None, "baseline")
            except (OSError, ValueError):
                emit({"type": "error", "code": "invalid_bias_profile"})
                return
            if not native.start(locale, profile, digest):
                return
            while not aborted.is_set():
                try:
                    event = native.read_event(timeout=.1)
                except EOFError:
                    if not aborted.is_set():
                        emit({"type": "error", "code": "native_worker_exited"})
                    break
                if event is None:
                    if not native.is_alive():
                        emit({"type": "error", "code": "native_worker_exited"})
                        break
                    continue
                if event["type"] == "ready":
                    native.ready.set()
                if event["type"] == "closed":
                    break
                emit(event)
        except Exception as exc:
            if not aborted.is_set():
                log.warning("Recognizer failed: locale=%s class=%s", locale, type(exc).__name__)
                emit({"type": "error", "code": "recognition_failed"})
        finally:
            try:
                native.close()
            except RuntimeError:
                log.error("Native session cleanup failed")
                emit({"type": "error", "code": "native_cleanup_failed"})
                if native.is_alive():
                    return  # Keep capacity reserved if the OS cannot reap it.
            worker_finished.set()
            emit({"type": "closed"})

    async def watch_native() -> None:
        started = time.monotonic()
        final_started = None
        while not worker_finished.is_set():
            if aborted.is_set():
                await asyncio.to_thread(native.close)
                return
            now = time.monotonic()
            if input_finished.is_set() and final_started is None:
                final_started = now
            code = None
            if final_started is not None and now - final_started > NATIVE_FINAL_TIMEOUT_SECONDS:
                code = "native_finalization_timeout"
            elif not native.ready.is_set() and now - started > NATIVE_READY_TIMEOUT_SECONDS:
                code = "native_initialization_timeout"
            if code:
                emit({"type": "error", "code": code})
                aborted.set()
                await asyncio.to_thread(native.close)
                return
            # Silence during an open microphone session has no final deadline.
            await asyncio.sleep(.1)

    async def receive_audio() -> None:
        try:
            while not aborted.is_set():
                message = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_SECONDS)
                if message["type"] == "websocket.disconnect":
                    aborted.set()
                    break
                data = message.get("bytes")
                if data is not None:
                    if not data or len(data) > MAX_FRAME or len(data) % 2:
                        emit({"type": "error", "code": "invalid_pcm_frame"})
                        aborted.set()
                        break
                    try:
                        audio_queue.put_nowait(data)
                    except queue.Full:
                        emit({"type": "error", "code": "audio_backpressure"})
                        aborted.set()
                        break
                elif message.get("text") == "stop":
                    break
                else:
                    emit({"type": "error", "code": "expected_pcm_or_stop"})
                    aborted.set()
                    break
        except asyncio.TimeoutError:
            emit({"type": "error", "code": "audio_idle_timeout"})
            aborted.set()
        except (WebSocketDisconnect, RuntimeError):
            aborted.set()
        finally:
            # Out-of-band completion cannot deadlock behind a full audio queue.
            input_finished.set()

    def release_slot() -> None:
        global active_workers
        active_workers -= 1
        slots.release()

    async def release_when_finished() -> None:
        await asyncio.to_thread(worker_finished.wait)
        release_slot()

    thread = threading.Thread(target=worker, name="soda-session", daemon=True)
    receiver: asyncio.Task | None = None
    try:
        thread.start()
        receiver = asyncio.create_task(receive_audio())
        asyncio.create_task(watch_native())
        while True:
            event = await asyncio.wait_for(events.get(), timeout=IDLE_TIMEOUT_SECONDS + 10)
            await asyncio.wait_for(ws.send_json(event), timeout=10)
            if event["type"] == "closed":
                break
    except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
        pass
    finally:
        sender_finished = True
        aborted.set()
        input_finished.set()
        if receiver:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver
        with suppress(Exception):
            await ws.close()
        if thread.ident is None or worker_finished.is_set():
            release_slot()
        else:
            # Hold capacity until native cleanup actually finishes. A disconnected
            # client cannot create unlimited helpers by reconnecting rapidly.
            asyncio.create_task(release_when_finished())
