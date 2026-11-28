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
from training.personal_vocabulary import load_vocabulary, build_effective_profile
from training.dataset import dataset_count, save_sample
from native_session import NativeSession
from soda_languages import LANGUAGES
from vi_enhance.postprocessor import PostProcessConfig, VietnamesePostProcessor
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


def worker_terminal_error(*, worker_status_seen: bool, worker_status_ok: bool,
                          closed_seen: bool, exitcode: int | None) -> str | None:
    """Return the one protocol error required for an incomplete worker.

    A zero exit code is not enough: normal completion requires both an
    explicit successful worker_status and the closed marker. Nonzero exits
    retain their typed native_worker_exited diagnosis.
    """
    # A process exit status is authoritative even when buffered protocol
    # messages contained a misleading success marker.
    if exitcode not in (None, 0):
        return "native_worker_exited"
    if worker_status_seen and worker_status_ok and closed_seen:
        return None
    if closed_seen and worker_status_seen:
        return "recognition_failed"
    return "native_worker_protocol"


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
app = FastAPI(title="Google SODA Native STT", version="0.3.0")
slots = asyncio.Semaphore(MAX_CONCURRENT)
active_workers = 0
BIAS_FILE = Path(os.environ.get("STT_BIAS_PROFILE", str(ACTIVE_PATH)))


def bias_snapshot(*, vocabulary: dict | None = None) -> tuple[dict | None, str]:
    """Build the one effective profile snapshot used by a session."""
    try:
        profile = load_profile(BIAS_FILE)
    except FileNotFoundError:
        # Rollback may remove the file between is_file() and read_text().
        profile = None
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    effective = build_effective_profile(vocab, curated_profile=profile)
    # The smart builder is the single profile boundary.  Do not fall back to
    # the legacy full-vocabulary conversion here: it would reintroduce a
    # session-fatal validation error for one unsafe candidate.  A baseline
    # digest is an explicit diagnostic for the all-rejected case; this is not
    # equivalent to silently disabling adaptation.
    if effective is None and vocab["phrases"]:
        log.warning("Vietnamese adaptation baseline: no validator-safe vocabulary candidates")
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
        "checkout_id": os.getenv("SODA_CHECKOUT_ID", "unknown"),
        "channels": 1,
        "languages": sorted(LANGUAGES),
        "active_sessions": active_workers,
        "max_sessions": MAX_CONCURRENT,
        "model_status": "inspect_per_session",
        "vi_enhancements": "active_model_agnostic",
        "note": "This is Chrome SODA, not verified as identical to Android ML Kit.",
    }


@app.get("/v1/enhancement/status")
def enhancement_status() -> dict:
    return {
        "language": "vi-VN",
        "model_weights_modified": False,
        "modules": [
            "tonal_safe_audio_frontend_oa",
            "hat_trailing_silence_flush",
            "smart_dynamic_bias_ranker_40cap",
            "vietnamese_phonetic_vocab_rescue",
            "vietnamese_code_switch_rescue",
            "vietnamese_itn_safe_and_full",
            "boundary_overlap_stitcher",
            "closed_loop_confusion_miner",
        ],
    }


@app.get("/v1/adaptation/status")
def adaptation_status() -> dict:
    try:
        vocabulary = load_vocabulary()
        profile, digest = bias_snapshot(vocabulary=vocabulary)
    except (OSError, ValueError) as exc:
        raise HTTPException(503, detail="invalid_bias_profile") from exc
    return {"language": "vi-VN", "enabled": profile is not None,
            "profile_sha256": digest,
            "phrase_count": len(profile["terms"]) if profile else 0,
            "user_phrase_count": len(vocabulary["phrases"]),
            "type": "smart_context_biasing_not_acoustic_finetuning"}


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
    mode_param = ws.query_params.get("mode", "caption")
    if mode_param not in ("caption", "ime"):
        mode_param = "caption"
    itn_param = ws.query_params.get("itn", "safe")
    if itn_param not in ("off", "safe", "full"):
        itn_param = "safe"
    # Post-processing is opt-in for raw WebSocket clients. The desktop client
    # explicitly requests it; this keeps protocol consumers from receiving an
    # unexpected rewrite while still exposing the enhancement pipeline.
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
    closed_emitted = False
    event_overflow = False
    worker_status_ok = False
    worker_status_seen = False
    closed_seen = False
    terminal_error_sent = False
    # This is deliberately separate from NativeSession.aborted: close() sets
    # the latter during every cleanup, including successful completion.  Only
    # client/network cancellation suppresses the terminal worker diagnosis.
    terminal_suppressed = threading.Event()
    # Positive allowlist: unknown and missing values preserve faithful native
    # text rather than unexpectedly enabling semantic rewrites.
    postprocess_enabled = ws.query_params.get("postprocess", "0").strip().casefold() in {"1", "true", "on"}
    vi_processor = (
        VietnamesePostProcessor(PostProcessConfig(itn_mode=itn_param))
        if (locale == "vi-VN" and postprocess_enabled)
        else None
    )

    def emit(event: dict[str, Any]) -> None:
        def add() -> None:
            nonlocal event_overflow, closed_emitted
            if sender_finished or event_overflow:
                return
            if events.full():
                # A slow recipient cannot retain an unbounded result backlog.
                # Wake the sender with a terminal error instead of leaving it blocked.
                event_overflow = True
                closed_emitted = True
                terminal_suppressed.set()
                aborted.set()
                while not events.empty():
                    events.get_nowait()
                events.put_nowait({"type": "error", "code": "result_backpressure"})
                events.put_nowait({"type": "closed"})
                return
            if event.get("type") == "closed":
                if closed_emitted:
                    return
                closed_emitted = True
            events.put_nowait(event)
        if not loop.is_closed():
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(add)

    def worker() -> None:
        nonlocal worker_status_ok, worker_status_seen, closed_seen, terminal_error_sent
        try:
            try:
                if locale == "vi-VN":
                    vocab_snapshot = load_vocabulary()
                    profile, digest = bias_snapshot(vocabulary=vocab_snapshot)
                else:
                    profile, digest = None, "baseline"
                    vocab_snapshot = None
            except (OSError, ValueError):
                # Configuration is a terminal pre-worker failure. Mark it as
                # diagnosed before returning so the finally block emits only
                # the protocol's single error followed by closed; the native
                # worker was never started and must not be reported as a
                # protocol failure.
                emit({"type": "error", "code": "invalid_bias_profile"})
                terminal_error_sent = True
                return
            soda_opts = {
                "recognition_mode": mode_param,
                "enable_formatting": True,
                "trailing_silence_ms": int(os.getenv("SODA_TRAILING_SILENCE_MS", "0")),
            }
            if not native.start(locale, profile, digest, options=soda_opts):
                if not terminal_suppressed.is_set() and not terminal_error_sent:
                    emit({"type": "error", "code": "native_worker_protocol"})
                    terminal_error_sent = True
                return
            while not aborted.is_set():
                try:
                    event = native.read_event(timeout=.1)
                except EOFError:
                    break
                if event is None:
                    if not native.is_alive():
                        # Defer diagnosis until NativeSession.close() has
                        # reaped the worker. A closed marker can be buffered
                        # immediately before a nonzero process exit.
                        break
                    continue
                if event["type"] == "ready":
                    native.ready.set()
                if event["type"] == "worker_status":
                    worker_status_seen = True
                    worker_status_ok = bool(event.get("ok", False))
                    if not worker_status_ok and not terminal_suppressed.is_set() and not terminal_error_sent:
                        error = {"type": "error", "code": event.get("code", "recognition_failed")}
                        if event.get("message"):
                            error["message"] = event["message"]
                        emit(error)
                        terminal_error_sent = True
                    continue
                if event["type"] == "closed":
                    closed_seen = True
                    # The final worker diagnosis is emitted after cleanup
                    # reaps the process, so a buffered closed marker cannot
                    # hide a nonzero exit.
                    break
                if event["type"] == "final" and event.get("text"):
                    if vi_processor is None:
                        emit(event)
                    else:
                        # SODA finals are independent committed utterances;
                        # never stitch them into a synthetic transcript.
                        raw_text = event["text"]
                        postprocess_status = "unchanged"
                        try:
                            cleaned, _meta = vi_processor.process(raw_text, vocabulary=vocab_snapshot)
                            postprocess_status = "applied" if cleaned != raw_text else "unchanged"
                        except Exception:
                            # Preserve the native text for observability, but mark
                            # the delivery as failed; clients must not treat it as
                            # a successful postprocessed final.
                            cleaned = raw_text
                            postprocess_status = "failed"
                        emit({**event, "text": cleaned, "raw_text": raw_text,
                              "postprocess_applied": postprocess_status == "applied",
                              "postprocess_status": postprocess_status})
                    continue
                emit(event)
        except Exception as exc:
            if not terminal_suppressed.is_set() and not terminal_error_sent:
                log.warning("Recognizer failed: locale=%s class=%s", locale, type(exc).__name__)
                emit({"type": "error", "code": "recognition_failed"})
                terminal_error_sent = True
        finally:
            cleanup_reported = False
            while True:
                try:
                    native.close()
                    break
                except RuntimeError:
                    if not cleanup_reported:
                        log.error("Native session cleanup failed")
                        if not terminal_error_sent:
                            emit({"type": "error", "code": "native_cleanup_failed"})
                            terminal_error_sent = True
                        cleanup_reported = True
                    # Retry cleanup rather than returning from finally. The
                    # worker_finished event is the sole release signal, so the
                    # capacity slot remains reserved until all owned resources
                    # have actually been reaped.
                    time.sleep(.1)
            # close() has reaped the owned worker, so this is the final
            # authoritative exit-code check.  In particular, a child may have
            # sent worker_status/closed before dying nonzero; that must not be
            # observable as a successful ready->closed session.
            if not terminal_suppressed.is_set() and not terminal_error_sent:
                code = worker_terminal_error(
                    worker_status_seen=worker_status_seen,
                    worker_status_ok=worker_status_ok,
                    closed_seen=closed_seen,
                    exitcode=native.exitcode,
                )
                if code:
                    emit({"type": "error", "code": code})
                    terminal_error_sent = True
            worker_finished.set()
            # Always enqueue through emit: its event-loop boundary is the
            # single closed-event guard, so a native closed event and this
            # cleanup path cannot race into duplicate terminal markers.
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
                terminal_suppressed.set()
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
                    terminal_suppressed.set()
                    aborted.set()
                    break
                data = message.get("bytes")
                if data is not None:
                    if not data or len(data) > MAX_FRAME or len(data) % 2:
                        emit({"type": "error", "code": "invalid_pcm_frame"})
                        terminal_suppressed.set()
                        aborted.set()
                        break
                    try:
                        audio_queue.put_nowait(data)
                    except queue.Full:
                        emit({"type": "error", "code": "audio_backpressure"})
                        terminal_suppressed.set()
                        aborted.set()
                        break
                else:
                    text_message = message.get("text")
                    is_eof = text_message == "stop"
                    if isinstance(text_message, str) and not is_eof:
                        try:
                            control = json.loads(text_message)
                            is_eof = isinstance(control, dict) and control.get("type") == "eof"
                        except (TypeError, ValueError, AttributeError, RecursionError):
                            is_eof = False
                    if is_eof:
                        break
                    emit({"type": "error", "code": "expected_pcm_or_stop"})
                    terminal_suppressed.set()
                    aborted.set()
                    break
        except asyncio.TimeoutError:
            emit({"type": "error", "code": "audio_idle_timeout"})
            terminal_suppressed.set()
            aborted.set()
        except (WebSocketDisconnect, RuntimeError):
            terminal_suppressed.set()
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
        # Sender-side cancellation/disconnect is cleanup, not a native worker
        # diagnosis. The receiver normally marks this too, but send failures
        # can reach this boundary first.
        terminal_suppressed.set()
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
