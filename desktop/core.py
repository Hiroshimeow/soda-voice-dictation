"""Pure state plus asynchronous audio/WebSocket pipeline for local Chrome SODA."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import sounddevice as sd
from websockets.asyncio.client import connect

from desktop.audio import gain_pcm16, validate_gain

SAMPLE_RATE = 16000
BLOCK_FRAMES = 1600
MAX_SAMPLE_BYTES = 30 * SAMPLE_RATE * 2
STARTUP_BUFFER_SECONDS = 4
MAX_PENDING_FRAMES = int(STARTUP_BUFFER_SECONDS * SAMPLE_RATE / BLOCK_FRAMES)


@dataclass
class DictationBuffer:
    segments: list[str] = field(default_factory=list)
    pending: str = ""

    def partial(self, text: str) -> None:
        self.pending = text.strip()

    def final(self, text: str) -> None:
        text = text.strip()
        if text:
            self.segments.append(text)
        self.pending = ""

    @property
    def committed(self) -> str:
        return " ".join(self.segments)

    @property
    def preview(self) -> str:
        return " ".join(filter(None, (*self.segments, self.pending)))


@dataclass
class AudioSample:
    """Only an entire <=30-second session may be paired with its corrected text."""
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    over_limit: bool = False

    def add(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > MAX_SAMPLE_BYTES:
            self.chunks.clear()
            self.over_limit = True
        if not self.over_limit:
            self.chunks.append(chunk)

    def get_if_eligible(self) -> bytes | None:
        if not self.over_limit and SAMPLE_RATE <= self.size <= MAX_SAMPLE_BYTES:
            return b"".join(self.chunks)
        return None


class DictationSession:
    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:18765/v1/stt?lang=vi-VN",
        microphone: Callable = sd.RawInputStream,
        on_event: Callable[[dict], None] | None = None,
        device: int | None = None,
        gain_db: float = 0,
        final_timeout: float = 45,
    ):
        self.url = url
        self.microphone = microphone
        self.emit = on_event or (lambda _event: None)
        self.device = device
        self.gain_db = validate_gain(gain_db)
        self.final_timeout = final_timeout
        self.stop_requested = threading.Event()
        self.cancel_requested = threading.Event()
        self.buffer = DictationBuffer()
        self.sample = AudioSample()
        self.error: str | None = None
        self._loop = None
        self._runner_task = None

    def stop(self) -> None:
        self.stop_requested.set()

    def cancel(self) -> None:
        if self.cancel_requested.is_set():
            return
        self.cancel_requested.set()
        self.stop_requested.set()
        # Wake a connection, final-result wait, or pending microphone read promptly.
        if self._loop is not None and self._runner_task is not None:
            try:
                self._loop.call_soon_threadsafe(self._runner_task.cancel)
            except RuntimeError:
                pass  # The loop has already completed and released its resources.

    def _event(self, kind: str, **fields) -> None:
        self.emit({"type": kind, **fields})

    async def _read_event(self, socket, timeout: float | None = None) -> dict:
        raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("type"), str):
            raise RuntimeError("Invalid SODA response")
        if data["type"] in ("partial", "final") and not isinstance(data.get("text"), str):
            raise RuntimeError("Invalid SODA transcript")
        return data

    async def _read_results(self, socket) -> None:
        while True:
            # Silence is valid during recording. WebSocket ping checks liveness;
            # a final-result deadline is applied only after the user presses Stop.
            item = await self._read_event(socket)
            kind = item["type"]
            if kind == "partial":
                self.buffer.partial(item["text"])
                self._event("transcript", text=self.buffer.preview, final=False)
            elif kind == "final":
                self.buffer.final(item["text"])
                self._event("transcript", text=self.buffer.preview, final=True)
            elif kind == "error":
                self.error = str(item.get("code", "soda_error"))
                self.stop_requested.set()
                self._event("status", text=f"SODA error: {self.error}")
                return
            elif kind == "closed":
                if not self.stop_requested.is_set():
                    raise RuntimeError("SODA closed during the session; the transcript was preserved")
                return

    async def _wait_ready(self, socket, deadline: float) -> None:
        # Stop ends microphone capture but still delivers the complete buffered
        # utterance. Cancel interrupts the runner and discards the entire session.
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError("SODA was not ready after 4 seconds; the session was stopped. Try again.")
            try:
                item = await self._read_event(socket, min(.25, remaining))
            except asyncio.TimeoutError:
                continue
            if item["type"] == "ready":
                return
            if item["type"] == "error":
                raise RuntimeError(f"SODA: {item.get('code', 'unknown')}")
            if item["type"] == "closed":
                raise RuntimeError("SODA closed before becoming ready")

    async def _capture_audio(self, pending: asyncio.Queue, opened: asyncio.Future) -> None:
        if self.stop_requested.is_set():
            return
        with self.microphone(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=BLOCK_FRAMES, device=self.device,
        ) as stream:
            if self.stop_requested.is_set():
                return
            opened.set_result(asyncio.get_running_loop().time())
            self._event("status", text="Listening - starting SODA...",
                        capture_ready=True)
            while not self.stop_requested.is_set():
                read_task = asyncio.create_task(asyncio.to_thread(stream.read, BLOCK_FRAMES))
                try:
                    raw, overflow = await asyncio.shield(read_task)
                except asyncio.CancelledError:
                    abort = getattr(stream, "abort", None)
                    if abort is not None:
                        abort()
                    try:
                        await asyncio.wait_for(read_task, timeout=1)
                    except (Exception, asyncio.CancelledError):
                        pass
                    raise
                if overflow:
                    raise RuntimeError("Microphone overflow; audio is incomplete")
                chunk = bytes(raw)
                if len(chunk) != BLOCK_FRAMES * 2:
                    raise RuntimeError("Microphone returned an invalid frame size")
                chunk, levels = gain_pcm16(chunk, self.gain_db)
                try:
                    pending.put_nowait(chunk)
                except asyncio.QueueFull as exc:
                    raise RuntimeError(
                        "The 4-second audio buffer filled up; the session was stopped to avoid losing the beginning."
                    ) from exc
                self.sample.add(chunk)
                self._event("level", **levels)
        # Exiting the context stops physical capture even while the network
        # still connects, drains the buffered utterance, or awaits final text.

    async def _stream_audio(self, pending: asyncio.Queue, capture_task, deadline: float) -> None:
        result_task = None
        ready = False
        try:
            async with connect(
                self.url, open_timeout=max(.01, deadline - asyncio.get_running_loop().time()),
                close_timeout=2, max_size=262144, ping_interval=15, ping_timeout=15,
            ) as socket:
                await self._wait_ready(socket, deadline)
                ready = True
                self._event(
                    "status", text="Finalizing recognition..." if self.stop_requested.is_set()
                    else "Listening...", backend_ready=True,
                )
                result_task = asyncio.create_task(self._read_results(socket))
                next_send = time.perf_counter()
                timer_resolution = time.get_clock_info("monotonic").resolution
                while not (capture_task.done() and pending.empty()):
                    if result_task.done():
                        result_task.result()
                        return
                    try:
                        chunk = await asyncio.wait_for(pending.get(), timeout=.1)
                    except asyncio.TimeoutError:
                        continue
                    delay = next_send - time.perf_counter()
                    if delay > 0:
                        # Windows asyncio timers may wake one clock-resolution
                        # early. Include that resolution to keep the 2x bound.
                        await asyncio.sleep(delay + timer_resolution)
                    if result_task.done():
                        result_task.result()
                        return
                    await asyncio.wait_for(socket.send(chunk), timeout=3)
                    # Catch up retained 100 ms frames at no more than 2x.
                    # Once the queue is empty, microphone reads pace live audio.
                    next_send = time.perf_counter() + BLOCK_FRAMES / SAMPLE_RATE / 2
                if result_task.done():
                    result_task.result()
                else:
                    self._event("status", text="Finalizing recognition...")
                    await asyncio.wait_for(socket.send("stop"), timeout=3)
                    try:
                        await asyncio.wait_for(result_task, timeout=self.final_timeout)
                    except asyncio.TimeoutError as exc:
                        raise RuntimeError(
                            "Timed out waiting for the final result; the transcript was preserved"
                        ) from exc
        except asyncio.TimeoutError as exc:
            if not ready:
                raise RuntimeError(
                    "SODA was not ready after 4 seconds; the session was stopped. Try again."
                ) from exc
            raise RuntimeError("Timed out sending audio to SODA; the session was stopped.") from exc
        finally:
            if result_task:
                if not result_task.done():
                    result_task.cancel()
                await asyncio.gather(result_task, return_exceptions=True)

    async def run(self) -> None:
        capture_task = network_task = None
        self._loop = asyncio.get_running_loop()
        self._runner_task = asyncio.current_task()
        opened = self._loop.create_future()
        try:
            if self.cancel_requested.is_set() or self.stop_requested.is_set():
                return
            self._event("status", text="Opening microphone...")
            pending = asyncio.Queue(maxsize=MAX_PENDING_FRAMES)
            capture_task = asyncio.create_task(self._capture_audio(pending, opened))
            await asyncio.wait((capture_task, opened), return_when=asyncio.FIRST_COMPLETED)
            if capture_task.done():
                capture_task.result()
            if not opened.done():
                return  # Stop arrived while the device was opening.
            network_task = asyncio.create_task(
                self._stream_audio(pending, capture_task, opened.result() + STARTUP_BUFFER_SECONDS)
            )
            done, _ = await asyncio.wait(
                (capture_task, network_task), return_when=asyncio.FIRST_COMPLETED
            )
            if capture_task in done:
                capture_task.result()  # Surface capture/overflow failures promptly.
            await network_task
            self.stop_requested.set()
            await capture_task
        except asyncio.CancelledError:
            self.cancel_requested.set()
            self.stop_requested.set()
        except Exception as exc:
            if not self.cancel_requested.is_set():
                self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self.stop_requested.set()
            tasks = [task for task in (capture_task, network_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if not opened.done():
                opened.cancel()
            cancelled = self.cancel_requested.is_set()
            self._runner_task = None
            self._loop = None
            self._event(
                "complete",
                text="" if cancelled or self.error else self.buffer.committed,
                preview=self.buffer.preview,
                cancelled=cancelled,
                error=None if cancelled else self.error,
                pcm=None if cancelled or self.error else self.sample.get_if_eligible(),
                sample_eligible=not self.sample.over_limit,
            )
