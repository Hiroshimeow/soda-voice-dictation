"""Own a SODA worker process group without modifying the upstream adapter.

The service may terminate only this freshly spawned group on cancellation or a
finalization deadline. PCM and results cross private pipes; no cloud request or
transcript persistence is introduced.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
import os
import struct
import queue
import signal
import tempfile
import threading
from contextlib import suppress
from typing import Callable

from soda_languages import LANGUAGES

TRAILING_SILENCE_CHUNK_MS = 100


def _require_native_terminal(received_audio: bool, saw_final: bool, aborted: bool) -> None:
    """Compatibility hook; terminal success is owned by worker/process status.

    Audio presence and transcript text are intentionally not terminal signals.
    """
    return None


class _CheckedStdout:
    """Check the installed adapter's length-prefixed helper protocol."""
    def __init__(self, stream):
        self._stream = stream
        self.truncated = False
        self._payload_length = None

    def read(self, size=-1):
        data = self._stream.read(size)
        if self._payload_length is None:
            if size == 4 and data and len(data) != 4:
                self.truncated = True
            if len(data) == 4:
                self._payload_length = struct.unpack("<I", data)[0]
        else:
            if len(data) != self._payload_length:
                self.truncated = True
            self._payload_length = None
        return data

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _CheckedPopen:
    def __init__(self, process):
        self._process = process
        self.stdout = _CheckedStdout(process.stdout)

    def __getattr__(self, name):
        return getattr(self._process, name)


def _native_stream_status(model, chunks, *, realtime_factor: float = 2.0):
    """Yield results and validate helper exit/protocol without editing soda_stt.

    The installed adapter silently returns on a short frame and ignores the
    helper return code.  Its Popen is wrapped only while this child consumes the
    iterator, making those two boundary failures visible to NativeSession.
    Adapters that provide an explicit status iterator are also supported.
    """
    stream = getattr(model, "stream_pcm_status", None)
    if stream is not None:
        results = stream(chunks, realtime_factor=realtime_factor)
        for item in results:
            if isinstance(item, dict) and item.get("type") == "adapter_status":
                if item.get("returncode") not in (None, 0):
                    raise RuntimeError("native helper exited with status " + str(item["returncode"]))
                if item.get("truncated"):
                    raise RuntimeError("native helper emitted a truncated frame")
                continue
            yield item
        return

    checked = []
    stream_globals = getattr(model.stream_pcm, "__globals__", {})
    subprocess_module = stream_globals.get("subprocess")
    original_popen = getattr(subprocess_module, "Popen", None)
    if subprocess_module is None or original_popen is None:
        yield from model.stream_pcm(chunks, realtime_factor=realtime_factor)
        return

    def checked_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        wrapper = _CheckedPopen(process)
        checked.append(wrapper)
        return wrapper

    subprocess_module.Popen = checked_popen
    try:
        yield from model.stream_pcm(chunks, realtime_factor=realtime_factor)
    finally:
        subprocess_module.Popen = original_popen
    for process in checked:
        returncode = process.returncode
        if returncode not in (None, 0):
            raise RuntimeError("native helper exited with status " + str(returncode))
        if process.stdout.truncated or process.stdout._payload_length is not None:
            raise RuntimeError("native helper emitted a truncated frame")


def _recognize(incoming: Connection, outgoing: Connection, aborted,
               locale: str, profile: dict | None, digest: str,
               options: dict | None = None) -> None:
    from soda_stt import SodaRecognizer, _config
    from training.context_profile import apply_to_soda

    opts = options or {}
    mode_name = opts.get("recognition_mode", "caption")
    enable_formatting = bool(opts.get("enable_formatting", True))
    if "trailing_silence_ms" in opts:
        trailing_silence_ms = int(opts["trailing_silence_ms"])
    else:
        # Legacy boolean is retained for callers during migration; the safe
        # default is no synthetic audio unless explicitly requested.
        trailing_silence_ms = 400 if bool(opts.get("trailing_silence_flush", False)) else 0
    if trailing_silence_ms not in (0, 200, 400, 600):
        raise ValueError("trailing_silence_ms must be one of 0, 200, 400, 600")

    _config.LANGUAGE_PACK_IDS.update({locale: cfg["component_id"] for locale, cfg in LANGUAGES.items()})
    soda_kwargs = {
        "recognition_mode": _config.MODE_IME if mode_name == "ime" else _config.MODE_CAPTION,
        "enable_formatting": enable_formatting,
    }
    model = SodaRecognizer.for_locale(locale, **soda_kwargs)
    if locale == "vi-VN":
        apply_to_soda(model, profile)
    if aborted.is_set():
        return
    outgoing.send({"type": "ready", "locale": locale, "engine": "soda",
                   "bias_profile_sha256": digest, "recognition_mode": mode_name,
                   "enable_formatting": enable_formatting})

    def chunks():
        had_audio = False
        while not aborted.is_set():
            if not incoming.poll(.1):
                continue
            try:
                data = incoming.recv_bytes(65536)
            except EOFError:
                break
            if data:
                had_audio = True
            for offset in range(0, len(data), 3200):
                if aborted.is_set():
                    return
                yield data[offset:offset + 3200]
        # Flush trailing silence so SODA's HAT lookahead commits the final syllable
        if had_audio and trailing_silence_ms and not aborted.is_set():
            silence = b"\x00" * 3200
            for _ in range(trailing_silence_ms // TRAILING_SILENCE_CHUNK_MS):
                if aborted.is_set():
                    return
                yield silence

    for item in _native_stream_status(model, chunks()):
        if aborted.is_set():
            break
        outgoing.send({"type": "final" if item.is_final else "partial",
                       "text": item.text, "locale": locale})


def _owned_group(target, incoming, outgoing, aborted, group_ready,
                 locale, profile, digest, temp_directory: str | None = None,
                 options: dict | None = None) -> None:
    """Run the target and publish its outcome before the terminal marker.

    The upstream adapter does not currently expose its helper return code.  We
    therefore make the owned worker boundary authoritative: target exceptions
    are reported and re-raised (giving the process a nonzero exit code), while
    a target that returns normally—including an empty native stop—has exit code
    zero.  The parent can also inspect ``Process.exitcode`` if the pipe dies.
    """
    failure = None
    try:
        os.setsid()
        if temp_directory is not None:
            tempfile.tempdir = temp_directory
        group_ready.set()
        if options is None:
            target(incoming, outgoing, aborted, locale, profile, digest)
        else:
            target(incoming, outgoing, aborted, locale, profile, digest, options)
    except BaseException as exc:
        message = str(exc)[:160]
        failure = {"type": "worker_status", "ok": False,
                   "exception": type(exc).__name__,
                   "message": message}
        if message.startswith("native helper "):
            failure["code"] = "native_worker_exited"
        # The parent maps this worker status to one protocol error. Do not
        # send a second error here; terminal delivery remains exactly once.
        raise
    finally:
        with suppress(OSError, EOFError):
            outgoing.send(failure or {"type": "worker_status", "ok": True})
            outgoing.send({"type": "closed"})
        incoming.close()
        outgoing.close()


class NativeSession:
    """Bounded PCM queue plus an explicitly owned worker and helper group."""

    def __init__(self, max_chunks: int = 100, *, target: Callable | None = None):
        context = multiprocessing.get_context("spawn")
        self.audio: queue.Queue[bytes] = queue.Queue(maxsize=max_chunks)
        self.input_finished = threading.Event()
        self.aborted = context.Event()
        self.ready = threading.Event()
        self.group_ready = context.Event()
        self._child_input, self._send_audio = context.Pipe(duplex=False)
        self._read_events, self._child_output = context.Pipe(duplex=False)
        self._context = context
        self._target = target or _recognize
        self._temp_directory = tempfile.TemporaryDirectory(prefix="soda-native-session-")
        self._process = None
        self._feeder = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    @property
    def exitcode(self) -> int | None:
        """Return the owned worker exit status once it has stopped."""
        if self._process is None:
            return None
        # ``Process.exitcode`` is not populated until the multiprocessing
        # handle polls/reaps the child; a terminal pipe event means it is safe
        # to perform a non-blocking join before exposing the status.
        with suppress(AssertionError):
            self._process.join(timeout=0)
        return self._process.exitcode

    def start(self, locale: str, profile: dict | None, digest: str,
              options: dict | None = None) -> bool:
        with self._lock:
            if self._closed or self.aborted.is_set():
                return False
            # Test and embedding targets historically accept the original
            # seven worker arguments. Only the built-in recognizer has the
            # optional SODA configuration argument.
            child_options = options if self._target is _recognize else None
            self._process = self._context.Process(
                target=_owned_group,
                args=(self._target, self._child_input, self._child_output,
                      self.aborted, self.group_ready, locale, profile, digest,
                      self._temp_directory.name, child_options),
                name="soda-native-session",
            )
            self._process.start()
            # These ends belong only to the child. Closing the parent copies
            # ensures EOF/BrokenPipe wakes readers/writers when that group dies.
            self._child_input.close()
            self._child_output.close()
            self._feeder = threading.Thread(target=self._feed, name="soda-input", daemon=True)
            self._feeder.start()
            return True

    def _feed(self) -> None:
        try:
            while not self.aborted.is_set():
                try:
                    frame = self.audio.get(timeout=.1)
                except queue.Empty:
                    if self.input_finished.is_set():
                        return
                    continue
                self._send_audio.send_bytes(frame)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            with suppress(OSError):
                self._send_audio.close()

    def read_event(self, timeout: float = .1) -> dict | None:
        if not self._read_events.poll(timeout):
            return None
        return self._read_events.recv()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _signal_owned(self, signum: int) -> None:
        process = self._process
        if process is None or process.pid is None:
            return
        if self.group_ready.is_set():
            # setsid() ran in this child before it could create a SODA helper.
            # This is the exact owned group, never a process-name/global kill.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
        elif process.is_alive():
            with suppress(ProcessLookupError):
                os.kill(process.pid, signum)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Keep the session retryable if a process or feeder cannot yet be
            # reaped.  The server holds its capacity slot until this method
            # completes successfully; marking closed before cleanup would make
            # a later retry falsely report cleanup complete.
            self.aborted.set()
            self.input_finished.set()
            process = self._process
            if process is not None and process.pid is not None:
                process.join(timeout=.3)
                self._signal_owned(signal.SIGTERM)
                process.join(timeout=1)
                # Also clean a helper orphaned by an unexpectedly exited worker.
                self._signal_owned(signal.SIGKILL)
                process.join(timeout=2)
                if process.is_alive():
                    raise RuntimeError("Owned native process could not be reaped")
            if self._feeder is not None:
                self._feeder.join(timeout=1)
                if self._feeder.is_alive():
                    raise RuntimeError("Owned input feeder did not exit")
            for connection in (self._send_audio, self._read_events,
                               self._child_input, self._child_output):
                with suppress(OSError):
                    connection.close()
            self._temp_directory.cleanup()
            self._closed = True
