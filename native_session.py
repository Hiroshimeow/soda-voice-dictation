"""Own a SODA worker process group without modifying the upstream adapter.

The service may terminate only this freshly spawned group on cancellation or a
finalization deadline. PCM and results cross private pipes; no cloud request or
transcript persistence is introduced.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
import os
import queue
import signal
import tempfile
import threading
from contextlib import suppress
from typing import Callable

from soda_languages import LANGUAGES


def _recognize(incoming: Connection, outgoing: Connection, aborted,
               locale: str, profile: dict | None, digest: str) -> None:
    from soda_stt import SodaRecognizer, _config
    from training.context_profile import apply_to_soda

    _config.LANGUAGE_PACK_IDS.update({locale: cfg["component_id"] for locale, cfg in LANGUAGES.items()})
    model = SodaRecognizer.for_locale(locale)
    if locale == "vi-VN":
        apply_to_soda(model, profile)
    if aborted.is_set():
        return
    outgoing.send({"type": "ready", "locale": locale, "engine": "soda",
                   "bias_profile_sha256": digest})

    def chunks():
        while not aborted.is_set():
            if not incoming.poll(.1):
                continue
            try:
                data = incoming.recv_bytes(65536)
            except EOFError:
                return
            for offset in range(0, len(data), 3200):
                if aborted.is_set():
                    return
                yield data[offset:offset + 3200]

    for item in model.stream_pcm(chunks(), realtime_factor=2.0):
        if aborted.is_set():
            break
        outgoing.send({"type": "final" if item.is_final else "partial",
                       "text": item.text, "locale": locale})


def _owned_group(target, incoming, outgoing, aborted, group_ready,
                 locale, profile, digest, temp_directory: str | None = None) -> None:
    try:
        os.setsid()
        if temp_directory is not None:
            tempfile.tempdir = temp_directory
        group_ready.set()
        target(incoming, outgoing, aborted, locale, profile, digest)
    except Exception:
        if not aborted.is_set():
            with suppress(OSError, EOFError):
                outgoing.send({"type": "error", "code": "recognition_failed"})
    finally:
        with suppress(OSError, EOFError):
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

    def start(self, locale: str, profile: dict | None, digest: str) -> bool:
        with self._lock:
            if self._closed or self.aborted.is_set():
                return False
            self._process = self._context.Process(
                target=_owned_group,
                args=(self._target, self._child_input, self._child_output,
                      self.aborted, self.group_ready, locale, profile, digest,
                      self._temp_directory.name),
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
            self._closed = True
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
