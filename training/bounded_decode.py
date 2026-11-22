"""Bound file evaluation and reap only its own native process group on timeout."""
from __future__ import annotations
import multiprocessing, os, queue, signal, tempfile, time

def _worker(recognizer, pcm, pace, result_queue, session_temp):
    if os.name == "posix":
        os.setsid()
    # Upstream writes a protobuf .cfg through Python tempfile. Keep it scoped.
    tempfile.tempdir = session_temp
    os.environ["TMPDIR"] = session_temp
    os.environ["TEMP"] = session_temp
    os.environ["TMP"] = session_temp
    try:
        start = time.monotonic()
        first = None
        finals = []
        partials = 0
        for event in recognizer.stream_pcm(
            (pcm[i:i + 3200] for i in range(0, len(pcm), 3200)),
            realtime_factor=pace,
        ):
            if event.text and first is None:
                first = time.monotonic() - start
            if event.is_final and event.text:
                finals.append(event.text)
            else:
                partials += 1
        result_queue.put({
            "ok": True, "text": " ".join(finals).strip(),
            "first_result_seconds": first, "elapsed_seconds": time.monotonic() - start,
            "final_segments": len(finals), "partial_events": partials,
        })
    except Exception as exc:
        result_queue.put({"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})

def _stop(process):
    if process.pid is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
    elif process.is_alive():
        process.terminate()
    process.join(1)
    if os.name == "posix":
        # Reap helpers even if their group leader exited before they did.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
    elif process.is_alive():
        process.kill()
    if process.is_alive():
        process.join(1)

def recognize_pcm(recognizer, pcm: bytes, *, pace: float = 2, timeout: float | None = None):
    if not 0.5 <= pace <= 4:
        raise ValueError("File feed pace must be 0.5..4")
    limit = timeout if timeout is not None else len(pcm) / 32000 / pace + 12
    if limit <= 0:
        raise ValueError("Timeout must be positive")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue(maxsize=1)
    with tempfile.TemporaryDirectory(prefix="soda-file-eval-") as session_temp:
        process = ctx.Process(target=_worker, args=(recognizer, pcm, pace, results, session_temp), daemon=True)
        try:
            process.start()
            deadline = time.monotonic() + limit
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"Native SODA file evaluation exceeded {limit:.1f}s")
                try:
                    result = results.get(timeout=min(0.2, left))
                    break
                except queue.Empty:
                    if not process.is_alive():
                        raise RuntimeError(f"Native decode worker exited {process.exitcode} without a result")
            process.join(1)
            if process.is_alive():
                raise RuntimeError("Decode worker did not exit after delivering result")
            if not result["ok"]:
                raise RuntimeError(result["error_type"] + ": " + result["error"])
            if not result["text"].strip():
                raise RuntimeError("Native decode returned no final text")
            return result
        finally:
            _stop(process)
            results.close()
            results.join_thread()
    # TemporaryDirectory cleanup runs only after the owned process group is reaped.
