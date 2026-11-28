"""Evaluate local development audio; references never enter recognition.

Input manifest JSON rows: id, audio (absolute PCM WAV path), reference.
The named numeric-equivalence score expands digits/percent to Vietnamese
spoken forms; raw syllable error and character error remain in the report.
Supports A/B testing of model-agnostic Vietnamese audio front-end (--enhance-audio)
and zero-hallucination post-processing (--postprocess / --itn).
"""
from __future__ import annotations
import argparse, array, hashlib, json, math, os, pathlib, re, tempfile, time, unicodedata, wave
from datetime import datetime, timezone
from training.calibrate import rate, normalized
from training.context_profile import apply_to_soda, validate_profile, profile_sha256
from training.personal_vocabulary import load_vocabulary
from vi_enhance.audio_frontend import AudioFrontendConfig, enhance_pcm16
from vi_enhance.postprocessor import PostProcessConfig, VietnamesePostProcessor

DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def speak_integer(value: str) -> str:
    if len(value) > 3 or (len(value) > 1 and value.startswith("0")):
        return " ".join(DIGITS[int(c)] for c in value)
    n = int(value)
    if n < 10:
        return DIGITS[n]
    if n < 20:
        return "mười" + (" " + ("lăm" if n % 10 == 5 else DIGITS[n % 10]) if n % 10 else "")
    if n < 100:
        return DIGITS[n // 10] + " mươi" + (" " + ("mốt" if n % 10 == 1 else "lăm" if n % 10 == 5 else DIGITS[n % 10]) if n % 10 else "")
    return DIGITS[n // 100] + " trăm" + (" " + ("linh " if n % 100 < 10 else "") + speak_integer(str(n % 100)) if n % 100 else "")


def numeric_equivalence(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"(\d)\s*%", r"\1 phần trăm", text)
    # Digit groups adjacent in a phone number are each read digit-by-digit.
    text = re.sub(r"(?<!\d)0\d{2,}(?:[ \-]\d{2,})*", lambda m: " ".join(DIGITS[int(c)] for c in m[0] if c.isdigit()), text)
    text = re.sub(r"\d+", lambda m: speak_integer(m[0]), text)
    return normalized(text)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pcm(path, gain_db, *, enhance_audio: bool = False):
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError("16k mono s16 WAV required")
        pcm = w.readframes(w.getnframes())
    if enhance_audio:
        pcm, summary = enhance_pcm16(
            pcm,
            AudioFrontendConfig(manual_gain_db=gain_db),
            pad_silence=True,
        )
        return pcm, {"seconds": len(pcm) / 32000, **summary}
    samples = array.array("h")
    samples.frombytes(pcm)
    if os.sys.byteorder != "little":
        samples.byteswap()
    peak = max((abs(v) for v in samples), default=0)
    rms = math.sqrt(sum(v * v for v in samples) / max(1, len(samples)))
    gain = 10 ** (gain_db / 20)
    clipped = sum(abs(v * gain) > 32767 for v in samples)
    if gain_db:
        samples = array.array("h", (max(-32768, min(32767, round(v * gain))) for v in samples))
        if os.sys.byteorder != "little":
            samples.byteswap()
        pcm = samples.tobytes()
    return pcm, {"seconds": len(pcm) / 32000, "peak_dbfs": 20 * math.log10(max(peak, 1) / 32768), "rms_dbfs": 20 * math.log10(max(rms, 1e-9) / 32768), "clipped_samples": clipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=("caption", "ime"), default="caption")
    ap.add_argument("--no-formatting", action="store_true")
    ap.add_argument("--gain-db", type=float, default=0)
    ap.add_argument("--pace", type=float, default=2)
    ap.add_argument("--profile", type=pathlib.Path)
    ap.add_argument("--context-name", choices=("palace-vi-safe-context-v1", "android-speech-api-generic-phrases"), default="palace-vi-safe-context-v1")
    ap.add_argument("--model-dir", type=pathlib.Path)
    ap.add_argument("--enhance-audio", action="store_true", help="Enable tonal-safe Vietnamese audio front-end (HPF + Wiener OA + AGC + HAT silence pad)")
    ap.add_argument("--postprocess", action="store_true", help="Enable zero-hallucination Vietnamese post-processing")
    ap.add_argument("--itn", choices=("off", "safe", "full"), default="safe")
    ap.add_argument("--include-reference", action="store_true", help="Include clear references in the report for explicitly authorized local mining")
    a = ap.parse_args()
    if not 0.5 <= a.pace <= 4:
        ap.error("Evaluation pace must be 0.5..4; 8x lost audio in the measured development setup.")
    if not -12 <= a.gain_db <= 30:
        ap.error("Gain must be -12..30dB")
    from soda_stt import SodaRecognizer, _config
    _config.LANGUAGE_PACK_IDS["vi-VN"] = "bpnckojighgibleldbofdjmfdipfeiak"
    options = {"recognition_mode": _config.MODE_IME if a.mode == "ime" else _config.MODE_CAPTION, "enable_formatting": not a.no_formatting}
    rec = SodaRecognizer(str(a.model_dir.resolve()), **options) if a.model_dir else SodaRecognizer.for_locale("vi-VN", **options)
    from soda_stt import soda_api_pb2
    serialized_dir = soda_api_pb2.ExtendedSodaConfigMsg.FromString(rec._config_bytes).language_pack_directory
    if pathlib.Path(serialized_dir).resolve() != pathlib.Path(rec.language_pack_directory).resolve():
        raise RuntimeError("Serialized model path mismatch")
    model_hashes = {str(p.relative_to(pathlib.Path(rec.language_pack_directory))): sha(p) for p in pathlib.Path(rec.language_pack_directory).rglob("*.tflite")}
    effective_profile = validate_profile(json.loads(a.profile.read_text(encoding="utf-8"))) if a.profile else None
    if effective_profile:
        apply_to_soda(rec, effective_profile, context_name=a.context_name)
    processor = VietnamesePostProcessor(PostProcessConfig(itn_mode=a.itn)) if a.postprocess else None
    vocab = load_vocabulary() if a.postprocess else None
    rows = json.loads(a.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty list required")
    profile_fingerprint = profile_sha256(effective_profile)
    a.report.parent.mkdir(parents=True, exist_ok=True)

    def write_report_atomic(payload: dict) -> None:
        """Publish each state transition atomically, including failure states."""
        fd, temporary = tempfile.mkstemp(prefix=f".{a.report.name}.", suffix=".tmp", dir=str(a.report.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, a.report)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def failed_report(stage: str, row_id: object, exc: BaseException) -> None:
        write_report_atomic({
            "schema_version": 2, "status": "failed", "failure_stage": stage,
            "failed_sample": row_id, "error_type": type(exc).__name__,
            "error": str(exc)[:500], "expected_samples": len(rows),
            "manifest_sha256": sha(a.manifest), "results": results,
        })

    results = []
    write_report_atomic({"schema_version": 2, "status": "running", "expected_samples": len(rows), "results": results, "manifest_sha256": sha(a.manifest)})
    for row in rows:
        path = pathlib.Path(row["audio"])
        try:
            pcm, levels = read_pcm(path, a.gain_db, enhance_audio=a.enhance_audio)
        except Exception as exc:
            failed_report("wav", row["id"], exc)
            raise
        from training.bounded_decode import recognize_pcm
        try:
            result_data = recognize_pcm(rec, pcm, pace=a.pace)
        except Exception as exc:
            failed_report("helper", row["id"], exc)
            raise
        elapsed = result_data["elapsed_seconds"]
        first = result_data["first_result_seconds"]
        raw_prediction = result_data["text"]
        native_finals = list(result_data.get("finals", [raw_prediction] if raw_prediction else []))
        postprocess_status = "disabled" if processor is None else "unchanged"
        processed_finals: list[str] = []
        if processor is None:
            processed_finals = native_finals
        else:
            try:
                for final in native_finals:
                    processed_finals.append(processor.process(final, vocabulary=vocab)[0])
                if any(a != b for a, b in zip(native_finals, processed_finals)):
                    postprocess_status = "applied"
            except Exception as exc:
                postprocess_status = "failed"
                failed_report("postprocess", row["id"], exc)
                raise RuntimeError(f"Postprocessing failed for sample {row['id']}: {exc}") from exc
        prediction = " ".join(processed_finals).strip()
        partials = result_data["partial_events"]
        final_count = result_data["final_segments"]
        try:
            raw_we, raw_wn = rate(row["reference"], raw_prediction)
            raw_ce, raw_cn = rate(row["reference"], raw_prediction, "char")
            raw_ne, raw_nn = rate(numeric_equivalence(row["reference"]), numeric_equivalence(raw_prediction))
            delivered_we, delivered_wn = rate(row["reference"], prediction)
            delivered_ce, delivered_cn = rate(row["reference"], prediction, "char")
            delivered_ne, delivered_nn = rate(numeric_equivalence(row["reference"]), numeric_equivalence(prediction))
        except Exception as exc:
            failed_report("metric", row["id"], exc)
            raise
        final_results = []
        for native_final, processed_final in zip(native_finals, processed_finals):
            try:
                final_raw_we, final_raw_wn = rate(row["reference"], native_final)
                final_delivered_we, final_delivered_wn = rate(row["reference"], processed_final)
            except Exception as exc:
                failed_report("metric", row["id"], exc)
                raise
            final_results.append({
                "native": native_final,
                "delivered": processed_final,
                "postprocess_status": "applied" if native_final != processed_final else "unchanged",
                "raw_syllable_errors": final_raw_we,
                "raw_reference_syllables": final_raw_wn,
                "delivered_syllable_errors": final_delivered_we,
                "delivered_reference_syllables": final_delivered_wn,
            })
        result = {
            "id": row["id"], "audio_sha256": sha(path),
            "outcome": "no_final" if final_count == 0 else "ok",
            "no_final": final_count == 0,
            "helper_status": "ok",
            "postprocess_status": postprocess_status,
            "final_results": final_results,
            "configuration_provenance": {"profile_sha256": profile_fingerprint, "enhance_audio": a.enhance_audio, "postprocess": a.postprocess, "itn": a.itn},
            "levels": levels, "native_finals": native_finals, "processed_finals": processed_finals,
            "raw_recognition": raw_prediction,
            "delivered_recognition": prediction,
            "raw_syllable_errors": raw_we, "raw_reference_syllables": raw_wn,
            "raw_ser": raw_we / max(1, raw_wn),
            "delivered_syllable_errors": delivered_we,
            "delivered_reference_syllables": delivered_wn,
            "delivered_ser": delivered_we / max(1, delivered_wn),
            "raw_char_errors": raw_ce, "raw_reference_chars": raw_cn,
            "raw_cer": raw_ce / max(1, raw_cn),
            "delivered_char_errors": delivered_ce, "delivered_reference_chars": delivered_cn,
            "delivered_cer": delivered_ce / max(1, delivered_cn),
            "raw_numeric_normalized_errors": raw_ne,
            "raw_numeric_normalized_reference_syllables": raw_nn,
            "raw_numeric_normalized_ser": raw_ne / max(1, raw_nn),
            "delivered_numeric_normalized_errors": delivered_ne,
            "delivered_numeric_normalized_reference_syllables": delivered_nn,
            "delivered_numeric_normalized_ser": delivered_ne / max(1, delivered_nn),
            "elapsed_seconds": elapsed, "first_result_seconds": first,
            "final_segments": final_count, "partial_events": partials,
        }
        if a.include_reference:
            result["reference"] = row["reference"]
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        report = {"schema_version": 2, "status": "completed" if len(results) == len(rows) else "running", "expected_samples": len(rows), "created_at": datetime.now(timezone.utc).isoformat(), "evaluation_type": "development_acceptance_not_independent_holdout", "evaluation_changes_weights": False, "references_included": a.include_reference, "model_sha256s": model_hashes, "serialized_language_pack_directory": serialized_dir, "engine": "Google Chrome SODA", "engine_sha256": sha(pathlib.Path(rec.libsoda_path)), "model_path": rec.language_pack_directory, "manifest_sha256": sha(a.manifest), "configuration": {"mode": a.mode, "formatting": not a.no_formatting, "gain_db": a.gain_db, "pace": a.pace, "context_name": a.context_name, "profile_sha256": profile_fingerprint, "enhance_audio": a.enhance_audio, "postprocess": a.postprocess, "itn": a.itn, "include_reference": a.include_reference}, "metric_definition": {"raw_ser": "native raw_recognition against the manifest reference", "delivered_ser": "delivered_recognition after optional postprocessing against the manifest reference", "numeric_normalized_ser": "reference-independent integer/percent/phone-digit expansion; not a semantic accuracy score"}, "results": results, "aggregate_raw_ser": sum(x["raw_syllable_errors"] for x in results) / sum(x["raw_reference_syllables"] for x in results), "aggregate_delivered_ser": sum(x["delivered_syllable_errors"] for x in results) / sum(x["delivered_reference_syllables"] for x in results), "aggregate_raw_cer": sum(x["raw_char_errors"] for x in results) / sum(x["raw_reference_chars"] for x in results), "aggregate_delivered_cer": sum(x["delivered_char_errors"] for x in results) / sum(x["delivered_reference_chars"] for x in results)}
        write_report_atomic(report)
    print("REPORT", str(a.report), flush=True)


if __name__ == "__main__":
    main()
