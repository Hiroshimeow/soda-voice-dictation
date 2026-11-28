"""Comprehensive unit & integration test suite for vi_enhance and repo integrations."""
from __future__ import annotations

from array import array
import math
import struct
import sys
import tempfile
import unittest
import unicodedata
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop.audio import gain_pcm16, validate_gain
from training.confusion_miner import mine_confusions
from training.context_profile import validate_profile
from training.evaluate_samples import read_pcm
from vi_enhance.audio_frontend import (
    AudioFrontendConfig,
    StreamingAudioFrontend,
    _irfft_pure,
    _rfft_pure,
    enhance_pcm16,
    pad_utterance_pcm,
)
from vi_enhance.dynamic_bias import build_smart_bias_profile, score_bias_phrase
from vi_enhance.long_audio import plan_vad_segments
from vi_enhance.phonetics import (
    normalize_vietnamese_tone_placement,
    parse_vietnamese_syllable,
    phrase_phonetic_similarity,
    vietnamese_phonetic_distance,
)
from vi_enhance.postprocessor import (
    GatedLLMCorrector,
    PostProcessConfig,
    VietnamesePostProcessor,
    apply_vietnamese_itn,
    phonetic_vocab_rescue,
    polish_vietnamese_punctuation,
    stitch_segments,
)


def _synth_sine(freq_hz: float, duration_s: float, amp: float = 0.25, dc: float = 0.0, sr: int = 16000) -> bytes:
    n = int(sr * duration_s)
    arr = array("h")
    for i in range(n):
        val = dc + amp * math.sin(2.0 * math.pi * freq_hz * i / sr)
        arr.append(max(-32768, min(32767, round(val * 32767))))
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def _rms_linear(pcm: bytes) -> float:
    arr = array("h")
    arr.frombytes(pcm)
    if sys.byteorder != "little":
        arr.byteswap()
    return math.sqrt(sum(v * v for v in arr) / max(1, len(arr))) / 32768.0


class TestAudioFrontend(unittest.TestCase):
    def test_highpass_rejects_dc_and_50hz_hum_while_preserving_vietnamese_f0(self):
        # 50 Hz AC hum + DC offset vs 200 Hz Vietnamese fundamental frequency
        hum = _synth_sine(50.0, 1.0, amp=0.2, dc=0.15)
        f0 = _synth_sine(200.0, 1.0, amp=0.2, dc=0.0)
        cfg = AudioFrontendConfig(enable_highpass=True, enable_denoise_oa=False, enable_agc=False)
        hum_out, _ = enhance_pcm16(hum, cfg)
        f0_out, _ = enhance_pcm16(f0, cfg)
        # Skip initial 200 ms filter transient
        hum_rms_in = _rms_linear(hum[6400:])
        hum_rms_out = _rms_linear(hum_out[6400:])
        f0_rms_in = _rms_linear(f0[6400:])
        f0_rms_out = _rms_linear(f0_out[6400:])
        hum_atten_db = 20.0 * math.log10(max(hum_rms_out, 1e-6) / hum_rms_in)
        f0_ratio = f0_rms_out / f0_rms_in
        self.assertLess(hum_atten_db, -10.0, f"Expected >10 dB hum+DC attenuation, got {hum_atten_db:.2f} dB")
        self.assertGreater(f0_ratio, 0.88, f"Vietnamese F0 (200 Hz) must be preserved, got ratio={f0_ratio:.3f}")

    def test_bypass_is_sample_aligned_for_arbitrary_fragmentation(self):
        values = [0] * 4096
        values[1234] = 30000
        pcm = struct.pack("<" + "h" * len(values), *values)
        frontend = StreamingAudioFrontend(AudioFrontendConfig(
            enable_highpass=False, enable_denoise_oa=False, enable_agc=False,
        ))
        chunks = [1, 137, 911, 2, 2048, 7, 990]
        output = bytearray()
        offset = 0
        for sample_count in chunks:
            part = pcm[offset * 2:(offset + sample_count) * 2]
            output.extend(frontend.process_chunk(part)[0])
            offset += sample_count
        output.extend(frontend.process_chunk(pcm[offset * 2:])[0])
        self.assertEqual(bytes(output), pcm)

    def test_enabled_frontend_preserves_length_and_impulse_order(self):
        values = [0] * 12000
        values[5000] = 28000
        pcm = struct.pack("<" + "h" * len(values), *values)
        frontend = StreamingAudioFrontend()
        output = bytearray()
        offset = 0
        for sample_count in (17, 1600, 311, 4097, 23, 5952):
            part = pcm[offset * 2:(offset + sample_count) * 2]
            output.extend(frontend.process_chunk(part)[0])
            offset += sample_count
        output.extend(frontend.process_chunk(pcm[offset * 2:])[0])
        out_values = struct.unpack("<" + "h" * len(values), output)
        self.assertEqual(len(output), len(pcm))
        self.assertGreater(max(abs(v) for v in out_values[4500:5500]), 1000)
        self.assertLess(max(abs(v) for v in out_values[:3500]), 1000)

    def test_enabled_frontend_is_chunking_stable(self):
        values = [int(5000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(10000)]
        pcm = struct.pack("<" + "h" * len(values), *values)

        def run(sizes):
            frontend = StreamingAudioFrontend()
            output = bytearray()
            offset = 0
            for size in sizes:
                if offset >= len(values):
                    break
                end = min(len(values), offset + size)
                output.extend(frontend.process_chunk(pcm[offset * 2:end * 2])[0])
                offset = end
            if offset < len(values):
                output.extend(frontend.process_chunk(pcm[offset * 2:])[0])
            return bytes(output)

        whole = run([len(values)])
        fragmented = run([1, 137, 911, 2, 2048, 7, 311, 4097, 23, 5952])
        self.assertEqual(len(whole), len(fragmented))
        whole_values = struct.unpack("<" + "h" * len(values), whole)
        fragmented_values = struct.unpack("<" + "h" * len(values), fragmented)
        self.assertGreater(_rms_linear(fragmented), 0.001)
        self.assertLess(abs(_rms_linear(whole) - _rms_linear(fragmented)), 0.03)
        self.assertLess(abs(max(whole_values)) - abs(max(fragmented_values)), 2000)

    def test_quiet_speech_agc_recovers_without_hard_clipping(self):
        # Simulate -46 dBFS quiet recording (like docs/VI_DEVELOPMENT_RESULTS.md)
        quiet = _synth_sine(220.0, 1.5, amp=0.007)
        out, summary = enhance_pcm16(quiet, AudioFrontendConfig(enable_denoise_oa=False, enable_agc=True))
        self.assertGreater(summary["mean_output_dbfs"], summary["mean_input_dbfs"] + 12.0)
        self.assertEqual(summary["clipped_samples"], 0)
        self.assertEqual(len(out), len(quiet))

    def test_soft_limiter_prevents_hard_clipping_on_hot_signal(self):
        hot = _synth_sine(300.0, 0.5, amp=0.95)
        out, summary = enhance_pcm16(hot, AudioFrontendConfig(enable_agc=False, manual_gain_db=12.0))
        self.assertEqual(summary["clipped_samples"], 0)
        self.assertEqual(len(out), len(hot))

    def test_oa_blends_on_noisy_and_preserves_clean_speech(self):
        # Clean speech preceded by silence -> high SNR -> oa_alpha near 0
        silence = b"\x00" * (16000 * 2 // 2)
        clean_speech = _synth_sine(250.0, 0.8, amp=0.25)
        _, summary_clean = enhance_pcm16(silence + clean_speech)
        self.assertLess(summary_clean["mean_oa_alpha"], 0.15)

        # The release streaming path intentionally does not apply a spectral
        # transform; telemetry must honestly report the no-op.
        import random
        rng = random.Random(42)
        n = 16000
        noise_arr = array("h", (int(rng.gauss(0, 1800)) for _ in range(n)))
        noisy_speech = array(
            "h",
            (
                max(-32768, min(32767, int(rng.gauss(0, 1800) + 4200 * math.sin(2 * math.pi * 220 * i / 16000))))
                for i in range(n)
            ),
        )
        if sys.byteorder != "little":
            noise_arr.byteswap()
            noisy_speech.byteswap()
        _, summary_noisy = enhance_pcm16(noise_arr.tobytes() + noisy_speech.tobytes())
        self.assertEqual(summary_noisy["mean_oa_alpha"], 0.0)

    def test_pure_fft_roundtrip(self):
        frame = [math.sin(2 * math.pi * 5 * i / 64) + 0.3 * math.cos(2 * math.pi * 11 * i / 64) for i in range(64)]
        rec = _irfft_pure(_rfft_pure(frame), 64)
        max_err = max(abs(a - b) for a, b in zip(frame, rec))
        self.assertLess(max_err, 1e-10)

    def test_pad_utterance_pcm(self):
        pcm = b"\x01\x00" * 1600
        padded = pad_utterance_pcm(pcm, leading_ms=100, trailing_ms=400)
        self.assertEqual(len(padded), len(pcm) + (1600 + 6400) * 2)

    def test_backward_compatible_gain_pcm16(self):
        self.assertEqual(validate_gain(6), 6.0)
        pcm = _synth_sine(440.0, 0.1, amp=0.2)
        out, levels = gain_pcm16(pcm, 6.0)
        self.assertEqual(len(out), len(pcm))
        self.assertGreater(levels["rms_dbfs"], levels["input_dbfs"] + 5.0)


class TestVietnamesePhonetics(unittest.TestCase):
    def test_tone_placement_and_nfc_normalization(self):
        nfd = unicodedata.normalize("NFD", "hoà bình thuý kiều khoẻ mạnh hoàn toàn toán học")
        norm = normalize_vietnamese_tone_placement(nfd)
        self.assertEqual(norm, "hòa bình thúy kiều khỏe mạnh hoàn toàn toán học")

    def test_syllable_parser_six_tones(self):
        tones = [parse_vietnamese_syllable(w).tone for w in ("ma", "mà", "má", "mả", "mã", "mạ")]
        self.assertEqual(tones, [0, 1, 2, 3, 4, 5])
        s = parse_vietnamese_syllable("nghiêng")
        self.assertEqual((s.onset, s.nucleus, s.coda, s.tone), ("ngh", "iê", "ng", 0))

    def test_regional_dialect_confusion_distances(self):
        # Regional confusions should be much closer than unrelated words
        self.assertLess(vietnamese_phonetic_distance("chuyển", "truyển"), 0.10)
        self.assertLess(vietnamese_phonetic_distance("sản", "xản"), 0.10)
        self.assertLess(vietnamese_phonetic_distance("việt", "diệt"), 0.10)
        self.assertLess(vietnamese_phonetic_distance("bản", "bãn"), 0.10)  # hoi/nga merge
        self.assertLess(vietnamese_phonetic_distance("bán", "báng"), 0.10)  # -n/-ng merge
        self.assertGreater(vietnamese_phonetic_distance("bán", "trời"), 0.70)


class TestVietnamesePostProcessor(unittest.TestCase):
    def test_stitch_segments_removes_boundary_overlap(self):
        segs = [
            "Hôm nay chúng ta sẽ",
            "chúng ta sẽ triển khai phiên bản mới",
            "phiên bản mới cho hệ thống.",
        ]
        self.assertEqual(
            stitch_segments(segs, overlap_windows=True),
            "Hôm nay chúng ta sẽ triển khai phiên bản mới cho hệ thống.",
        )

    def test_independent_finals_are_not_deduplicated(self):
        self.assertEqual(stitch_segments(["Yes.", "Yes."]), "Yes. Yes.")

    def test_safe_processing_preserves_urls_and_ambiguous_numbers(self):
        text = "Xem https://example.com/path?a=1, và nói một triệu năm."
        self.assertEqual(apply_vietnamese_itn(text, mode="safe"), text)
        self.assertEqual(polish_vietnamese_punctuation(text), text)

    def test_literal_and_ambiguous_text_stays_exact_and_explicit_write_is_final(self):
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        text = "https://example.com/hoà?di=1 a@b.example C:\\Temp\\server.py server.py"
        out, _ = proc.process(text, vocabulary={"phrases": [], "replacements": []})
        self.assertEqual(out, text)
        for phrase in ("ba đồng nghiệp", "âm hai phẩy năm đồng"):
            self.assertEqual(proc.process(phrase, vocabulary={"phrases": [], "replacements": []})[0], phrase.capitalize())
        out, _ = proc.process(
            "xê đê pê a",
            vocabulary={"phrases": [], "replacements": [{"heard": "xê đê pê a", "write": "ba đồng nghiệp"}]},
        )
        self.assertEqual(out, "ba đồng nghiệp")

    def test_spaced_windows_path_and_replacement_are_collision_safe(self):
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        path = r"C:\\Program Files\\hai mươi\\tool.exe"
        out, _ = proc.process(
            path + " và sô đa",
            vocabulary={"phrases": [], "replacements": [{"heard": "sô đa", "write": "SODA"}]},
        )
        self.assertEqual(out, path + " và SODA")

    def test_short_acronym_and_syllable_rescue_is_conservative(self):
        vocab = ["API", "Git"]
        self.assertEqual(phonetic_vocab_rescue("AI", vocab)[0], "AI")
        self.assertEqual(phonetic_vocab_rescue("ai", vocab)[0], "ai")
        self.assertEqual(phonetic_vocab_rescue("di", vocab)[0], "di")
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        out, _ = proc.process("AI và di", vocabulary={"phrases": vocab, "replacements": []})
        self.assertEqual(out, "AI và di")

    def test_itn_placeholders_cannot_collide_with_user_text(self):
        text = "Mã SODAITN0END và hai đồng nghiệp."
        self.assertEqual(apply_vietnamese_itn(text, mode="safe"), text)

    def test_version_requires_complete_supported_span(self):
        self.assertEqual(
            apply_vietnamese_itn("phiên bản hai chấm ba mươi", mode="safe"),
            "phiên bản hai chấm ba mươi",
        )
        self.assertEqual(
            apply_vietnamese_itn("phiên bản hai chấm ba", mode="safe"),
            "phiên bản 2.3",
        )
        for text in (
            "phiên bản hai chấm ba chấm bốn chấm năm",
            "phiên bản hai chấm ba chấm bốn point five",
        ):
            self.assertEqual(apply_vietnamese_itn(text, mode="safe"), text)

    def test_reviewer_safe_itn_examples_and_immutable_spans(self):
        self.assertEqual(apply_vietnamese_itn("một phẩy năm việt nam đồng", mode="safe"), "một phẩy năm việt nam đồng")
        self.assertEqual(apply_vietnamese_itn("hai mươi bốn giờ rưỡi", mode="safe"), "hai mươi bốn giờ rưỡi")
        invalid = "ngày ba mươi mốt tháng hai năm hai nghìn không trăm hai mươi bốn"
        self.assertEqual(apply_vietnamese_itn(invalid, mode="safe"), invalid)
        long_number = "không một hai ba bốn năm sáu bảy tám chín không một hai"
        self.assertEqual(apply_vietnamese_itn(long_number, mode="safe"), long_number)
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        value, _ = proc.process(
            "github /tmp/hoà.txt https://example.com/q?a=1",
            vocabulary={"phrases": [], "replacements": [{"heard": "github", "write": "GitHub"}]},
        )
        self.assertEqual(value, "GitHub /tmp/hoà.txt https://example.com/q?a=1")

    def test_safe_processing_never_rewrites_prose_or_literals(self):
        for text in (
            "Bệnh nhân có hai triệu chứng.",
            "Chúng tôi gặp hai tư vấn viên.",
            "Đó là hai tỷ lệ khác nhau.",
        ):
            self.assertEqual(apply_vietnamese_itn(text, mode="safe"), text)
        self.assertEqual(apply_vietnamese_itn("tôi mua hai đồng hồ", mode="safe"), "tôi mua hai đồng hồ")
        self.assertEqual(apply_vietnamese_itn("sáu giờ ba người đã tới", mode="safe"), "sáu giờ ba người đã tới")
        self.assertEqual(apply_vietnamese_itn("sáu giờ hai mươi phút nữa", mode="safe"), "sáu giờ hai mươi phút nữa")
        self.assertEqual(apply_vietnamese_itn("cuộc họp kéo dài hai giờ ba mươi phút", mode="safe"), "cuộc họp kéo dài hai giờ ba mươi phút")
        self.assertEqual(apply_vietnamese_itn("lúc sáu giờ hai mươi phút", mode="safe"), "lúc 6:20")
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        self.assertEqual(proc.process("sô đa là nước uống", vocabulary={"phrases": [], "replacements": []})[0], "Sô đa là nước uống")
        for value in ("server.py", "https://example.com/a", "example.com", "a@b.example", r"C:\\Temp\\server.py", r"C:\\Dữ liệu\\hoà.txt"):
            self.assertEqual(polish_vietnamese_punctuation(value), value)
        self.assertEqual(
            VietnamesePostProcessor(PostProcessConfig(itn_mode="safe")).process(
                r"C:\\Dữ liệu\\hoà.txt", vocabulary={"phrases": [], "replacements": []}
            )[0],
            r"C:\\Dữ liệu\\hoà.txt",
        )
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        value, _ = proc.process("xê đê pê a", vocabulary={"replacements": [{"heard": "xê đê pê a", "write": "server.py"}]})
        self.assertEqual(value, "server.py")

    def test_vietnamese_itn_safe_and_full(self):
        s = "Ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu lúc chín giờ ba mươi phút đạt hai mươi lăm phần trăm với giá một trăm năm mươi nghìn đồng và số không chín không một hai ba bốn năm sáu bảy cho phiên bản hai chấm ba."
        out = apply_vietnamese_itn(s, mode="safe")
        self.assertIn("15/11/2026", out)
        self.assertIn("9:30", out)
        self.assertIn("25%", out)
        self.assertIn("một trăm năm mươi nghìn đồng", out)
        self.assertIn("0901234567", out)
        self.assertIn("phiên bản 2.3", out)

        # Safe mode must NOT corrupt natural single-digit Vietnamese prose!
        prose = "Có một người bạn đi hai năm nay với ba cuốn sách."
        self.assertEqual(apply_vietnamese_itn(prose, mode="safe"), prose)

    def test_phonetic_vocab_rescue_and_zero_hallucination(self):
        vocab = ["nhận dạng giọng nói", "chuyển đổi số", "bảo mật dữ liệu"]
        text = "Hệ thống nhận rạng giọng lói giúp truyển đổi số và bảo mặc dữ liệu."
        fixed, edits = phonetic_vocab_rescue(text, vocab, threshold=0.80)
        self.assertIn("nhận dạng giọng nói", fixed)
        self.assertIn("chuyển đổi số", fixed)
        self.assertIn("bảo mật dữ liệu", fixed)
        self.assertEqual(len(edits), 3)

        # Zero hallucination: unrelated clean text must remain untouched
        clean = "Hôm nay trời rất đẹp và tôi đi uống cà phê."
        fixed_clean, edits_clean = phonetic_vocab_rescue(clean, vocab, threshold=0.80)
        self.assertEqual(fixed_clean, clean)
        self.assertEqual(edits_clean, [])
        exact, exact_edits = phonetic_vocab_rescue("nhóm An Ninh", ["nhóm An Ninh", "nhóm ăn ninh"])
        self.assertEqual(exact, "nhóm An Ninh")
        self.assertEqual(exact_edits, [])

    def test_full_pipeline_code_switching_and_punctuation(self):
        proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))
        vocab = {
            "schema_version": 1,
            "language": "vi-VN",
            "phrases": ["nhận dạng giọng nói", "CDPA"],
            "replacements": [{"heard": "xê đê pê a", "write": "CDPA"}],
        }
        raw = "hôm nay mình đì ploy sô đa lên gít hắp dùng phát ây pi ai và xê đê pê a , tốc độ tăng hai mươi lăm phần trăm ."
        out, meta = proc.process(raw, vocabulary=vocab)
        self.assertIn("deploy", out)
        self.assertIn("SODA", out)
        self.assertIn("GitHub", out)
        self.assertIn("FastAPI", out)
        self.assertIn("CDPA", out)
        self.assertIn("25%", out)
        self.assertTrue(out.startswith("Hôm nay"))
        self.assertNotIn(" ,", out)
        self.assertNotIn(" .", out)
        self.assertTrue(meta["itn_applied"])

    def test_gated_llm_corrector_accepts_faithful_and_rejects_hallucination(self):
        # Faithful edit (only tone/diacritic fix) -> accepted
        faithful = GatedLLMCorrector(
            "http://127.0.0.1:9999/v1/chat/completions",
            "local-slm",
            requester=lambda _u, _p, _t: {
                "choices": [{"message": {"content": "Hệ thống nhận dạng giọng nói tiếng Việt"}}]
            },
        )
        out1, applied1 = faithful.correct("Hệ thống nhận rạng giọng nói tiếng Việt")
        self.assertTrue(applied1)
        self.assertEqual(out1, "Hệ thống nhận dạng giọng nói tiếng Việt")

        # Hallucinated rewrite -> rejected!
        hallucinator = GatedLLMCorrector(
            "http://127.0.0.1:9999/v1/chat/completions",
            "local-slm",
            requester=lambda _u, _p, _t: {
                "choices": [{"message": {"content": "Xin chào tôi là trợ lý AI có thể giúp gì cho bạn hôm nay"}}]
            },
        )
        orig = "Hệ thống nhận rạng giọng nói tiếng Việt"
        out2, applied2 = hallucinator.correct(orig)
        self.assertFalse(applied2)
        self.assertEqual(out2, orig)


class TestDynamicBiasAndLongAudioAndMiner(unittest.TestCase):
    def test_smart_bias_filters_unsafe_digit_candidates_and_keeps_snapshot(self):
        profile = build_smart_bias_profile(
            personal_phrases=["hệ thống", "mã 1234", "FastAPI"],
        )
        self.assertIsNotNone(profile)
        assert profile is not None
        phrases = [term["phrase"] for term in profile["terms"]]
        self.assertIn("hệ thống", phrases)
        self.assertIn("FastAPI", phrases)
        self.assertNotIn("mã 1234", phrases)
        self.assertEqual(profile["diagnostics"]["rejected_count"], 1)
        self.assertEqual(profile["diagnostics"]["rejected_reasons"]["unsafe phrase format"], 1)
        self.assertNotIn("mã 1234", repr(profile["diagnostics"]))
        validate_profile(profile)

    def test_rejected_profile_unicode_and_digit_entry_never_reaches_digest_or_output(self):
        from training.context_profile import profile_phrase_safety_error, profile_sha256
        self.assertEqual(profile_phrase_safety_error("mã 123456788"), "unsafe phrase format")
        self.assertEqual(profile_phrase_safety_error("mã\x00 bí mật"), "control character")
        self.assertEqual(profile_phrase_safety_error("mã\ud800 bí mật"), "invalid unicode")
        profile = build_smart_bias_profile(personal_phrases=["mã 123456788"])
        self.assertIsNone(profile)
        self.assertEqual(profile_sha256(profile), "baseline")

    def test_smart_bias_returns_no_profile_only_with_diagnostic_when_all_unsafe(self):
        with self.assertLogs("vi_enhance.dynamic_bias", level="WARNING") as logs:
            profile = build_smart_bias_profile(personal_phrases=["mã 1234"])
        self.assertIsNone(profile)
        self.assertIn("No validator-safe Vietnamese bias candidates", logs.output[0])
        self.assertNotIn("mã 1234", logs.output[0])

    def test_smart_bias_enforces_40_cap_and_prioritizes_oov(self):
        many_curated = {
            "schema_version": 1,
            "language": "vi-VN",
            "terms": [{"phrase": f"cụm từ mẫu số {i}", "boost": 1.0} for i in range(55)],
        }
        personal = ["CDPA", "FastAPI", "WebSocket", "nhận dạng giọng nói"]
        profile = build_smart_bias_profile(personal_phrases=personal, curated_profile=many_curated)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertLessEqual(len(profile["terms"]), 40)
        top_phrases = [t["phrase"] for t in profile["terms"][:4]]
        self.assertIn("CDPA", top_phrases)
        self.assertIn("FastAPI", top_phrases)
        # Acronym gets higher boost than generic 1-word term
        _, boost_acronym = score_bias_phrase("CDPA")
        _, boost_generic = score_bias_phrase("mô hình")
        self.assertGreater(boost_acronym, boost_generic)
        validate_profile(profile)

        curated_zero = build_smart_bias_profile(curated_profile={
            "schema_version": 1,
            "language": "vi-VN",
            "terms": [{"phrase": "Curated Term", "boost": 0.0}],
        })
        self.assertEqual(curated_zero["terms"][0]["boost"], 0.0)

    def test_positive_postprocess_allowlist_is_not_truthiness(self):
        allowed = lambda value: str(value).strip().casefold() in {"1", "true", "on"}
        for value in ("1", "true", "on", " TRUE "):
            self.assertTrue(allowed(value))
        for value in (None, "0", "false", "yes", ""):
            self.assertFalse(allowed(value))

    def test_long_audio_vad_splits_at_silence_valleys(self):
        # 75 seconds: 20s tone + 1s silence + 25s tone + 1s silence + 28s tone
        pcm = (
            _synth_sine(220.0, 20.0, amp=0.25)
            + (b"\x00" * 32000)
            + _synth_sine(240.0, 25.0, amp=0.25)
            + (b"\x00" * 32000)
            + _synth_sine(200.0, 28.0, amp=0.25)
        )
        plans = plan_vad_segments(pcm, target_seconds=22.0, max_seconds=30.0)
        self.assertEqual(len(plans), 3)
        for seg in plans:
            self.assertLessEqual(seg.duration_seconds, 31.0)
        # First boundary should fall inside the first silence valley (around 20..21s)
        self.assertTrue(19.5 <= plans[0].end_sample / 16000.0 <= 21.5)

    def test_confusion_miner_extracts_recurring_phonetic_pairs(self):
        pairs = [
            ("hệ thống nhận rạng giọng nói chạy tốt", "hệ thống nhận dạng giọng nói chạy tốt"),
            ("cần cải thiện nhận rạng giọng nói nhanh hơn", "cần cải thiện nhận dạng giọng nói nhanh hơn"),
            ("chúng ta sẽ đì ploy bản mới", "chúng ta sẽ deploy bản mới"),
            ("ngày mai đì ploy lên máy chủ", "ngày mai deploy lên máy chủ"),
            ("một lỗi ngẫu nhiên mèo", "một lỗi ngẫu nhiên chó"),  # count=1 & non-phonetic -> ignored
        ]
        mined = mine_confusions(pairs, min_count=2, min_phonetic_similarity=0.60)
        heard_set = {r["heard"] for r in mined["replacements"]}
        self.assertIn("rạng", heard_set)
        self.assertNotIn("mèo", heard_set)
        self.assertIsNotNone(mined["bias_profile"])

    def test_evaluate_samples_read_pcm_with_enhance(self):
        pcm = _synth_sine(220.0, 0.5, amp=0.01)
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "sample.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm)
            out_pcm, levels = read_pcm(wav_path, 0.0, enhance_audio=True)
            self.assertGreater(len(out_pcm), len(pcm))  # includes HAT silence padding
            self.assertEqual(levels["clipped_samples"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
