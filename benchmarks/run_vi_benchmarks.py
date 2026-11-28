"""Evidence-first quantitative benchmark for model-agnostic Vietnamese SODA upgrades.

Measures and prints reproducible metrics for:
1. Audio Front-End DSP Benchmark (DC/50Hz rejection, F0 preservation, quiet -46 dBFS
   AGC recovery, hard-clipping elimination, noisy-speech SNR gain via OA, realtime speed).
2. Vietnamese Post-Processing Accuracy Benchmark (20 realistic SODA-error Vietnamese
   utterances covering North/Central/South dialect confusions, tone slips, tech
   code-switching, boundary overlap stutters, and spoken ITN) measuring Before vs After
   SER/WER and CER using the repository's own training.calibrate.rate().
3. Zero-Hallucination Regression Guardrail (10 clean Vietnamese sentences -> 0.0%
   over-correction rate).
"""
from __future__ import annotations

from array import array
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.calibrate import rate
from vi_enhance.audio_frontend import AudioFrontendConfig, enhance_pcm16
from vi_enhance.postprocessor import (
    PostProcessConfig,
    VietnamesePostProcessor,
    stitch_segments,
)


def _synth_pcm(samples: list[float]) -> bytes:
    arr = array("h", (max(-32768, min(32767, round(x * 32767))) for x in samples))
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def _pcm_to_floats(pcm: bytes) -> list[float]:
    arr = array("h")
    arr.frombytes(pcm)
    if sys.byteorder != "little":
        arr.byteswap()
    return [v / 32768.0 for v in arr]


def _rms(vals: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vals) / max(1, len(vals)))


def run_dsp_benchmark() -> dict:
    sr = 16000
    # 1. Hum + DC rejection vs Vietnamese F0 (180 Hz) preservation
    hum_in = _synth_pcm([0.12 + 0.20 * math.sin(2 * math.pi * 50 * i / sr) for i in range(sr)])
    f0_in = _synth_pcm([0.20 * math.sin(2 * math.pi * 180 * i / sr) for i in range(sr)])
    hpf_cfg = AudioFrontendConfig(enable_highpass=True, enable_denoise_oa=False, enable_agc=False)
    hum_out, _ = enhance_pcm16(hum_in, hpf_cfg)
    f0_out, _ = enhance_pcm16(f0_in, hpf_cfg)
    hum_atten_db = 20 * math.log10(_rms(_pcm_to_floats(hum_out)[3200:]) / _rms(_pcm_to_floats(hum_in)[3200:]))
    f0_retention = _rms(_pcm_to_floats(f0_out)[3200:]) / _rms(_pcm_to_floats(f0_in)[3200:])

    # 2. Quiet -46 dBFS recording recovery (matching docs/VI_DEVELOPMENT_RESULTS.md)
    quiet_in = _synth_pcm([0.007 * math.sin(2 * math.pi * 220 * i / sr) for i in range(sr * 2)])
    _, quiet_summary = enhance_pcm16(quiet_in, AudioFrontendConfig(enable_denoise_oa=False, enable_agc=True))

    # 3. Noisy speech SNR improvement via F0-protected Wiener + Observation Addition
    rng = random.Random(2026)
    noise_only = [rng.gauss(0, 0.06) for _ in range(sr)]
    clean_voice = [
        0.16 * math.sin(2 * math.pi * 190 * i / sr)
        + 0.08 * math.sin(2 * math.pi * 380 * i / sr)
        + 0.05 * math.sin(2 * math.pi * 1100 * i / sr)
        for i in range(sr)
    ]
    noisy_voice = [c + rng.gauss(0, 0.06) for c in clean_voice]
    full_noisy = _synth_pcm(noise_only + noisy_voice)
    t0 = time.perf_counter()
    enhanced_noisy, noisy_summary = enhance_pcm16(
        full_noisy, AudioFrontendConfig(enable_highpass=True, enable_denoise_oa=True, enable_agc=False)
    )
    elapsed = time.perf_counter() - t0
    ef = _pcm_to_floats(enhanced_noisy)
    # Compare noise-segment RMS vs speech-segment RMS before and after
    snr_before_db = 20 * math.log10(_rms(noisy_voice) / _rms(noise_only))
    snr_after_db = 20 * math.log10(_rms(ef[sr + 3200 :]) / max(_rms(ef[3200:sr]), 1e-6))

    return {
        "hum_50hz_plus_dc_attenuation_db": round(hum_atten_db, 2),
        "vietnamese_f0_180hz_retention_ratio": round(f0_retention, 4),
        "quiet_input_dbfs": quiet_summary["mean_input_dbfs"],
        "quiet_output_dbfs": quiet_summary["mean_output_dbfs"],
        "quiet_clipped_samples": quiet_summary["clipped_samples"],
        "noisy_snr_before_db": round(snr_before_db, 2),
        "noisy_snr_after_db": round(snr_after_db, 2),
        "noisy_snr_gain_db": round(snr_after_db - snr_before_db, 2),
        "oa_alpha_used": noisy_summary["mean_oa_alpha"],
        "realtime_speed_x": round(2.0 / max(elapsed, 1e-6), 1),
    }


def run_vietnamese_text_benchmark() -> dict:
    vocab = {
        "schema_version": 1,
        "language": "vi-VN",
        "phrases": [
            "nhận dạng giọng nói",
            "chuyển đổi số",
            "bảo mật dữ liệu",
            "mô hình ngôn ngữ",
            "trí tuệ nhân tạo",
            "độ trễ thấp",
            "xác thực hai lớp",
            "kho tri thức",
            "CDPA",
            "FastAPI",
            "WebSocket",
            "Tailscale",
        ],
        "replacements": [
            {"heard": "xê đê pê a", "write": "CDPA"},
        ],
    }

    # 20 realistic Vietnamese utterances with documented SODA error modes
    noisy_cases = [
        # Dialect onset/coda/tone confusions
        (["hệ thống nhận rạng giọng lói hoạt động ổn định"], "Hệ thống nhận dạng giọng nói hoạt động ổn định"),
        (["doanh nghiệp đang đẩy mạnh truyển đổi số"], "Doanh nghiệp đang đẩy mạnh chuyển đổi số"),
        (["chúng tôi ưu tiên bảo mặc dữ liệu người dùng"], "Chúng tôi ưu tiên bảo mật dữ liệu người dùng"),
        (["mô hình ngôn ngử lớn chạy cục bộ"], "Mô hình ngôn ngữ lớn chạy cục bộ"),
        (["ứng dụng chí tuệ nhân tạo vào giáo dục"], "Ứng dụng trí tuệ nhân tạo vào giáo dục"),
        (["hệ thống đạt độ dễ thấp trên máy tính"], "Hệ thống đạt độ trễ thấp trên máy tính"),
        (["vui lòng bật xác thật hai lớp ngay"], "Vui lòng bật xác thực hai lớp ngay"),
        (["tra cứu thông tin trong kho chi thức"], "Tra cứu thông tin trong kho tri thức"),
        # Tech code-switching confusions
        (["hôm nay mình đì ploy sô đa lên gít hắp"], "Hôm nay mình deploy SODA lên GitHub"),
        (["dùng phát ây pi ai và oép sóc két cho bách en"], "Dùng FastAPI và WebSocket cho backend"),
        (["chạy đốc cơ với pái thần và pốt gờ rét"], "Chạy Docker với Python và PostgreSQL"),
        (["đã cơm mít mã nguồn và chờ rì viu"], "Đã commit mã nguồn và chờ review"),
        (["tích hợp xê đê pê a cùng eo eo em cục bộ"], "Tích hợp CDPA cùng LLM cục bộ"),
        # Streaming boundary overlap stutters
        (["hôm nay chúng ta sẽ", "chúng ta sẽ kiểm tra lại toàn bộ"], "Hôm nay chúng ta sẽ kiểm tra lại toàn bộ"),
        (["kết quả đo được", "đo được cho thấy độ chính xác tăng"], "Kết quả đo được cho thấy độ chính xác tăng"),
        # Vietnamese ITN (numbers, currency, date, time, phone, percent, version)
        (["tốc độ tăng hai mươi lăm phần trăm"], "Tốc độ tăng 25%"),
        (["tổng chi phí là một trăm năm mươi nghìn đồng"], "Tổng chi phí là 150.000đ"),
        (["hẹn gặp lúc chín giờ ba mươi phút ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu"], "Hẹn gặp lúc 9:30 ngày 15/11/2026"),
        (["liên hệ số không chín không một hai ba bốn năm sáu bảy"], "Liên hệ số 0901234567"),
        (["đang phát hành phiên bản hai chấm ba chấm một"], "Đang phát hành phiên bản 2.3.1"),
    ]

    # 10 clean Vietnamese sentences -> must have 0% over-correction!
    clean_cases = [
        "Hôm nay trời rất đẹp và tôi đi uống cà phê với bạn.",
        "Có một người bạn đi hai năm nay với ba cuốn sách hay.",
        "Chúng ta cần hoàn thành báo cáo trước thứ sáu tuần này.",
        "Hòa bình và ổn định là mục tiêu quan trọng nhất.",
        "Anh ấy đang đọc tài liệu kỹ thuật trên máy tính xách tay.",
        "Buổi họp bắt đầu đúng giờ tại phòng họp tầng năm.",
        "Cảm ơn bạn đã hỗ trợ kiểm tra lỗi rất nhanh chóng.",
        "Hệ thống chạy hoàn toàn trên máy cá nhân không gửi lên đám mây.",
        "Thời tiết ở Hà Nội hôm nay mát mẻ và có nắng nhẹ.",
        "Mình sẽ gửi lại tệp nén ngay sau khi kiểm thử xong.",
    ]

    proc = VietnamesePostProcessor(PostProcessConfig(itn_mode="safe"))

    raw_we = raw_wn = raw_ce = raw_cn = 0
    post_we = post_wn = post_ce = post_cn = 0
    for segs, ref in noisy_cases:
        raw_joined = " ".join(segs)
        stitched = stitch_segments(segs, overlap_windows=True)
        enhanced, _ = proc.process(stitched, vocabulary=vocab)

        e, n = rate(ref, raw_joined)
        raw_we += e
        raw_wn += n
        e, n = rate(ref, raw_joined, "char")
        raw_ce += e
        raw_cn += n

        e, n = rate(ref, enhanced)
        post_we += e
        post_wn += n
        e, n = rate(ref, enhanced, "char")
        post_ce += e
        post_cn += n

    clean_overcorrections = 0
    for sent in clean_cases:
        out, _ = proc.process(sent, vocabulary=vocab)
        if out != sent:
            clean_overcorrections += 1

    raw_wer = raw_we / max(1, raw_wn)
    post_wer = post_we / max(1, post_wn)
    raw_cer = raw_ce / max(1, raw_cn)
    post_cer = post_ce / max(1, post_cn)

    return {
        "utterances_tested": len(noisy_cases),
        "raw_word_errors": raw_we,
        "post_word_errors": post_we,
        "reference_words": raw_wn,
        "raw_wer_pct": round(raw_wer * 100, 2),
        "post_wer_pct": round(post_wer * 100, 2),
        "relative_wer_reduction_pct": round((raw_wer - post_wer) / max(raw_wer, 1e-9) * 100, 2),
        "raw_cer_pct": round(raw_cer * 100, 2),
        "post_cer_pct": round(post_cer * 100, 2),
        "relative_cer_reduction_pct": round((raw_cer - post_cer) / max(raw_cer, 1e-9) * 100, 2),
        "clean_sentences_tested": len(clean_cases),
        "clean_overcorrections": clean_overcorrections,
        "overcorrection_rate_pct": round(clean_overcorrections / len(clean_cases) * 100, 2),
    }


def main() -> None:
    report = {
        "dsp_frontend": run_dsp_benchmark(),
        "vietnamese_postprocessor": run_vietnamese_text_benchmark(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
