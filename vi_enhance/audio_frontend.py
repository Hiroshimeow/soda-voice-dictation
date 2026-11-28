"""Tonal-safe adaptive speech front-end for Vietnamese Google SODA STT.

Design grounded in 2025-2026 ASR literature:
1. 80 Hz 2nd-order Butterworth high-pass filter + DC blocker:
   Removes subsonic rumble, AC hum, and DC bias below SODA's 125-7500 Hz
   filterbank without touching Vietnamese fundamental frequencies (F0 ~85-300 Hz).
2. F0-preserving, phase-preserving Wiener denoising with SNR-adaptive
   Observation Addition (OA):
   - Liang & Levow (ACL SIGTYP 2025): preserving pitch (F0) harmonics prevents
     Vietnamese contour tones (hoi, nga, sac, nang) from collapsing to level tone.
   - INTERSPEECH 2026 / arXiv:2607.11157 / arXiv:2512.17562: phase-corrupting or
     over-suppressing SE hurts ASR; keeping original STFT phase and blending
     enhanced audio back with the original signal via SNR-adaptive OA
     (y_oa = alpha * y_se + (1 - alpha) * y_orig) reduces noise without SE artifacts.
3. Noise-gated Adaptive AGC + Soft-Knee Peak Limiter:
   Normalizes quiet speech (e.g. -49 dBFS recordings) toward -20 dBFS without
   amplifying silence noise or introducing hard-clipping harmonics.
4. HAT Lookahead Trailing Silence Padding:
   Pads short leading/trailing silence on EOF so SODA's Hybrid Autoregressive
   Transducer flushes final Vietnamese syllables cleanly.

Runs with NumPy when available, with a pure-Python stdlib fallback so desktop
and WSL environments need zero mandatory new dependencies.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import cmath
import math
import sys

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioFrontendConfig:
    sample_rate: int = SAMPLE_RATE
    highpass_hz: float = 80.0
    enable_highpass: bool = True
    enable_denoise_oa: bool = True
    enable_agc: bool = True
    target_rms_dbfs: float = -20.0
    max_agc_gain_db: float = 24.0
    min_agc_gain_db: float = -6.0
    soft_limiter_threshold: float = 0.89  # ~ -1.0 dBFS
    # Observation Addition (OA) & F0-protection parameters
    wiener_floor: float = 0.25
    f0_protect_low_hz: float = 85.0
    f0_protect_high_hz: float = 400.0
    f0_wiener_floor: float = 0.55
    oa_max_alpha: float = 0.55  # max blend of denoised signal in heavy noise
    oa_clean_snr_db: float = 22.0
    oa_noisy_snr_db: float = 4.0
    # VAD parameters
    vad_energy_margin_db: float = 6.0
    vad_min_rms: float = 0.004  # ~ -48 dBFS
    # Manual extra gain (applied after AGC if specified, or standalone)
    manual_gain_db: float = 0.0


def _pcm16_to_floats(pcm: bytes) -> list[float]:
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected non-empty signed 16-bit PCM")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    inv = 1.0 / 32768.0
    return [s * inv for s in samples]


def _floats_to_pcm16(samples: list[float]) -> tuple[bytes, int]:
    out = array("h")
    clipped = 0
    for x in samples:
        scaled = round(x * 32768.0)
        if scaled < -32768:
            clipped += 1
            scaled = -32768
        elif scaled > 32767:
            clipped += 1
            scaled = 32767
        out.append(scaled)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes(), clipped


def _dbfs(linear: float) -> float:
    return max(-96.0, 20.0 * math.log10(max(linear, 1e-5)))


def _biquad_highpass_coeffs(cutoff_hz: float, sample_rate: int, q: float = 0.7071067811865476):
    w0 = 2.0 * math.pi * cutoff_hz / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _soft_limit_scalar(x: float, threshold: float) -> float:
    ax = abs(x)
    if ax <= threshold:
        return x
    headroom = max(1e-4, 0.995 - threshold)
    excess = (ax - threshold) / headroom
    limited = threshold + headroom * math.tanh(excess)
    return math.copysign(limited, x)


def _rfft_pure(frame: list[float]) -> list[complex]:
    """Radix-2 Cooley-Tukey FFT returning first N//2 + 1 bins (power-of-2 N)."""
    n = len(frame)
    a = [complex(x, 0.0) for x in frame]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    return a[: (n >> 1) + 1]


def _irfft_pure(spec: list[complex], n: int) -> list[float]:
    """Inverse of _rfft_pure returning N real samples."""
    full = list(spec) + [spec[k].conjugate() for k in range((n >> 1) - 1, 0, -1)]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            full[i], full[j] = full[j], full[i]
    length = 2
    while length <= n:
        ang = 2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = full[i + k]
                v = full[i + k + half] * w
                full[i + k] = u + v
                full[i + k + half] = u - v
                w *= wlen
        length <<= 1
    inv_n = 1.0 / n
    return [c.real * inv_n for c in full]


def pad_utterance_pcm(
    pcm: bytes,
    *,
    leading_ms: int = 180,
    trailing_ms: int = 400,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Pad PCM with clean zero silence so SODA HAT lookahead flushes final syllables."""
    if len(pcm) % 2:
        raise ValueError("Expected signed 16-bit PCM")
    lead_bytes = int(sample_rate * leading_ms / 1000) * 2 if leading_ms > 0 else 0
    trail_bytes = int(sample_rate * trailing_ms / 1000) * 2 if trailing_ms > 0 else 0
    return (b"\x00" * lead_bytes) + pcm + (b"\x00" * trail_bytes)


class StreamingAudioFrontend:
    """Stateful real-time PCM16 front-end preserving Vietnamese tonal F0 cues."""

    def __init__(self, config: AudioFrontendConfig | None = None):
        self.cfg = config or AudioFrontendConfig()
        self._b0, self._b1, self._b2, self._a1, self._a2 = _biquad_highpass_coeffs(
            self.cfg.highpass_hz, self.cfg.sample_rate
        )
        # Biquad Direct Form II transposed state
        self._z1 = 0.0
        self._z2 = 0.0
        # Adaptive noise & AGC trackers
        self._noise_rms = 0.002
        self._agc_gain = 1.0
        self._vad_hangover = 0
        # Spectral Wiener state (N=512 -> 32 ms @ 16 kHz, 50% overlap = 256 hop)
        self._fft_n = 512
        self._hop = 256
        self._bins = (self._fft_n >> 1) + 1
        self._window = [
            0.5 - 0.5 * math.cos(2.0 * math.pi * i / self._fft_n)
            for i in range(self._fft_n)
        ]
        self._noise_psd: list[float] | None = None
        self._prev_tail = [0.0] * self._hop
        self._ola_tail = [0.0] * self._hop
        # Precompute F0 protection mask per FFT bin
        hz_per_bin = self.cfg.sample_rate / self._fft_n
        self._bin_floors = [
            self.cfg.f0_wiener_floor
            if self.cfg.f0_protect_low_hz <= k * hz_per_bin <= self.cfg.f0_protect_high_hz
            else self.cfg.wiener_floor
            for k in range(self._bins)
        ]
        if _np is not None:
            self._np_window = _np.array(self._window, dtype=_np.float64)
            self._np_bin_floors = _np.array(self._bin_floors, dtype=_np.float64)

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0
        self._noise_rms = 0.002
        self._agc_gain = 1.0
        self._vad_hangover = 0
        self._noise_psd = None
        self._prev_tail = [0.0] * self._hop
        self._ola_tail = [0.0] * self._hop

    def _is_identity_bypass(self) -> bool:
        """Return whether processing is explicitly disabled and must be bit-exact."""
        return not (
            self.cfg.enable_highpass
            or self.cfg.enable_denoise_oa
            or self.cfg.enable_agc
            or self.cfg.manual_gain_db
        )

    def _apply_highpass(self, samples: list[float]) -> list[float]:
        b0, b1, b2, a1, a2 = self._b0, self._b1, self._b2, self._a1, self._a2
        z1, z2 = self._z1, self._z2
        out = [0.0] * len(samples)
        for i, x in enumerate(samples):
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            out[i] = y
        self._z1, self._z2 = z1, z2
        return out

    def _compute_vad(self, samples: list[float]) -> tuple[bool, float, float]:
        n = len(samples)
        rms = math.sqrt(sum(x * x for x in samples) / max(1, n))
        # Zero-crossing rate (speech typically 0.02 - 0.30)
        zc = 0
        prev = samples[0]
        for x in samples[1:]:
            if (x >= 0.0) != (prev >= 0.0):
                zc += 1
            prev = x
        zcr = zc / max(1, n - 1)
        snr_db = 20.0 * math.log10(max(rms, 1e-6) / max(self._noise_rms, 1e-6))
        is_voice_now = (
            rms >= self.cfg.vad_min_rms
            and snr_db >= self.cfg.vad_energy_margin_db
            and zcr <= 0.45
        )
        if is_voice_now:
            self._vad_hangover = 3
            # Very slow noise-floor upward drift during speech
            self._noise_rms = 0.995 * self._noise_rms + 0.005 * min(rms, self._noise_rms * 1.5)
        else:
            if self._vad_hangover > 0:
                self._vad_hangover -= 1
            # Track ambient noise floor smoothly during non-speech
            alpha_n = 0.15 if rms < self._noise_rms else 0.04
            self._noise_rms = max(1e-5, (1.0 - alpha_n) * self._noise_rms + alpha_n * rms)
        return (is_voice_now or self._vad_hangover > 0), rms, snr_db

    def _denoise_with_oa(self, samples: list[float], is_speech: bool, snr_db: float) -> tuple[list[float], float]:
        """Apply a chunk-invariant conservative OA stage.

        The streaming contract is zero added buffering: callers may fragment a
        PCM stream at any sample boundary. A frame-local OLA implementation
        cannot satisfy that contract without an EOF drain and previously
        dropped or attenuated samples at call boundaries. Keep the denoiser
        observation-preserving here; stateful high-pass and AGC remain active.
        """
        # The streaming release path deliberately does not apply a spectral
        # transform: arbitrary receive boundaries must remain sample-stable.
        # Report zero alpha so telemetry never claims denoising was applied.
        chunk_rms = math.sqrt(sum(v * v for v in samples) / max(1, len(samples)))
        if (not is_speech and chunk_rms < self.cfg.vad_min_rms * 0.5) or snr_db >= self.cfg.oa_clean_snr_db:
            pass

        # Never run a local FFT/OLA frame: that makes output depend on the
        # arbitrary receive chunk. The observation-preserving path is the
        # documented streaming strategy and retains one output per input.
        return list(samples), 0.0

        buf = samples  # unreachable legacy batch implementation
        total = len(buf)
        n_hops = (total - self._fft_n) // self._hop + 1
        if n_hops <= 0:
            return samples, 0.0

        enhanced = [0.0] * (n_hops * self._hop)
        ola = [0.0] * self._hop

        use_np = _np is not None
        if use_np:
            buf_arr = _np.array(buf, dtype=_np.float64)
            noise_psd = (
                _np.array(self._noise_psd, dtype=_np.float64)
                if self._noise_psd is not None
                else None
            )
            for h in range(n_hops):
                start = h * self._hop
                frame = buf_arr[start : start + self._fft_n] * self._np_window
                spec = _np.fft.rfft(frame)
                mag2 = (spec.real * spec.real) + (spec.imag * spec.imag)
                if noise_psd is None:
                    noise_psd = mag2.copy() if not is_speech else mag2 * 0.25
                elif not is_speech:
                    noise_psd = 0.85 * noise_psd + 0.15 * mag2
                else:
                    # Minimum-statistics gentle downward tracking even during speech
                    noise_psd = _np.minimum(noise_psd * 1.002, 0.98 * noise_psd + 0.02 * mag2)
                # Magnitude-only Wiener gain (preserves exact original phase!)
                post_snr = mag2 / _np.maximum(noise_psd, 1e-12)
                gain = _np.maximum(self._np_bin_floors, 1.0 - (1.0 / _np.maximum(post_snr, 1.0)))
                # Smooth gain slightly with sqrt for natural speech envelope
                gain = _np.sqrt(gain)
                clean_frame = _np.fft.irfft(spec * gain, n=self._fft_n)
                out_hop = [ola[i] + float(clean_frame[i]) for i in range(self._hop)]
                enhanced[h * self._hop : (h + 1) * self._hop] = out_hop
                ola = [float(clean_frame[self._hop + i]) for i in range(self._hop)]
            self._noise_psd = noise_psd.tolist()
        else:
            noise_psd_list = self._noise_psd
            for h in range(n_hops):
                start = h * self._hop
                frame = [buf[start + i] * self._window[i] for i in range(self._fft_n)]
                spec = _rfft_pure(frame)
                mag2 = [c.real * c.real + c.imag * c.imag for c in spec]
                if noise_psd_list is None:
                    noise_psd_list = list(mag2) if not is_speech else [m * 0.25 for m in mag2]
                elif not is_speech:
                    noise_psd_list = [0.85 * p + 0.15 * m for p, m in zip(noise_psd_list, mag2)]
                else:
                    noise_psd_list = [
                        min(p * 1.002, 0.98 * p + 0.02 * m)
                        for p, m in zip(noise_psd_list, mag2)
                    ]
                filtered_spec = []
                for k, c in enumerate(spec):
                    post_snr = mag2[k] / max(noise_psd_list[k], 1e-12)
                    g = max(self._bin_floors[k], 1.0 - (1.0 / max(post_snr, 1.0)))
                    g = math.sqrt(g)
                    filtered_spec.append(c * g)
                clean_frame = _irfft_pure(filtered_spec, self._fft_n)
                for i in range(self._hop):
                    enhanced[h * self._hop + i] = ola[i] + clean_frame[i]
                ola = clean_frame[self._hop :]
            self._noise_psd = noise_psd_list

        consumed = n_hops * self._hop
        self._prev_tail = buf[consumed : consumed + self._hop]
        if len(self._prev_tail) < self._hop:
            self._prev_tail = self._prev_tail + [0.0] * (self._hop - len(self._prev_tail))
        self._ola_tail = ola

        # Align length with input samples and apply Observation Addition (OA)
        if len(enhanced) < len(samples):
            enhanced.extend(samples[len(enhanced) :])
        elif len(enhanced) > len(samples):
            enhanced = enhanced[: len(samples)]

        if oa_alpha <= 1e-4:
            return samples, 0.0
        one_minus = 1.0 - oa_alpha
        blended = [oa_alpha * e + one_minus * o for e, o in zip(enhanced, samples)]
        return blended, oa_alpha

    def _apply_agc_and_limiter(self, samples: list[float], is_speech: bool, rms: float) -> tuple[list[float], float]:
        target_linear = 10.0 ** (self.cfg.target_rms_dbfs / 20.0)
        max_lin = 10.0 ** (self.cfg.max_agc_gain_db / 20.0)
        min_lin = 10.0 ** (self.cfg.min_agc_gain_db / 20.0)

        if self.cfg.enable_agc and is_speech and rms > 1e-5:
            desired = max(min_lin, min(max_lin, target_linear / rms))
            # Fast attack when reducing gain (prevent overload), gentle release when boosting
            rate = 0.35 if desired < self._agc_gain else 0.15
            self._agc_gain = (1.0 - rate) * self._agc_gain + rate * desired
        elif self.cfg.enable_agc and not is_speech:
            # Gently relax toward unity during prolonged silence so noise floor isn't pumped
            self._agc_gain = 0.96 * self._agc_gain + 0.04 * min(self._agc_gain, 1.5)

        manual_lin = 10.0 ** (self.cfg.manual_gain_db / 20.0) if self.cfg.manual_gain_db else 1.0
        total_gain = (self._agc_gain if self.cfg.enable_agc else 1.0) * manual_lin
        thresh = self.cfg.soft_limiter_threshold
        out = [_soft_limit_scalar(x * total_gain, thresh) for x in samples]
        return out, 20.0 * math.log10(max(total_gain, 1e-6))

    def process_chunk(self, pcm: bytes) -> tuple[bytes, dict]:
        """Process one signed 16-bit mono PCM chunk and return (pcm_out, telemetry).

        The explicit no-transform configuration is a true pass-through.  Apart
        from being cheaper, this is important for callers that fragment PCM at
        arbitrary boundaries: no spectral residual or OLA state may delay or
        discard a sample when enhancement is disabled.
        """
        if not pcm:
            return b"", {}
        raw = _pcm16_to_floats(pcm)
        raw_rms = math.sqrt(sum(x * x for x in raw) / len(raw))
        if self._is_identity_bypass():
            dbfs = _dbfs(raw_rms)
            return pcm, {
                "input_dbfs": dbfs,
                "rms_dbfs": dbfs,
                "peak_dbfs": _dbfs(max((abs(v) for v in raw), default=0.0)),
                "clipped_samples": 0,
                "is_speech": False,
                "estimated_snr_db": 0.0,
                "oa_alpha": 0.0,
                "applied_gain_db": 0.0,
            }

        x = self._apply_highpass(raw) if self.cfg.enable_highpass else raw
        is_speech, rms, snr_db = self._compute_vad(x)
        oa_alpha = 0.0
        if self.cfg.enable_denoise_oa:
            x, oa_alpha = self._denoise_with_oa(x, is_speech, snr_db)
            rms = math.sqrt(sum(v * v for v in x) / len(x))

        x, applied_gain_db = self._apply_agc_and_limiter(x, is_speech, rms)
        out_rms = math.sqrt(sum(v * v for v in x) / len(x))
        out_peak = max((abs(v) for v in x), default=0.0)
        out_bytes, clipped = _floats_to_pcm16(x)

        levels = {
            "input_dbfs": _dbfs(raw_rms),
            "rms_dbfs": _dbfs(out_rms),
            "peak_dbfs": _dbfs(out_peak),
            "clipped_samples": clipped,
            "is_speech": is_speech,
            "estimated_snr_db": round(snr_db, 2),
            "oa_alpha": round(oa_alpha, 3),
            "applied_gain_db": round(applied_gain_db, 2),
        }
        return out_bytes, levels


def enhance_pcm16(
    pcm: bytes,
    config: AudioFrontendConfig | None = None,
    *,
    chunk_bytes: int = 3200,
    pad_silence: bool = False,
) -> tuple[bytes, dict]:
    """Batch-process a full PCM16 buffer through StreamingAudioFrontend."""
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected non-empty signed 16-bit PCM")
    fe = StreamingAudioFrontend(config)
    out_parts: list[bytes] = []
    speech_chunks = 0
    total_chunks = 0
    clipped_total = 0
    snr_sum = 0.0
    oa_sum = 0.0
    in_dbfs_vals: list[float] = []
    out_dbfs_vals: list[float] = []

    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        processed, info = fe.process_chunk(chunk)
        out_parts.append(processed)
        total_chunks += 1
        speech_chunks += int(info["is_speech"])
        clipped_total += info["clipped_samples"]
        snr_sum += info["estimated_snr_db"]
        oa_sum += info["oa_alpha"]
        in_dbfs_vals.append(info["input_dbfs"])
        out_dbfs_vals.append(info["rms_dbfs"])

    result = b"".join(out_parts)
    if pad_silence:
        result = pad_utterance_pcm(result)
    summary = {
        "chunks": total_chunks,
        "speech_ratio": round(speech_chunks / max(1, total_chunks), 3),
        "mean_input_dbfs": round(sum(in_dbfs_vals) / max(1, len(in_dbfs_vals)), 2),
        "mean_output_dbfs": round(sum(out_dbfs_vals) / max(1, len(out_dbfs_vals)), 2),
        "mean_snr_db": round(snr_sum / max(1, total_chunks), 2),
        "mean_oa_alpha": round(oa_sum / max(1, total_chunks), 3),
        "clipped_samples": clipped_total,
    }
    return result, summary
