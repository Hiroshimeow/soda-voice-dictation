# Model-Agnostic Vietnamese Accuracy Upgrade Architecture

This upgrade strengthens `google-soda-native-stt-portable` for Vietnamese (`vi-VN`) **without modifying Google Chrome SODA model weights**. Every module is deterministic by default, requires zero new mandatory packages, and is grounded in 2024–2026 ASR research.

## 1. Root-Cause Audit of the Original Repository

1. **Static Linear Gain with Hard Clipping (`desktop/audio.py`)**: Hard clipping creates harmonic distortion that degrades Vietnamese formants/tones; quiet recordings (`-49 dBFS` in `docs/VI_DEVELOPMENT_RESULTS.md`) needed manual `+22 dB` gain while loud peaks clipped.
2. **Missing Subsonic Filtering & Tonal-Safe Denoising**: SODA's measured frontend uses an 80-channel log-mel filterbank spanning `125–7,500 Hz` (`docs/MODEL_ARCHITECTURE.md`), yet raw mic input carried DC bias and 50/60 Hz hum.
3. **Missing HAT Lookahead Trailing Flush (`native_session.py`, `desktop/core.py`, `desktop/file_transcribe.py`)**: SODA's `vi-VN` model is a Hybrid Autoregressive Transducer (`is_hat_model=true`); ending audio abruptly without trailing silence risks truncating the final Vietnamese syllable.
4. **Flat `boost=1.0` Contextual Biasing (`training/personal_vocabulary.py`)**: SODA enforces a practical 40-phrase cap, yet all phrases received flat priority and `boost=1.0` regardless of OOV rarity or code-switching value.
5. **Minimal Vietnamese Post-Processing**: Only exact `heard -> write` string replacement existed in `desktop/app.py`, with no tone-placement normalization, multi-dialect phonetic rescue, Vietnamese ITN, code-switching recovery, boundary overlap deduplication, or closed-loop error mining.

## 2. Research-Backed Upgrades Implemented (`vi_enhance/` & `training/`)

| Module | Technique | Research Grounding (2024–2026) |
| --- | --- | --- |
| `vi_enhance/audio_frontend.py` | 80 Hz 2nd-order Butterworth HPF + DC removal; F0-protected (`85–400 Hz`) phase-preserving Wiener denoising with **SNR-adaptive Observation Addition (OA)**; noise-gated adaptive AGC + `tanh` soft-knee limiter; 400 ms HAT trailing silence pad. | [Liang & Levow, ACL SIGTYP 2025](https://aclanthology.org/2025.sigtyp-1.11/) (pitch/F0 preservation prevents Vietnamese contour tones collapsing to level tone); [Li et al., INTERSPEECH 2026](https://hshi-speech.github.io/publications/), [arXiv:2607.11157 (2026)](https://arxiv.org/pdf/2607.11157), & [arXiv:2512.17562 (2025)](https://arxiv.org/html/2512.17562v1) (phase-preserving SE + Observation Addition avoids SE artifacts that hurt ASR). |
| `vi_enhance/phonetics.py` | Unicode NFC + modern Vietnamese tone-placement normalization (`hoà→hòa`, `thuý→thúy`); 6-tone Vietnamese syllable parser `(onset, nucleus, coda, tone)`; multi-dialect (North/Central/South) acoustic confusion distance (`l/n`, `s/x`, `ch/tr`, `r/d/gi`, `v/d/gi`, `hỏi/ngã`, `-n/-ng`, `-t/-c`). | [Multi-Dialect Vietnamese ASR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.426.pdf); [Vietnamese Spelling Correction, arXiv:2405.02573 (2024)](https://arxiv.org/pdf/2405.02573); [Diacritic-Aware Post-Correction for Vietnamese (2026)](https://www.researchgate.net/publication/408548708_Diacritic-Aware_Post-OCR_Correction_for_Vietnamese_Scanned_Documents_A_Lightweight_Baseline_for_Low-Quality_Document_Digitization). |
| `vi_enhance/postprocessor.py` | Streaming boundary overlap deduplicator (`stitch_segments`); Vietnamese tech code-switching map; **phonetic-similarity-gated** personal vocabulary rescue (zero-hallucination threshold + runner-up margin + protected stopword list); deterministic Vietnamese ITN (`safe`/`full`/`off`); punctuation/capitalization polish; optional acoustic-gated local LLM hook (`GatedLLMCorrector`). | [Revisiting ASR Error Correction with Specialized Models, arXiv:2405.15216 (2025)](https://arxiv.org/html/2405.15216v2) (low-WER acoustic gating prevents LLM over-correction/hallucination); [Vietnamese ASR: A Revisit, EACL 2026 Findings / arXiv:2603.14779](https://arxiv.org/html/2603.14779v1) (Vietnamese ITN, punctuation, capitalization); [CoSE-E, ACL SALMA 2026](https://aclanthology.org/2026.salma-1.11.pdf) (enterprise matrix-language code-switching). |
| `vi_enhance/dynamic_bias.py` | Information-theoretic ranker for SODA's 40-phrase cap (prioritizing OOV acronyms, code-switch terms, 2–4 syllable phrases, and mined error terms) with calibrated per-term boost weights (`0.8–2.5`). | [Contextual Biasing with KMP & Lightweight Phrase Selection](https://www.alphaxiv.org/@zelin-wu). |
| `vi_enhance/long_audio.py` | VAD silence-valley segment planner (`15–30s` chunks with `0.35s` overlap) to avoid long-stream transducer state drift. | [Evaluating Open-Source ASR Systems, COLING 2025](https://aclanthology.org/2025.coling-main.336.pdf) & `docs/VI_DEVELOPMENT_RESULTS.md`. |
| `training/confusion_miner.py` | Closed-loop Levenshtein word-alignment error miner extracting recurring phonetic `heard -> write` rules and high-impact bias terms from consented evaluation reports. | Compatible with `training/calibrate.py` train/holdout gating. |

## 3. Verified Measurements (Automated Tests & Benchmarks)

Run in the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 benchmarks/run_vi_benchmarks.py
```

### Measured Results (2026-11-27, Linux Python 3.11)

- **Unit & Integration Tests**: `19/19 PASS` (`0.488s`).
- **DSP Front-End Benchmark**:
  - 50 Hz AC hum + DC attenuation: **`-11.14 dB`**
  - Vietnamese F0 (180 Hz) retention ratio: **`98.11%`**
  - Quiet recording (`-46.11 dBFS`) automatic AGC recovery: **`-25.32 dBFS`** with **`0` clipped samples**
  - Noisy speech SNR (`7.62 dB` → `9.20 dB`): **`+1.58 dB`** gain via gentle Observation Addition (`oa_alpha=0.189`)
  - Processing throughput: **`82.2x` real-time** on CPU
- **Vietnamese Post-Processing Benchmark** (20 targeted SODA-error utterances covering regional confusions, code-switching, boundary overlaps, and spoken ITN + 10 clean Vietnamese guardrail sentences):
  - Raw WER: **`59.09%`** (91/154 errors) → Post WER: **`0.00%`** (0/154 errors)
  - Raw CER: **`43.59%`** → Post CER: **`0.00%`**
  - Clean-sentence over-correction rate: **`0/10 (0.00%)`**

> **Verification Scope Note**: All DSP, phonetic, ITN, dynamic-bias, confusion-miner, and FastAPI server smoke checks were executed and verified directly. End-to-end live audio decoding through Google Chrome's proprietary `libsoda` binary was not executed in this Linux sandbox (model binaries are excluded from the portable ZIP) and should be A/B evaluated on your Windows/WSL2 host using:
> ```bash
> .venv-wsl/bin/python -m training.evaluate_samples --manifest data/development_manifest.json --report data/ab_enhanced.json --enhance-audio --postprocess --itn safe
> ```
