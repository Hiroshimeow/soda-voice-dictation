# Vietnamese Google SODA: measured architecture and weight experiment

**Measured on Windows/WSL2 Ubuntu 22.04 on 2026-11-22** using Vietnamese Chrome SODA pack `vi-VN/1.3071.0`.

**A real gradient update to Google's copied output-bias parameters succeeded. Native SODA loaded that copied model, but recognition did not improve on the three development recordings.** This establishes a limited trainable component; it does not establish acoustic encoder fine-tuning, independent-set accuracy, or equivalence to the Android Live Transcribe/ML Kit weights. The installed production pack remains unchanged.

## Actual graph contracts

| Model in `endtoendmodel/` | Bytes / SHA-256 prefix | Input and output | Operators / quantized tensors |
| --- | ---: | --- | ---: |
| medium-encoder.tflite | 36,037,864 / `302a3eae9214` | 86 inputs and outputs; features `[1,1,482]` → `[1,1,512]`, plus 85 states | 1,762 / 134 |
| medium-decoder.tflite | 6,076,352 / `6706e2f1c829` | Current and previous token int32 → 640 features and next token state | 31 / 5 |
| medium-joint_posterior.tflite | 3,381,616 / `af015d2f7245` | 512 acoustic + 640 decoder features → 4,096 probabilities | 18 / 3 |
| medium-joint_prior.tflite | 3,052,000 / `39f0cc6175ae` | 640 decoder features → 4,095 nonblank probabilities | 5 / 2 |

The four core models have zero variable tensors, zero SignatureDefs and no bundled optimizer or gradient entry point. Their core operators are standard LiteRT operators. All seven installed TFLite models independently allocate under LiteRT 2.4.0. Text punctuation/capitalization graphs contain custom operators in additional subgraphs; allocation alone does not prove those custom operators can execute independently.

The native config explicitly sets `is_hat_model=true`: this is a hybrid autoregressive transducer with a separate blank gate. It declares BOS 1, EOS 2, decoder initial state 1 and `internal_lm_cost_scale=0.0`. A prior-only change would have zero scoring effect under this configuration. The experiment therefore changes the tied posterior and prior output bias while leaving the blank bias unchanged.

Evidence: `research/model_audit.py`, private `data/model_audit/graph_report.json`, and `research/frontend_audit/`.

## Frontend and tokenizer recovered

The tokenizer has 4,096 neural symbols: 4,026 portable wordpieces plus 70 special symbols. The MARISA inverse-permutation mapping matches every portable piece with zero mismatches. Greedy adjacent merges use the lowest resulting token ID, following the public Lingvo WordPiece algorithm. Both merge implementations agree on all 4,026 pieces; each piece reconstructs to its own ID. NFC, unknown-character behavior and public roundtrips are tested. Direct comparison with a callable native tokenizer remains unmeasured.

The config plus native protobuf descriptors establish this feature layout:

- 25 ms frames, 10 ms step, preemphasis 0.97, HANNING window, dither scale 8.
- 80 filterbank channels, 125–7,500 Hz; logarithmic output.
- Three-frame stack → 240 values, subsample 3, normalize with 240 means and 240 standard deviations.
- Append domain ID → 241, stack current plus one right frame → 482, subsample 2.
- The encoder splits the domain ID, one-hot encodes it into 16 values, joins the 240 acoustic values and projects 256 → 512.

The minimum FFT size is 512 and overdrive is true; public Lingvo would use 1,024 points. Native FFT scaling, dither sequence, window/padding numerics and initialization/feedback for all 85 encoder states have not yet been matched numerically. The complete evidence and exact remaining gaps are in [the frontend audit](../research/frontend_audit/README.md).

## Projection reconstruction

`research/joint_parity.py` reconstructs the two joint projections from actual FlatBuffer constants. It emulates the graphs' asymmetric dynamic activation quantization and fixed INT8 matrices.

On 32 seeded feature probes per head:

| Head | Top-1 agreement | Maximum absolute probability error | Mean L1 probability error |
| --- | ---: | ---: | ---: |
| Posterior | 32/32 | 6.0201e-6 | 2.1894e-6 |
| Prior | 32/32 | 6.0797e-6 | 2.8104e-6 |

These are projection-level numerical comparisons, not bit-exact waveform parity. No audio accuracy claim follows from random feature probes. Raw report: `data/joint_parity_asymmetric.json`.

## Supervised copied-weight update

`research/train_joint_bias.py` performs an actual cross-entropy gradient update, using the three user-provided corrected texts and frozen Google decoder features. It normalizes labels to NFC lowercase, removes sentence punctuation, tokenizes using the recovered mapping, appends EOS, and rejects unknown labels. The experiment trains a text-conditioned language-head objective, not an audio-alignment loss.

| Measurement | Result |
| --- | --- |
| Labels | 214 tokens from three development texts |
| Steps / learning rate / L2 | 100 / 1.0 / 0.01 |
| NLL before → after | 4.734789274 → 4.439559768 |
| Regularized objective before → after | 4.734789274 → 4.506710161 |
| Gradient finite-difference maximum error | 3.08e-11 |
| Changed FLOAT32 parameters | 3,745 in each tied bias buffer |
| Largest parameter change | 0.350000024 |
| Encoder, decoder and blank bias updated | No |
| Source pack unchanged | All 24 source-file hashes match |
| Copied posterior/prior allocation | Passed |

The copied posterior SHA-256 is `f44f808230008d9e3e2f37b6c6278c990dfdf91f432806384d5a64addfcf3be8`; the copied prior SHA-256 is `63c603eb31fb7ec049e4515a25823c08c1c7cd649a0f27bff4f78740d7f29da8`. Model copies and training history remain ignored under `data/weight_experiments/`.

Native A/B used an unchanged copied pack as control and the trained copied pack as candidate. Serialized recognizer config points to each intended directory; native-helper file-open traces corroborate that the modified posterior file was loaded by the helper, rather than merely hashed by Python.

Both complete three-recording runs scored **12/210 numeric-normalized syllable edits** (raw: 28/210). Therefore the candidate was **not promoted**. A later clean sample-3 run, with WSL kept alive and no strace, completed successfully in 8.466 seconds of decode time at 2x file feed, still with 6/72 edits. Those times are not live-microphone latency measurements.

Earlier traced attempts stalled on the third recording; one control and one candidate attempt were affected. The cause is unconfirmed because tracing, runner lifetime and WSL idle shutdown differed. Per-recording deadlines and owned-process-group cleanup now bound evaluation failures. A timeout is reported as failure, never scored as an empty successful transcript.

The earlier CDPA `research/weight_probe.py` is a separate synthetic one-bias sensitivity experiment. Its lower synthetic loss does not establish speech training. It is retained as an auditable small probe, distinct from the supervised text experiment above.

## What remains before acoustic fine-tuning

1. Measure the waveform frontend and streaming-state parity against native SODA.
2. Reconstruct the differentiable encoder/decoder forward pass with the observed quantization behavior.
3. Implement and gradient-check the HAT sequence objective and blank/alignment semantics.
4. Train on consented paired audio and corrected labels, then evaluate on stable untouched held-out recordings.
5. Require native-engine compatibility, quality improvement and acceptable resource use before activation.

The pack lacks a supported training API, but that is not proof that research reconstruction is impossible. The present result is a working, limited parameter-training path whose first candidate did not improve ASR.

## Reproduce without touching the active pack

Use an isolated WSL environment with `requirements-model-research.txt`; keep it separate from the server environment. From the repository:

```bash
PACK=$HOME/.cache/soda-stt/langpacks/vi-VN/1.3071.0/SODAModels
PY=$HOME/.cache/soda-model-research-venv/bin/python
"$PY" research/model_audit.py --pack "$PACK" --output data/model_audit/graph_report.json --check-core
"$PY" -m research.frontend_audit.audit --pack "$PACK" --output data/frontend_audit/verified_contract.json
"$PY" -m research.joint_parity --model-dir "$PACK/endtoendmodel" --probes 32 --report data/joint_parity_asymmetric.json
"$PY" -m research.train_joint_bias --model-dir "$PACK" --manifest data/development_manifest.json --output data/weight_experiments/text_head_new
```

The training command refuses an existing output directory or a directory overlapping the source pack. Evaluation uses the server's WSL Python:

```bash
.venv-wsl/bin/python -m training.evaluate_samples --manifest data/development_manifest.json --report data/weight_experiments/text_head_new/native_evaluation.json --model-dir data/weight_experiments/text_head_new/SODAModels --gain-db 22 --pace 2
```

The +22 dB setting is specific to these unusually quiet supplied recordings, whose RMS is near −49 dBFS. Do not apply it globally to an ordinary microphone.

## Primary references and scope

- [Google LiteRT on-device training](https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training) demonstrates explicit train/infer/save/restore functions in deliberately exported models.
- [Google LiteRT signatures](https://developers.google.com/edge/litert/conversion/tensorflow/signatures) documents named entry points.
- [TensorFlow Lingvo WordPiece](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/wpm_encoder.py) supports the merge-algorithm comparison.
- [TensorFlow Lingvo frontend](https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/frontend.py) supports comparison of named frontend settings, not full native parity.
- [Google ML Kit speech recognition](https://developers.google.com/ml-kit/genai/speech-recognition/android) describes an Android SDK; it does not establish that this Chrome pack is identical to the S20 Live Transcribe model.

No model binaries, private vocabulary or user transcripts are committed.
