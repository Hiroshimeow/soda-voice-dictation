# Vietnamese SODA frontend and tokenizer contract

## Scope and evidence

Read-only audit of the owned `vi-VN/1.3071.0/SODAModels` pack and the installed
Google `libsoda.so`. No model, runtime, authorization logic or production setting
was changed. The reusable scripts read the local pack; they do not bundle model
weights, vocabulary, recordings or private text.

The structural contract below is supported by the actual config protobufs,
TFLite tensor/operator metadata, and ten protobuf descriptors embedded in the
native library. Exact numeric equivalence of a replacement audio frontend
against native SODA is **NOT_MEASURED**.

## Verified neural symbol mapping

`medium.syms.compact` contains a MARISA trie followed by a 32-byte header and
4,096 little-endian uint16 values. The latter are an inverse permutation:

```python
vocabulary[neural_id] = trie.restore_key(permutation[neural_id] - 1)
```

All 4,026 portable wordpieces independently satisfy
`neural_id = portable_wordpiece_id + 69`; there are **zero mismatches**.
The 70 portable exception symbols equal the remaining 70 neural symbols.

| Symbol | Neural ID |
| --- | ---: |
| `<epsilon>` | 0 |
| `<S>` | 1 |
| `</S>` | 2 |
| `<UNK>` | 35 |
| `<noise>` | 54 |
| `<sil>` | 64 |
| Word boundary `▁` | 70 |

The portable model stores 3,605 merge rules. Each target is the concatenation
of its two source pieces; targets cover portable IDs 422 through 4,026.
The reconstructed encoder repeatedly merges the adjacent pair with the
**lowest resulting token ID**, following the published TensorFlow Lingvo
WordPiece algorithm. It does not use a longest-match substitution.

Verification on the actual pack:

- All 4,026 pieces reconstruct to their exact own token ID.
- Portable merge-table and vocabulary-concatenation merge algorithms agree
  on all 4,026 vocabulary pieces.
- Four public text roundtrips pass, including Vietnamese accents, numbers,
  punctuation and mixed English terms.
- Unknown Unicode characters map to `<UNK>`; decomposed and NFC Vietnamese
  input produce identical IDs.

This establishes mapping and internal segmentation consistency. Direct parity
with a callable native SODA tokenizer is **NOT_MEASURED**.

### API

```python
from research.frontend_audit.tokenizer import SodaTokenizer

tokenizer = SodaTokenizer(pack / "endtoendmodel")
ids = tokenizer.encode(text, lowercase=False, add_bos=False, add_eos=False)
restored = tokenizer.decode(ids)
```

NFC and whitespace normalization are explicit implementation behavior.
Lowercasing, BOS/EOS insertion and removal of punctuation are the caller's
training-label decisions. Digits and punctuation are preserved by default.
A raw acoustic transcript and a formatted display transcript are not
automatically interchangeable training labels.

## Verified audio pipeline structure

The config's directed edges give this order:

1. Decode PCM and frame at 25 ms with a 10 ms step.
2. Preemphasis/window, FFT, 80-channel filterbank.
3. Stack two left frames plus the current frame: 240 values.
4. Subsample every third frame.
5. Endpoint segmentation, then normalize with 240 means and 240 standard deviations.
6. Append one numeric domain ID: 241 values.
7. Stack the current and one right frame with zero-padding enabled: 482 values.
8. Subsample every second stacked item and feed the causal Conformer encoder.

The first encoder operators independently corroborate the layout: reshape into
241-feature frames, separate element 240 as the domain ID, convert that ID to
a 16-element one-hot vector, concatenate it with the first 240 values and project
the resulting 256 features to 512 dimensions.

| Parameter | Evidence from config plus native descriptor defaults |
| --- | --- |
| Frame size / step | 25 ms / 10 ms |
| Preemphasis padding | Enabled, one extra input sample |
| Preemphasis coefficient | 0.97 |
| HTK preemphasis compatibility | True |
| Window | HANNING, enum 1 |
| Dither noise scale | 8.0 |
| Minimum FFT size | 512 |
| FFT overdrive | True |
| FFT output | Magnitude squared |
| Filterbank summation | Magnitude |
| Filterbank output | Natural-log transform, floor 1 |
| Filterbank channels / band | 80 / 125–7,500 Hz |
| First stack | Left 2, right 0 |
| First subsample | 3 |
| Second stack | Input 241, left 0, right 1, zero-padding true |
| Second subsample | 2 |

The public Lingvo frontend doubles its minimum FFT resolution when
`fft_overdrive=True`, giving a **1,024-point FFT in that implementation**. The exact
native SODA FFT size/scaling must still be checked numerically; reporting
“512-point FFT” from the minimum-size field alone would be premature.

Both normalization files have **240 means plus 240 standard deviations**,
not 480 of each. Their means range from 7.995007 to 9.538423 and standard
deviations from 1.548844 to 1.822889.

The continuous ASR stream has forced domain ID 2, with FARFIELD mapped to 7
and TELEPHONY to 2; the short ASR stream has forced ID 7. The VAD stream has
a different domain mapping. Do not substitute VAD's CAPTION ID 4 into the
ASR encoder merely because the recognition mode is CAPTION.

## Decoder and joint semantics

Native descriptor names remove ambiguity from the config field numbers:

- `PieRnnFstDecoderParams.sos_id=1`, `eos_id=2`.
- `TfliteTransducerModelParams.state_as_input=true`.
- `decoder_initial_value=[1]`.
- `is_hat_model=true`: this is explicitly a hybrid autoregressive
  transducer (HAT), with a separate blank gate.
- `internal_lm_cost_scale=0.0`, `subtract_internal_lm=true`.

The decoder graph has current-token and previous-token integer inputs; its
next-state output copies the current token. Initial BOS 1 and previous state 1
are therefore grounded in config rather than guessed from a generic RNN-T.

A change confined to the prior graph would have no score effect with the
current zero internal-LM scale. A shared posterior/prior projection update
must preserve the separate blank gate and be evaluated through native SODA
before any release.

## Remaining numerical work

A fully equivalent replacement audio frontend still requires:

- Native comparison of frame boundary padding and the exact window formula.
- Verification of dither distribution, seed/state and ordering relative to
  preemphasis/windowing.
- Exact FFT normalization and mel-filter coefficients.
- Native comparison of normalization and left/right padding behavior.
- Verified initialization and ordered feedback for all 85 encoder states.
- Streaming/segment-reset equivalence, not merely allocation of the TFLite graph.

The graph exposes a position-counter state and streaming caches; this audit
does not claim that setting every encoder state to zero reproduces native
initialization. These unresolved details block a claim of verified acoustic
encoder fine-tuning, but do not make reconstruction impossible.

## Reproduce

Research environment used:
`$HOME/.cache/soda-model-research-venv/bin/python`;
`marisa-trie 1.4.1`, `tflite 2.18.0`, `ai-edge-litert 2.4.0`.

From the model-lab worktree in WSL:

```sh
$HOME/.cache/soda-model-research-venv/bin/python \
  -m research.frontend_audit.audit \
  --pack $HOME/.cache/soda-stt/langpacks/vi-VN/1.3071.0/SODAModels \
  --output data/frontend_audit/verified_contract.json

$HOME/.cache/soda-model-research-venv/bin/python \
  -m research.frontend_audit.descriptors \
  --library <repo>/.venv-wsl/lib/python3.12/site-packages/soda_stt/_bundle/SODAFiles/libsoda.so \
  --output data/frontend_audit/verified_descriptors.json
```

Both commands completed with exit code 0 on the development host. Reports contain SHA-256
digests for the exact source artifacts. Raw descriptors and exploratory logs
remain under ignored `data/frontend_audit/`.

## Primary implementation references

- TensorFlow Lingvo WordPiece:
  https://github.com/tensorflow/lingvo/blob/master/lingvo/core/wpm_encoder.py
- TensorFlow Lingvo audio frontend:
  https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/frontend.py

These public source files were retrieved read-only from GitHub and retained
under ignored research data for inspection. They support the algorithmic
comparison; they are not proof that every numerical detail matches SODA.
