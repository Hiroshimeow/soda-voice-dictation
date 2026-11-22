"""Research-only probe for SODA RecognitionContext Phrase boosts."""
from __future__ import annotations

import pathlib
import time
import wave
from soda_stt import SodaRecognizer, soda_api_pb2, _config

_config.LANGUAGE_PACK_IDS.update({
    "vi-VN": "bpnckojighgibleldbofdjmfdipfeiak",
    "ja-JP": "onhpjgkfgajmkkeniaoflicgokpaebfa",
})
wav_path = pathlib.Path("/mnt/e/git-project/live-transcribe-stt-lab/models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/test_wavs/0.wav")
with wave.open(str(wav_path), "rb") as wav:
    assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
    data = wav.readframes(wav.getnframes())

rec = SodaRecognizer.for_locale("vi-VN")
cfg = soda_api_pb2.ExtendedSodaConfigMsg.FromString(rec._config_bytes)
context = cfg.recognition_context.context.add()
context.name = "android-speech-api-generic-phrases"
for phrase in ("hỗ trợ", "lâu lâu", "cho gạo"):
    item = context.phrases.phrase.add()
    item.phrase = phrase
    item.boost = 4
rec._config_bytes = cfg.SerializeToString()
assert len(cfg.recognition_context.context) == 1
print("CONTEXT_PROTO_OK", [(p.phrase, p.boost) for p in context.phrases.phrase], flush=True)
start = time.monotonic()
try:
    result = rec.transcribe_pcm(data)
    print("WITH_HINTS", result, "elapsed", round(time.monotonic()-start, 2), flush=True)
except Exception as e:
    print("HINT_INFERENCE_ERROR", type(e).__name__, str(e)[:300], flush=True)
    raise
