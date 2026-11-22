"""Non-production smoke check of a Palace-trained contextual candidate on real SODA."""
from pathlib import Path
import time
import wave
from soda_stt import SodaRecognizer, _config
from training.context_profile import PROFILE_PATH, load_profile, apply_to_soda

_config.LANGUAGE_PACK_IDS["vi-VN"]="bpnckojighgibleldbofdjmfdipfeiak"
audio=Path("/mnt/e/git-project/live-transcribe-stt-lab/models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/test_wavs/0.wav")
profile=load_profile(PROFILE_PATH)
assert profile and len(profile["terms"])>=20
with wave.open(str(audio),"rb") as f:
    assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,1,2)
    pcm=f.readframes(f.getnframes())
for mode in ("baseline","candidate"):
    r=SodaRecognizer.for_locale("vi-VN")
    if mode=="candidate":apply_to_soda(r,profile)
    start=time.perf_counter()
    result=r.transcribe_pcm(pcm)
    print(mode,"seconds",round(time.perf_counter()-start,3),"text",result,flush=True)
print("PROFILE_SMOKE_OK")
