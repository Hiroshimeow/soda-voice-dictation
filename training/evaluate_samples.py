"""Evaluate local development audio; references never enter recognition.

Input manifest JSON rows: id, audio (absolute PCM WAV path), reference.
The named numeric-equivalence score expands digits/percent to Vietnamese
spoken forms; raw syllable error and character error remain in the report.
"""
from __future__ import annotations
import argparse, array, hashlib, json, math, os, pathlib, re, time, unicodedata, wave
from datetime import datetime, timezone
from training.calibrate import rate, normalized
from training.context_profile import apply_to_soda, validate_profile

DIGITS = ("không","một","hai","ba","bốn","năm","sáu","bảy","tám","chín")
def speak_integer(value: str) -> str:
    if len(value)>3 or (len(value)>1 and value.startswith("0")):
        return " ".join(DIGITS[int(c)] for c in value)
    n=int(value)
    if n<10:return DIGITS[n]
    if n<20:return "mười"+(" "+("lăm" if n%10==5 else DIGITS[n%10]) if n%10 else "")
    if n<100:return DIGITS[n//10]+" mươi"+(" "+("mốt" if n%10==1 else "lăm" if n%10==5 else DIGITS[n%10]) if n%10 else "")
    return DIGITS[n//100]+" trăm"+(" "+("linh " if n%100<10 else "")+speak_integer(str(n%100)) if n%100 else "")
def numeric_equivalence(text: str) -> str:
    text=unicodedata.normalize("NFC",text).casefold()
    text=re.sub(r"(\d)\s*%",r"\1 phần trăm",text)
    # Digit groups adjacent in a phone number are each read digit-by-digit.
    text=re.sub(r"(?<!\d)0\d{2,}(?:[ \-]\d{2,})*",lambda m:" ".join(DIGITS[int(c)] for c in m[0] if c.isdigit()),text)
    text=re.sub(r"\d+",lambda m:speak_integer(m[0]),text)
    return normalized(text)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_pcm(path,gain_db):
    with wave.open(str(path),"rb") as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):raise ValueError("16k mono s16 WAV required")
        pcm=w.readframes(w.getnframes())
    samples=array.array("h");samples.frombytes(pcm)
    if os.sys.byteorder!="little":samples.byteswap()
    peak=max((abs(v) for v in samples),default=0)
    rms=math.sqrt(sum(v*v for v in samples)/max(1,len(samples)))
    gain=10**(gain_db/20)
    clipped=sum(abs(v*gain)>32767 for v in samples)
    if gain_db:
        samples=array.array("h",(max(-32768,min(32767,round(v*gain))) for v in samples))
        if os.sys.byteorder!="little":samples.byteswap()
        pcm=samples.tobytes()
    return pcm,{"seconds":len(pcm)/32000,"peak_dbfs":20*math.log10(max(peak,1)/32768),"rms_dbfs":20*math.log10(max(rms,1e-9)/32768),"clipped_samples":clipped}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=pathlib.Path,required=True)
    ap.add_argument("--report",type=pathlib.Path,required=True)
    ap.add_argument("--mode",choices=("caption","ime"),default="caption")
    ap.add_argument("--no-formatting",action="store_true")
    ap.add_argument("--gain-db",type=float,default=0)
    ap.add_argument("--pace",type=float,default=2)
    ap.add_argument("--profile",type=pathlib.Path)
    ap.add_argument("--context-name",choices=("palace-vi-safe-context-v1","android-speech-api-generic-phrases"),default="palace-vi-safe-context-v1")
    ap.add_argument("--model-dir",type=pathlib.Path)
    a=ap.parse_args()
    if not 0.5<=a.pace<=4:ap.error("Evaluation pace must be 0.5..4; 8x lost audio in the measured development setup.")
    if not -12<=a.gain_db<=30:ap.error("Gain must be -12..30dB")
    from soda_stt import SodaRecognizer,_config
    _config.LANGUAGE_PACK_IDS["vi-VN"]="bpnckojighgibleldbofdjmfdipfeiak"
    options={"recognition_mode":_config.MODE_IME if a.mode=="ime" else _config.MODE_CAPTION,"enable_formatting":not a.no_formatting}
    rec=SodaRecognizer(str(a.model_dir.resolve()),**options) if a.model_dir else SodaRecognizer.for_locale("vi-VN",**options)
    from soda_stt import soda_api_pb2
    serialized_dir=soda_api_pb2.ExtendedSodaConfigMsg.FromString(rec._config_bytes).language_pack_directory
    if pathlib.Path(serialized_dir).resolve()!=pathlib.Path(rec.language_pack_directory).resolve():raise RuntimeError("Serialized model path mismatch")
    model_hashes={str(p.relative_to(pathlib.Path(rec.language_pack_directory))):sha(p) for p in pathlib.Path(rec.language_pack_directory).rglob("*.tflite")}
    if a.profile:apply_to_soda(rec,validate_profile(json.loads(a.profile.read_text(encoding="utf-8"))),context_name=a.context_name)
    rows=json.loads(a.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not rows:raise ValueError("nonempty list required")
    results=[]
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({"status":"running","expected_samples":len(rows),"results":[],"manifest_sha256":sha(a.manifest)}),encoding="utf-8")
    for row in rows:
        path=pathlib.Path(row["audio"])
        pcm,levels=read_pcm(path,a.gain_db)
        from training.bounded_decode import recognize_pcm
        try:
            result_data=recognize_pcm(rec,pcm,pace=a.pace)
        except Exception as exc:
            failed=json.loads(a.report.read_text(encoding="utf-8"))
            failed.update({"status":"failed","failed_sample":row["id"],"error_type":type(exc).__name__,"error":str(exc)[:500],"expected_samples":len(rows)})
            a.report.write_text(json.dumps(failed,ensure_ascii=False,indent=2),encoding="utf-8")
            raise
        elapsed=result_data["elapsed_seconds"];first=result_data["first_result_seconds"]
        prediction=result_data["text"]
        partials=result_data["partial_events"]
        final_count=result_data["final_segments"]
        we,wn=rate(row["reference"],prediction);ce,cn=rate(row["reference"],prediction,"char")
        ne,nn=rate(numeric_equivalence(row["reference"]),numeric_equivalence(prediction))
        result={"id":row["id"],"audio_sha256":sha(path),"reference_sha256":hashlib.sha256(row["reference"].encode()).hexdigest(),"levels":levels,"recognition":prediction,"raw_syllable_errors":we,"raw_reference_syllables":wn,"raw_ser":we/max(1,wn),"char_errors":ce,"reference_chars":cn,"cer":ce/max(1,cn),"numeric_normalized_errors":ne,"numeric_normalized_reference_syllables":nn,"numeric_normalized_ser":ne/max(1,nn),"elapsed_seconds":elapsed,"first_result_seconds":first,"final_segments":final_count,"partial_events":partials}
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        report={"schema_version":1,"status":"completed" if len(results)==len(rows) else "running","expected_samples":len(rows),"created_at":datetime.now(timezone.utc).isoformat(),"evaluation_type":"development_acceptance_not_independent_holdout","evaluation_changes_weights":False,"model_sha256s":model_hashes,"serialized_language_pack_directory":serialized_dir,"engine":"Google Chrome SODA","engine_sha256":sha(pathlib.Path(rec.libsoda_path)),"model_path":rec.language_pack_directory,"manifest_sha256":sha(a.manifest),"configuration":{"mode":a.mode,"formatting":not a.no_formatting,"gain_db":a.gain_db,"pace":a.pace,"context_name":a.context_name,"profile_sha256":sha(a.profile) if a.profile else None},"metric_definition":{"raw_ser":"NFC lowercase punctuation-stripped whitespace-delimited Vietnamese syllables","numeric_normalized_ser":"same after reference-independent integer/percent/phone-digit expansion; not a semantic accuracy score"},"results":results,"aggregate_raw_ser":sum(x["raw_syllable_errors"] for x in results)/sum(x["raw_reference_syllables"] for x in results),"aggregate_numeric_normalized_ser":sum(x["numeric_normalized_errors"] for x in results)/sum(x["numeric_normalized_reference_syllables"] for x in results),"aggregate_cer":sum(x["char_errors"] for x in results)/sum(x["reference_chars"] for x in results)}
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("REPORT",str(a.report),flush=True)
if __name__=="__main__":main()
