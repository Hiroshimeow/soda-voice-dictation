"""Supervised calibration of Vietnamese context profiles, not Google weights.

Persistent hash partitioning prevents old samples moving between train/holdout
when new corrections arrive. File inference is paced at 2x to avoid observed loss.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, time, unicodedata, wave
from datetime import datetime, timezone
from training.context_profile import DATA, PROFILE_PATH, ACTIVE_PATH,validate_profile,apply_to_soda,write_atomic,load_profile,profile_sha256
from training.storage import stable_split
from training.personal_vocabulary import load_vocabulary, build_effective_profile, vocabulary_fingerprint
DATASET=DATA/"eval_manifest.jsonl"
AUDIO_DIR=DATA/"audio"
REPORT=DATA/"calibration_report.json"
MIN_SAMPLES=30
BOOST_SCALES=(0.5,1.0,1.3)
EVALUATION_PACE=2.0

def normalized(text:str)->str:
    return " ".join(re.findall(r"[\w]+",unicodedata.normalize("NFC",text).casefold(),re.UNICODE))
def distance(a:list[str],b:list[str])->int:
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=cur
    return row[-1]
def rate(ref:str,hypothesis:str,unit:str="word")->tuple[int,int]:
    ref=normalized(ref);hypothesis=normalized(hypothesis)
    a=ref.split() if unit=="word" else list(ref.replace(" ",""))
    b=hypothesis.split() if unit=="word" else list(hypothesis.replace(" ",""))
    return distance(a,b),len(a)
def file_sha(path:pathlib.Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def read_dataset(path:pathlib.Path):
    if not path.is_file():raise FileNotFoundError("No corrected audio manifest: "+str(path))
    audio_dir=(path.parent/"audio").resolve()
    rows=[];seen=set()
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        item=json.loads(line)
        if not isinstance(item,dict) or item.get("language")!="vi-VN" or item.get("consent") is not True:
            raise ValueError(f"Line {n} requires explicit consent and vi-VN")
        if item.get("purpose","personalization")!="personalization":continue
        rel=item.get("audio");ref=item.get("reference")
        if not isinstance(rel,str) or not isinstance(ref,str) or not 2<=len(ref.strip())<=3000:
            raise ValueError(f"Invalid entry at line {n}")
        audio=(audio_dir/rel).resolve()
        if not audio.is_relative_to(audio_dir) or not audio.is_file() or audio.suffix.lower()!=".wav":
            raise ValueError(f"Invalid WAV path at line {n}")
        with wave.open(str(audio),"rb") as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):raise ValueError(f"16k mono s16 required at line {n}")
            seconds=w.getnframes()/16000;pcm=w.readframes(w.getnframes())
        if not 0.5<=seconds<=30:raise ValueError(f"Audio duration outside 0.5..30s at line {n}")
        identity=hashlib.sha256(pcm).hexdigest()
        if identity in seen:raise ValueError(f"Duplicate audio identity at line {n}")
        seen.add(identity)
        split=stable_split(identity)
        if item.get("pcm_sha256",identity)!=identity or item.get("split",split)!=split:
            raise ValueError(f"Sample identity or persisted split mismatch at line {n}")
        rows.append((identity,audio,ref.strip(),split))
    return sorted(rows,key=lambda row:row[0])
def dataset_identity(rows):
    records=[{"pcm_sha256":r[0],"reference_sha256":hashlib.sha256(r[2].encode()).hexdigest(),"split":r[3]} for r in rows]
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load_dataset(path:pathlib.Path):
    rows=read_dataset(path)
    if len(rows)<MIN_SAMPLES:raise ValueError(f"Need {MIN_SAMPLES} distinct corrected recordings; found {len(rows)}")
    train=[r[:3] for r in rows if r[3]=="train"];holdout=[r[:3] for r in rows if r[3]=="holdout"]
    if len(train)<15 or len(holdout)<5:raise ValueError(f"Need >=15 train and >=5 holdout; fixed partition has {len(train)}/{len(holdout)}. Collect more corrections; do not move samples.")
    return train,holdout
def decode_structured(recognizer,wav_path:pathlib.Path)->dict:
    with wave.open(str(wav_path),"rb") as w:pcm=w.readframes(w.getnframes())
    from training.bounded_decode import recognize_pcm
    return recognize_pcm(recognizer,pcm,pace=EVALUATION_PACE)


def decode(recognizer,wav_path:pathlib.Path)->str:
    """Preserve the historical public string contract; no-final is ``""``."""
    return decode_structured(recognizer, wav_path)["text"]
def scaled_profile(candidate,scale):
    return validate_profile({"schema_version":1,"language":"vi-VN","terms":[{"phrase":x["phrase"],"boost":min(3.5,round(x["boost"]*scale,2))} for x in candidate["terms"]]})
def make_recognizer(profile=None):
    from soda_stt import SodaRecognizer,_config
    _config.LANGUAGE_PACK_IDS["vi-VN"]="bpnckojighgibleldbofdjmfdipfeiak"
    rec=SodaRecognizer.for_locale("vi-VN",recognition_mode=_config.MODE_CAPTION,enable_formatting=True)
    apply_to_soda(rec,profile)
    return rec
def model_identity(rec):
    files=[pathlib.Path(rec.libsoda_path),pathlib.Path(rec._helper)]
    pack=pathlib.Path(rec.language_pack_directory)
    files.extend(sorted(p for p in pack.rglob("*") if p.is_file()))
    return [{"path":str(p.resolve()),"sha256":file_sha(p)} for p in files]
def score(samples,profile):
    rec=make_recognizer(profile);we=wn=ce=cn=0
    for _,audio,ref in samples:
        pred = decode(rec, audio)
        # No-final is a measured serving outcome, not a helper failure. Keep it
        # in the denominator as an empty prediction so calibration and serving
        # score the same final-segment contract.
        e,n=rate(ref,pred);we+=e;wn+=n
        e,n=rate(ref,pred,"char");ce+=e;cn+=n
    return {"wer":we/max(1,wn),"cer":ce/max(1,cn),"word_errors":we,"reference_words":wn,"char_errors":ce,"reference_chars":cn}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=pathlib.Path,default=DATASET)
    p.add_argument("--proposal",type=pathlib.Path,default=PROFILE_PATH)
    p.add_argument("--report",type=pathlib.Path,default=REPORT)
    a=p.parse_args()
    write_atomic(a.report,{"schema_version":2,"passed":False,"status":"running"})
    try:
        rows=read_dataset(a.dataset);train,holdout=load_dataset(a.dataset)
        candidate=validate_profile(json.loads(a.proposal.read_text(encoding="utf-8")))
        baseline=load_profile();baseline_hash=profile_sha256(baseline)
        vocabulary = load_vocabulary()
        personal_hash=vocabulary_fingerprint()
        effective_personal_hash=profile_sha256(build_effective_profile(vocabulary))
        manifest_hash=file_sha(a.dataset);data_hash=dataset_identity(rows)
        candidate_hash=file_sha(a.proposal);artifacts=model_identity(make_recognizer())
        print("TRAIN_SIZE",len(train),"HOLDOUT_SIZE",len(holdout),flush=True)
        train_scores={"active":score(train,build_effective_profile(vocabulary, curated_profile=baseline))}
        for scale in BOOST_SCALES:
            train_scores[str(scale)]=score(train,build_effective_profile(vocabulary, curated_profile=scaled_profile(candidate,scale)))
            print("TRAIN_SCORE",scale,train_scores[str(scale)]["wer"],flush=True)
        selected=min(train_scores,key=lambda k:(train_scores[k]["wer"],train_scores[k]["cer"],0 if k=="active" else float(k)))
        effective_before = build_effective_profile(vocabulary, curated_profile=baseline)
        before=score(holdout,effective_before)
        effective_after = (build_effective_profile(
            vocabulary, curated_profile=scaled_profile(candidate, float(selected)))
            if selected != "active" else effective_before)
        after=score(holdout,effective_after) if selected!="active" else before
        effective_profile_hash=profile_sha256(effective_after)
        passed=selected!="active" and after["wer"]<before["wer"] and after["cer"]<=before["cer"]
        if (file_sha(a.dataset)!=manifest_hash or file_sha(a.proposal)!=candidate_hash or
                profile_sha256(load_profile())!=baseline_hash or vocabulary_fingerprint()!=personal_hash):
            raise RuntimeError("Dataset, candidate or active profile changed during calibration; rerun")
        report={"schema_version":2,"status":"completed","language":"vi-VN","created_at":datetime.now(timezone.utc).isoformat(),"train_samples":len(train),"holdout_samples":len(holdout),"split_rule":"PCM SHA256 first8 hex modulo5; zero=holdout","selected_scale":float(selected) if selected!="active" else 0,"train_metrics":train_scores,"holdout_baseline":before,"holdout_candidate":after,"passed":passed,"method":"context_bias_calibration_not_Google_weight_training","candidate_sha256":candidate_hash,"active_profile_sha256":baseline_hash,"personal_vocabulary_sha256":personal_hash,"effective_personal_profile_sha256":effective_personal_hash,"effective_profile_sha256":effective_profile_hash,"dataset_path":str(a.dataset.resolve()),"dataset_sha256":data_hash,"manifest_sha256":manifest_hash,"engine_artifacts":artifacts,"evaluation_pace":EVALUATION_PACE}
        write_atomic(a.report,report)
        print("GATE","PASS" if passed else "FAIL","report",str(a.report),flush=True)
        if not passed:raise SystemExit(2)
    except Exception as exc:
        write_atomic(a.report,{"schema_version":2,"passed":False,"status":"failed","error_type":type(exc).__name__,"error":str(exc)[:500]})
        raise
if __name__=="__main__":main()
