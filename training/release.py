"""Promote only a current passing context profile; preserve version + rollback.

No model weights are modified by this command. Run in the same WSL environment
used for calibration so recorded artifact paths can be reverified.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,uuid
from datetime import datetime,timezone
from training.context_profile import (DATA, PROFILE_PATH, ACTIVE_PATH, validate_profile,
    write_atomic, load_profile, profile_sha256, profile_digest_matches)
from training.calibrate import read_dataset,dataset_identity,file_sha,EVALUATION_PACE,BOOST_SCALES,MIN_SAMPLES
from training.storage import exclusive_lock
from training.personal_vocabulary import vocabulary_fingerprint, load_vocabulary, build_effective_profile


def stored_profile_sha256(profile: dict | None) -> str:
    """Hash exactly what is stored, including terms filtered from serving."""
    if profile is None:
        return "baseline"
    return hashlib.sha256(json.dumps(
        profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _artifact_sha256(artifact: dict) -> str:
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _checked_rollback_artifact(raw: dict) -> dict | None:
    if raw.get("schema_version") == 3 and "artifact_sha256" in raw:
        if raw.get("artifact_sha256") != _artifact_sha256(raw):
            raise RuntimeError("Rollback artifact digest mismatch")
        profile = validate_profile(raw["profile"]) if raw.get("profile") is not None else None
        effective = profile_sha256(profile)
        stored = stored_profile_sha256(profile)
        if raw.get("profile_sha256") != effective or raw.get("stored_profile_sha256") != stored:
            raise RuntimeError("Rollback artifact digest mismatch")
        if profile is None:
            if effective != "baseline" or stored != "baseline":
                raise RuntimeError("Rollback baseline mismatch")
            return None
        # A storage-valid profile may contain only terms excluded at the native
        # boundary (for example WSL2). Its effective profile is then the
        # authenticated baseline even though the stored payload is non-empty.
        # The artifact and both payload digests above authenticate the complete
        # stored profile before accepting this rollback; do not reject it merely
        # because serving currently has no native-safe terms.
        return profile
    if raw.get("schema_version") == 2 and "profile" in raw:
        profile = validate_profile(raw["profile"]) if raw["profile"] is not None else None
        expected = raw.get("profile_sha256")
        if profile is None:
            if expected != "baseline":
                raise RuntimeError("Rollback baseline mismatch")
        else:
            legacy_match = expected == hashlib.sha256(json.dumps(
                raw["profile"], sort_keys=True, ensure_ascii=False
            ).encode("utf-8")).hexdigest()
            if not profile_digest_matches(profile, expected) and not legacy_match:
                raise RuntimeError("Rollback artifact digest mismatch")
            if expected == "baseline":
                raise RuntimeError("Rollback baseline mismatch")
        return profile
    return validate_profile(raw)


def verify_report(report,proposal_path,active_path):
    raw=proposal_path.read_bytes();profile=validate_profile(json.loads(raw))
    if not isinstance(report,dict) or report.get("schema_version")!=2 or report.get("status")!="completed" or report.get("passed") is not True:
        raise RuntimeError("A passing v2 evaluation report is required")
    if report.get("candidate_sha256")!=hashlib.sha256(raw).hexdigest():raise RuntimeError("Candidate changed after evaluation")
    if report.get("train_samples",0)<15 or report.get("holdout_samples",0)<5 or report.get("train_samples",0)+report.get("holdout_samples",0)<MIN_SAMPLES:raise RuntimeError("Insufficient supervised evaluation")
    if report.get("evaluation_pace")!=EVALUATION_PACE:raise RuntimeError("Unsafe or unknown evaluation pacing")
    if not profile_digest_matches(load_profile(active_path), report.get("active_profile_sha256")):
        raise RuntimeError("Active profile changed after evaluation")
    if report.get("personal_vocabulary_sha256")!=vocabulary_fingerprint():raise RuntimeError("Personal vocabulary changed or was not evaluated; rerun calibration")
    manifest=pathlib.Path(report["dataset_path"])
    rows=read_dataset(manifest)
    if file_sha(manifest)!=report.get("manifest_sha256") or dataset_identity(rows)!=report.get("dataset_sha256"):
        raise RuntimeError("Dataset or reference labels changed after evaluation")
    if sum(r[3]=="train" for r in rows)!=report["train_samples"] or sum(r[3]=="holdout" for r in rows)!=report["holdout_samples"]:
        raise RuntimeError("Evaluation sample counts do not match dataset")
    artifacts=report.get("engine_artifacts")
    if not isinstance(artifacts,list) or len(artifacts)<3:raise RuntimeError("Engine/model identity missing")
    for artifact in artifacts:
        if file_sha(pathlib.Path(artifact["path"]))!=artifact["sha256"]:raise RuntimeError("Engine/model changed after evaluation")
    scale=report.get("selected_scale")
    if isinstance(scale,bool) or scale not in BOOST_SCALES:raise RuntimeError("Invalid calibration scale")
    before=report["holdout_baseline"];after=report["holdout_candidate"]
    for metrics in (before,after):
        for key in ("wer","cer"):
            value=metrics.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise RuntimeError("Invalid holdout metric")
    if not(after["wer"]<before["wer"] and after["cer"]<=before["cer"]):raise RuntimeError("Holdout regression gate not passed")
    updated = validate_profile({"schema_version":1,"language":"vi-VN","terms":[{"phrase":x["phrase"],"boost":min(3.5,round(x["boost"]*scale,2))} for x in profile["terms"]]})
    effective = build_effective_profile(load_vocabulary(), curated_profile=updated)
    expected_effective = report.get("effective_profile_sha256")
    if not isinstance(expected_effective, str) or profile_sha256(effective) != expected_effective:
        raise RuntimeError("Scored effective profile does not match serving profile")
    return updated

def promote(report_path:pathlib.Path,proposal_path:pathlib.Path,*,data_dir:pathlib.Path=DATA):
    active=data_dir/"active_bias.json"
    with exclusive_lock(data_dir):
        report=json.loads(report_path.read_text(encoding="utf-8"))
        updated=verify_report(report,proposal_path,active)
        version=stored_profile_sha256(updated);previous=load_profile(active)
        versions=data_dir/"profile_versions"
        write_atomic(versions/(version+".json"),updated)
        previous_artifact={"schema_version":3,"profile":previous,
                           "profile_sha256":profile_sha256(previous),
                           "stored_profile_sha256":stored_profile_sha256(previous)}
        previous_artifact["artifact_sha256"]=_artifact_sha256(previous_artifact)
        write_atomic(data_dir/"active_bias.previous.json",previous_artifact)
        write_atomic(active,updated)
        receipt={"promoted":True,"version":version,"previous_version":profile_sha256(previous),"report_sha256":file_sha(report_path),"dataset_sha256":report["dataset_sha256"],"created_at":datetime.now(timezone.utc).isoformat(),"restart_required":False,"applies_from":"next_recognition_session","weights_updated":False}
        write_atomic(versions/(version+".receipt."+uuid.uuid4().hex[:8]+".json"),receipt)
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt
def rollback(*,data_dir:pathlib.Path=DATA):
    with exclusive_lock(data_dir):
        previous_path=data_dir/"active_bias.previous.json"
        marker=data_dir/"active_bias.previous.baseline"
        if previous_path.exists():
            raw=json.loads(previous_path.read_text(encoding="utf-8"))
            previous=_checked_rollback_artifact(raw)
            if previous is None:(data_dir/"active_bias.json").unlink(missing_ok=True)
            else:write_atomic(data_dir/"active_bias.json",previous)
        elif marker.exists():(data_dir/"active_bias.json").unlink(missing_ok=True)
        else:raise RuntimeError("No verified rollback artifact")
    print("Rollback complete; effective from the next recognition session")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("promote","rollback"))
    p.add_argument("--approve",action="store_true")
    p.add_argument("--report",type=pathlib.Path,default=DATA/"calibration_report.json")
    p.add_argument("--proposal",type=pathlib.Path,default=PROFILE_PATH)
    a=p.parse_args()
    if not a.approve:raise SystemExit("Use --approve to apply this local profile change")
    if a.action=="promote":promote(a.report,a.proposal)
    else:rollback()
if __name__=="__main__":main()
