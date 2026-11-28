"""Behavior tests for append-stable splits, safe feed and tamper-safe release."""
import hashlib,json,pathlib,tempfile,unittest,wave
from unittest.mock import patch
from training.dataset import save_sample
from training.storage import stable_split
from training.calibrate import read_dataset,dataset_identity,decode,file_sha
from training.context_profile import write_atomic, profile_sha256, legacy_profile_sha256, validate_profile
from training.personal_vocabulary import build_effective_profile, load_vocabulary
from training.release import promote,rollback,stored_profile_sha256,_artifact_sha256
from training.personal_vocabulary import vocabulary_fingerprint
class WorkflowTests(unittest.TestCase):
    def make_sample(self,root,index):
        pcm=index.to_bytes(2,"little")*8000
        save_sample({"reference":"xin chào Việt Nam","language":"vi-VN","consent":True},pcm,data_dir=root)
    def test_split_survives_append_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            for i in range(8):self.make_sample(root,i)
            manifest=root/"eval_manifest.jsonl";before=read_dataset(manifest)
            for i in range(8,12):self.make_sample(root,i)
            after={r[0]:r[3] for r in read_dataset(manifest)}
            self.assertTrue(all(after[r[0]]==r[3] for r in before))
            rows=[json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
            rows[0]["split"]="holdout" if rows[0]["split"]=="train" else "train"
            manifest.write_text("\n".join(json.dumps(x) for x in rows),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"split mismatch"):read_dataset(manifest)
    def test_recovers_wav_written_before_manifest_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);self.make_sample(root,42)
            manifest=root/"eval_manifest.jsonl"
            manifest.unlink()
            self.make_sample(root,42)
            self.assertEqual(len(read_dataset(manifest)),1)
            with self.assertRaisesRegex(ValueError,"Duplicate"):self.make_sample(root,42)
    def test_decode_sets_safe_pace(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);self.make_sample(root,1)
            audio=next((root/"audio").glob("*.wav"));rec=object()
            with patch("training.bounded_decode.recognize_pcm",return_value={"text":"xin chào"}) as bounded:
                self.assertEqual(decode(rec,audio),"xin chào")
                bounded.return_value = {"text": ""}
                self.assertEqual(decode(rec,audio), "")  # valid no-final, not helper failure
                self.assertEqual(bounded.call_args.kwargs["pace"],2.0)
    def test_promote_then_rollback_and_stale_dataset_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            for i in range(80):self.make_sample(root,i)
            manifest=root/"eval_manifest.jsonl";rows=read_dataset(manifest)
            proposal=root/"candidate.json"
            write_atomic(proposal,{"schema_version":1,"language":"vi-VN","terms":[{"phrase":"trí tuệ nhân tạo","boost":1.2}]})
            artifacts=[]
            for i in range(3):
                artifact=root/("engine"+str(i));artifact.write_bytes(b"fixture artifact "+bytes([i]))
                artifacts.append({"path":str(artifact),"sha256":file_sha(artifact)})
            candidate = json.loads(proposal.read_text(encoding="utf-8"))
            report={"schema_version":2,"status":"completed","passed":True,"candidate_sha256":file_sha(proposal),"train_samples":sum(r[3]=="train" for r in rows),"holdout_samples":sum(r[3]=="holdout" for r in rows),"evaluation_pace":2.0,"active_profile_sha256":"baseline","personal_vocabulary_sha256":vocabulary_fingerprint(),"dataset_path":str(manifest),"dataset_sha256":dataset_identity(rows),"manifest_sha256":file_sha(manifest),"engine_artifacts":artifacts,"selected_scale":1.0,"holdout_baseline":{"wer":0.2,"cer":0.1},"holdout_candidate":{"wer":0.1,"cer":0.09},"effective_profile_sha256":profile_sha256(build_effective_profile(load_vocabulary(), curated_profile=candidate))}
            path=root/"report.json";write_atomic(path,report)
            with patch("training.release.vocabulary_fingerprint", return_value="changed"):
                with self.assertRaisesRegex(RuntimeError,"Personal vocabulary"):
                    promote(path,proposal,data_dir=root)
            receipt=promote(path,proposal,data_dir=root)
            self.assertTrue((root/"profile_versions"/(receipt["version"]+".json")).is_file())
            self.assertTrue((root/"active_bias.json").is_file())
            rollback(data_dir=root);self.assertFalse((root/"active_bias.json").exists())
            self.make_sample(root,81)
            with self.assertRaisesRegex(RuntimeError,"Dataset"):promote(path,proposal,data_dir=root)
            self.assertFalse((root/"active_bias.json").exists())

    def test_v3_rollback_accepts_authenticated_native_unsafe_baseline_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            profile = validate_profile({"schema_version": 1, "language": "vi-VN",
                       "terms": [{"phrase": "WSL2", "boost": 2.0}]})
            artifact = {"schema_version": 3, "profile": profile,
                        "profile_sha256": "baseline",
                        "stored_profile_sha256": stored_profile_sha256(profile)}
            artifact["artifact_sha256"] = _artifact_sha256(artifact)
            write_atomic(root / "active_bias.previous.json", artifact)
            rollback(data_dir=root)
            self.assertEqual(json.loads((root / "active_bias.json").read_text())["terms"][0]["phrase"], "WSL2")
            tampered = dict(artifact)
            tampered["profile"] = {**profile, "terms": [{"phrase": "WSL2", "boost": 2.1}]}
            write_atomic(root / "active_bias.previous.json", tampered)
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                rollback(data_dir=root)

    def test_legacy_schema_v2_rollback_digest_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            profile = {"schema_version": 1, "language": "vi-VN",
                       "terms": [{"phrase": "FastAPI", "boost": 1.5}]}
            write_atomic(root / "active_bias.previous.json", {
                "schema_version": 2, "profile": profile,
                "profile_sha256": legacy_profile_sha256(profile),
            })
            rollback(data_dir=root)
            self.assertEqual(json.loads((root / "active_bias.json").read_text())["terms"][0]["boost"], 1.5)
            write_atomic(root / "active_bias.previous.json", {
                "schema_version": 2, "profile": profile,
                "profile_sha256": legacy_profile_sha256(profile)[:-1] + "0",
            })
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                rollback(data_dir=root)
if __name__=="__main__":unittest.main()
