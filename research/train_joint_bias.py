"""Supervised text-conditioned update of Google's tied HAT output bias.

This updates actual FLOAT32 parameters in copied TFLite artifacts using gradients.
It is not acoustic encoder fine-tuning. No live pack, service or profile changes.
A development-set run must never be described as held-out ASR improvement.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re,shutil,time,unicodedata
import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from research.joint_parity import constants,hybrid_linear,softmax
from research.frontend_audit.tokenizer import SodaTokenizer

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def speech_text(text):
    return " ".join(re.findall(r"[^\W_]+",unicodedata.normalize("NFC",text).casefold()))
def decoder_features(model_path,sequences):
    interp=Interpreter(model_path=str(model_path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    ins={x["name"]:x["index"] for x in interp.get_input_details()}
    outs={x["name"]:x["index"] for x in interp.get_output_details()}
    features=[];targets=[]
    for sequence in sequences:
        previous=1;state=np.array([1],np.int32)
        for target in sequence+[2]:
            interp.set_tensor(ins["inputs"],np.array([previous],np.int32))
            interp.set_tensor(ins["previous_decoder_state_0"],state)
            interp.invoke()
            features.append(interp.get_tensor(outs["outputs"])[0].copy())
            state=interp.get_tensor(outs["next_decoder_state_0"]).copy()
            targets.append(target);previous=target
    targets=np.asarray(targets,dtype=np.int64)
    if np.any(targets<=0) or np.any(targets>=4096):raise ValueError("Blank or out-of-range text label")
    return np.asarray(features,np.float32),targets-1
def objective(logits,labels,delta,regularization):
    z=logits.astype(np.float64)+delta
    logsum=z.max(1)+np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))
    nll=float(np.mean(logsum-z[np.arange(len(labels)),labels]))
    return nll+regularization*float(np.dot(delta,delta))/2,nll
def gradient(logits,labels,delta,regularization):
    p=softmax(logits.astype(np.float64)+delta)
    p[np.arange(len(labels)),labels]-=1
    return p.mean(0)+regularization*delta
def replace_float_buffer(path,index,values):
    blob=bytearray(path.read_bytes());m=tflite.Model.GetRootAsModel(blob,0)
    t=m.Subgraphs(0).Tensors(index)
    if t.Type()!=tflite.TensorType.FLOAT32:raise ValueError("Only verified FLOAT32 bias buffer can be updated")
    buffer=m.Buffers(t.Buffer());field=buffer._tab.Offset(4)
    if not field:raise ValueError("Missing constant buffer")
    start=buffer._tab.Vector(field);payload=np.asarray(values,dtype="<f4").tobytes()
    if len(payload)!=buffer.DataLength():raise ValueError("Parameter shape mismatch")
    blob[start:start+len(payload)]=payload
    path.write_bytes(blob)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",type=pathlib.Path,required=True)
    ap.add_argument("--manifest",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--learning-rate",type=float,default=1)
    ap.add_argument("--max-delta",type=float,default=0.35)
    ap.add_argument("--regularization",type=float,default=0.01)
    a=ap.parse_args()
    if not 1<=a.steps<=500 or not 0<a.learning_rate<=5 or not 0<a.max_delta<=0.5 or not 0<=a.regularization<=1:ap.error("Research bounds exceeded")
    source=a.model_dir.resolve();out=a.output.resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):ap.error("Output must be a new isolated directory outside source")
    rows=json.loads(a.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not 1<=len(rows)<=10000:raise ValueError("Expected a bounded list of corrected texts")
    tok=SodaTokenizer(source/"endtoendmodel")
    sequences=[tok.encode(speech_text(r["reference"])) for r in rows]
    if not all(sequences):raise ValueError("Empty label sequence")
    if any(35 in s for s in sequences):raise ValueError("Unknown token in labels; resolve before training")
    posterior=source/"endtoendmodel/medium-joint_posterior.tflite"
    prior=source/"endtoendmodel/medium-joint_prior.tflite"
    before={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()}
    cp,_,_=constants(posterior);cl,_,_=constants(prior)
    if not np.array_equal(cp[8],cl[5]):raise ValueError("Posterior/prior biases are not tied in source")
    d,labels=decoder_features(source/"endtoendmodel/medium-decoder.tflite",sequences)
    hidden=np.tanh(hybrid_linear(d,cl[7])+cl[6])
    logits=(hybrid_linear(hidden,cl[4])+cl[5])[:,1:]
    delta=np.zeros(4095,np.float64);loss0,nll0=objective(logits,labels,delta,a.regularization)
    grad=gradient(logits,labels,delta,a.regularization)
    checks=[]
    for index in np.unique(labels)[:8]:
        eps=1e-4;plus=delta.copy();minus=delta.copy();plus[index]+=eps;minus[index]-=eps
        numerical=(objective(logits,labels,plus,a.regularization)[0]-objective(logits,labels,minus,a.regularization)[0])/(2*eps)
        checks.append(abs(numerical-grad[index]))
    if max(checks)>1e-6:raise RuntimeError("Gradient finite-difference check failed")
    history=[]
    for step in range(a.steps):
        delta=np.clip(delta-a.learning_rate*gradient(logits,labels,delta,a.regularization),-a.max_delta,a.max_delta)
        if step%10==0 or step==a.steps-1:
            loss,nll=objective(logits,labels,delta,a.regularization);history.append({"step":step+1,"objective":loss,"nll":nll})
    loss1,nll1=objective(logits,labels,delta,a.regularization)
    if not loss1<loss0:raise RuntimeError("Supervised objective did not improve")
    updated=cl[5].copy();updated[1:]+=delta.astype(np.float32)
    changed=int(np.count_nonzero(updated!=cl[5]))
    if changed==0 or updated[0]!=cl[5][0]:raise RuntimeError("No weight change or blank bias changed")
    pack=out/"SODAModels";shutil.copytree(source,pack)
    replace_float_buffer(pack/"endtoendmodel/medium-joint_posterior.tflite",8,updated)
    replace_float_buffer(pack/"endtoendmodel/medium-joint_prior.tflite",5,updated)
    for filename in ("medium-joint_posterior.tflite","medium-joint_prior.tflite"):
        interpreter=Interpreter(model_path=str(pack/"endtoendmodel"/filename),num_threads=1)
        interpreter.allocate_tensors()
    after_source={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()}
    if before!=after_source:raise RuntimeError("Source pack changed during experiment")
    report={"schema_version":1,"experiment":"text_conditioned_HAT_output_bias_training","weights_updated":True,"acoustic_encoder_updated":False,"decoder_updated":False,"blank_bias_updated":False,"changed_float_parameters_per_tied_file":changed,"training_texts":len(rows),"training_label_tokens":int(len(labels)),"unique_target_ids":int(len(np.unique(labels))),"label_normalization":"NFC lowercase; punctuation removed; whitespace; verified WPM piece IDs; unknown labels refused","tokenizer_native_api_parity":"NOT_MEASURED; portable merge table versus official algorithm verified separately","dataset_role":"development_only_no_independent_holdout","manifest_sha256":sha(a.manifest),"nll_before":nll0,"nll_after":nll1,"objective_before":loss0,"objective_after":loss1,"gradient_max_absolute_error":float(max(checks)),"max_parameter_delta":float(np.max(abs(updated-cl[5]))),"steps":a.steps,"learning_rate":a.learning_rate,"regularization":a.regularization,"history":history,"source_unchanged":True,"source_hashes":before,"changed_files":[{"path":str(p.relative_to(pack)),"sha256":sha(p)} for p in (pack/"endtoendmodel/medium-joint_posterior.tflite",pack/"endtoendmodel/medium-joint_prior.tflite")],"copied_models_load":"passed","native_ASR_evaluation":"pending","promoted":False}
    (out/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("history","source_hashes")},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
