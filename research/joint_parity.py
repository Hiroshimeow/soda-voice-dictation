"""Reconstruct SODA's posterior/prior projections from actual TFLite constants.

This is a bounded research tool. It never changes the active Google pack.
Random feature probes establish numeric parity only, not ASR accuracy or training.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, platform
import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter,OpResolverType

def constants(path: pathlib.Path):
    model=tflite.Model.GetRootAsModel(path.read_bytes(),0)
    graph=model.Subgraphs(0); result={}
    dtypes={tflite.TensorType.FLOAT32:np.dtype("<f4"),tflite.TensorType.INT8:np.dtype("i1"),tflite.TensorType.INT32:np.dtype("<i4")}
    for i in range(graph.TensorsLength()):
        t=graph.Tensors(i);buf=model.Buffers(t.Buffer())
        if not buf.DataLength():continue
        shape=tuple(int(t.Shape(j)) for j in range(t.ShapeLength()))
        arr=np.frombuffer(buf.DataAsNumpy().tobytes(),dtype=dtypes[t.Type()]).reshape(shape).copy()
        q=t.Quantization()
        if q and q.ScaleLength():
            scales=q.ScaleAsNumpy().astype(np.float32);zero=q.ZeroPointAsNumpy().astype(np.float32)
            if len(scales)>1:
                s=[1]*arr.ndim;s[q.QuantizedDimension()]=len(scales);scales=scales.reshape(s);zero=zero.reshape(s)
            arr=(arr.astype(np.float32)-zero)*scales
        result[i]=arr
    return result,model,graph
def softmax(z):
    e=np.exp(z-z.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
def hybrid_linear(x,weight):
    lo=np.minimum(x.min(axis=-1,keepdims=True),0)
    hi=np.maximum(x.max(axis=-1,keepdims=True),0)
    scale=np.maximum((hi-lo)/255,np.finfo(np.float32).tiny)
    zero=np.clip(np.rint(-128-lo/scale),-128,127)
    shifted=x/scale+zero
    q=np.clip(np.floor(abs(shifted)+0.5)*np.sign(shifted),-128,127)
    return ((q-zero)*scale)@weight.T

def posterior(a,d,c):
    h=np.tanh(hybrid_linear(a,c[10])+hybrid_linear(d,c[7])+c[9])
    z=hybrid_linear(h,c[6])+c[8]
    # The actual graph slices the first logit for the blank gate.
    blank_index=int(c[5][-1]);end=int(c[4][-1])
    blank=z[:,blank_index:end if end else None]
    begin=int(c[4][-1]);finish=int(c[5][-1])
    labels=z[:,begin:finish if finish else None]
    # Assert the graph-specific HAT partition instead of assuming token order.
    if blank.shape[1]!=1 or labels.shape[1]!=4095:raise ValueError((blank_index,end,begin,finish,c[3],c[4],c[5]))
    p_blank=1/(1+np.exp(-blank))
    return np.concatenate((p_blank,(1-p_blank)*softmax(labels)),axis=-1),h,z
def prior(d,c):
    h=np.tanh(hybrid_linear(d,c[7])+c[6]);z=hybrid_linear(h,c[4])+c[5]
    # Derive the nonblank slice from graph constants.
    begin=int(c[3][-1]);end=int(c[2][-1])
    return softmax(z[:,begin:end if end else None]),h,z
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",type=pathlib.Path,required=True)
    ap.add_argument("--report",type=pathlib.Path,required=True)
    ap.add_argument("--probes",type=int,default=12)
    a=ap.parse_args()
    if not 1<=a.probes<=100:ap.error("probes must be 1..100")
    rng=np.random.default_rng(20261122);rows=[]
    for name,fn,inputs in (("medium-joint_posterior.tflite",posterior,[512,640]),("medium-joint_prior.tflite",prior,[640])):
        path=a.model_dir/name;c,model,graph=constants(path)
        interpreter=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,experimental_preserve_all_tensors=True)
        interpreter.allocate_tensors()
        metrics=[]
        for n in range(a.probes):
            vals=[rng.normal(0,0.1 if n<a.probes//2 else 1,(1,size)).astype(np.float32) for size in inputs]
            for meta,value in zip(interpreter.get_input_details(),vals):interpreter.set_tensor(meta["index"],value)
            interpreter.invoke()
            expected=interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            actual,hidden,logits=fn(*vals,c)
            metrics.append({"max_abs_probability_error":float(np.max(abs(expected-actual))),"l1_probability_error":float(np.sum(abs(expected-actual))),"top1_matches":bool(expected.argmax()==actual.argmax()),"reference_sum":float(expected.sum()),"reconstruction_sum":float(actual.sum())})
        row={"model":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"probes":a.probes,"max_abs_probability_error":max(m["max_abs_probability_error"] for m in metrics),"mean_l1_probability_error":float(np.mean([m["l1_probability_error"] for m in metrics])),"top1_matches":sum(m["top1_matches"] for m in metrics),"all":metrics}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!="all"}),flush=True)
    report={"status":"projection_reconstruction_only","weights_updated":False,"actual_user_audio_used":False,"limitation":"Reconstruction emulates observed asymmetric dynamic activation quantization and fixed INT8 weights. Small kernel rounding differences remain; this is probability parity on seeded feature probes, not bit-exact or waveform-to-text parity.","results":rows}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__":main()
