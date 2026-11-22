"""Read-only, reproducible TFLite model audit of an installed SODA pack.

Run from the isolated WSL research environment. No audio/model bytes are emitted.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import json
import pathlib

import tflite

TYPES = {v: k for k, v in vars(tflite.TensorType).items() if isinstance(v, int)}
OPS = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
TRAINING_OPS = {"VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "CALL_ONCE", "RANDOM_STANDARD_NORMAL",
                "RESOURCE_GATHER", "RESOURCE_APPLY_ADAM", "RESOURCE_APPLY_GRADIENT_DESCENT"}

def name(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tensor_info(tensor, model):
    q = tensor.Quantization()
    buf = model.Buffers(tensor.Buffer())
    raw = buf.DataAsNumpy()
    blob = raw.tobytes() if raw is not None and buf.DataLength() else b""
    return {
        "name": name(tensor.Name()),
        "shape": [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())],
        "shape_signature": [int(tensor.ShapeSignature(i)) for i in range(tensor.ShapeSignatureLength())],
        "dtype": TYPES.get(tensor.Type(), str(tensor.Type())),
        "is_variable": bool(tensor.IsVariable()),
        "buffer_index": int(tensor.Buffer()),
        "buffer_bytes": len(blob),
        "buffer_sha256": digest(blob) if blob else None,
        "quantization": {
            "scales": [float(q.Scale(i)) for i in range(q.ScaleLength())] if q else [],
            "zero_points": [int(q.ZeroPoint(i)) for i in range(q.ZeroPointLength())] if q else [],
            "axis": int(q.QuantizedDimension()) if q else None,
        },
    }

def audit_model(path):
    data = path.read_bytes()
    if data[4:8] != b"TFL3":
        raise ValueError(f"Not a TFL3 FlatBuffer: {path}")
    model = tflite.Model.GetRootAsModel(data, 0)
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        op = model.OperatorCodes(i)
        opcodes.append({"builtin": OPS.get(op.BuiltinCode(), str(op.BuiltinCode())),
                        "custom": name(op.CustomCode()), "version": int(op.Version())})
    signatures = []
    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        signatures.append({"key": name(sig.SignatureKey()), "subgraph": int(sig.SubgraphIndex()),
                           "inputs": [name(sig.Inputs(j).Name()) for j in range(sig.InputsLength())],
                           "outputs": [name(sig.Outputs(j).Name()) for j in range(sig.OutputsLength())]})
    graphs = []
    for i in range(model.SubgraphsLength()):
        graph = model.Subgraphs(i)
        tensors = [tensor_info(graph.Tensors(j), model) for j in range(graph.TensorsLength())]
        operators = [opcodes[graph.Operators(j).OpcodeIndex()] for j in range(graph.OperatorsLength())]
        counts = collections.Counter(o["custom"] or o["builtin"] for o in operators)
        graphs.append({
            "name": name(graph.Name()),
            "tensor_count": len(tensors),
            "operator_count": len(operators),
            "operator_counts": dict(sorted(counts.items())),
            "variable_tensors": [t["name"] for t in tensors if t["is_variable"]],
            "quantized_tensors": sum(bool(t["quantization"]["scales"]) for t in tensors),
            "tensors": tensors,
            "inputs": [{"index": int(graph.Inputs(j)), **tensors[graph.Inputs(j)]}
                       for j in range(graph.InputsLength())],
            "outputs": [{"index": int(graph.Outputs(j)), **tensors[graph.Outputs(j)]}
                        for j in range(graph.OutputsLength())],
            "training_op_candidates": sorted(set(counts).intersection(TRAINING_OPS)),
        })
    return {"file": path.name, "file_bytes": len(data), "file_sha256": digest(data),
            "schema_version": int(model.Version()), "operator_codes": opcodes,
            "signatures": signatures, "graphs": graphs}

def interpreter_probe(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        return {"status": "allocated", "signatures": interpreter.get_signature_list()}
    except Exception as exc:
        return {"status": "failed", "type": type(exc).__name__, "message": str(exc)[:2000]}

def test_observed_contract(report):
    by_name = {m["relative_path"]: m for m in report["models"]}
    required = {"medium-encoder.tflite": (86, 86, [1, 1, 482], [1, 1, 512]),
                "medium-decoder.tflite": (2, 2, [1], [1, 640]),
                "medium-joint_posterior.tflite": (2, 1, [1, 512], [1, 4096]),
                "medium-joint_prior.tflite": (1, 1, [1, 640], [1, 4095])}
    for filename, (ni, no, inp, out) in required.items():
        matched = [v for p, v in by_name.items() if p.endswith("/" + filename)]
        assert len(matched) == 1, filename
        graph = matched[0]["graphs"][0]
        assert len(graph["inputs"]) == ni and len(graph["outputs"]) == no, filename
        assert graph["inputs"][0]["shape"] == inp and graph["outputs"][0]["shape"] == out, filename
        assert not graph["variable_tensors"] and not matched[0]["signatures"], filename
    assert len(by_name) >= 4
    return "passed: 4 actual core graph interfaces, signatures, and variables"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--check-core", action="store_true")
    args = p.parse_args()
    paths = sorted(args.pack.rglob("*.tflite"))
    if not paths:
        p.error("no TFLite models found")
    models = []
    for path in paths:
        info = audit_model(path)
        info["relative_path"] = path.relative_to(args.pack).as_posix()
        info["interpreter"] = interpreter_probe(path)
        models.append(info)
        print(json.dumps({"model": info["relative_path"], "sha256": info["file_sha256"],
                          "subgraphs": len(info["graphs"]), "interpreter": info["interpreter"]["status"]}),
              flush=True)
    report = {"format": 1,
              "package_versions": {k: importlib.metadata.version(k)
                                   for k in ("tflite", "ai-edge-litert", "flatbuffers", "numpy")},
              "models": models}
    if args.check_core:
        report["contract_check"] = test_observed_contract(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": len(models),
                      "contract_check": report.get("contract_check")}), flush=True)

if __name__ == "__main__":
    main()
