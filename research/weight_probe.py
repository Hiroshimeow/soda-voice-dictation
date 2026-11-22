"""Isolated supervised *feature-level* weight-edit feasibility probe.

NOT speech fine-tuning: the target ID and feature vectors are synthetic.
Writes a copied joint-posterior TFLite under --out; never edits source pack.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib

import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter

def sha(data):
    return hashlib.sha256(data).hexdigest()

def run(model_bytes, encoder, decoder):
    interpreter = Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    assert [list(v["shape"]) for v in inputs] == [[1, 512], [1, 640]]
    interpreter.set_tensor(inputs[0]["index"], encoder)
    interpreter.set_tensor(inputs[1]["index"], decoder)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--label-id", type=int, default=1, help="synthetic non-blank posterior ID (1..4095)")
    parser.add_argument("--step", type=float, default=0.5, help="positive learning rate <=1")
    a = parser.parse_args()
    if not 1 <= a.label_id <= 4095 or not 0 < a.step <= 1:
        parser.error("invalid label-id or step")
    if a.out.resolve() == a.model.resolve():
        parser.error("refusing in-place mutation of source pack")
    source = a.model.read_bytes()
    modified = bytearray(source)
    model = tflite.Model.GetRootAsModel(modified, 0)
    graph = model.Subgraphs(0)
    # Observed graph contract: 4096-element *float32 bias* for final projection.
    bias = graph.Tensors(8)
    if bias.Type() != tflite.TensorType.FLOAT32 or list(bias.ShapeAsNumpy()) != [4096]:
        raise ValueError("unexpected output bias layout")
    buf = model.Buffers(bias.Buffer())
    raw = buf.DataAsNumpy()
    if raw.nbytes != 4096 * 4 or not raw.flags.writeable:
        raise ValueError("bias is not directly editable")
    bias_values = np.frombuffer(raw, dtype="<f4")
    original_bias = float(bias_values[a.label_id])
    rng = np.random.default_rng(20261122)
    encoder = rng.normal(0, 0.1, (1, 512)).astype("float32")
    decoder = rng.normal(0, 0.1, (1, 640)).astype("float32")
    original = run(source, encoder, decoder)
    p = float(original[a.label_id])
    nonblank_mass = float(original[1:].sum())
    gradient = 1.0 - p / nonblank_mass
    bias_values[a.label_id] = original_bias + a.step * gradient
    updated = bytes(modified)
    assert source != updated and len(source) == len(updated)
    changed = np.nonzero(np.frombuffer(source, dtype="u1") != np.frombuffer(updated, dtype="u1"))[0]
    assert 0 < len(changed) <= 4, "unexpected edits outside one float"
    candidate = run(updated, encoder, decoder)
    old_loss, new_loss = -float(np.log(original[a.label_id])), -float(np.log(candidate[a.label_id]))
    assert new_loss < old_loss, "supervised feature-level loss did not decrease"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(updated)
    assert a.out.read_bytes() == updated and a.model.read_bytes() == source
    report = {"classification": "synthetic feature-target gradient step ONLY; not acoustic model training",
              "label_id": a.label_id, "target_is_audio_transcript": False,
              "speech_accuracy_measured": False, "native_soda_pack_mutated": False,
              "source_sha256": sha(source), "copy_sha256": sha(updated),
              "altered_bytes": len(changed), "bias_before": original_bias,
              "bias_after": float(bias_values[a.label_id]),
              "source_target_probability": p, "copy_target_probability": float(candidate[a.label_id]),
              "cross_entropy_before": old_loss, "cross_entropy_after": new_loss,
              "max_abs_output_change": float(np.max(np.abs(candidate - original))),
              "interpreter_load": "success (standalone LiteRT)",
              "native_soda_inference": "NOT_TESTED", "rnnt_loss": "NOT_IMPLEMENTED"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
