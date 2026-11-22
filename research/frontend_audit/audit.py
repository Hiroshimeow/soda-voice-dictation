"""Run bounded, read-only checks against a locally owned SODA language pack."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path
from .tokenizer import SodaTokenizer
from .wire import fields

def report(pack: Path) -> dict:
    tokenizer = SodaTokenizer(pack / "endtoendmodel")
    result = tokenizer.mapping_report()
    pieces = tokenizer.vocabulary[tokenizer.special_count:]
    disagreements = sum(
        tokenizer._merge_characters(piece, True) != tokenizer._merge_characters(piece, False)
        for piece in pieces
    )
    reconstruction_failures = sum(
        tokenizer._merge_characters(piece, True) != [tokenizer.symbol_to_id[piece]]
        for piece in pieces
    )
    public_examples = ("xin chào thế giới", "tôi muốn nhập bằng giọng nói",
                       "hẹn lúc 7 giờ", "AI cloud data 60%")
    roundtrip_failures = sum(
        tokenizer.decode(tokenizer.encode(text)) != text for text in public_examples
    )
    result["checks"] = {
        "wordpiece_merge_equivalence_count": len(pieces),
        "merge_algorithm_disagreements": disagreements,
        "wordpiece_self_reconstruction_failures": reconstruction_failures,
        "public_roundtrip_examples": len(public_examples),
        "public_roundtrip_failures": roundtrip_failures,
        "unknown_character_maps_to_unk": tokenizer.unk_id in tokenizer.encode(chr(0x10FFFF)),
        "nfc_equivalence": tokenizer.encode("tiếng") == tokenizer.encode("tie\u0302\u0301ng"),
    }
    configs = {}
    for name in ("CONTINUOUS", "SHORT"):
        path = pack / "configs" / f"ONDEVICE_MEDIUM_{name}.config"
        nodes = []
        for outer, wire, value in fields(path.read_bytes()):
            if outer not in (1, 3):
                continue
            entry = list(fields(value))
            named = {f.number: f.value for f in entry if f.number in (1, 2)}
            node_name = named.get(1, b"").decode()
            if not any(part in node_name for part in (
                    "audio_features", "filter_bank_params", "conformer_encoder")):
                continue
            options = []
            for field in entry:
                if field.number == 3:
                    for extension in fields(field.value):
                        decoded = []
                        for option in fields(extension.value):
                            val = option.value
                            if option.wire == 5:
                                val = struct.unpack("<f", val)[0]
                            elif isinstance(val, bytes):
                                try:
                                    val = val.decode("utf-8")
                                except UnicodeDecodeError:
                                    val = {"hex": val.hex()}
                            decoded.append({"number": option.number, "wire": option.wire, "value": val})
                        options.append({"extension_id": extension.number, "values": decoded})
            nodes.append({"kind": outer, "name": node_name, "type": named.get(2, b"").decode(), "options": options})
        configs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "nodes": nodes}
    norm = {}
    for name in ("CONTINUOUS", "SHORT"):
        path = pack / "endtoendmodel" / f"ONDEVICE_MEDIUM_{name}.mean_stddev"
        vectors = {}
        for field in fields(path.read_bytes()):
            values = struct.unpack(f"<{len(field.value)//4}f", field.value)
            vectors[{1: "mean", 2: "std_dev"}.get(field.number, str(field.number))] = {
                "count": len(values), "minimum": min(values), "maximum": max(values)}
        norm[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "vectors": vectors}
    result["configs"] = configs
    result["normalization"] = norm
    result["feature_native_parity"] = "NOT_MEASURED"
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = result["checks"]
    print(json.dumps({"output": str(args.output), "checks": checks}, ensure_ascii=True))
    if any(checks[key] for key in ("merge_algorithm_disagreements", "wordpiece_self_reconstruction_failures", "public_roundtrip_failures")):
        raise SystemExit(1)
    if not checks["unknown_character_maps_to_unk"] or not checks["nfc_equivalence"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
