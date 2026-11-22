"""Inspect selected public-format protobuf descriptors embedded in libsoda.

Read-only: this neither loads native code nor changes the model or runtime.
Only the named architecture descriptors are extracted; credentials and runtime
configuration are outside this inspector's scope.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .wire import fields, varint

DESCRIPTORS = (
    "speech/greco3/frontend/framer.proto",
    "speech/greco3/frontend/window.proto",
    "speech/greco3/frontend/fft.proto",
    "speech/greco3/frontend/filter_bank.proto",
    "speech/greco3/frontend/frame_stacker.proto",
    "speech/greco3/frontend/append_clusterid_stream_params.proto",
    "speech/greco3/tflite/encoder_stream_params.proto",
    "speech/lstm/mean_stddev.proto",
    "speech/languagemodel/wordpiecemodel/portable_wordpiecemodel_data.proto",
    "speech/decoder2/dual/pie/pie_rnn_fst_decoder_params.proto",
)

def encode_varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)

def extract_descriptor(binary: bytes, name: str) -> bytes:
    encoded = name.encode("utf-8")
    start = binary.find(b"\x0a" + encode_varint(len(encoded)) + encoded)
    if start < 0:
        raise ValueError(f"Descriptor absent: {name}")
    end = start
    while end < min(len(binary), start + 100_000):
        tag, cursor = varint(binary, end)
        number, wire = tag >> 3, tag & 7
        if number not in set(range(1, 13)) | {14} or (number == 1 and end != start):
            break
        if wire == 0:
            _, cursor = varint(binary, cursor)
        elif wire in (1, 5):
            cursor += 8 if wire == 1 else 4
        elif wire == 2:
            size, cursor = varint(binary, cursor)
            cursor += size
        else:
            break
        if cursor > min(len(binary), start + 100_000):
            raise ValueError("Descriptor exceeds bounded region")
        end = cursor
    data = binary[start:end]
    if not data:
        raise ValueError("Empty descriptor")
    return data

def first(message: bytes, number: int, default=None):
    return next((item.value for item in fields(message) if item.number == number), default)

def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value

def enum(message: bytes) -> dict:
    return {
        "name": text(first(message, 1)),
        "values": [{"name": text(first(item.value, 1)), "number": first(item.value, 2)}
                   for item in fields(message) if item.number == 2],
    }

def message_descriptor(message: bytes) -> dict:
    result = {"name": text(first(message, 1)), "fields": [], "enums": []}
    for item in fields(message):
        if item.number in (2, 6):
            raw = item.value
            result["fields"].append({
                "name": text(first(raw, 1)), "number": first(raw, 3),
                "label": first(raw, 4), "type": first(raw, 5),
                "type_name": text(first(raw, 6)), "default": text(first(raw, 7)),
            })
        elif item.number == 4:
            result["enums"].append(enum(item.value))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.library.read_bytes()
    records = []
    for name in DESCRIPTORS:
        data = extract_descriptor(binary, name)
        if text(first(data, 1)) != name:
            raise ValueError("Descriptor filename mismatch")
        records.append({
            "filename": name, "descriptor_sha256": hashlib.sha256(data).hexdigest(),
            "messages": [message_descriptor(item.value) for item in fields(data) if item.number == 4],
        })
    result = {"library_sha256": hashlib.sha256(binary).hexdigest(), "descriptors": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"descriptors": len(records), "output": str(args.output)}))

if __name__ == "__main__":
    main()
