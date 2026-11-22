"""Bounded protobuf-wire reader for read-only model metadata inspection."""
from __future__ import annotations
from collections.abc import Iterator
from typing import NamedTuple

class Field(NamedTuple):
    number: int
    wire: int
    value: int | bytes

def varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value > 2**64 - 1:
                raise ValueError("Protobuf varint exceeds uint64")
            return value, offset
    raise ValueError("Invalid protobuf varint")

def fields(data: bytes) -> Iterator[Field]:
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if not number or number >= 2**29:
            raise ValueError("Invalid protobuf field number")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if offset > len(data):
            raise ValueError("Truncated protobuf field")
        yield Field(number, wire, value)
