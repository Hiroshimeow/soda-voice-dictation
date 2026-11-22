"""Reconstruct SODA neural symbol IDs and portable WordPiece segmentation.

The ID mapping is checked against every portable wordpiece. Segmentation uses
the lowest-result-ID adjacent merge algorithm documented in TensorFlow Lingvo.
Native SODA tokenizer-call parity is NOT_MEASURED. No vocabulary is bundled.
"""
from __future__ import annotations
import hashlib
import struct
import unicodedata
from pathlib import Path
import marisa_trie
from .wire import fields

class SodaTokenizer:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        compact = (model_dir / "medium.syms.compact").read_bytes()
        portable = (model_dir / "medium.wpm.portable").read_bytes()
        start = compact.index(b"We love Marisa.")
        trie = marisa_trie.Trie().frombytes(compact[start:])
        trailer = compact[start + len(trie.tobytes()):]
        count = len(trie)
        if len(trailer) != 32 + 2 * count:
            raise ValueError("Unrecognized compact symbol trailer")
        permutation = struct.unpack(f"<{count}H", trailer[32:])
        if set(permutation) != set(range(1, count + 1)):
            raise ValueError("Compact symbol permutation is not bijective")
        self.vocabulary = tuple(trie.restore_key(value - 1) for value in permutation)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.vocabulary)}
        pieces = {}
        portable_merges = {}
        exceptions = []
        for number, wire, data in fields(portable):
            if number == 3:
                entry = {f.number: f.value for f in fields(data)}
                pieces[int(entry[1])] = entry[2].decode("utf-8")
            elif number == 4:
                entry = {f.number: f.value for f in fields(data)}
                portable_merges[(int(entry[1]), int(entry[2]))] = int(entry[3])
            elif number == 7:
                exceptions.append(data.decode("utf-8"))
        offsets = {self.symbol_to_id[symbol] - index for index, symbol in pieces.items()}
        if len(offsets) != 1:
            raise ValueError("Portable and neural symbol mappings disagree")
        self.portable_to_neural_offset = offsets.pop()
        if set(exceptions) != set(self.vocabulary) - set(pieces.values()):
            raise ValueError("Special-symbol sets disagree")
        self.merges = {}
        for (left, right), merged in portable_merges.items():
            if pieces[left] + pieces[right] != pieces[merged]:
                raise ValueError("Merge table does not concatenate its source pieces")
            off = self.portable_to_neural_offset
            self.merges[left + off, right + off] = merged + off
        self.unk_id = self.symbol_to_id["<UNK>"]
        self.bos_id = self.symbol_to_id["<S>"]
        self.eos_id = self.symbol_to_id["</S>"]
        self.epsilon_id = self.symbol_to_id["<epsilon>"]
        self.piece_count = len(pieces)
        self.special_count = len(exceptions)
        self.source_hashes = {
            "medium.syms.compact": hashlib.sha256(compact).hexdigest(),
            "medium.wpm.portable": hashlib.sha256(portable).hexdigest(),
        }

    def _merge_characters(self, text: str, use_merge_table: bool = True) -> list[int]:
        tokens = [self.symbol_to_id.get(char, self.unk_id) for char in text]
        while len(tokens) > 1:
            options = []
            for index, (left, right) in enumerate(zip(tokens, tokens[1:])):
                if use_merge_table:
                    merged = self.merges.get((left, right))
                else:
                    merged = self.symbol_to_id.get(self.vocabulary[left] + self.vocabulary[right])
                if merged is not None:
                    options.append((merged, index))
            if not options:
                break
            merged, index = min(options)
            tokens[index:index + 2] = [merged]
        return tokens

    def encode(self, text: str, *, lowercase: bool = False,
               add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode NFC text; callers explicitly choose casing and BOS/EOS."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if len(text) > 100_000:
            raise ValueError("Encode text in bounded segments")
        normalized = unicodedata.normalize("NFC", text)
        if lowercase:
            normalized = normalized.lower()
        result = [self.bos_id] if add_bos else []
        for word in normalized.split():
            if len(word) > 4096:
                raise ValueError("Word exceeds research tokenizer bound")
            result.extend(self._merge_characters("▁" + word))
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(self, ids: list[int], *, remove_control_tokens: bool = True) -> str:
        controls = {self.bos_id, self.eos_id, self.epsilon_id} if remove_control_tokens else set()
        return "".join(self.vocabulary[index] for index in ids if index not in controls).replace("▁", " ").strip()

    def mapping_report(self) -> dict:
        return {
            "vocabulary_count": len(self.vocabulary),
            "wordpiece_count": self.piece_count,
            "special_count": self.special_count,
            "merge_count": len(self.merges),
            "portable_to_neural_offset": self.portable_to_neural_offset,
            "mapping_mismatches": 0,
            "bos_id": self.bos_id, "eos_id": self.eos_id,
            "epsilon_id": self.epsilon_id, "unk_id": self.unk_id,
            "source_sha256": self.source_hashes,
            "segmentation_native_parity": "NOT_MEASURED",
        }
