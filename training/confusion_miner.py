"""Closed-loop Vietnamese phonetic confusion miner (no SODA weight changes).

Aligns (prediction, reference) pairs from consented local evaluation reports or
manifests via Levenshtein word alignment, then extracts:
1. High-precision recurring phonetic confusion pairs (heard -> write) filtered
   by Vietnamese phonetic similarity so coincidental errors are never promoted.
2. High-error OOV / rare Vietnamese phrases ranked for SODA's 40-phrase
   contextual biasing profile.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from training.context_profile import DATA, MAX_TERMS, validate_profile, write_atomic
from vi_enhance.dynamic_bias import build_smart_bias_profile
from vi_enhance.phonetics import phrase_phonetic_similarity

MINED_OUTPUT_PATH = DATA / "mined_vi_adaptations.json"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold(), re.UNICODE)


def align_words(pred_words: list[str], ref_words: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Return optimal word alignment ops: ('eq'|'sub'|'del'|'ins', pred_tok, ref_tok)."""
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[int, int, str]]] = [[(0, 0, "")] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "ins")
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "del")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = (i - 1, j - 1, "eq")
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i - 1][j] + 1
                dele = dp[i][j - 1] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                if best == sub:
                    back[i][j] = (i - 1, j - 1, "sub")
                elif best == ins:
                    back[i][j] = (i - 1, j, "ins")
                else:
                    back[i][j] = (i, j - 1, "del")

    ops: list[tuple[str, str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        pi, pj, op = back[i][j]
        if op in ("eq", "sub"):
            ops.append((op, pred_words[i - 1], ref_words[j - 1]))
        elif op == "ins":
            ops.append(("ins", pred_words[i - 1], None))
        else:
            ops.append(("del", None, ref_words[j - 1]))
        i, j = pi, pj
    ops.reverse()
    return ops


def mine_confusions(
    pairs: list[tuple[str, str]],
    *,
    min_count: int = 2,
    min_phonetic_similarity: float = 0.65,
) -> dict:
    """Mine recurring phonetic confusion rules and priority bias phrases from (pred, ref) pairs."""
    sub_spans: Counter[tuple[str, str]] = Counter()
    missed_terms: Counter[str] = Counter()

    for pred, ref in pairs:
        p_toks = _tokenize(pred)
        r_toks = _tokenize(ref)
        ops = align_words(p_toks, r_toks)

        # Group contiguous non-equal spans (captures multi-syllable Vietnamese errors!)
        buf_p: list[str] = []
        buf_r: list[str] = []

        def _flush() -> None:
            if buf_p and buf_r and len(buf_p) <= 4 and len(buf_r) <= 4:
                heard = " ".join(buf_p)
                write = " ".join(buf_r)
                if heard != write:
                    sub_spans[(heard, write)] += 1
                    missed_terms[write] += 1
            elif buf_r and len(buf_r) <= 4:
                missed_terms[" ".join(buf_r)] += 1
            buf_p.clear()
            buf_r.clear()

        for op, p_tok, r_tok in ops:
            if op == "eq":
                _flush()
            else:
                if p_tok is not None:
                    buf_p.append(p_tok)
                if r_tok is not None:
                    buf_r.append(r_tok)
        _flush()

    replacements: list[dict] = []
    seen_heard: set[str] = set()
    for (heard, write), count in sub_spans.most_common():
        if count < min_count:
            continue
        if heard in seen_heard or not (2 <= len(heard) <= 75 and 2 <= len(write) <= 75):
            continue
        sim = phrase_phonetic_similarity(heard, write)
        if sim < min_phonetic_similarity:
            continue
        seen_heard.add(heard)
        replacements.append({
            "heard": heard,
            "write": write,
            "count": count,
            "phonetic_similarity": sim,
        })

    mined_bias_inputs = [
        {"phrase": phrase, "count": count}
        for phrase, count in missed_terms.most_common(MAX_TERMS)
        if count >= min_count and 2 <= len(phrase) <= 75
    ]
    bias_profile = build_smart_bias_profile(mined_terms=mined_bias_inputs) if mined_bias_inputs else None

    return {
        "schema_version": 1,
        "language": "vi-VN",
        "pairs_analyzed": len(pairs),
        "replacements": replacements[:64],
        "bias_profile": validate_profile(bias_profile) if bias_profile else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine Vietnamese phonetic confusions from evaluation reports")
    ap.add_argument("--report", type=Path, required=True, help="Schema-v2 evaluation report with results[].raw_recognition & reference")
    ap.add_argument("--output", type=Path, default=MINED_OUTPUT_PATH)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--min-similarity", type=float, default=0.65)
    args = ap.parse_args()

    raw = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("Unsupported evaluation report: expected schema_version 2")
    if raw.get("status") != "completed":
        raise ValueError("Evaluation report is not completed")
    if not raw.get("references_included"):
        raise ValueError("Evaluation report has no clear references; rerun evaluation with --include-reference")
    rows = raw.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation report has no results[]")
    expected = raw.get("expected_samples")
    if not isinstance(expected, int) or expected != len(rows):
        raise ValueError("Evaluation report is incomplete: expected_samples does not match results[]")
    if any(not isinstance(row, dict) or row.get("helper_status") != "ok" for row in rows):
        raise ValueError("Evaluation report contains a non-successful helper row")
    missing = [str(row.get("id", index)) for index, row in enumerate(rows)
               if not isinstance(row, dict) or not isinstance(row.get("raw_recognition"), str)
               or not isinstance(row.get("reference"), str)]
    if missing:
        raise ValueError("Evaluation report is missing raw_recognition/reference for samples: " + ", ".join(missing[:5]))
    pairs = []
    skipped = []
    for index, row in enumerate(rows):
        raw_text, reference = row["raw_recognition"], row["reference"]
        if not raw_text.strip() or not reference.strip():
            skipped.append(str(row.get("id", index)))
            continue
        # Never mine candidate rules from literal-looking output; these are
        # paths/URLs/credentials, not Vietnamese recognition corrections.
        if re.search(r"(?i)(https?://|[A-Za-z]:\\|\\\\|@|\b(?:token|secret|password)\b)", raw_text + " " + reference):
            skipped.append(str(row.get("id", index)))
            continue
        pairs.append((raw_text, reference))
    if not pairs:
        raise ValueError("Evaluation report contains no safe non-empty raw/reference pairs")
    mined = mine_confusions(pairs, min_count=args.min_count, min_phonetic_similarity=args.min_similarity)
    mined["samples_skipped"] = skipped
    write_atomic(args.output, mined)
    print(f"MINED rules={len(mined['replacements'])} bias_terms={len((mined['bias_profile'] or {}).get('terms', []))} -> {args.output}")


if __name__ == "__main__":
    main()
