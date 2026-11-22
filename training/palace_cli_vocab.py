"""Query authorized live Palace CLI rather than bypassing a running SQLite owner.

No content leaves the local machine; only a conservative curated vocabulary and aggregate
counts are retained. Search hits are capped, so counts are NOT global frequency.
"""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os
import pathlib
import subprocess
import sys

from training.seed_terms import TERMS

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = "palace.cmd" if os.name == "nt" else "palace"
def run(term: str, limit: int) -> tuple[int,int]:
    p = subprocess.run([CLI, "search", term, "--limit", str(limit), "--max-bytes", "30000"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=25, check=False)
    if p.returncode:
        raise RuntimeError(f"Palace CLI query failed ({term}): exit={p.returncode}")
    data = json.loads(p.stdout)
    if not data.get("ok"):
        raise RuntimeError(f"Palace CLI error: {data.get('code', 'unknown')}")
    items = data.get("items",[])
    # Only observed cards; avoid storing summaries or full JSON.
    return sum(x.get("status") == "observed" for x in items), int(data.get("candidates",len(items)))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-terms",type=int,default=55)
    ap.add_argument("--limit",type=int,default=5)
    ap.add_argument("--out",type=pathlib.Path,default=ROOT/"data"/"palace_candidates.json")
    a=ap.parse_args()
    if not 1<=a.max_terms<=150 or not 1<=a.limit<=5:
        raise ValueError("invalid bounds")
    records=[]
    for term in TERMS[:a.max_terms]:
        try:
            count,total=run(term,a.limit)
        except Exception as e:
            print("QUERY_FAILED",term,type(e).__name__,file=sys.stderr,flush=True)
            continue
        print("QUERY_OK",term,count,total,flush=True)
        if count>=1:
            records.append({"phrase":term,"observed_search_hits":count,
                            "boost":round(min(3.5,1.0+math.log1p(count)/3.5),2),
                            "retrieval_capped":total>a.limit})
    records.sort(key=lambda x:(-x["observed_search_hits"],x["phrase"].casefold()))
    result={"schema_version":1,"language":"vi-VN",
            "source":"live Palace official CLI (read-only searches)",
            "frequency_type":"observed_retrieved_hits_not_full_corpus_frequency",
            "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scanned_queries":min(len(TERMS),a.max_terms),
            "candidate_count":len(records),"terms":records,
            "privacy":"no retrieved summaries, body, card ids or secrets stored",
            "training_type":"context bias prior estimation; no acoustic weights updated"}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    temp=a.out.with_suffix(".tmp")
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(a.out)
    print("RESULT",json.dumps({"count":len(records),"path":str(a.out)},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
