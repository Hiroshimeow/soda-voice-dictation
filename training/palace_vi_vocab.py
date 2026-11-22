"""Retrieve Vietnamese Palace cards through its official read-only CLI.

Only candidate phrases/provenance hashes are saved. This is not acoustic training
and sampled search counts are not population frequencies. No card text executes.
"""
from __future__ import annotations
import argparse, collections, datetime, hashlib, json, os, pathlib, re, shutil, subprocess, unicodedata
from training.context_profile import validate_profile,write_atomic

QUERIES=("tôi muốn","tiếng Việt","nhận dạng","hệ thống","công việc","dữ liệu","ứng dụng","kiến thức","quy trình","người dùng","tự động","cá nhân","phát triển","trí tuệ","giao tiếp","nội dung")
MARKERS=set("tôi bạn muốn có không được cho của là và các một những với trong trên đang khi cần để thì đã sẽ phải này đó từ như việc nhưng cũng mình chúng hãy đến bằng về hơn".split())
DIACRITICS=re.compile("[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",re.I)
SENSITIVE=re.compile(r"https?://|www\.|@|(?:[A-Z]:)?[\\/]|(?:api.?key|token|secret|password|credential|authorization)\s*[:=]|\b(?:sk|ghp|gho|xoxb)[-_]",re.I)
CJK=re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
CODE=re.compile(r"[{}<>]|\\n|\b(?:def|function|const|import|SELECT|INSERT|Traceback)\b")
CLI=shutil.which("palace.cmd" if os.name=="nt" else "palace")
def call(*args):
    if not CLI:raise RuntimeError("Official Palace CLI is not installed")
    p=subprocess.run([CLI,*args],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=25)
    if p.returncode:raise RuntimeError("CLI failed: exit "+str(p.returncode))
    data=json.loads(p.stdout)
    if not data.get("ok"):raise RuntimeError("CLI error "+str(data.get("code","unknown")))
    return data
def words(text):return re.findall(r"[^\W\d_]+",unicodedata.normalize("NFC",text).casefold())
def vietnamese_line(line):
    if SENSITIVE.search(line) or CJK.search(line) or CODE.search(line):return False
    ws=words(line)
    return len(ws)>=6 and sum(bool(DIACRITICS.search(w)) for w in ws)>=2 and sum(w in MARKERS for w in ws)/len(ws)>=0.12
def eligible_card(card):
    body=card.get("body","")
    if not isinstance(body,str) or card.get("status")!="observed":return []
    # Observation cards often wrap source output in JSON with escaped Unicode.
    # Decode text escapes only; never execute or evaluate a card as code.
    slash=chr(92)
    for _ in range(2):
        body=re.sub(re.escape(slash)+"u([0-9a-fA-F]{4})",lambda m:chr(int(m[1],16)),body)
        body=body.replace(slash+"r"+slash+"n",chr(10)).replace(slash+"n",chr(10)).replace(slash+"t"," ").replace(slash+'"','"')
    lines=[line.strip() for line in body.splitlines() if len(words(line))>=6 and not CODE.search(line) and not SENSITIVE.search(line)]
    if not lines:return []
    accepted=[line for line in lines if vietnamese_line(line)]
    accepted_size=sum(len(words(s)) for s in accepted)
    total=sum(len(words(s)) for s in lines)
    # Conservative card-level filter; mixed source/code logs are excluded.
    if accepted_size<20 or accepted_size/max(1,total)<0.6:return []
    return accepted
def phrase_candidates(lines):
    found=set()
    for line in lines:
        for sentence in re.split("[.!?;:,\\n]",line):
            ws=words(sentence)
            for n in (2,3,4):
                for i in range(len(ws)-n+1):
                    part=ws[i:i+n]
                    if part[0] in MARKERS or part[-1] in MARKERS:continue
                    if not any(DIACRITICS.search(w) for w in part):continue
                    if all(w in MARKERS for w in part):continue
                    phrase=" ".join(part)
                    if len(phrase)<=64:found.add(phrase)
    return found
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("data/palace_vi_candidates.json"))
    ap.add_argument("--profile",type=pathlib.Path,default=pathlib.Path("data/palace_vi_bias.json"))
    ap.add_argument("--max-queries",type=int,default=16)
    ap.add_argument("--per-query",type=int,default=5)
    ap.add_argument("--max-cards",type=int,default=60)
    a=ap.parse_args()
    if not 1<=a.max_queries<=len(QUERIES) or not 1<=a.per_query<=5 or not 1<=a.max_cards<=120:ap.error("bounds exceeded")
    status=call("status")
    if status.get("pluginState")!="enabled":raise RuntimeError("Palace unavailable")
    seen={};failures=[];queries=[]
    for query in QUERIES[:a.max_queries]:
        try:
            result=call("search",query,"--limit",str(a.per_query),"--max-bytes","14000")
            queries.append({"query":query,"returned":result.get("returned",len(result.get("items",[]))),"candidates":result.get("candidates"),"sampled":True})
            for item in result.get("items",[]):
                if item.get("status")=="observed" and item.get("id"):seen[item["id"]]=item
        except Exception as e:failures.append({"operation":"search","query":query,"error":str(e)})
    phrases=collections.defaultdict(set);provenance={};accepted_cards=0;rejected=0
    for cardid in list(seen)[:a.max_cards]:
        try:
            card=call("read",cardid,"--max-bytes","16000").get("card",{})
            lines=eligible_card(card)
            if not lines:rejected+=1;continue
            accepted_cards+=1
            provenance[cardid]={"id":cardid,"version":card.get("version"),"content_hash":card.get("contentHash"),"evidence_verified_now":False}
            for phrase in phrase_candidates(lines):phrases[phrase].add(cardid)
        except Exception as e:failures.append({"operation":"read","card_id":cardid,"error":str(e)})
        print("READ_PROGRESS",accepted_cards,rejected,len(failures),flush=True)
    ranked=sorted(((p,ids) for p,ids in phrases.items() if len(ids)>=2),key=lambda v:(-len(v[1]),-len(v[0].split()),v[0]))
    terms=[]
    for phrase,ids in ranked:
        if any(phrase in t["phrase"] or t["phrase"] in phrase for t in terms):continue
        terms.append({"phrase":phrase,"boost":1.2,"retrieved_card_count":len(ids),"source_card_ids":sorted(ids)})
        if len(terms)>=64:break
    report={"schema_version":1,"language":"vi-VN","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source":"official Palace read-only CLI","filter":"Vietnamese diacritic/function-word heuristic; minimum 60% natural Vietnamese card tokens; Japanese/CJK, code, URLs, paths and secret-like lines rejected","counts":{"queried":len(queries),"retrieved_unique_cards":len(seen),"attempted_cards":min(len(seen),a.max_cards),"accepted_vietnamese_cards":accepted_cards,"rejected_cards":rejected,"candidate_phrases":len(terms)},"limitations":["sampled retrieval, not exhaustive corpus frequency","language heuristic requires human review","card provenance does not validate factual truth","phrases unproven until audio A/B; inactive by default","no acoustic weights changed"],"queries":queries,"provenance":list(provenance.values()),"terms":terms,"failures":failures}
    write_atomic(a.out,report)
    if terms:
        profile=validate_profile({"schema_version":1,"language":"vi-VN","terms":[{"phrase":t["phrase"],"boost":t["boost"]} for t in terms]})
        write_atomic(a.profile,profile)
    print(json.dumps({"report":str(a.out),"profile":str(a.profile) if terms else None,"counts":report["counts"],"failures":len(failures)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
