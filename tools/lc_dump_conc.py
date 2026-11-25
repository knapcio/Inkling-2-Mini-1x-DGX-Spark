#!/usr/bin/env python3
"""lc_dump_conc.py OUT.jsonl N [ROUNDS]: lc_dump.py's greedy request (T=0, effort low, 160 tokens, top-2 logprobs)
for the 30 speedbench quick prompts with N requests in flight, so the target-verify graph runs at off-bucket batch
sizes (N=9: the c10 graph with one padded request; the tail of the round passes 8..1). Each prompt is primed once
(prefix cache) before the timed round. Output records as lc_dump.py, plus 'round'; compare two arms with
lc_diverge.py / cmp.py by (id, round)."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request

TOPK = int(os.environ.get("LC_TOPK", "2"))
P = json.load(open(os.environ.get("SPEEDBENCH_PROMPTS", "speedbench/prompts/speedbench-v1.json")))["prompts"]
ids = [x["id"] for x in P if x.get("quick")]
out_path, n = sys.argv[1], int(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def call(p, tokens):
    body = {"model": "inkling-2-mini", "messages": p["messages"], "max_tokens": tokens, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}, "logprobs": True, "top_logprobs": TOPK,
            "separate_reasoning": False}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def one(pid):
    p = next(x for x in P if x["id"] == pid)
    d = call(p, 160)
    c = d["choices"][0]
    m = c["message"]
    txt = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    toks = [{"t": e["token"], "lp": e["logprob"],
             "top": [(t["token"], t["logprob"]) for t in (e.get("top_logprobs") or [])]}
            for e in ((c.get("logprobs") or {}).get("content") or [])]
    return {"id": pid, "sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "text": txt, "tokens": toks,
            "usage": d.get("usage")}


with cf.ThreadPoolExecutor(n) as ex:
    list(ex.map(lambda pid: call(next(x for x in P if x["id"] == pid), 1), ids))  # prime
    with open(out_path, "w") as out:
        for r in range(rounds):
            for rec in ex.map(one, ids):
                rec["round"] = r
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"round {r}: {len(ids)} prompts at concurrency {n}", flush=True)
