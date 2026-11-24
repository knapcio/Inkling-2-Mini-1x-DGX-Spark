#!/usr/bin/env python3
"""lc_dump.py OUT.jsonl [IDS]: greedy outputs with per-token top-2 logprobs for speedbench-v1 prompts (default the
four longctx prompts whose hashes differed between the DSpark and DFlash2 stock arms; default = the 30
speedbench quick prompts). Same request as speedbench
quick (T=0, effort low, 160 tokens); each prompt is sent once to warm the prefix cache (as speedbench's prime) and
then measured. Compare two arms with lc_diverge.py."""
import hashlib
import json
import sys
import os
import urllib.request

TOPK = int(os.environ.get("LC_TOPK", "2"))  # more alternatives for divergence analysis

P = json.load(open(os.environ.get('SPEEDBENCH_PROMPTS', 'speedbench/prompts/speedbench-v1.json')))['prompts']
ids = (sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != 'quick'
       else [x['id'] for x in P if x.get('quick')])  # 'quick' (default) = the 30 speedbench quick prompts
out = open(sys.argv[1], 'w')


def call(p, n):
    body = {"model": "inkling-2-mini", "messages": p['messages'], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}, "logprobs": True, "top_logprobs": TOPK,
            "separate_reasoning": False}  # keep the reasoning tokens in content so logprobs cover every token
    req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=900))


for pid in ids:
    p = next(x for x in P if x['id'] == pid)
    call(p, 1)  # prime
    d = call(p, 160)
    c = d['choices'][0]
    m = c['message']
    txt = (m.get('reasoning_content') or '') + '\x00' + (m.get('content') or '')
    toks = [{'t': e['token'], 'lp': e['logprob'],
             'top': [(t['token'], t['logprob']) for t in (e.get('top_logprobs') or [])]}
            for e in ((c.get('logprobs') or {}).get('content') or [])]
    out.write(json.dumps({'id': pid, 'sha': hashlib.sha256(txt.encode()).hexdigest()[:16], 'text': txt,
                          'tokens': toks, 'usage': d.get('usage')}, ensure_ascii=False) + '\n')
    out.flush()
    print(pid, len(toks), flush=True)
