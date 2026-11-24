#!/usr/bin/env python3
"""cmp.py A_DIR [B_DIR]: speedbench quick per category (geomean tok/s, median tokens/step, median step ms), greedy
sha256 matches, decode table (prose/code) and thinking for one or two r2 arm directories."""
import json
import os
import sys

dirs = sys.argv[1:]


def J(d, f):
    p = os.path.join(d, f)
    return json.load(open(p)) if os.path.exists(p) else None


cats = ["ALL", "prose", "code", "structured", "agentic", "multilingual", "longctx"]
sb = {d: J(d, "sbquick.json") for d in dirs}
print("speedbench quick: geomean tok/s | median tokens/step | median step ms")
for c in cats:
    row = [f"{c:12s}"]
    for d in dirs:
        s = (sb[d] or {}).get("summary", {}).get(c) or {}
        row.append(f"{s.get('geomean_tok_s', float('nan')):7.2f} {s.get('median_tokens_per_step', float('nan')):6.3f} "
                   f"{s.get('median_step_ms', float('nan')):6.2f}")
    if len(dirs) == 2:
        a = (sb[dirs[0]] or {}).get("summary", {}).get(c) or {}
        b = (sb[dirs[1]] or {}).get("summary", {}).get(c) or {}
        try:
            row.append(f"ratio tok/s {b['geomean_tok_s'] / a['geomean_tok_s']:.3f} tps {b['median_tokens_per_step'] / a['median_tokens_per_step']:.3f}"
                       f" step {b['median_step_ms'] - a['median_step_ms']:+.2f} ms")
        except Exception:
            pass
    print("  ".join(row))
for d in dirs:
    print(d, "failed", (sb[d] or {}).get("summary", {}).get("failed"))
if len(dirs) == 2 and all(sb.values()):
    ha = {r["key"]: r.get("sha256") for r in sb[dirs[0]]["requests"] if r.get("kind") == "measure"}
    hb = {r["key"]: r.get("sha256") for r in sb[dirs[1]]["requests"] if r.get("kind") == "measure"}
    keys = sorted(set(ha) & set(hb))
    same = [k for k in keys if ha[k] == hb[k]]
    print(f"greedy sha256 equal {len(same)}/{len(keys)}; differ: {[k for k in keys if ha[k] != hb[k]]}")
for d in dirs:
    c = J(d, "decode-pc.json")
    if c and isinstance(c.get("table"), dict):
        cells = []
        for typ, lv in c["table"].items():
            for k, cell in lv.items():
                v = cell.get("per_stream_tok_s") if str(k) == "1" else cell.get("aggregate_tok_s")
                cells.append(f"{typ}c{k}={v}")
        print(os.path.basename(d.rstrip('/')), "decode", " ".join(cells))
    p = os.path.join(d, "think.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                t = json.loads(line)
                print(os.path.basename(d.rstrip('/')), "think", t["label"], "agg", t["aggregate_tok_s"], "per-req median",
                      t.get("per_request_decode_tok_s_median"), "failed", t.get("failed"))
            except Exception:
                pass
    p = os.path.join(d, "startup.json")
    s = json.loads(open(p).readline()) if os.path.exists(p) else None  # line 2 is appended at the end of the arm
    if s:
        print(os.path.basename(d.rstrip('/')), "startup_s", s.get("startup_s"), "memavail_ready", s.get("memavail_ready_gib"),
              s.get("kv", "")[:160])
