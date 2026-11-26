#!/usr/bin/env python3
"""ctx8_lcpair.py A_DIR B_DIR [--min-prompt-len 16000] [--split 65536] [--json out.json]: per-prompt paired tokens per
step and drafted-token agreement between two DFLASH steplog captures ("inkling-dflash-steplog/1") of the same greedy
long-context prompts (e.g. speedbench longctx with a fixed --salt-nonce), one arm with a draft-side change and one
without.

Requests are paired by (prompt_hash, prompt_len, k), k = the request's rank among the requests of that prompt in its
capture, in send order (longctx sends each document cold, then prefix-cached). For each pair the two committed
token streams are compared; n = their common length before the first differing token (greedy output is the target's
argmax chain, so the streams agree up to a verify-schedule near-tie, if any). Over those n tokens:
- tokens per step = n / (steps that start inside the first n tokens), per arm; reported per request (the B/A step
  ratio) and pooled per bucket (prompt_len below / at or above --split, i.e. 32K and 128K);
- drafted-token agreement per slot on the steps that start at the same context in both arms (same target prefix).
Only greedy requests of at least --min-prompt-len prompt tokens with complete step chains (no missing step, no -1 id)
are used. Run it on two A-arm captures (or two B-arm captures) for the noise floor."""
import argparse
import glob
import json
import os
import sys

import numpy as np

FORMAT = "inkling-dflash-steplog/1"


def load(d, min_len):
    files = sorted(glob.glob(os.path.join(d, "**", "dflash1-*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"no dflash1-*.npz under {d}")
    reqs = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        if str(z["format"]) != FORMAT:
            raise SystemExit(f"{f}: format {z['format']} != {FORMAT}")
        step = np.repeat(z["step"], z["bs"])
        walks = np.repeat(z["walks"], z["bs"])
        for i in np.flatnonzero(z["greedy"] & (z["commit_len"] > 0) & (z["prompt_len"] >= min_len)):
            cl = int(z["commit_len"][i])
            t = z["committed"][i][:cl]
            r = reqs.setdefault(str(z["rid"][i]), {"p": (int(z["prompt_hash"][i]), int(z["prompt_len"][i])),
                                                   "first": int(step[i]), "s": []})
            r["first"] = min(r["first"], int(step[i]))
            r["s"].append((int(z["context"][i]), None if (t < 0).any() else t.astype(np.int64),
                           z["drafted"][i].copy() if walks[i] == 1 else None))
    out, incomplete = {}, 0
    by_prompt = {}
    for r in reqs.values():
        r["s"].sort(key=lambda x: x[0])
        s = r["s"]
        if any(t is None for _, t, _ in s) or any(s[j + 1][0] != s[j][0] + len(s[j][1]) for j in range(len(s) - 1)):
            incomplete += 1
            continue
        r["start"] = s[0][0]
        r["tok"] = np.concatenate([t for _, t, _ in s])
        by_prompt.setdefault(r["p"], []).append(r)
    for p, rs in by_prompt.items():
        for k, r in enumerate(sorted(rs, key=lambda x: x["first"])):
            out[(p[0], p[1], k)] = r
    return out, incomplete


def compare(A, B, split):
    keys = sorted(set(A) & set(B))
    buckets = {}
    for key in keys:
        a, b = A[key], B[key]
        if a["start"] != b["start"]:
            continue
        n = min(len(a["tok"]), len(b["tok"]))
        diff = np.flatnonzero(a["tok"][:n] != b["tok"][:n])
        diverged = len(diff) > 0
        if diverged:
            n = int(diff[0])
        if n < 8:
            continue
        end = a["start"] + n
        sa = [x for x in a["s"] if x[0] < end]
        sb = [x for x in b["s"] if x[0] < end]
        da = {c: d for c, _, d in sa if d is not None}
        db = {c: d for c, _, d in sb if d is not None}
        common = sorted(set(da) & set(db))
        bk = "lt" if key[1] < split else "ge"
        B_ = buckets.setdefault(bk, {"req": [], "tok": 0, "steps_a": 0, "steps_b": 0, "slot_eq": None, "slot_n": None,
                                      "block_eq": 0, "matched": 0, "diverged": 0})
        B_["req"].append({"prompt_len": key[1], "k": key[2], "n": n, "tps_a": n / len(sa), "tps_b": n / len(sb),
                          "diverged": diverged})
        B_["tok"] += n
        B_["steps_a"] += len(sa)
        B_["steps_b"] += len(sb)
        B_["diverged"] += diverged
        for c in common:
            x, y = da[c], db[c]
            ok = (x >= 0) & (y >= 0)
            eq = (x == y) & ok
            if B_["slot_eq"] is None:
                B_["slot_eq"] = np.zeros(len(x))
                B_["slot_n"] = np.zeros(len(x))
            B_["slot_eq"] += eq
            B_["slot_n"] += ok
            B_["block_eq"] += bool(eq.all() and ok.all())
            B_["matched"] += 1
    res = {}
    for bk, v in buckets.items():
        ratios = [r["tps_b"] / r["tps_a"] for r in v["req"]]
        res[bk] = {
            "requests": len(v["req"]), "diverged": v["diverged"], "tokens": v["tok"],
            "tps_a": v["tok"] / v["steps_a"], "tps_b": v["tok"] / v["steps_b"],
            "tps_b_over_a_pooled": v["steps_a"] / v["steps_b"],
            "tps_b_over_a_median_request": float(np.median(ratios)),
            "requests_b_higher": int(sum(x > 1 for x in ratios)),
            "matched_steps": v["matched"],
            "slot_agreement": [] if v["slot_eq"] is None else (v["slot_eq"] / np.maximum(v["slot_n"], 1)).tolist(),
            "block_agreement": v["block_eq"] / max(1, v["matched"]),
            "per_request": v["req"],
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-prompt-len", type=int, default=16000)
    ap.add_argument("--split", type=int, default=65536)
    ap.add_argument("--json", default="")
    x = ap.parse_args()
    A, ia = load(x.a, x.min_prompt_len)
    B, ib = load(x.b, x.min_prompt_len)
    res = compare(A, B, x.split)
    print(f"A {x.a}: {len(A)} long requests ({ia} incomplete); B {x.b}: {len(B)} ({ib} incomplete)")
    for bk in sorted(res):
        r = res[bk]
        name = f"prompt < {x.split}" if bk == "lt" else f"prompt >= {x.split}"
        print(f"{name}: {r['requests']} paired requests ({r['diverged']} diverge before their end), {r['tokens']} tokens; "
              f"tokens/step A {r['tps_a']:.4f} B {r['tps_b']:.4f}: B/A pooled {r['tps_b_over_a_pooled']:.4f}, median "
              f"per request {r['tps_b_over_a_median_request']:.4f}, B higher on {r['requests_b_higher']}/{r['requests']}")
        print("  drafted agreement per slot: " + " ".join(f"{100 * s:.1f}" for s in r["slot_agreement"])
              + f" %; whole block {100 * r['block_agreement']:.1f} % over {r['matched_steps']} matched steps")
    if x.json:
        with open(x.json, "w") as fh:
            json.dump(res, fh, indent=1)
    if not res:
        sys.exit(1)


if __name__ == "__main__":
    main()
