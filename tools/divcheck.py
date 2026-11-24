#!/usr/bin/env python3
"""divcheck.py REF_DIR POL_DIR POLNR_DIR: the verdict on the longctx-04 divergence.

1. determinism: REF lc04a vs lc04b, POL lc04a vs lc04b (tokens and logprobs bitwise).
2. the first divergence REF vs POL on longctx-04 with top-5 logprobs: both tokens, their logprobs and ranks in each
   run, the top-5 lists, and the widened near-tie test (each arm's token in the other's top-5, within 0.5 nats of that
   run's top-1).
3. decisive: POL (remap on) vs POLNR (same policy, remap off => same schedule) on the 30 quick prompts, bitwise.
   Identical => the cap's divergence from REF is schedule-only (without the remap every committed token is the target
   argmax of its own fully computed row), i.e. floating-point order under the project rules."""
import json
import os
import sys


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def same(a, b):
    ta, tb = a["tokens"], b["tokens"]
    return len(ta) == len(tb) and all(x["t"] == y["t"] and x["lp"] == y["lp"] and x["top"] == y["top"]
                                      for x, y in zip(ta, tb))


def rank(top, tok):
    for i, (t, lp) in enumerate(top):
        if t == tok:
            return i + 1, lp
    return None, None


def main(ref_dir, pol_dir, polnr_dir):
    ok = True
    print("1. determinism (longctx-04, top-5, two dumps per boot)")
    for name, d in (("REF", ref_dir), ("POL", pol_dir)):
        try:
            a = load(os.path.join(d, "lc04a.jsonl"))["longctx-04"]
            b = load(os.path.join(d, "lc04b.jsonl"))["longctx-04"]
            print(f"   {name}: {'bit-identical' if same(a, b) else 'DIFFERENT'} across the two dumps")
        except Exception as exc:  # noqa: BLE001
            print(f"   {name}: missing ({exc!r})")
            ok = False
    print("2. first divergence REF vs POL on longctx-04 (top-5)")
    try:
        r = load(os.path.join(ref_dir, "lc04a.jsonl"))["longctx-04"]["tokens"]
        p = load(os.path.join(pol_dir, "lc04a.jsonl"))["longctx-04"]["tokens"]
        i = next((k for k in range(min(len(r), len(p))) if r[k]["t"] != p[k]["t"]), None)
        if i is None:
            print("   no divergence in this window (identical over the dump)")
        else:
            tr, tp = r[i]["top"], p[i]["top"]
            rk_p_in_r, lp_p_in_r = rank(tr, p[i]["t"])
            rk_r_in_p, lp_r_in_p = rank(tp, r[i]["t"])
            print(f"   token {i}: REF {r[i]['t']!r} ({r[i]['lp']:.3f}) vs POL {p[i]['t']!r} ({p[i]['lp']:.3f})")
            print(f"   REF top-5 {[(t, round(lp, 3)) for t, lp in tr]}")
            print(f"   POL top-5 {[(t, round(lp, 3)) for t, lp in tp]}")
            print(f"   POL's token in REF: rank {rk_p_in_r}, logprob {lp_p_in_r}; REF's token in POL: rank {rk_r_in_p}, "
                  f"logprob {lp_r_in_p}")
            near = (lp_p_in_r is not None and lp_r_in_p is not None and tr[0][1] - lp_p_in_r <= 0.5
                    and tp[0][1] - lp_r_in_p <= 0.5)
            print(f"   widened near-tie (top-5, 0.5 nats both ways): {'YES' if near else 'NO'}")
            drift = [abs(r[k]["lp"] - p[k]["lp"]) for k in range(i)]
            print(f"   shared prefix {i} tokens: |dlogprob| max {max(drift) if drift else 0:.3f}, "
                  f"nonzero on {sum(x > 1e-6 for x in drift)}")
    except Exception as exc:  # noqa: BLE001
        print(f"   missing ({exc!r})")
        ok = False
    print("3. decisive: POL vs POLNR (remap off, same schedule), 30 quick prompts, bitwise")
    try:
        a, b = load(os.path.join(pol_dir, "lcd.jsonl")), load(os.path.join(polnr_dir, "lcd.jsonl"))
        if len(a) != 30 or set(a) != set(b):
            print(f"   INCOMPLETE: {len(a)} vs {len(b)} prompts")
            ok = False
        else:
            diff = [pid for pid in sorted(a) if not same(a[pid], b[pid])]
            print(f"   {30 - len(diff)}/30 bit-identical" + (f"; different: {diff}" if diff else ""))
            print("VERDICT " + ("SCHEDULE-ONLY (remap not involved)" if not diff else "REMAP-DEPENDENT (investigate)"))
            ok = ok and not diff
    except Exception as exc:  # noqa: BLE001
        print(f"   missing ({exc!r})")
        print("VERDICT INCOMPLETE")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
