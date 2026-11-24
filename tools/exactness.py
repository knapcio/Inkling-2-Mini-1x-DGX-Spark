#!/usr/bin/env python3
"""exactness.py REF.jsonl ARM.jsonl [--max-gap 0.5]: greedy exactness verdict of a capped arm against the uncapped
reference, from lc_dump.py outputs (greedy, per-token top-2 logprobs, same prompts).

Per prompt: identical token sequence, or the FIRST divergence must be a near-tie: both arms' top-2 at that position are
the same two tokens (swapped choice) and the reference's top-1 minus top-2 logprob gap is <= --max-gap (default 0.5
nats; the round-2 precedent: gaps <= 0.24). Verdict line: EXACT (all identical), NEAR-TIES-ONLY (every difference is a
near-tie), FAIL (a difference that is not a near-tie, or a missing / failed prompt). Exit 0 unless FAIL."""
import argparse
import json
import sys


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("arm")
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--expect", type=int, default=30, help="prompts both dumps must hold (speedbench quick: 30)")
    a = ap.parse_args()
    R, A = load(a.ref), load(a.arm)
    ident = ties = wides = bad = 0
    rows = []
    if len(R) != a.expect or set(R) != set(A):
        print(f"VERDICT FAIL: incomplete dumps (ref {len(R)} prompts, arm {len(A)}, expected {a.expect} in both)")
        return 1
    for pid in sorted(set(R) | set(A)):
        if pid not in R or pid not in A or not R[pid]["tokens"] or not A[pid]["tokens"]:
            bad += 1
            rows.append(f"{pid}: MISSING")
            continue
        r, x = R[pid]["tokens"], A[pid]["tokens"]
        i = next((k for k in range(min(len(r), len(x))) if r[k]["t"] != x[k]["t"]), None)
        if i is None:
            if len(r) == len(x):
                ident += 1
                continue
            bad += 1
            rows.append(f"{pid}: same prefix, different lengths {len(r)} vs {len(x)}")
            continue
        tr, tx = r[i]["top"], x[i]["top"]
        gap = (tr[0][1] - tr[1][1]) if len(tr) > 1 else float("inf")
        # a near-tie is a SWAP: the reference emits its top-1, the arm emits exactly the reference's runner-up, and
        # the arm's top-2 is the same pair in the other order
        swap = (len(tr) > 1 and len(tx) > 1 and r[i]["t"] == tr[0][0] and x[i]["t"] == tr[1][0]
                and tx[0][0] == tr[1][0] and tx[1][0] == tr[0][0])
        ok = swap and gap <= a.max_gap
        # widened rule: each run's token within the other run's top-k and within max_gap of its top-1
        lr = dict((t, lp) for t, lp in tr)
        lx = dict((t, lp) for t, lp in tx)
        wide = (not ok and x[i]["t"] in lr and r[i]["t"] in lx and tr[0][1] - lr[x[i]["t"]] <= a.max_gap
                and tx[0][1] - lx[r[i]["t"]] <= a.max_gap)
        ties += ok
        wides += wide
        bad += not (ok or wide)
        rows.append(f"{pid}: first divergence at token {i}: ref {r[i]['t']!r} vs arm {x[i]['t']!r}, ref top-2 "
                    f"{[(t, round(lp, 3)) for t, lp in tr]}, arm top-2 {[(t, round(lp, 3)) for t, lp in tx]}, "
                    f"ref gap {gap:.3f} -> {'near-tie' if ok else ('near-tie (top-k)' if wide else 'NOT A NEAR-TIE')}")
    for row in rows:
        print(row)
    n = ident + ties + wides + bad
    verdict = "FAIL" if bad else ("EXACT" if ties + wides == 0 else "NEAR-TIES-ONLY")
    print(f"VERDICT {verdict}: {ident}/{n} identical, {ties} near-ties (top-2 swap), {wides} near-ties (top-k), "
          f"{bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
