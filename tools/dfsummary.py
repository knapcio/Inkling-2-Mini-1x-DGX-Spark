#!/usr/bin/env python3
"""Descriptive summary of DFLASH steplogs (complete steps only: walks == 1, visits == 48):

    python3 tools/dfsummary.py DIR [DIR ...] [--visits 48]

Per mode (greedy / sampled): request-steps, tokens/step, conditional acceptance per slot. Per batch size: steps,
median draft / verify / period ms, tokens/step, summed full-width union and the uniform union curve (fraction of
union(8) at widths 1..8). Feature separation: AUC of each per-slot feature for "draft j accepted | reached j"."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfreplay  # noqa: E402


def auc(score, label):
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--visits", type=int, default=48)
    a = ap.parse_args(argv)
    R, S = dfreplay.load(a.dirs)
    dfreplay.prepare(R, S)
    good = (S["walks"] == 1) & (S["visits"] == a.visits)
    req_good = np.repeat(good, S["bs"])
    out = {"steps": int(len(good)), "complete_steps": int(good.sum()), "request_steps": int(req_good.sum())}
    acc = R["accepted"].astype(int)
    for mode, sel in (("greedy", R["greedy"] & req_good), ("sampled", ~R["greedy"] & req_good)):
        if not sel.any():
            continue
        a_ = acc[sel]
        cond = [float((a_ > j).sum() / max(1, (a_ >= j).sum())) for j in range(dfreplay.GAMMA)]
        out[mode] = {"request_steps": int(sel.sum()), "tokens_per_step": float((1 + a_).mean()),
                     "cond_accept_per_slot": [round(c, 4) for c in cond]}
    per_bs = {}
    period = np.diff(S["t_start_ms"])
    for bs in sorted(set(S["bs"].tolist())):
        idx = np.nonzero(good & (S["bs"] == bs))[0]
        if len(idx) < 5:
            continue
        rows = np.concatenate([R["by_step"][s] for s in idx])
        u = np.stack([S["union"][s][:a.visits].astype(float).sum(0) for s in idx])
        nxt = idx[idx < len(period)]
        per_bs[int(bs)] = {
            "steps": int(len(idx)), "tokens_per_step": float((1 + acc[rows]).sum() / len(idx)),
            "draft_ms": float(np.median(S["t_draft_ms"][idx])), "verify_ms": float(np.median(S["t_verify_ms"][idx])),
            "period_ms_p50": float(np.median(period[nxt])) if len(nxt) else None,
            "union8_sum": float(np.median(u[:, -1])),
            "union_curve": [round(float(x), 3) for x in np.median(u / u[:, -1:], axis=0)]}
    out["per_bs"] = per_bs
    m, gap = dfreplay.features(R)
    bins = dfreplay.feature_bins(R, m, gap)
    feats = {"m": m, "gap": gap}
    if "unary" in R and "lse" in R:
        tail = 1.0 - np.exp(R["unary"] - R["lse"][..., None]).sum(-1)
        feats["tail(-)"] = -tail
        feats["m*(1-tail)"] = m * (1 - tail)
    feats["prev"] = bins["prev"][0].astype(float)
    sep = {}
    for mode, msel in (("greedy", R["greedy"] & req_good), ("sampled", ~R["greedy"] & req_good)):
        if not msel.any():
            continue
        sep[mode] = {}
        for name, f in feats.items():
            vals = []
            for j in range(dfreplay.GAMMA):
                reach = msel & (acc >= j)
                x = f[reach, j] if f.ndim == 2 else f[reach]
                ok = np.isfinite(x)
                vals.append(round(auc(x[ok], (acc[reach] > j)[ok]), 3))
            sep[mode][name] = vals
    out["auc_per_slot"] = sep
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
