#!/usr/bin/env python3
"""Cross-fitted held-out replay of a verify-width policy on DFLASH full-width steplogs ("inkling-dflash-steplog/1").

    python3 tools/dfreplay.py DIR [DIR ...] [--json out.json] [--export-policy policy.json]
                              [--draft-ms 1=2.5,4=3.0,...] [--beta-scale 1.0] [--row-ms 0]

Model (all fit on training folds only; the report scores held-out clusters):
- labels: request-step with A accepted drafts (full width) -> drafts 0..A-1 accepted, draft A rejected (A < 7),
  later drafts unlabeled. a(mode, band, j, mbin, gbin) = P(d_j accepted | d_<j accepted, features of the walked
  row j), binned on m and gap; sparse cells borrow PSEUDO observations from (mode, j, mbin) and that from (mode, j).
- width rule (a stopping rule): S_j = a_0 * ... * a_j; draft j is verified while S_j >= theta_band * rowcost_band[j]
  (leading run); L = 1 + #verified drafts. theta is a price in tokens per ms; rowcost[j] is the modeled verify ms of
  row j + 1 (beta_band * mean marginal union of that row + row_ms).
- step time: t_draft + t_verify measured per step at full width, minus beta_band * sum_visits (U_8 - U_L) - row_ms *
  dead rows (+ policy overhead). beta_band (ms per expert of union) is regressed from t_verify on sum_visits U_8 over
  the training steps of the band (natural step-to-step variation). U_L is exact for mixed widths when the step kept
  its raw ids (bs <= 8); otherwise the uniform-width union curve at the mean width (interpolated).
- tokens: sum_r 1 + min(A_r, L_r - 1) on the captured trajectory.
- draft ms: measured per step, or replaced per band (--draft-ms) to price a faster draft (draft bytes).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FORMAT = "inkling-dflash-steplog/1"
POLICY_FORMAT = "inkling-vcap-policy/df1"
BANDS = (1, 2, 4, 8, 16, 24, 48)
M_EDGES = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
G_EDGES = (0.05, 0.2, 0.5, 0.8)
PSEUDO = 32.0
GAMMA = 7
STRIDE = 8


def band_of(bs):
    for i, b in enumerate(BANDS):
        if bs <= b:
            return i
    return len(BANDS) - 1


# ------------------------------------------------------------------ load
def load(dirs):
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "dflash1-*.npz")))
    if not files:
        raise SystemExit(f"no dflash1-*.npz under {dirs}")
    req = defaultdict(list)
    steps = defaultdict(list)
    raw = {}
    nreq = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        if str(z["format"]) != FORMAT:
            raise SystemExit(f"{f}: format {z['format']} != {FORMAT}")
        run = str(z["run"])
        n = len(z["rid"])
        bs = z["bs"]
        for key in ("rid", "context", "greedy", "temp", "t_eff", "rows", "m", "gap", "accepted"):
            req[key].append(z[key])
        for key in ("unary", "lse"):  # lattice fields (captures since the section-3 fields)
            if key in z.files:
                req[key].append(z[key])
        req["run"].append(np.full(n, run))
        for key in ("step", "bs", "union", "visits", "walks", "t_draft_ms", "t_verify_ms", "t_start_ms"):
            steps[key].append(z[key])
        steps["run"].append(np.full(len(bs), run))
        for k, r in zip(z["ids_req"], z["ids"]):
            raw[nreq + int(k)] = r
        nreq += n
    R = {k: np.concatenate(v) for k, v in req.items() if len(v) == len(files)}
    S = {k: np.concatenate(v) for k, v in steps.items()}
    R["raw"] = raw
    return R, S


def features(R, t_override=None):
    """(m, gap) recomputed from the walked rows (at t_eff, or a fixed T for every row)."""
    rows = R["rows"].astype(np.float64)
    t = R["t_eff"] if t_override is None else np.full(len(rows), t_override)
    x = rows / t[:, None, None]
    x = x - x.max(axis=-1, keepdims=True)
    q = np.exp(x)
    q /= q.sum(axis=-1, keepdims=True)
    top = -np.sort(-q, axis=-1)[..., :2]
    return top[..., 0], top[..., 0] - top[..., 1]


# ------------------------------------------------------------------ calibration
TAIL_EDGES = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
PREV_EDGES = (0, 1, 2, 3, 5)  # previous step's accepted drafts of the request; -1 (first step) -> bin 0
FEATURES = ("slot", "m", "gap", "tail", "mt", "prev")


def feature_bins(R, m, gap):
    """{name: ([n_req, G] int bins, n_bins)} for every feature that the data supports. All are stopping-rule
    legal: slot index; m, gap of the walked row j (chosen by d_{j-1}); tail = 1 - sum_k exp(unary_jk - lse_j) of slot
    j's candidate plane (the draft's hidden at j only); mt = m * (1 - tail); prev = the request's accepted drafts at
    its previous logged step (known before drafting)."""
    n = len(m)
    out = {"slot": (np.broadcast_to(np.arange(GAMMA), (n, GAMMA)), GAMMA),
           "m": (np.digitize(m, M_EDGES), len(M_EDGES) + 1),
           "gap": (np.digitize(gap, G_EDGES), len(G_EDGES) + 1)}
    if "unary" in R and "lse" in R:
        tail = 1.0 - np.exp(R["unary"] - R["lse"][..., None]).sum(-1)
        nan = ~np.isfinite(tail)
        tb = np.digitize(np.nan_to_num(tail, nan=0.0), TAIL_EDGES)
        tb[nan] = len(TAIL_EDGES) + 1  # own bin for "plane not produced"
        out["tail"] = (tb, len(TAIL_EDGES) + 2)
        mt = m * (1.0 - np.nan_to_num(tail, nan=0.0))
        out["mt"] = (np.digitize(mt, M_EDGES), len(M_EDGES) + 1)
    prev = np.full(n, -1, dtype=np.int64)
    last = {}
    for i in range(n):  # files are read in name order = run, then chunk: chronological within a run
        key = (R["run"][i], R["rid"][i])
        prev[i] = last.get(key, -1)
        last[key] = int(R["accepted"][i])
    pb = np.where(prev < 0, 0, 1 + np.digitize(prev, PREV_EDGES[1:]))
    out["prev"] = (np.broadcast_to(pb[:, None], (n, GAMMA)), len(PREV_EDGES) + 1)
    return out


class Calib:
    """a = P(d_j accepted | d_<j accepted, cell): a survival table over at most 3 binned features (a design limit),
    stratified by greedy / sampled (and optionally by batch band). Sparse cells borrow PSEUDO observations from the
    table without the last feature, recursively down to the stratum."""

    def __init__(self, features=("slot", "m", "gap"), band=False):
        if len(features) > 3:
            raise ValueError("at most 3 binned features (a design limit)")
        self.features = tuple(features)
        self.band = band

    def _cells(self, R, idx, bins):
        n = len(idx)
        mode = (~R["greedy"][idx]).astype(np.int64)
        strata = mode
        nstrata = 2
        if self.band:
            band = np.array([band_of(b) for b in R["bs_req"][idx]], dtype=np.int64)
            strata = mode * len(BANDS) + band
            nstrata = 2 * len(BANDS)
        levels = [np.broadcast_to(strata[:, None], (n, GAMMA)).astype(np.int64)]
        sizes = [nstrata]
        cell, size = levels[0], nstrata
        for f in self.features:
            b, nb = bins[f]
            cell = cell * nb + b[idx]
            size *= nb
            levels.append(cell)
            sizes.append(size)
        return levels, sizes

    def fit(self, R, idx, bins):
        levels, sizes = self._cells(R, idx, bins)
        acc = R["accepted"][idx].astype(int)
        j = np.arange(GAMMA)
        lab = acc[:, None] >= j[None, :]      # reached draft j: labeled
        ok = (acc[:, None] > j[None, :]) & lab
        self.tables = []
        est_parent = None
        for li, (cell, size) in enumerate(zip(levels, sizes)):
            nn = np.bincount(cell[lab], minlength=size).astype(float)
            kk = np.bincount(cell[ok], minlength=size).astype(float)
            if li == 0:
                est = (kk + 1.0) / (nn + 2.0)
            else:
                nb = size // sizes[li - 1]
                prior = np.repeat(est_parent, nb)
                est = (kk + PSEUDO * prior) / (nn + PSEUDO)
            self.tables.append(est)
            est_parent = est
        self.support = nn
        return self

    def survival(self, R, idx, bins):
        levels, _ = self._cells(R, idx, bins)
        a = self.tables[-1][levels[-1]]
        return np.cumprod(a, axis=1)


# ------------------------------------------------------------------ cost
def union_for(widths, raw_rows, union_uniform):
    """sum over visits of the union of rows 0..L_r-1 of every request (exact from raw ids), or interpolated."""
    if raw_rows is not None:
        total = 0
        for v in range(raw_rows[0].shape[0]):
            if raw_rows[0][v, 0, 0] == 255:  # past the step's last MoE visit
                break
            block = np.concatenate([r[v, :L].reshape(-1) for L, r in zip(widths, raw_rows)])
            total += len(np.unique(block[block != 255]))
        return float(total)
    u = union_uniform[union_uniform[:, 0] >= 0].astype(float)  # [visits, 8]
    w = float(np.mean(widths))
    lo = int(np.floor(w))
    hi = min(STRIDE, lo + 1)
    f = w - lo
    col = lambda L: u[:, L - 1].sum()  # noqa: E731
    return col(lo) if lo == hi else (1 - f) * col(lo) + f * col(hi)


def cached_union(R, S, s, widths):
    key = (s, widths)
    hit = R["ucache"].get(key)
    if hit is None:
        rows = R["by_step"][s]
        raw = [R["raw"].get(int(r)) for r in rows]
        raw = raw if all(x is not None for x in raw) else None
        hit = R["ucache"][key] = union_for(np.asarray(widths), raw, S["union"][s])
    return hit


def fit_beta(S, sidx_train):
    """ms of verify per expert of summed union (full width), per band: OLS slope of t_verify on sum_v U_8."""
    beta = {}
    for b in range(len(BANDS)):
        sel = [i for i in sidx_train if band_of(int(S["bs"][i])) == b and np.isfinite(S["t_verify_ms"][i])]
        if len(sel) < 30:
            continue
        u8 = np.array([S["union"][i][S["union"][i][:, 0] >= 0, 7].sum() for i in sel], dtype=float)
        t = S["t_verify_ms"][sel].astype(float)
        if u8.std() < 1e-6:
            continue
        A = np.c_[u8, np.ones_like(u8)]
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        pred = A @ coef
        r2 = 1 - ((t - pred) ** 2).sum() / max(1e-9, ((t - t.mean()) ** 2).sum())
        beta[b] = dict(beta=float(coef[0]), intercept=float(coef[1]), r2=float(r2), n=len(sel))
    return beta


# ------------------------------------------------------------------ replay
def widths_for(surv, theta, rowcost):
    ok = surv >= theta * rowcost[None, :]
    return 1 + np.cumprod(ok, axis=1).sum(axis=1)


def replay(R, S, steps_idx, surv_all, policy, args, beta):
    """(tokens, ms) for the full-width baseline and the policy over the given steps."""
    by_step = R["by_step"]
    out = dict(tok_full=0.0, ms_full=0.0, tok_pol=0.0, ms_pol=0.0, width=0.0, nreq=0)
    for s in steps_idx:
        rows = by_step[s]
        bs = len(rows)
        b = band_of(bs)
        acc = R["accepted"][rows].astype(int)
        draft = S["t_draft_ms"][s] if b not in args.draft_ms else args.draft_ms[b]
        t_full = draft + S["t_verify_ms"][s]
        out["tok_full"] += float((1 + acc).sum())
        out["ms_full"] += t_full
        th = policy["theta"].get(b, 0.0)
        if th <= 0 or bs > policy["bsmax"]:
            L = np.full(bs, STRIDE)
        else:
            L = widths_for(surv_all[rows], th, policy["rowcost"][b])
        bt = beta.get(b, {}).get("beta", args.beta_default) * args.beta_scale
        u8 = cached_union(R, S, s, (STRIDE,) * bs)
        uL = cached_union(R, S, s, tuple(int(x) for x in L))
        saving = bt * (u8 - uL) + args.row_ms * float((STRIDE - L).sum())
        overhead = args.policy_ms if th > 0 else 0.0
        out["tok_pol"] += float((1 + np.minimum(acc, L - 1)).sum())
        out["ms_pol"] += t_full - saving + overhead
        out["width"] += float(L.sum())
        out["nreq"] += bs
    return out


def mean_rowcost(S, steps_idx, beta, args):
    """per band: modeled ms of row j+1 = beta * mean marginal uniform union per request + row_ms."""
    rc = {}
    for b in range(len(BANDS)):
        sel = [s for s in steps_idx if band_of(int(S["bs"][s])) == b]
        if not sel:
            continue
        marg = np.zeros(GAMMA)
        for s in sel:
            u = S["union"][s]
            u = u[u[:, 0] >= 0].astype(float).sum(axis=0)
            marg += np.diff(u) / int(S["bs"][s])
        marg /= len(sel)
        bt = beta.get(b, {}).get("beta", args.beta_default) * args.beta_scale
        rc[b] = bt * marg + args.row_ms
    return rc


def clusters(R, S):
    """connected components of requests that shared a step (union-find over (run, rid))."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, rows in enumerate(R["by_step"]):
        keys = [(R["run"][r], R["rid"][r]) for r in rows]
        for k in keys[1:]:
            parent[find(k)] = find(keys[0])
        find(keys[0])
    comp = {}
    step_comp = np.zeros(len(R["by_step"]), dtype=np.int64)
    for s, rows in enumerate(R["by_step"]):
        root = find((R["run"][rows[0]], R["rid"][rows[0]]))
        step_comp[s] = comp.setdefault(root, len(comp))
    return step_comp


def search_theta(R, S, train, surv_all, beta, rowcost, args):
    policy = dict(theta={}, rowcost=rowcost, bsmax=args.bsmax)
    grid = np.r_[0.0, np.geomspace(1e-4, 2.0, 40)]
    for b in sorted(rowcost):
        sel = [s for s in train if band_of(int(S["bs"][s])) == b]
        if len(sel) < args.min_steps:
            policy["theta"][b] = 0.0
            continue
        best, best_rate = 0.0, None
        for th in grid:
            policy["theta"][b] = th
            o = replay(R, S, sel, surv_all, policy, args, beta)
            rate = o["tok_pol"] / o["ms_pol"]
            if best_rate is None or rate > best_rate * (1 + 1e-9):
                best, best_rate = th, rate
        policy["theta"][b] = best
    return policy


def prepare(R, S):
    by_step = []
    start = 0
    for bs in S["bs"]:
        by_step.append(np.arange(start, start + int(bs)))
        start += int(bs)
    R["by_step"] = by_step
    R["bs_req"] = np.repeat(S["bs"], S["bs"])
    R["ucache"] = {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--export-policy")
    ap.add_argument("--draft-ms", default="", help="band_bs=ms,... replaces the measured draft time per band")
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--beta-default", type=float, default=0.0, help="ms/expert where a band has no fit")
    ap.add_argument("--row-ms", type=float, default=0.0, help="extra ms saved per dead row (NOEXP rows)")
    ap.add_argument("--policy-ms", type=float, default=0.05)
    ap.add_argument("--bsmax", type=int, default=48)
    ap.add_argument("--min-steps", type=int, default=60)
    ap.add_argument("--visits", type=int, default=48, help="MoE visits of a complete step (Inkling-2-Mini: 48)")
    ap.add_argument("--no-gap", action="store_true", help="ablation: calibrate on m only (one gap bin)")
    ap.add_argument("--features", default="slot,m,gap", help=f"<= 3 of {','.join(FEATURES)}")
    ap.add_argument("--band-calib", action="store_true", help="also stratify the calibration by batch band")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args(argv)
    args.draft_ms = {band_of(int(k)): float(v) for k, v in
                     (kv.split("=") for kv in args.draft_ms.split(",") if kv)}

    R, S = load(args.dirs)
    prepare(R, S)
    m, gap = features(R)
    if args.no_gap:
        gap = np.zeros_like(gap)
    bins = feature_bins(R, m, gap)
    feats = tuple(f for f in args.features.split(",") if f)
    missing = [f for f in feats if f not in bins]
    if missing:
        raise SystemExit(f"features {missing} are not in this capture (available: {sorted(bins)})")
    check = float(np.abs(m - R["m"]).max()) if len(m) else 0.0
    visits = args.visits  # complete steps: every MoE visit journaled (partial journals are written as 0)
    good_steps = np.nonzero((S["walks"] == 1) & (S["visits"] == visits) & np.isfinite(S["t_verify_ms"])
                            & (S["union"][:, 0, 0] >= 0))[0]
    comp = clusters(R, S)
    rng = np.random.default_rng(args.seed)
    ncomp = int(comp.max()) + 1
    fold_of = rng.integers(0, args.folds, ncomp)
    report = dict(input=FORMAT, files=None, request_steps=int(len(R["rid"])), steps=int(len(S["bs"])),
                  usable_steps=int(len(good_steps)), visits=visits, clusters=ncomp, features=list(feats),
                  band_calib=bool(args.band_calib), m_recompute_maxerr=check, folds=[])
    held = defaultdict(lambda: defaultdict(float))
    per_cluster = defaultdict(lambda: np.zeros(4))
    for f in range(args.folds):
        train = [s for s in good_steps if fold_of[comp[s]] != f]
        test = [s for s in good_steps if fold_of[comp[s]] == f]
        tr_rows = np.concatenate([R["by_step"][s] for s in train]) if train else np.zeros(0, int)
        calib = Calib(feats, band=args.band_calib).fit(R, tr_rows, bins)
        beta = fit_beta(S, train)
        rowcost = mean_rowcost(S, train, beta, args)
        surv_all = calib.survival(R, np.arange(len(R["rid"])), bins)
        policy = search_theta(R, S, train, surv_all, beta, rowcost, args)
        report["folds"].append(dict(theta={BANDS[b]: t for b, t in policy["theta"].items()},
                                    beta={BANDS[b]: v for b, v in beta.items()}))
        for s in test:
            o = replay(R, S, [s], surv_all, policy, args, beta)
            for key in ("all", f"bs={BANDS[band_of(int(S['bs'][s]))]}",
                        "greedy" if R["greedy"][R["by_step"][s]].all() else "sampled/mixed"):
                for k2, v in o.items():
                    held[key][k2] += v
            per_cluster[comp[s]] += [o["tok_full"], o["ms_full"], o["tok_pol"], o["ms_pol"]]
    scores = {}
    keys = np.array(sorted(per_cluster))
    mat = np.array([per_cluster[k] for k in keys]) if len(keys) else np.zeros((0, 4))
    for key, o in sorted(held.items()):
        if o["ms_full"] <= 0:
            continue
        full = 1000 * o["tok_full"] / o["ms_full"]
        pol = 1000 * o["tok_pol"] / o["ms_pol"]
        scores[key] = dict(full_tok_s=full, policy_tok_s=pol, gain=pol / full - 1,
                           tokens_ratio=o["tok_pol"] / o["tok_full"], ms_ratio=o["ms_pol"] / o["ms_full"],
                           mean_width=o["width"] / max(1, o["nreq"]))
    if len(mat) > 1 and "all" in scores:
        boots = []
        for _ in range(args.bootstrap):
            pick = mat[rng.integers(0, len(mat), len(mat))].sum(axis=0)
            boots.append((pick[2] / pick[3]) / (pick[0] / pick[1]) - 1)
        scores["all"]["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    report["scores"] = scores
    # context: fixed widths K for every request (beta from all usable steps), per band
    beta_all = fit_beta(S, list(good_steps))
    fixed = {}
    for K in range(1, STRIDE + 1):
        per = defaultdict(lambda: np.zeros(2))
        for s in good_steps:
            rows = R["by_step"][s]
            bs = len(rows)
            b = band_of(bs)
            acc = R["accepted"][rows].astype(int)
            draft = S["t_draft_ms"][s] if b not in args.draft_ms else args.draft_ms[b]
            bt = beta_all.get(b, {}).get("beta", args.beta_default) * args.beta_scale
            saving = bt * (cached_union(R, S, s, (STRIDE,) * bs) - cached_union(R, S, s, (K,) * bs)) \
                + args.row_ms * (STRIDE - K) * bs
            t = draft + S["t_verify_ms"][s] - saving
            for key in ("all", f"bs={BANDS[b]}"):
                per[key] += [float((1 + np.minimum(acc, K - 1)).sum()), t]
        fixed[K] = {k: 1000 * v[0] / v[1] for k, v in per.items()}
    report["fixed_width_tok_s"] = fixed
    report["beta_all"] = {BANDS[b]: v for b, v in beta_all.items()}
    if args.export_policy:
        rows = np.concatenate([R["by_step"][s] for s in good_steps])
        calib = Calib(feats, band=args.band_calib).fit(R, rows, bins)
        beta = fit_beta(S, list(good_steps))
        rowcost = mean_rowcost(S, list(good_steps), beta, args)
        surv_all = calib.survival(R, np.arange(len(R["rid"])), bins)
        policy = search_theta(R, S, list(good_steps), surv_all, beta, rowcost, args)
        with open(args.export_policy, "w") as fh:
            json.dump(dict(format=POLICY_FORMAT + "+features", stride=STRIDE, bands=list(BANDS),
                           features=list(feats), band_calib=bool(args.band_calib),
                           edges=dict(m=list(M_EDGES), gap=list(G_EDGES), tail=list(TAIL_EDGES), mt=list(M_EDGES),
                                      prev=list(PREV_EDGES)),
                           calib=calib.tables[-1].tolist(),
                           theta=[[BANDS[b], float(t)] for b, t in sorted(policy["theta"].items())],
                           rowcost={BANDS[b]: v.tolist() for b, v in rowcost.items()}, bsmax=args.bsmax,
                           source=dict(dirs=args.dirs, draft_ms={BANDS[b]: v for b, v in args.draft_ms.items()},
                                       beta={BANDS[b]: v for b, v in beta.items()}, beta_scale=args.beta_scale,
                                       row_ms=args.row_ms)), fh)
    text = json.dumps(report, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(json.dumps(dict(scores=report["scores"], fixed=report["fixed_width_tok_s"],
                          beta=report["beta_all"]), indent=1, default=float))


if __name__ == "__main__":
    main()
