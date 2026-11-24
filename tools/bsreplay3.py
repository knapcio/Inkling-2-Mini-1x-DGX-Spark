#!/usr/bin/env python3
"""Cross-fit the Inkling-2-Mini verify-width policy on full-width steplogs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from bsreplay import EDGES, GAMMA, STRIDE, width_from

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "overlay"))
from inkling_vcap.policyfile import RICH_FORMAT, load_policy, parse_policy

SHIPPED = ROOT / "overlay/inkling_vcap/policies/r2-theta004.json"
LOG_FORMAT = "inkling-vcap-steplog/3"
BANDS = (1, 4, 8, 16, 24)
CONTEXT = (2048, 8192, 32768)
TEMPERATURE = (0.6, 0.9, 1.1)
GAPS = (0.05, 0.2, 0.5, 0.8)
PRICES = np.r_[0.0, np.geomspace(0.003, 32.0, 48)]
# From a c1/c4/c8/c24 step profile (v2.6). Greedy exclusive MoE GEMM ms.
PROFILE_BS = (1, 4, 8, 24)
PROFILE_WALL = (34.604, 63.497, 84.025, 153.720)
PROFILE_MOE = (16.376, 39.435, 52.295, 95.754)
SAMPLED_BS = (1, 8, 24)
SAMPLED_EXTRA = (43.934 - 34.604, 102.633 - 84.025, 154.195 - 153.720)


@dataclass
class Batch:
    bs: int
    step: np.ndarray
    run: np.ndarray
    rid: np.ndarray
    m: np.ndarray
    gap: np.ndarray
    context: np.ndarray
    temp: np.ndarray
    greedy: np.ndarray
    accepted: np.ndarray
    union: np.ndarray
    cluster: np.ndarray

    def __len__(self):
        return len(self.step)

    def take(self, idx):
        return Batch(self.bs, *(getattr(self, f.name)[idx] for f in fields(self)[1:]))


def input_files(inputs, legacy):
    pattern = "steplog-*.npz" if legacy else "vcap3-*.npz"
    files = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob(pattern))
        else:
            files.update(Path(p).resolve() for p in glob.glob(value))
    if not files:
        raise ValueError(f"No {pattern} files found")
    return sorted(files)


def read_chunks(files, legacy):
    """Yield complete batches with request-major features and mean union per visit."""
    seen = set()
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if legacy:
                run = str(path.parent / re.sub(r"-\d{6}\.npz$", "", path.name))
                index, start, count = np.unique(z["step"], return_index=True, return_counts=True)
                rows = z["row"]
                bs = z["bs"][start]
                if (bs != count).any() or not np.array_equal(
                    rows, np.arange(len(rows)) - np.repeat(start, count)
                ):
                    raise ValueError(f"{path}: incomplete or unordered legacy step")
                if (z["live"] != STRIDE).any() or (z["greedy"] & ~z["argmax_ok"]).any():
                    raise ValueError(f"{path}: legacy rows lack full width or valid greedy logits")
                unions = z["union_k"][start]
                m = z["m"]
                gap = np.zeros_like(m)
                context, accepted = z["seq_len"], z["accept"]
            else:
                if str(z["format"]) != LOG_FORMAT or int(z["live"]) != STRIDE:
                    raise ValueError(f"{path}: expected full-width {LOG_FORMAT}")
                run = str(z["run"])
                index, bs = z["step"], z["bs"]
                start = np.r_[0, np.cumsum(bs)[:-1]]
                if len(index) != len(bs) or bs.sum() != len(z["rid"]):
                    raise ValueError(f"{path}: incomplete steplog chunk")
                raw = z["union"]
                if raw.ndim != 3 or raw.shape[0] != len(bs) or raw.shape[2] != STRIDE:
                    raise ValueError(f"{path}: invalid per-visit union shape")
                valid = (raw >= 0).all(axis=-1)
                if ((raw < 0).any(axis=-1) & (raw >= 0).any(axis=-1)).any():
                    raise ValueError(f"{path}: incomplete routing visit")
                if (np.diff(raw[valid], axis=-1) < 0).any():
                    raise ValueError(f"{path}: non-monotonic routing prefix union")
                count = valid.sum(axis=1, keepdims=True)
                unions = np.where(count, np.where(valid[..., None], raw, 0).sum(axis=1)
                                  / count.clip(min=1), np.nan)
                m, gap = z["m"][:, :GAMMA], z["gap"][:, :GAMMA]
                context, accepted = z["context"], z["accepted"]
            temp, greedy, rid = z["temp"], z["greedy"], z["rid"]
            if (bs < 1).any():
                raise ValueError(f"{path}: invalid batch size")
            if ((accepted < 0) | (accepted > GAMMA)).any() or (context < 0).any():
                raise ValueError(f"{path}: invalid full-width acceptance or context length")
            if any(not np.isfinite(a).all() for a in (m, gap, temp)) or (
                ((m < 0) | (m > 1) | (gap < 0) | (gap > m + 1e-6)).any()
            ):
                raise ValueError(f"{path}: invalid confidence features")
            if (not np.isfinite(context).all() or (temp < 0).any() or (rid == "").any()
                    or m.shape != (len(rid), GAMMA) or gap.shape != m.shape
                    or any(a.shape != rid.shape for a in (context, accepted, temp, greedy))):
                raise ValueError(f"{path}: invalid request metadata")
            for step, begin, n, union in zip(index, start, bs, unions):
                key = run, int(step)
                if key in seen:
                    raise ValueError(f"{path}: duplicate step {key}")
                seen.add(key)
                sl = slice(begin, begin + n)
                yield (int(n), int(step), run, rid[sl], m[sl], gap[sl], context[sl], temp[sl],
                       greedy[sl], accepted[sl], union)


def load_batches(inputs, legacy=False):
    files = input_files(inputs, legacy)
    by_bs = defaultdict(list)
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    # The complete batch stays together. All batches connected by any request
    # (including a request moving across bs bands) stay in the same cluster.
    for bs, step, run, rid, *rest in read_chunks(files, legacy):
        anchor = root((run, str(rid[0])))
        for name in rid[1:]:
            parent[root((run, str(name)))] = anchor
        by_bs[bs].append((step, run, rid, *rest))
    groups = {}
    names = {}
    for bs, rows in sorted(by_bs.items()):
        columns = [np.stack(col) for col in zip(*rows)]
        cluster = [names.setdefault(root((run, str(rid[0]))), len(names))
                   for run, rid in zip(columns[1], columns[2])]
        groups[bs] = Batch(bs, *columns, np.asarray(cluster))
    if len(names) < 2:
        raise ValueError("Cross-fitting needs at least two request-disjoint batch clusters")
    if not any(np.isfinite(g.union).all(axis=1).any() for g in groups.values()):
        raise ValueError("No usable target expert-union data in these steplogs")
    return groups, len(files)


def band_for(bs):
    return min(int(np.searchsorted(BANDS, bs)), len(BANDS) - 1)


def bins(edges, data):
    return np.searchsorted(np.asarray(edges, dtype=np.float32), data, side="right")


def feature_indices(g, gap_edges):
    return (np.asarray(~g.greedy, dtype=int)[..., None],
            bins(CONTEXT, g.context)[..., None], bins(TEMPERATURE, g.temp)[..., None],
            np.arange(GAMMA), bins(EDGES, g.m), bins(gap_edges, g.gap))


def fit_calibration(groups, gap_edges):
    shape = (len(BANDS), 2, len(CONTEXT) + 1, len(TEMPERATURE) + 1, GAMMA,
             len(EDGES) + 1, len(gap_edges) + 1)
    trials, hits = np.zeros(shape), np.zeros(shape)
    for g in groups.values():
        if g.bs > BANDS[-1]:
            continue
        idx = (band_for(g.bs), *feature_indices(g, gap_edges))
        np.add.at(trials, idx, g.accepted[..., None] >= np.arange(GAMMA))
        np.add.at(hits, idx, g.accepted[..., None] > np.arange(GAMMA))
    # Risk sets: only rows that reached j contribute. Back off to successively
    # broader populations with 32 pseudo-observations from the parent.
    estimate = 0.5
    for keep in ((4,), (1, 4), (1, 4, 5), (1, 4, 5, 6),
                 (0, 1, 4, 5, 6), (0, 1, 2, 4, 5, 6), tuple(range(len(shape)))):
        drop = tuple(axis for axis in range(len(shape)) if axis not in keep)
        n = trials.sum(axis=drop, keepdims=True)
        k = hits.sum(axis=drop, keepdims=True)
        estimate = (k + 32 * estimate) / (n + 32)
    return estimate.astype(np.float32)


def shipped_width(g, policy, bsmax):
    if (bsmax and g.bs > bsmax) or policy.theta_for(g.bs) == 0:
        return np.full(g.accepted.shape, STRIDE)
    tab = np.asarray(policy.calib, dtype=np.float32)
    a = tab[(~g.greedy).astype(int)[..., None], np.arange(GAMMA), bins(policy.edges, g.m)]
    threshold = np.float32(policy.theta_for(g.bs)) * np.asarray(policy.row_cost, dtype=np.float32)
    return width_from(a.cumprod(axis=-1) >= threshold)


def apply_policy(g, survival, price, costs):
    if price == 0 or g.bs > BANDS[-1]:
        return np.full(g.accepted.shape, STRIDE)
    return width_from(survival >= np.float32(price) * costs)


def interpolate(keys_values, x):
    keys = sorted(keys_values)
    return np.interp(x, keys, [keys_values[k] for k in keys])


class Cost:
    """Profile-anchored fixed cost plus a partially union-sensitive MoE term."""

    def __init__(self, groups, shipped, bsmax, elasticity, rich_overhead, shipped_overhead):
        self.shipped, self.bsmax = shipped, bsmax
        self.elasticity = elasticity
        self.rich_overhead, self.shipped_overhead = rich_overhead, shipped_overhead
        self.curves = {}
        for bs, g in groups.items():
            valid = np.isfinite(g.union).all(axis=1) & (g.union[:, -1] > 0)
            if valid.any():
                self.curves[bs] = g.union[valid].mean(axis=0)
        if not self.curves:
            raise ValueError("Training fold has no measured expert-union curves")
        self.u1 = self.curve(1) if 1 in self.curves else np.asarray(shipped.u1)
        self.reference = {}
        for bs, g in groups.items():
            if not self.shipped_dynamic(bs):
                continue
            greedy = g.greedy.all(axis=1)
            if greedy.any():
                self.reference[bs] = float(self.fraction(g, shipped_width(g, shipped, bsmax))[
                    greedy].mean())
        if not self.reference:
            self.reference = {bs: float(self.fraction(g, shipped_width(g, shipped, bsmax)).mean())
                              for bs, g in groups.items() if len(g) and self.shipped_dynamic(bs)}

    def shipped_dynamic(self, bs):
        return self.shipped.theta_for(bs) > 0 and (not self.bsmax or bs <= self.bsmax)

    def reference_at(self, bs):
        if not self.shipped_dynamic(bs) or not self.reference:
            return 1.0
        return interpolate(self.reference, bs)

    def curve(self, bs):
        return np.array([interpolate({n: u[k] for n, u in self.curves.items()}, bs)
                         for k in range(STRIDE)])

    def fraction(self, g, widths):
        # The same step's curve is exact for uniform widths. Mixed widths use
        # the curve at the mean c1 expert footprint of the requests' widths.
        lengths = np.interp(self.u1[widths - 1].mean(axis=1), self.u1,
                            np.arange(STRIDE))
        lo = np.minimum(lengths.astype(int), STRIDE - 2)
        hi = lo + 1
        curve = g.union
        missing = ~np.isfinite(curve).all(axis=1) | (curve[:, -1] <= 0)
        if missing.any():
            curve = np.where(missing[:, None], self.curve(g.bs), curve)
        idx = np.arange(len(g))
        u = curve[idx, lo] + (lengths - lo) * (curve[idx, hi] - curve[idx, lo])
        # Enforce the endpoints and exact uniform widths, including flat c1 curves.
        uniform = (widths == widths[:, :1]).all(axis=1)
        u[uniform] = curve[idx[uniform], widths[uniform, 0] - 1]
        return u / curve[:, -1]

    def marginal(self, bs):
        curve = self.curve(bs)
        moe = np.interp(bs, PROFILE_BS, PROFILE_MOE)
        ref = self.reference_at(bs)
        return self.elasticity * moe * np.diff(curve) / (ref * curve[-1] * bs)

    def time(self, g, widths, *, dynamic=None, elasticity=None):
        bs = g.bs
        wall = np.interp(bs, PROFILE_BS, PROFILE_WALL)
        moe = np.interp(bs, PROFILE_BS, PROFILE_MOE)
        sample_extra = np.interp(bs, SAMPLED_BS, SAMPLED_EXTRA) * (~g.greedy).mean(axis=1)
        ref = self.reference_at(bs)
        scale = self.elasticity if elasticity is None else elasticity
        ms = wall + sample_extra + scale * moe * (self.fraction(g, widths) / ref - 1.0)
        if dynamic is not None:
            ms += self.rich_overhead * dynamic - self.shipped_overhead * self.shipped_dynamic(bs)
        return ms


def emitted(g, widths):
    return (np.minimum(g.accepted, widths - 1) + 1).sum(axis=1)


def survival(g, calibration, gap_edges):
    return calibration[band_for(g.bs)][feature_indices(g, gap_edges)].cumprod(axis=-1)


def fit(groups, shipped, gap_edges, args):
    cost = Cost(groups, shipped, args.shipped_bsmax, args.moe_elasticity,
                args.rich_overhead_ms, args.shipped_overhead_ms)
    calibration = fit_calibration(groups, gap_edges)
    row_costs, theta, support = [], [], []
    for band, upper in enumerate(BANDS):
        widths = [g for g in groups.values() if g.bs <= BANDS[-1] and band_for(g.bs) == band]
        nsteps = sum(map(len, widths))
        nclusters = len({int(c) for g in widths for c in g.cluster})
        support.append({"steps": nsteps, "clusters": nclusters})
        # Smooth zero differences at saturated prefixes. Keep the export and
        # parameter search in the same float32 arithmetic as the runtime.
        if nsteps:
            marginal = sum((len(g) * cost.marginal(g.bs) for g in widths)) / nsteps
        else:
            marginal = cost.marginal(upper)
        row_cost = np.maximum(marginal, 1e-6).astype(np.float32)
        row_costs.append(row_cost)
        price = 0.0
        if nsteps >= args.min_steps and nclusters >= 2:
            total_tokens = np.zeros_like(PRICES)
            total_ms = np.zeros_like(PRICES)
            for g in widths:
                s = survival(g, calibration, gap_edges)
                for i, p in enumerate(PRICES):
                    l = apply_policy(g, s, p, row_cost)
                    total_tokens[i] += emitted(g, l).sum()
                    total_ms[i] += cost.time(g, l, dynamic=p > 0).sum()
            price = float(PRICES[(total_tokens / total_ms).argmax()])
        theta.append([upper, price])
    return calibration, np.asarray(row_costs), theta, cost, support


def split_folds(groups, folds, seed):
    nclusters = 1 + max(int(g.cluster.max()) for g in groups.values() if len(g))
    if nclusters < folds:
        raise ValueError(f"Need {folds} independent batch clusters, found {nclusters}")
    counts = np.zeros((nclusters, len(BANDS)))
    for g in groups.values():
        ids, steps = np.unique(g.cluster, return_counts=True)
        counts[ids, band_for(g.bs)] += steps
    # Equal weight per populated band plus an overall workload balance term.
    # Largest indivisible clusters go first so they can land in different folds.
    counts = np.column_stack((counts, counts.sum(axis=1)))
    weights = counts / counts.sum(axis=0).clip(min=1)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(nclusters), key=lambda c: np.linalg.norm(weights[c]),
                   reverse=True)
    assignment = np.empty(nclusters, dtype=int)
    load = np.zeros((folds, weights.shape[1]))
    for c in order:
        fold = int(np.argmin(load @ weights[c]))
        assignment[c] = fold
        load[fold] += weights[c]
    return assignment


def aggregate(records, bootstrap, seed):
    # Columns: candidate tokens/ms, shipped tokens/ms, candidate/shipped width,
    # then candidate/shipped ms at zero/full union elasticity.
    cluster = np.concatenate([c for c, _ in records])
    data = np.concatenate([x for _, x in records])
    ids, inv = np.unique(cluster, return_inverse=True)
    totals = np.stack([np.bincount(inv, x) for x in data.T], axis=1)
    ct, cm, st, sm, _, _, cm0, sm0, cm1, sm1 = totals.sum(axis=0)
    gain = ct / cm / (st / sm) - 1
    ci = None
    if bootstrap and len(ids) >= 2:
        rng = np.random.default_rng(seed)
        draws = np.empty(bootstrap)
        for i in range(bootstrap):
            t, m, bt, bm = totals[rng.integers(len(ids), size=len(ids)), :4].sum(axis=0)
            draws[i] = t / m / (bt / bm) - 1
        ci = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
    return dict(steps=len(data), clusters=len(ids), candidate_tok_s=float(1000 * ct / cm),
                shipped_tok_s=float(1000 * st / sm), vs_shipped=float(gain), ci95=ci,
                mean_width=[float(x) for x in data[:, 4:6].mean(axis=0)],
                sensitivity_gain=[float(ct / cm0 / (st / sm0) - 1),
                                  float(ct / cm1 / (st / sm1) - 1)])


def evaluate(groups, shipped, gap_edges, args):
    folds = split_folds(groups, args.folds, args.seed)
    records = defaultdict(list)
    details = []
    for fold in range(args.folds):
        train = {bs: g.take(folds[g.cluster] != fold) for bs, g in groups.items()}
        table, costs, theta, cost, support = fit(train, shipped, gap_edges, args)
        details.append({"theta": theta, "train_support": support, "union_reference": cost.reference})
        for bs, all_g in groups.items():
            g = all_g.take(folds[all_g.cluster] == fold)
            if not len(g):
                continue
            band = band_for(bs)
            price = theta[band][1]
            l3 = apply_policy(g, survival(g, table, gap_edges), price, costs[band])
            ls = shipped_width(g, shipped, args.shipped_bsmax)
            dyn = price > 0 and bs <= BANDS[-1]
            data = np.column_stack((emitted(g, l3), cost.time(g, l3, dynamic=dyn),
                                    emitted(g, ls), cost.time(g, ls),
                                    l3.mean(axis=1), ls.mean(axis=1),
                                    cost.time(g, l3, dynamic=dyn, elasticity=0),
                                    cost.time(g, ls, elasticity=0),
                                    cost.time(g, l3, dynamic=dyn, elasticity=1),
                                    cost.time(g, ls, elasticity=1)))
            records["all"].append((g.cluster, data))
            records[f"bs={bs}"].append((g.cluster, data))
            for label, mask in (("greedy", g.greedy.all(axis=1)),
                                ("sampled", (~g.greedy).all(axis=1)),
                                ("mixed", g.greedy.any(axis=1) & (~g.greedy).any(axis=1))):
                if mask.any():
                    records[label].append((g.cluster[mask], data[mask]))
    return {name: aggregate(r, args.bootstrap, args.seed) for name, r in records.items()}, details


def write_json(path, obj, *, compact=False):
    with open(path, "w") as f:
        json.dump(obj, f, allow_nan=False, indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="capture directories, NPZ files or quoted globs")
    ap.add_argument("--legacy", action="store_true", help="historical round-2 NPZ, no gap feature")
    ap.add_argument("--shipped-policy", type=Path, default=SHIPPED)
    ap.add_argument("--shipped-bsmax", type=int, default=16, help="v2.6 launch cutoff; 0 disables")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-steps", type=int, default=60, help="minimum train steps per batch band")
    ap.add_argument("--moe-elasticity", type=float, default=0.5)
    ap.add_argument("--rich-overhead-ms", type=float, default=0.25)
    ap.add_argument("--shipped-overhead-ms", type=float, default=0.10)
    ap.add_argument("--json", help="write machine-readable cross-fitted report")
    ap.add_argument("--export-policy", help="write v2 policy fitted on all input rows")
    args = ap.parse_args(argv)
    if (args.folds < 2 or args.bootstrap < 0 or args.min_steps < 1 or args.shipped_bsmax < 0
            or not 0 <= args.moe_elasticity <= 1
            or any(not np.isfinite(x) or not 0 <= x <= 10
                   for x in (args.rich_overhead_ms, args.shipped_overhead_ms))):
        ap.error("Invalid folds, bootstrap, minimum steps, bsmax, elasticity or overhead")
    try:
        shipped = load_policy(args.shipped_policy)
        if shipped.rich or shipped.rounds or shipped.stride != STRIDE:
            raise ValueError("The baseline must be a per-request v1 policy with stride 8")
        groups, files = load_batches(args.inputs, legacy=args.legacy)
        gap_edges = () if args.legacy else GAPS
        scores, folds = evaluate(groups, shipped, gap_edges, args)
        report = {
            "input": "round-2 stock (historical, no gap)" if args.legacy else LOG_FORMAT,
            "files": files, "request_steps": sum(g.bs * len(g) for g in groups.values()),
            "missing_union_steps": sum(int((~np.isfinite(g.union).all(axis=1)).sum()) for g in groups.values()),
            "profile": {"bs": PROFILE_BS, "wall_ms": PROFILE_WALL, "moe_ms": PROFILE_MOE,
                        "sampled_bs": SAMPLED_BS, "sampled_offsets_ms": SAMPLED_EXTRA,
                        "moe_elasticity": args.moe_elasticity,
                        "rich_overhead_ms": args.rich_overhead_ms,
                        "shipped_overhead_ms": args.shipped_overhead_ms,
                        "shipped_bsmax": args.shipped_bsmax},
            "scores": scores, "folds": folds,
        }
        print(f"{report['input']} | {report['request_steps']} request-steps | modeled tokens/s")
        print(f"{'slice':>8} {'steps':>7} {'clust.':>6} {'v3':>9} {'shipped':>9} {'gain':>9}  95% paired CI")
        names = ["all", *[f"bs={bs}" for bs in sorted(groups)], "greedy", "sampled", "mixed"]
        for name in names:
            if name not in scores:
                continue
            s = scores[name]
            ci = s["ci95"]
            interval = f"[{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci is not None else "unavailable"
            print(f"{name:>8} {s['steps']:7} {s['clusters']:6} {s['candidate_tok_s']:9.2f} "
                  f"{s['shipped_tok_s']:9.2f} {s['vs_shipped']:+9.2%}  {interval}")
        if args.json:
            write_json(args.json, report)
        if args.export_policy:
            calibration, costs, theta, _, support = fit(groups, shipped, gap_edges, args)
            policy = dict(format=RICH_FORMAT, stride=STRIDE, theta=theta, row_cost_ms=costs.tolist(),
                          bin_edges=EDGES, gap_edges=gap_edges, context_edges=CONTEXT,
                          temp_edges=TEMPERATURE, calib=calibration.tolist(),
                          source={"input": report["input"], "request_steps": report["request_steps"],
                                  "profile": report["profile"], "train_support": support})
            parse_policy(policy, args.export_policy)
            write_json(args.export_policy, policy, compact=True)
            print(f"Exported {args.export_policy} with theta {theta}")
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
