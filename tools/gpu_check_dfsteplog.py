"""GPU checks of the DFLASH capture (run in the SGLang image, no server, before the capture boot):

  PYTHONPATH=/work/overlay python3 tools/gpu_check_dfsteplog.py

1. route journal: v3_kernels.record_route (write kernel + separate cursor advance) inside a CUDA graph of 48 visits,
   replayed with fresh ids and different real batch sizes (padding rows) at graph rows 8 .. 384 (the 8-warp tile);
   every slab must equal the gate ids of the real rows, unused lanes -1, the cursor 48; a zero real batch writes
   nothing.
2. walked-row features (dflash.DflashFeatures.record) recorded inside a CUDA graph after the image's
   selector_walk_triton, against a float64 CPU reference on the same tokens; seq advances once per replay.
3. DfStepLog end to end on CUDA: mark / start / graph replay of 48 routed visits / finish per step, bs varying, then
   the npz: unions recomputed from the ids, visits 48, walks 1, accepted, finite positive timings, status drained.
Prints PASS/FAIL lines; exit 1 on any FAIL."""

import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "overlay"))

from inkling_vcap import dfsteplog, dflash  # noqa: E402
from inkling_vcap.v3_kernels import record_route  # noqa: E402

DEV = torch.device("cuda")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def ref_rows_features(cand, scores, tokens, t_eff):
    """Independent CPU reference (loops, float64): walked rows and (m, gap); no overlay helper is used."""
    bs, G, K = cand.shape
    rows = torch.empty(bs, G, K, dtype=torch.float64)
    m = torch.empty(bs, G, dtype=torch.float64)
    gap = torch.empty(bs, G, dtype=torch.float64)
    for b in range(bs):
        prev = None
        for j in range(G):
            row = scores[b, 0, 0] if j == 0 else scores[b, j, prev]
            rows[b, j] = row.double()
            x = row.double() / float(t_eff[b])
            q = torch.exp(x - x.max())
            q = q / q.sum()
            top = sorted(q.tolist(), reverse=True)
            m[b, j], gap[b, j] = top[0], top[0] - top[1]
            prev = [int(c) for c in cand[b, j]].index(int(tokens[b, j]))
    return rows, m, gap


def route_graph_check():
    gen = torch.Generator(device="cpu").manual_seed(0)
    V, S, LANES, MAXV = 48, 8, 8, 64
    for graph_bs in (1, 4, 16, 48):
        rows = graph_bs * S
        cap = 48
        journal = torch.full((MAXV * cap * S * LANES,), -1, dtype=torch.int16, device=DEV)
        control = torch.zeros(2, dtype=torch.int32, device=DEV)
        ids = [torch.zeros((rows, 8), dtype=torch.int32, device=DEV) for _ in range(V)]
        for t in ids:  # warm up / compile outside the capture
            record_route(t, journal, control, S, LANES, MAXV)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for t in ids:
                record_route(t, journal, control, S, LANES, MAXV)
        for real in sorted({1, max(1, graph_bs - 1), graph_bs}):
            for rep in range(3):
                vals = [torch.stack([torch.randperm(128, generator=gen)[:8] for _ in range(rows)]).to(torch.int32)
                        for _ in range(V)]
                for t, v in zip(ids, vals):
                    t.copy_(v)
                journal.fill_(-7)
                control[0] = 0
                control[1] = real
                g.replay()
                torch.cuda.synchronize()
                cursor = int(control[0])
                n = real * S
                jr = journal[:V * n * LANES].view(V, n, LANES).cpu()
                ok = cursor == V and all(torch.equal(jr[v].to(torch.int32), vals[v][:n]) for v in range(V))
                ok = ok and bool((journal[V * n * LANES:] == -7).all())  # nothing past the real rows' slabs
                check(f"route graph_bs={graph_bs} real={real} rep={rep}", ok, f"cursor={cursor}")
        control[0] = 0
        control[1] = 0
        journal.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        check(f"route graph_bs={graph_bs} real=0 writes nothing", bool((journal == -7).all()) and int(control[0]) == V)
        # a real batch larger than the routed rows (a sharded path): the cursor is marked invalid (> VISITS)
        control[0] = 0
        control[1] = graph_bs + 1
        g.replay()
        torch.cuda.synchronize()
        check(f"route graph_bs={graph_bs} real>rows marks invalid", int(control[0]) == MAXV + 1,
              f"cursor={int(control[0])}")


def feature_graph_check():
    try:
        from sglang.kernels.ops.speculative.dflash import selector_walk_triton
    except Exception as exc:  # noqa: BLE001
        check("features: image selector_walk_triton import", False, repr(exc))
        return
    gen = torch.Generator(device="cpu").manual_seed(1)
    bs, G, K = 12, 7, 16
    feats = dflash.DflashFeatures(capacity=16, gamma=G, top_k=K, device=DEV, keep_rows=True)
    cand = torch.zeros((bs, G, K), dtype=torch.int64, device=DEV)
    scores = torch.zeros((bs, G, K, K), dtype=torch.float32, device=DEV)
    uni = torch.zeros((bs, G), dtype=torch.float32, device=DEV)
    temps = torch.ones(bs, dtype=torch.float32, device=DEV)
    greedy = torch.zeros(bs, dtype=torch.bool, device=DEV)
    out_tok = torch.zeros((bs, G), dtype=torch.int64, device=DEV)

    def body():
        tok, q = selector_walk_triton(candidate_ids=cand, scores=scores, uniforms=uni, temperatures=temps,
                                      greedy_mask=greedy)
        feats.record(candidate_ids=cand, scores=scores, tokens=tok, temperatures=temps, greedy_mask=greedy)
        out_tok.copy_(tok)

    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for rep in range(4):
        cand.copy_(torch.stack([torch.stack([torch.randperm(201000, generator=gen)[:K] for _ in range(G)])
                                for _ in range(bs)]))
        scores.copy_(torch.randn(bs, G, K, K, generator=gen) * 3)
        uni.copy_(torch.rand(bs, G, generator=gen))
        temps.copy_(torch.tensor([0.6, 0.8, 1.0, 1.3] * 3))
        greedy.copy_(torch.tensor([True, False, False, True] * 3))
        seq0 = int(feats.seq)
        g.replay()
        torch.cuda.synchronize()
        c, sc, tk = cand.cpu(), scores.cpu(), out_tok.cpu()
        teff = torch.where(greedy.cpu(), torch.ones(bs), temps.cpu().clamp_min(1e-5)).double()
        ref_rows, m, gap = ref_rows_features(c, sc, tk, teff)
        ok = (int(feats.seq) == seq0 + 1 and torch.equal(feats.rows[:bs].cpu().double(), ref_rows)
              and float((feats.m[:bs].cpu().double() - m).abs().max()) < 1e-5
              and float((feats.gap[:bs].cpu().double() - gap).abs().max()) < 1e-5
              and torch.equal(feats.greedy[:bs].cpu(), greedy.cpu()))
        check(f"features in graph rep={rep}", ok,
              f"m_err={float((feats.m[:bs].cpu().double() - m).abs().max()):.2e}")


def lattice_reductions_check():
    """record_candidates' chunked lse of transform(plane) and DfStepLog._target_stats on CUDA vs float64 CPU."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    G, K, V = 7, 16, 201024
    feats = dflash.DflashFeatures(capacity=8, gamma=G, top_k=K, device=DEV, keep_rows=True)

    def transform(x):  # the DFlash2 unary transform shape: multiplier then softcap, in place
        x.mul_(0.75)
        return x.div_(30.0).tanh_().mul_(30.0)

    for bs in (1, 3, 8):
        n = bs * G
        plane = (torch.randn(n, V, generator=gen) * 4).to(torch.bfloat16)
        plane[:, 200058:] = 0.0
        ids = torch.randint(0, V, (n, K), generator=gen)
        unary = torch.randn(n, K, generator=gen)
        feats._logits = plane.to(DEV)
        feats.record_candidates(n_rows=n, ids=ids.to(DEV), unary=unary.to(DEV), transform=transform)
        ref = torch.logsumexp(transform(plane.double()), dim=-1)
        err = float((feats.lse[:n].cpu().double() - ref).abs().max())
        check(f"lse bs={bs}", err < 1e-3 and feats._logits is None and torch.equal(feats.unary[:n].cpu(), unary),
              f"err={err:.2e}")
    log = dfsteplog.DfStepLog(tempfile.mkdtemp(prefix="dfsl-ts-"), stride=8, top_k=K, capacity=8, device=DEV,
                              features=feats)
    for bs in (1, 5):
        logits = torch.randn(bs * 8, V, generator=gen) * 5
        p1, gap = log._target_stats(logits.to(DEV), bs)
        pr = torch.softmax(logits.double(), dim=-1).topk(2, dim=-1).values.view(bs, 8, 2)
        e1 = float((p1.cpu().double() - pr[..., 0]).abs().max())
        e2 = float((gap.cpu().double() - (pr[..., 0] - pr[..., 1])).abs().max())
        check(f"target stats bs={bs}", e1 < 1e-5 and e2 < 1e-5, f"err={max(e1, e2):.2e}")
    p1, gap = log._target_stats(None, 2)
    check("target stats without logits -> NaN", bool(torch.isnan(p1).all() and torch.isnan(gap).all()))
    log.close()


def steplog_e2e_check():
    tmp = tempfile.mkdtemp(prefix="dfsl-gpu-")
    gen = torch.Generator(device="cpu").manual_seed(2)
    G, K, S, V = 7, 16, 8, 48
    cap = 8
    feats = dflash.DflashFeatures(capacity=cap, gamma=G, top_k=K, device=DEV, keep_rows=True)
    log = dfsteplog.DfStepLog(tmp, stride=S, top_k=K, capacity=cap, device=DEV, features=feats)
    graph_bs = 4
    rows = graph_bs * S
    ids = [torch.zeros((rows, 8), dtype=torch.int32, device=DEV) for _ in range(V)]
    from types import SimpleNamespace as NS

    def verify_body():
        for t in ids:
            log.route(t)

    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        verify_body()  # capture: the route kernels are recorded although no step is active
    ref = []
    for step in range(40):
        bs = 1 + step % graph_bs
        cand = torch.stack([torch.stack([torch.randperm(1000, generator=gen)[:K] for _ in range(G)])
                            for _ in range(bs)]).to(DEV)
        scores = (torch.randn(bs, G, K, K, generator=gen) * 2).to(DEV)
        tok = cand[:, :, 0].contiguous()
        log.mark_draft_start(bs)
        feats.record(candidate_ids=cand, scores=scores, tokens=tok, temperatures=torch.ones(bs, device=DEV),
                     greedy_mask=torch.ones(bs, dtype=torch.bool, device=DEV))
        log.start(batch=NS(reqs=[NS(rid=f"r{i}") for i in range(bs)], sampling_info=None),
                  prefix_lens=torch.arange(bs, device=DEV) + 100 + step)
        vals = [torch.stack([torch.randperm(128, generator=gen)[:8] for _ in range(rows)]).to(torch.int32)
                for _ in range(V)]
        for t, v in zip(ids, vals):
            t.copy_(v)
        g.replay()
        acc = torch.randint(0, 8, (bs,), generator=gen)
        log.finish(accept_len=acc.to(DEV))
        log.end_step()
        ref.append((bs, vals, acc))
    deadline = time.time() + 20
    while log.written < 40 and time.time() < deadline:
        time.sleep(0.2)
    log.close()
    files = sorted(glob.glob(os.path.join(tmp, "dflash1-*.npz")))
    zs = [np.load(f) for f in files]
    union = np.concatenate([z["union"] for z in zs]) if zs else np.zeros((0,))
    visits = np.concatenate([z["visits"] for z in zs]) if zs else np.zeros((0,))
    walks = np.concatenate([z["walks"] for z in zs]) if zs else np.zeros((0,))
    acc = np.concatenate([z["accepted"] for z in zs]) if zs else np.zeros((0,))
    tv = np.concatenate([z["t_verify_ms"] for z in zs]) if zs else np.zeros((0,))
    td = np.concatenate([z["t_draft_ms"] for z in zs]) if zs else np.zeros((0,))
    check("e2e steps written", len(union) == 40 and log.written == 40, f"{len(union)} / written {log.written}")
    if len(union) != 40:
        return
    check("e2e visits == 48, walks == 1", bool((visits == V).all() and (walks == 1).all()))
    ok_u, ok_a, off = True, True, 0
    for s, (bs, vals, a) in enumerate(ref):
        for v in range(V):
            x = vals[v][:bs * S].view(bs, S, 8)
            for w in range(1, S + 1):
                if int(union[s][v, w - 1]) != len(set(x[:, :w].reshape(-1).tolist())):
                    ok_u = False
        ok_a = ok_a and np.array_equal(acc[off:off + bs], a.numpy())
        off += bs
    check("e2e unions == gate ids", ok_u)
    check("e2e accepted", ok_a)
    check("e2e timings finite, >= 0", bool(np.isfinite(tv).all() and np.isfinite(td).all() and (tv >= 0).all()
                                           and (td >= 0).all()), f"verify median {np.median(tv):.3f} ms")
    st = json.load(open(glob.glob(os.path.join(tmp, "status-*.json"))[0]))
    check("e2e status drained", st["logged"] == st["written"] == 40 and st["idle"] and st["failed"] is None, str(st))


def main():
    route_graph_check()
    feature_graph_check()
    lattice_reductions_check()
    steplog_e2e_check()
    print(f"SUMMARY {'PASS' if not FAILS else 'FAIL'} fails={FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
