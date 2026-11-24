# Usage (in the SGLang image): INKLING_RSFIX=1 PYTHONPATH=/work/overlay python3 tools/gpu_check_dflash2_cap_law.py
"""GPU Monte Carlo law test of the DFLASH 2 sampled path UNDER THE VERIFY CAP, through the REAL image code:
selector walk (selector_walk_triton) -> DFlashWorkerV2._selector_sampling_accept (dense q scatter + accept_sampling,
chain or block verification) with cutoff_verify_lens. An extension of the uncapped DFlash2 law test with:

- each request's live length L from a stopping rule on the walked q rows: draft j live while the running product of
  max q_0 .. max q_j >= CAP_THR (decided by d_0..d_{j-1}, never by d_j);
- the dead target rows (>= L) carry a deliberately WRONG distribution (what the overlay's expert remap leaves there);
- block-verification arms hand the accept the sparse q masked by inkling_vcap.logic.mask_dead_q_rows_sparse, as the
  overlay's _selector_sampling_accept wrapper does;
- the cutoff contract: a request never accepts more than L - 1 drafts (counted as invalid otherwise).
Arms: chain / bv-mask at T 1.0 and 0.7, target top-k 5, bv without the mask (also exact: the stock cutoff alone keeps
the law); negative controls that must FAIL: the cutoff withheld (drafts accepted against dead rows), and
a look-ahead rule (L from q of the drafted token itself). Run with INKLING_RSFIX=1 (the serving CDF fix).
Verdict as the original: pooled chi-square z < 4 and no invalid output for law arms, z > 10 for negative controls."""
import os
import sys
import types

import torch

from inkling_vcap import logic
from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

dev = "cuda"
V, K, SLOTS, OUT = 24, 16, 7, 3
N = int(os.environ.get("LAW_N", "400000"))
THR = float(os.environ.get("CAP_THR", "0.15"))
g0 = torch.Generator(device=dev).manual_seed(1234)
P = torch.softmax(2.0 * torch.randn(V, V, generator=g0, device=dev, dtype=torch.float64), -1)


def target_law(T, topk=None):
    Pt = torch.softmax(torch.log(P) / T, -1)
    if topk:
        keep = torch.zeros_like(Pt, dtype=torch.bool).scatter_(1, Pt.topk(topk, -1).indices, True)
        Pt = torch.where(keep, Pt, torch.zeros_like(Pt))
        Pt = Pt / Pt.sum(-1, keepdim=True)
    return Pt, torch.einsum("a,ab,bc->abc", Pt[0], Pt, Pt)


class FakeWorker:
    """Only the attributes DFlashWorkerV2._selector_sampling_accept reads."""

    def __init__(self, bv):
        self._draft_probs_buf = None
        self._block_verification = bv


def draft(n, gen):
    cand = torch.rand(n, SLOTS, V, generator=gen, device=dev).argsort(-1)[..., :K].contiguous()
    scores = (1.5 * torch.randn(n, SLOTS, K, K, generator=gen, device=dev)).float().contiguous()
    return cand, scores


def live_lens(q, tok, cand, lookahead=False):
    """[N] 1 + leading run of cumprod(feature_j) >= THR; feature = max q_j (a stopping rule) or, for the negative
    control, q_j(d_j) (the drafted token's own probability: look-ahead)."""
    if lookahead:
        lane = (cand == tok[..., None]).int().argmax(-1)
        f = q.gather(-1, lane[..., None])[..., 0]
    else:
        f = q.max(-1).values
    ok = (torch.cumprod(f, dim=1) >= THR).long()
    return 1 + torch.cumprod(ok, dim=1).sum(dim=1)


def simulate(bv, T, seed, topk=None, mask=True, lookahead=False, nocut=False):
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    w = FakeWorker(bv)
    out = torch.full((N, OUT), -1, dtype=torch.long, device=dev)
    filled = torch.zeros(N, dtype=torch.long, device=dev)
    anchor = torch.zeros(N, dtype=torch.long, device=dev)
    temps = torch.full((N,), T, device=dev)
    si = types.SimpleNamespace(temperatures=temps.view(N, 1), is_all_greedy=False, is_any_greedy=False,
                               need_top_k_sampling=bool(topk), need_top_p_sampling=False, need_min_p_sampling=False,
                               top_ks=torch.full((N,), topk or V, dtype=torch.int32, device=dev),
                               top_ps=torch.ones(N, device=dev))
    di = types.SimpleNamespace(max_top_k=topk or 1, uniform_top_k_value=topk)
    acc_sum, steps, invalid, lens_sum, lens_n = 0, 0, 0, 0.0, 0
    while bool((filled < OUT).any()):
        cand, scores = draft(N, gen)
        u = torch.rand(N, SLOTS, generator=gen, device=dev)
        tok, q = selector_walk_triton(candidate_ids=cand, scores=scores, uniforms=u, temperatures=temps,
                                      greedy_mask=torch.zeros(N, dtype=torch.bool, device=dev))
        row_tok = torch.cat([anchor[:, None], tok], 1)  # [N, 8]
        L = live_lens(q, tok, cand, lookahead=lookahead)
        dead = torch.arange(SLOTS + 1, device=dev)[None, :] >= L[:, None]
        logits = torch.log(P[row_tok]).float()  # [N, 8, V]
        junk = 3.0 * torch.randn(N, SLOTS + 1, V, generator=gen, device=dev)  # dead rows: remapped garbage
        logits = torch.where(dead[..., None], junk, logits).reshape(N * (SLOTS + 1), V).contiguous()
        q_in = logic.mask_dead_q_rows_sparse(q, cand, tok, L) if (bv and mask) else q
        lens_sum += float(L.float().sum())
        lens_n += N
        correct, bonus, _ = DFlashWorkerV2._selector_sampling_accept(
            w, candidates=row_tok.contiguous(), next_token_logits=logits, candidate_ids=cand, q_rows=q_in,
            sampling_info=si, draft_input=di, cutoff_verify_lens=None if nocut else L)
        correct, bonus = correct.long(), bonus.long()
        bad = (bonus < 0) | (bonus >= V) | (correct < 0) | ((correct > L - 1) & (not nocut))
        nb = int(bad.sum())
        if nb:
            invalid += nb
            bonus = bonus.clamp(0, V - 1)
            correct = correct.clamp(min=0) if nocut else torch.minimum(correct.clamp(min=0), L - 1)
        emitted = torch.cat([tok, bonus[:, None]], 1)
        n_emit = correct + 1
        active = filled < OUT
        acc_sum += int(correct[active].sum())
        steps += int(active.sum())
        for k in range(SLOTS + 1):
            take = active & (k < n_emit) & (filled + k < OUT)
            t = torch.where(k == n_emit - 1, bonus, emitted[:, k])
            idx = (filled + k).clamp(max=OUT - 1)
            out[take, idx[take]] = t[take]
        filled = torch.where(active, filled + n_emit, filled)
        anchor = torch.where(active, bonus, anchor)
    emp = torch.zeros(V, V, V, dtype=torch.float64, device=dev)
    emp.index_put_((out[:, 0], out[:, 1], out[:, 2]), torch.ones(N, dtype=torch.float64, device=dev), accumulate=True)
    emp /= N
    _, exact = target_law(T, topk)
    cnt, expc = emp.flatten() * N, exact.flatten() * N
    big = expc >= 5
    obs = torch.cat([cnt[big], cnt[~big].sum().view(1)])
    ex = torch.cat([expc[big], expc[~big].sum().view(1)])
    keep = ex > 0
    chi2 = (((obs - ex) ** 2)[keep] / ex[keep]).sum()
    df = int(keep.sum()) - 1
    z = (chi2 - df) / (2 * df) ** 0.5
    if int(cnt[expc == 0].sum()):
        z = torch.tensor(1e9)
    return float((emp - exact).abs().max()), float(z), acc_sum / steps, invalid, steps, lens_sum / lens_n


ok = True
arms = [("cap-chain", False, 1.0, 0, {}, "lt"), ("cap-bv-mask", True, 1.0, 1, {}, "lt"),
        ("cap-chain", False, 0.7, 2, {}, "lt"), ("cap-bv-mask", True, 0.7, 3, {}, "lt"),
        ("cap-chain-topk5", False, 1.0, 6, {"topk": 5}, "lt"), ("cap-bv-topk5", True, 0.7, 7, {"topk": 5}, "lt"),
        # stock block verification with the cutoff also preserves the law (dead rows integrate out);
        # the mask only stops the kernel from scanning dead levels -> a positive control
        ("cap-bv-nomask", True, 1.0, 8, {"mask": False}, "lt"),
        # negative controls: the cutoff withheld (drafts accepted against the dead rows' garbage); look-ahead rule
        ("cap-nocutoff", False, 1.0, 10, {"nocut": True}, "gt"),
        # the chain cutoff keeps the law even for a look-ahead rule (the cut row's token is still
        # verified against its live row; the rule only decides whether the block continues) -> positive arm
        ("cap-lookahead", False, 1.0, 9, {"lookahead": True}, "lt")]
for name, bv, T, seed, kw, want in arms:
    err, z, acc, inv, nst, mean_l = simulate(bv, T, seed, **kw)
    good = (z < 4 and inv == 0) if want == "lt" else z > 10
    ok &= good
    print(f"{name:16s} T={T} seed {seed}: mean L {mean_l:.2f} max abs err {err:.2e} chi2-z {z:.2f} "
          f"accepted/step {acc:.4f} invalid {inv}/{nst} steps {'PASS' if good else 'FAIL'} (want {want})", flush=True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
