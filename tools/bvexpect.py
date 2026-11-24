"""Exact expected accepted drafts of block verification for every verify width (stopped draft), integrated over
the acceptance coins. Used by dfreplay for sampled requests: under block verification the capped yield is NOT
1 + min(A_full, L - 1) (a longer block can accept an earlier low-ratio token because a later token compensates).

The engine's block-verification branch (mirrored in overlay/tests/vcap/test_block_verify.py, block_verify_ref) with
the stopped draft at live length L (g = L - 1 live drafts; the cut row and q(d_j), j >= g, are zero):
  prefix pp_0 = 1, pp_s = min(pp_{s-1} P_{s-1}(d_{s-1}) / q_{s-1}(d_{s-1}), 1)          (s = 1..g; 0 if q not in (0, 1])
  walk s = g .. 1: level s accepts (first, from the top) when coin_s < min(pp_s, h_s), with
      h_s = pp_s                                    for s = g (the last live level: the cut row is all zero),
      h_s = r_s / (r_s + 1 - pp_s), r_s = sum_x max(pp_s P_s(x) - q_s(x), 0)   for s < g when 0 < pp_s < 1,
      h_s = pp_s                                    otherwise.
  q_s has support C_s (the 16 candidates of slot s) and P_s is normalized, so
      r_s = pp_s - sum_{c in C_s} min(pp_s P_s(c), q_s(c))     (exact; computed directly on the support).
  With independent uniform coins, P(tau = s) = a_s prod_{k > s} (1 - a_k), a_s = min(pp_s, h_s), and
  E[accepted drafts | L] = sum_s s P(tau = s).

Inputs per request (G = 7 slots, K = 16 candidates): q [n, G, K] (the walked rows' probabilities at the request
temperature), P [n, G, K] (target probabilities of row s = the row predicting slot s, at the request temperature, on
slot s's candidates), lane [n, G] (the drafted token's candidate index per slot). Output [n, G + 1]: expected accepted
drafts for L = 1..G+1."""

from __future__ import annotations

import numpy as np


def expected_accepts(q: np.ndarray, P: np.ndarray, lane: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    n, G, K = q.shape
    idx = np.arange(n)
    qd = np.stack([q[idx, j, lane[:, j]] for j in range(G)], axis=1)  # q_j(d_j)
    Pd = np.stack([P[idx, j, lane[:, j]] for j in range(G)], axis=1)  # P_j(d_j)
    pp = np.ones((n, G + 1))                                          # pp[:, s], s = 0..G
    for s in range(1, G + 1):
        ok = (qd[:, s - 1] > 0.0) & (qd[:, s - 1] <= 1.0)
        ratio = pp[:, s - 1] * Pd[:, s - 1] / np.where(ok, qd[:, s - 1], 1.0)
        pp[:, s] = np.where(ok, np.minimum(ratio, 1.0), 0.0)
    # h_s for an interior level s (< g): uses row s / slot s (P_s, q_s over C_s)
    h_int = pp.copy()
    for s in range(1, G):
        p = pp[:, s]
        r = p - np.minimum(p[:, None] * P[:, s], q[:, s]).sum(axis=1)
        r = np.maximum(r, 0.0)
        need = (p > 0.0) & (p < 1.0)
        h_int[:, s] = np.where(need, r / np.maximum(r + 1.0 - p, 1e-300), p)
    out = np.zeros((n, G + 1))
    for L in range(2, G + 2):                     # L = 1: no live draft -> 0 accepted
        g = L - 1
        surv = np.ones(n)                         # probability that no higher level accepted
        e = np.zeros(n)
        for s in range(g, 0, -1):
            h = pp[:, s] if s == g else h_int[:, s]
            a = np.minimum(pp[:, s], h)
            e += s * a * surv
            surv *= 1.0 - a
        out[:, L - 1] = e
    return out


def truncated_estimate(accepted_full: np.ndarray, G: int = 7) -> np.ndarray:
    """The truncation estimate the replay used: min(A_full, L - 1) for L = 1..G+1."""
    a = np.asarray(accepted_full)[:, None]
    return np.minimum(a, np.arange(G + 1)[None, :]).astype(np.float64)
