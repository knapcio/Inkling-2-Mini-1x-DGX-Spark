"""bvexpect against Monte Carlo through the engine-mirroring block-verification reference (CPU).

  python -m unittest tools/test_bvexpect.py   (from the repository root)

1. counterexample: ratios 1/2 then 2 -> the full 2-draft block always accepts both, the stopped 1-draft block
   accepts its token with probability 1/2: truncating the full-width count says 1, the exact expectation 1/2.
2. random sparse-support problems (K-candidate q per slot inside a larger vocabulary, stopped drafts as the overlay
   masks them): E[accepted | L] from bvexpect equals the Monte Carlo mean of block_verify_ref for every L."""
import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "overlay"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "overlay", "tests", "vcap"))

import bvexpect  # noqa: E402
from inkling_vcap import logic  # noqa: E402
from test_block_verify import block_verify_ref  # noqa: E402


class CounterexampleTest(unittest.TestCase):
    def test_truncation_is_wrong_for_block_verification(self):
        # 2 slots over 2 candidates; drafted lanes 0, 0. Ratios P/q: 1/2 at slot 0, 2 at slot 1.
        q = np.array([[[0.8, 0.2], [0.25, 0.75]]])
        P = np.array([[[0.4, 0.6], [0.5, 0.5]]])
        lane = np.array([[0, 0]])
        e = bvexpect.expected_accepts(q, P, lane)  # L = 1, 2, 3
        self.assertAlmostEqual(e[0, 2], 2.0)        # full block: pp_1 = 1/2, pp_2 = min(1/2 * 2, 1) = 1
        self.assertAlmostEqual(e[0, 1], 0.5)        # stopped after 1 draft: accept w.p. pp_1 = 1/2
        trunc = bvexpect.truncated_estimate(np.array([2]), G=2)
        self.assertEqual(trunc[0, 1], 1.0)          # what min(A_full, L - 1) would have said


def _mc(qk, Pfull, cand, lane, L, N, g):
    G, V = cand.shape[0], Pfull.shape[1]
    drafts = torch.tensor(cand[np.arange(G), lane], dtype=torch.long)[None].expand(N, G).contiguous()
    ps = torch.tensor(Pfull, dtype=torch.float64)[None].expand(N, G + 1, V).contiguous()
    qd = np.zeros((G, V))
    for j in range(G):
        qd[j, cand[j]] = qk[j]
    qs = torch.tensor(qd, dtype=torch.float64)[None].expand(N, G, V).contiguous().clone()
    logic.mask_dead_draft_rows(qs, drafts.clone(), torch.full((N,), L, dtype=torch.long))  # the stopped draft
    acc, _ = block_verify_ref(drafts, ps, qs, torch.rand(N, G, generator=g, dtype=torch.float64),
                              torch.rand(N, generator=g, dtype=torch.float64))
    return torch.minimum(acc, torch.tensor(L - 1)).double()


class MonteCarloTest(unittest.TestCase):
    N = 300_000

    def _check(self, qk, Pfull, cand, lane, g):
        G = cand.shape[0]
        P = np.stack([Pfull[j, cand[j]] for j in range(G)])
        want = bvexpect.expected_accepts(qk[None], P[None], lane[None])[0]
        full = _mc(qk, Pfull, cand, lane, G + 1, self.N, g)
        gaps = []
        for L in range(1, G + 2):
            acc = _mc(qk, Pfull, cand, lane, L, self.N, g)
            se = float(acc.std()) / self.N ** 0.5 + 1e-9
            self.assertLess(abs(float(acc.mean()) - want[L - 1]), 5 * se + 1e-4, (L, float(acc.mean()), want[L - 1]))
            gaps.append(abs(float(torch.minimum(full, torch.tensor(L - 1)).mean()) - want[L - 1]))
        return max(gaps)

    def test_counterexample_through_the_reference(self):
        g = torch.Generator().manual_seed(0)
        cand = np.array([[0, 1], [0, 1]])          # 2 slots, 2 candidates, vocabulary of 3
        qk = np.array([[0.8, 0.2], [0.25, 0.75]])
        Pfull = np.array([[0.4, 0.6, 0.0], [0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
        gap = self._check(qk, Pfull, cand, np.array([0, 0]), g)
        self.assertGreater(gap, 0.4)              # truncation is off by 1/2 at L = 2

    def test_random_sparse_problems(self):
        g = torch.Generator().manual_seed(1)
        rng = np.random.default_rng(0)
        G, K, V = 4, 5, 9
        worst = 0.0
        for _ in range(4):
            cand = np.stack([rng.permutation(V)[:K] for _ in range(G)])
            qk = rng.dirichlet(np.ones(K) * 2.0, size=G)
            Pfull = rng.dirichlet(np.ones(V) * 1.5, size=G + 1)
            lane = np.array([rng.integers(K) for _ in range(G)])
            worst = max(worst, self._check(qk, Pfull, cand, lane, g))
        self.assertGreater(worst, 0.02)           # the truncation estimate differs on these problems


if __name__ == "__main__":
    unittest.main()
