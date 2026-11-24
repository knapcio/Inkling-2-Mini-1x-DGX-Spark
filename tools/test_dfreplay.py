"""dfreplay on synthetic steplogs with a known acceptance law and a known verify cost (CPU).

  python -m unittest tools/test_dfreplay.py   (from the repository root)

Acceptance of draft j given d_<j accepted is m_j itself for greedy rows (the walked-row max at T = 1 is calibrated
by construction); verify ms = 10 + 0.25 * sum_visits U_8 + noise. The replay must recover beta, report a calibrated
table, and find a positive held-out gain when rows are expensive (the policy cuts low-survival rows) and ~0 when they
are free."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfreplay  # noqa: E402

V, K, G, S, TOPK = 8, 16, 7, 8, 4  # visits, candidates, drafts, stride, top-k router


def synth(directory, *, runs=4, steps=400, bs_list=(1, 4), seed=0, beta=0.25):
    rng = np.random.default_rng(seed)
    os.makedirs(directory, exist_ok=True)
    for run in range(runs):
        bs = bs_list[run % len(bs_list)]
        rids = [f"r{run}-{i}" for i in range(bs)]
        recs = dict(step=[], bs=[], rid=[], rows=[], m=[], gap=[], t_eff=[], greedy=[], temp=[], context=[],
                    accepted=[], union=[], visits=[], walks=[], t_start_ms=[], t_draft_ms=[], t_verify_ms=[],
                    host_t=[], ids=[], ids_req=[])
        nreq = 0
        for st in range(steps):
            rows = rng.normal(0, rng.uniform(0.5, 4.0, size=(bs, 1, 1)), size=(bs, G, K))
            m, gap = dfreplay.features(dict(rows=rows, t_eff=np.ones(bs)))
            acc = np.zeros(bs, dtype=np.int64)
            for r in range(bs):
                a = 0
                while a < G and rng.random() < m[r, a]:
                    a += 1
                acc[r] = a
            # routed ids: each row draws TOPK experts from a request-specific pool (correlated within a request)
            ids = np.zeros((bs, 64, S, 8), dtype=np.uint8) + 255
            union = np.full((64, S), -1, dtype=np.int16)
            for v in range(V):
                pool = [rng.choice(128, 24, replace=False) for _ in range(bs)]
                for r in range(bs):
                    for row in range(S):
                        ids[r, v, row, :TOPK] = rng.choice(pool[r], TOPK, replace=False)
                for w in range(1, S + 1):
                    ex = set(ids[:, v, :w, :TOPK].reshape(-1).tolist())
                    union[v, w - 1] = len(ex)
            u8 = union[:V, 7].sum()
            recs["step"].append(st)
            recs["bs"].append(bs)
            recs["rid"].extend(rids)
            recs["rows"].append(rows.astype(np.float32))
            recs["m"].append(m.astype(np.float32))
            recs["gap"].append(gap.astype(np.float32))
            recs["t_eff"].append(np.ones(bs, np.float32))
            recs["greedy"].append(np.ones(bs, bool))
            recs["temp"].append(np.zeros(bs, np.float32))
            recs["context"].append(np.full(bs, 100 + st))
            recs["accepted"].append(acc.astype(np.int8))
            recs["union"].append(union)
            recs["visits"].append(V)
            recs["walks"].append(1)
            recs["t_start_ms"].append(st * 50.0)
            recs["t_draft_ms"].append(5.0)
            recs["t_verify_ms"].append(10.0 + beta * u8 + rng.normal(0, 0.3))
            recs["host_t"].append(0.0)
            recs["ids"].append(ids)
            recs["ids_req"].extend(range(nreq, nreq + bs))
            nreq += bs
        np.savez_compressed(
            os.path.join(directory, f"dflash1-run{run}-000000.npz"), format=dfreplay.FORMAT, run=f"run{run}",
            stride=S, top_k=K, step=np.array(recs["step"]), bs=np.array(recs["bs"], np.int32),
            rid=np.array(recs["rid"]), rows=np.concatenate(recs["rows"]), m=np.concatenate(recs["m"]),
            gap=np.concatenate(recs["gap"]), t_eff=np.concatenate(recs["t_eff"]),
            greedy=np.concatenate(recs["greedy"]), temp=np.concatenate(recs["temp"]),
            context=np.concatenate(recs["context"]), accepted=np.concatenate(recs["accepted"]),
            union=np.stack(recs["union"]), visits=np.array(recs["visits"], np.int16),
            walks=np.array(recs["walks"], np.int32), t_start_ms=np.array(recs["t_start_ms"]),
            t_draft_ms=np.array(recs["t_draft_ms"], np.float32), t_verify_ms=np.array(recs["t_verify_ms"], np.float32),
            host_t=np.array(recs["host_t"]), ids=np.concatenate(recs["ids"]), ids_req=np.array(recs["ids_req"]),
            unary=np.concatenate(recs["rows"]), lse=np.log(np.exp(np.concatenate(recs["rows"])).sum(-1) / 0.8))


def run(dirs, *extra):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    subprocess.run([sys.executable, os.path.join(HERE, "dfreplay.py"), *dirs, "--json", out, "--bootstrap", "100",
                    "--visits", str(V),
                    *extra], check=True, capture_output=True, text=True)
    with open(out) as fh:
        return json.load(fh)


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfreplay-")
        cls.dir = os.path.join(cls.tmp, "a")
        synth(cls.dir, runs=6, steps=300)
        cls.rep = run([cls.dir])

    def test_beta_recovered(self):
        for fold in self.rep["folds"]:
            for band in ("1", "4"):
                self.assertAlmostEqual(fold["beta"][band]["beta"], 0.25, delta=0.02)

    def test_features_recomputed(self):
        self.assertLess(self.rep["m_recompute_maxerr"], 1e-5)

    def test_gain_when_rows_cost(self):
        s = self.rep["scores"]["all"]
        self.assertGreater(s["gain"], 0.0)
        self.assertLess(s["mean_width"], 8.0)
        self.assertLess(s["tokens_ratio"], 1.0)
        self.assertIn("ci95", s)

    def test_no_gain_when_rows_free(self):
        rep = run([self.dir], "--beta-scale", "0", "--policy-ms", "0")
        s = rep["scores"]["all"]
        self.assertAlmostEqual(s["gain"], 0.0, delta=1e-9)
        self.assertAlmostEqual(s["mean_width"], 8.0, delta=1e-9)

    def test_feature_sets_and_boundary(self):
        rep = run([self.dir], "--features", "slot,mt,prev")
        self.assertEqual(rep["features"], ["slot", "mt", "prev"])
        self.assertGreater(rep["scores"]["all"]["gain"], 0.0)
        R, S_ = dfreplay.load([self.dir])
        dfreplay.prepare(R, S_)
        m, gap = dfreplay.features(R)
        bins = dfreplay.feature_bins(R, m, gap)
        tail = 1 - np.exp(R["unary"] - R["lse"][..., None]).sum(-1)
        self.assertTrue(np.allclose(tail, 0.2, atol=1e-5))  # synthetic plane holds 20 % outside the top-K
        self.assertEqual(set(bins), set(dfreplay.FEATURES))
        with self.assertRaises(ValueError):
            dfreplay.Calib(("slot", "m", "gap", "tail"))
        out = subprocess.run([sys.executable, os.path.join(HERE, "dfreplay.py"), self.dir, "--features",
                              "slot,m,gap,tail", "--visits", str(V)], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)

    def test_exact_bv_scoring_for_sampled(self):
        # sampled synthetic requests with target probabilities on the candidates: the replay scores them by the exact
        # stopped-block expectation (bvexpect), not by min(A_full, L-1)
        d = os.path.join(self.tmp, "bv")
        synth(d, runs=2, steps=120, bs_list=(1,), seed=4)
        for f in os.listdir(d):
            z = dict(np.load(os.path.join(d, f)))
            n = len(z["rid"])
            rng = np.random.default_rng(1)
            z["greedy"] = np.zeros(n, bool)
            z["t_eff"] = np.ones(n, np.float32)
            z["cand_ids"] = np.tile(np.arange(K, dtype=np.int32), (n, G, 1))
            z["drafted"] = rng.integers(0, K, size=(n, G)).astype(np.int32)
            z["tgt_cand_p"] = rng.dirichlet(np.ones(K), size=(n, G)).astype(np.float32) * 0.9
            np.savez_compressed(os.path.join(d, f), **z)
        R, S_ = dfreplay.load([d])
        dfreplay.prepare(R, S_)
        self.assertIsNotNone(R["bvx"])
        self.assertTrue(np.isfinite(R["bvx"]).all())
        # (no monotonicity in L per fixed realized block: block verification is optimal in expectation over the
        # draft's own sampling, not for one conditioned block)
        self.assertTrue((R["bvx"][:, 0] == 0).all() and (R["bvx"] <= np.arange(8)[None, :] + 1e-12).all())
        exact = run([d])
        trunc = run([d], "--no-bv-exact")
        self.assertNotAlmostEqual(exact["scores"]["all"]["full_tok_s"], trunc["scores"]["all"]["full_tok_s"])

    def test_union_exact_vs_interpolated(self):
        R, S_ = dfreplay.load([self.dir])
        dfreplay.prepare(R, S_)
        s = 5
        rows = R["by_step"][s]
        raw = [R["raw"][int(r)] for r in rows]
        full = dfreplay.union_for(np.full(len(rows), 8), raw, S_["union"][s])
        self.assertEqual(full, float(S_["union"][s][:V, 7].sum()))
        for w in range(1, 9):
            self.assertEqual(dfreplay.union_for(np.full(len(rows), w), raw, S_["union"][s]),
                             float(S_["union"][s][:V, w - 1].sum()))
            self.assertAlmostEqual(dfreplay.union_for(np.full(len(rows), w), None, S_["union"][s]),
                                   float(S_["union"][s][:V, w - 1].sum()))


if __name__ == "__main__":
    unittest.main()
