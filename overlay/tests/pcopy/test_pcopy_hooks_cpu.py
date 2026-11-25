"""CPU tests of inkling_pcopy's history and hooks on stand-in worker / batch objects (torch on CPU; the matcher runs its
reference): seeding (fresh, chunked, gap, rewind), the per-step coverage, anchor / block writes, the override of
tokens and of the sampled selector sample, bypasses, the verify-cap minimum, the estimator mask, the failure phases,
refusals and installation.

    python3 -m unittest discover -s tests/pcopy
"""
import io
import os
import sys
import types
import unittest
from array import array
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from inkling_pcopy import history as hmod
    from inkling_pcopy import patch, rule
    import inkling_pcopy as pc
    from inkling_pcopy import _hook, _sources

G = 7
K = 4


def quiet():
    return redirect_stderr(io.StringIO())


def prompt_of_case():
    """the gate tests' prompt (doc(200) + 2 + 30 tokens)"""
    return [0] * 232


def make_req(tokens, slot, extend_end=None, streaming=False):
    return types.SimpleNamespace(full_untruncated_fill_ids=array("q", tokens), origin_input_ids=array("q", tokens),
                                 extend_end=extend_end, kv=types.SimpleNamespace(req_pool_idx=slot),
                                 session=types.SimpleNamespace(streaming=True) if streaming else None)


class FakeWorker:
    def __init__(self, rows=4, cap=512):
        self.gen = torch.zeros(rows, dtype=torch.int64)
        self.model_runner = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_generation=self.gen),
            model_config=types.SimpleNamespace(vocab_size=1000))
        self._inkling_pcopy = {"hist": hmod.History(rows, cap, torch.device("cpu")), "forced_any": None,
                               "vocab": 1000}
        self._draft_block_tokens_buf = torch.zeros((8, 1 + G), dtype=torch.int64)
        self._selector_sample = None


def decode_batch(reqs, slots, greedy=True, **kw):
    si = types.SimpleNamespace(is_all_greedy=greedy, has_custom_logit_processor=kw.get("clp", False),
                               need_min_p_sampling=kw.get("min_p", False))
    return types.SimpleNamespace(reqs=reqs, req_pool_indices=torch.tensor(slots), sampling_info=si,
                                 has_grammar=kw.get("grammar", False))


def doc(n, seed=1):
    return [(i * 7919 + seed * 104729) % 900 + 1 for i in range(n)]


@unittest.skipIf(torch is None, "needs torch")
class History(unittest.TestCase):
    def test_seed_fresh_chunk_gap_rewind(self):
        h = hmod.History(2, 64, torch.device("cpu"))
        t = list(range(100, 160))
        h.seed([(0, 1, 0, 20, array("q", t))])
        self.assertEqual(h.hist[0, :20].tolist(), t[:20])
        self.assertEqual(int(h.cov[0]), 20)
        h.seed([(0, 1, 20, 40, array("q", t))])           # next chunk appends
        self.assertEqual(h.hist[0, :40].tolist(), t[:40])
        self.assertEqual(int(h.cov[0]), 40)
        h.hist[0, :40] = 0
        h.seed([(0, 1, 50, 55, array("q", t))])           # gap: reseed from 0
        self.assertEqual(h.hist[0, :55].tolist(), t[:55])
        h.seed([(0, 2, 30, 35, array("q", t[::-1]))])      # new generation: reseed from 0
        self.assertEqual(h.hist[0, :35].tolist(), t[::-1][:35])
        h.seed([(0, 2, 10, 20, array("q", t))])            # rewind inside the same generation: rewrite from prefix
        self.assertEqual(h.hist[0, 10:20].tolist(), t[10:20])
        self.assertEqual(int(h.cov[0]), 20)

    def test_capacity_and_bad_slots(self):
        h = hmod.History(2, 16, torch.device("cpu"))
        self.assertEqual(h.seed([(5, 1, 0, 10, array("q", range(10))), (1, 1, 0, 40, array("q", range(40)))]), [])
        self.assertEqual((h.gen[1], int(h.cov[1])), (-1, 0))   # a span past the capacity: never matched
        self.assertEqual(h.seed([(0, 1, 0, 12, array("q", range(10)))]), [])  # past the fill ids
        self.assertEqual(h.gen[0], -1)

    def test_empty_rewind(self):
        h = hmod.History(1, 64, torch.device("cpu"))
        h.seed([(0, 1, 0, 40, array("q", range(1, 41)))])
        h.seed([(0, 1, 20, 20, array("q", range(1, 41)))])   # an empty extend rewound to 20
        self.assertEqual((h.valid_end[0], int(h.cov[0])), (20, 20))

    def test_coverage_and_block_writes(self):
        h = hmod.History(2, 64, torch.device("cpu"))
        h.seed([(0, 1, 0, 10, array("q", range(1, 11)))])
        slots, p0 = torch.tensor([0, 1]), torch.tensor([10, 5])
        ok = h.covered(slots, p0, torch.tensor([True, True]))
        self.assertEqual(ok.tolist(), [True, False])        # slot 1 never seeded
        h.write_block(slots, p0, torch.tensor([[77], [88]]), ok, offset=0)
        h.write_block(slots, p0, torch.arange(14).view(2, 7) + 200, ok, offset=1)
        self.assertEqual(int(h.hist[0, 10]), 77)
        self.assertEqual(h.hist[0, 11:18].tolist(), list(range(200, 207)))
        self.assertEqual(int(h.cov[0]), 18)
        self.assertEqual(int(h.cov[1]), 0)
        self.assertEqual(h.hist[1].abs().sum().item(), 0)    # nothing written for the uncovered row
        # next step: 3 drafts accepted -> p0 = 14 is covered (<= 18), p0 = 19 is not
        self.assertTrue(bool(h.covered(torch.tensor([0]), torch.tensor([14]), torch.tensor([True]))[0]))
        self.assertFalse(bool(h.covered(torch.tensor([0]), torch.tensor([19]), torch.tensor([True]))[0]))

    def test_negative_anchor_and_distinct_sinks(self):
        h = hmod.History(3, 64, torch.device("cpu"))
        h.seed([(0, 1, 0, 20, array("q", range(1, 21))), (1, 1, 0, 20, array("q", range(1, 21)))])
        ok = h.covered(torch.tensor([0, 1, 2]), torch.tensor([-1, 20, 3]), torch.tensor([True, True, True]))
        self.assertEqual(ok.tolist(), [False, True, False])
        before = h.hist.clone()
        h.write_block(torch.tensor([0, 2, 1]), torch.tensor([-1, 3, 20]), torch.tensor([[7], [8], [9]]),
                      torch.tensor([False, False, True]))
        self.assertTrue(torch.equal(h.hist[0], before[0]) and torch.equal(h.hist[2], before[2]))
        self.assertEqual(int(h.hist[1, 20]), 9)
        tail = h.flat[h.rows * h.cap:]
        self.assertEqual(sorted(tail[:2].tolist()), [7, 8])  # each masked token in its own sink cell

    def test_block_past_capacity(self):
        h = hmod.History(1, 16, torch.device("cpu"))
        h.seed([(0, 1, 0, 12, array("q", range(1, 13)))])
        ok = h.covered(torch.tensor([0]), torch.tensor([12]), torch.tensor([True]))
        h.write_block(torch.tensor([0]), torch.tensor([12]), torch.arange(7)[None] + 50, ok, offset=1)
        self.assertEqual(h.hist[0, 13:16].tolist(), [50, 51, 52])  # the rest went to the scratch sinks
        self.assertEqual(int(h.cov[0]), 16)                         # coverage clamped to the capacity


@unittest.skipIf(torch is None, "needs torch")
class Step(unittest.TestCase):
    def setUp(self):
        patch.STATE["inert"] = ""
        self.saved_min_ctx = patch.STATE["min_ctx"]
        patch.STATE["min_ctx"] = 0                           # the core on small fixtures; the gate has its own tests
        vstate = types.ModuleType("inkling_vcap.state")
        vstate.STATE = types.SimpleNamespace(df_min_live=None, steplog=None)
        self.saved = sys.modules.get("inkling_vcap.state")
        sys.modules["inkling_vcap.state"] = vstate
        self.vstate = vstate.STATE

    def tearDown(self):
        patch.STATE["min_ctx"] = self.saved_min_ctx
        if self.saved is None:
            sys.modules.pop("inkling_vcap.state", None)
        else:
            sys.modules["inkling_vcap.state"] = self.saved

    def prefill(self, w, req, slot, prompt_len):
        w.gen[slot] += 1
        req.extend_end = prompt_len
        batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_extend=lambda: True), reqs=[req],
                                      prefix_lens=[0], extend_lens=[prompt_len])
        with quiet():
            patch.make_forward(lambda self, b: "out")(w, batch)

    def run_step(self, w, batch, prefix, anchors, drafts):
        bs = len(prefix)
        w._draft_block_tokens_buf[:bs, 0] = torch.tensor(anchors)
        w._draft_block_tokens_buf[:bs, 1:] = torch.tensor(drafts)
        seen = {}

        def orig(self, *, batch, draft_input, prefix_lens):
            seen["min_live"] = self_vstate.df_min_live.clone() if self_vstate.df_min_live is not None else None
            return (None, None, None)

        self_vstate = self.vstate
        out = patch.make_plan_verify(orig)(w, batch=batch, draft_input=None, prefix_lens=torch.tensor(prefix))
        self.assertIsNone(self.vstate.df_min_live)          # cleared after the verify planning
        return seen

    def test_forced_copy_from_the_prompt(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]                    # the last 30 tokens repeat d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)           # the last prompt token is the anchor (bonus position)
        p0 = len(prompt) - 1
        seen = self.run_step(w, decode_batch([req], [0]), [p0], [prompt[-1]], [[0] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), d[130:137])
        self.assertEqual(seen["min_live"].tolist(), [G + 1])
        self.assertTrue(bool(w._inkling_pcopy["forced_any"][0]))
        h = w._inkling_pcopy["hist"]
        self.assertEqual(h.hist[0, p0:p0 + 8].tolist(), [prompt[-1]] + d[130:137])
        self.assertEqual(int(h.cov[0]), p0 + 8)

    def test_short_match_not_forced(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:113]                    # only 13 matching tokens: below the force length
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        seen = self.run_step(w, decode_batch([req], [0]), [len(prompt) - 1], [prompt[-1]], [[3] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [3] * G)
        self.assertEqual(seen["min_live"].tolist(), [0])

    def test_sampled_rows_get_point_masses(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        ids = torch.arange(G * K).view(1, G, K) + 2000
        q = torch.full((1, G, K), 1.0 / K)
        w._selector_sample = (ids, q)
        self.run_step(w, decode_batch([req], [0], greedy=False), [len(prompt) - 1], [prompt[-1]], [[0] * G])
        ids2, q2 = w._selector_sample
        self.assertIsNot(ids2, ids)
        self.assertEqual(ids2[0, :, 0].tolist(), d[130:137])
        self.assertTrue(torch.equal(q2[0, :, 0], torch.ones(G)))
        self.assertEqual(int(ids[0, 0, 0]), 2000)            # the sampler's buffers untouched

    def test_bypasses_still_write_history(self):
        for kw in ({"grammar": True}, {"clp": True}, {"min_p": True}):
            w = FakeWorker()
            d = doc(200)
            prompt = d + [5, 6] + d[100:130]
            req = make_req(prompt, 0)
            self.prefill(w, req, 0, len(prompt) - 1)
            p0 = len(prompt) - 1
            self.run_step(w, decode_batch([req], [0], **kw), [p0], [prompt[-1]], [[9] * G])
            self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [9] * G, msg=kw)
            self.assertEqual(int(w._inkling_pcopy["hist"].cov[0]), p0 + 8, msg=kw)
        w = FakeWorker()
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        self.run_step(w, decode_batch([req], [0], greedy=False), [len(prompt) - 1], [prompt[-1]], [[9] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [9] * G)  # sampled without a selector sample

    def test_streaming_and_stale_generation_not_forced(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]
        req = make_req(prompt, 0, streaming=True)
        self.prefill(w, req, 0, len(prompt) - 1)
        self.run_step(w, decode_batch([req], [0]), [len(prompt) - 1], [prompt[-1]], [[9] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [9] * G)
        w = FakeWorker()
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        w.gen[0] += 1                                        # slot reallocated, history not reseeded
        self.run_step(w, decode_batch([req], [0]), [len(prompt) - 1], [prompt[-1]], [[9] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [9] * G)

    def test_multi_step_bookkeeping(self):
        w = FakeWorker()
        d = doc(300)
        prompt = d
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        h = w._inkling_pcopy["hist"]
        p0, anchor = len(prompt) - 1, prompt[-1]
        committed = list(prompt)
        for step in range(5):
            drafts = [(step * 13 + j) % 900 + 1 for j in range(G)]
            self.run_step(w, decode_batch([req], [0]), [p0], [anchor], [drafts])
            block = w._draft_block_tokens_buf[0, 1:].tolist()
            a = step % 4                                      # accept a drafts, then a bonus
            bonus = 950 + step
            committed = committed + block[:a] + [bonus]
            p0, anchor = p0 + a + 1, bonus
        L = p0 + 1
        self.assertEqual(h.hist[0, :p0].tolist(), committed[:p0])  # every position below the anchor committed

    def test_estimator_mask(self):
        w = FakeWorker()
        calls = {}
        w._acceptance_estimator = types.SimpleNamespace(
            observe=lambda **kw: calls.update(kw))
        w._inkling_pcopy["forced_any"] = torch.tensor([True, False])

        def orig(self, **kw):
            self._acceptance_estimator.observe(features=None, num_correct_drafts=torch.tensor([3, 2]),
                                               num_graded_drafts=torch.tensor([7, 7]))

        patch.make_observe(orig)(w)
        self.assertEqual(calls["num_graded_drafts"].tolist(), [0, 7])
        self.assertFalse("observe" in w._acceptance_estimator.__dict__ and
                         w._acceptance_estimator.observe.__name__ == "masked")

    def test_failure_before_override_is_stock(self):
        w = FakeWorker()
        req = make_req(doc(50), 0)
        self.prefill(w, req, 0, 49)                          # a row that may be matched (else the step is skipped)
        w._inkling_pcopy["hist"].covered = None              # any failure before the override
        batch = decode_batch([req], [0])
        w._draft_block_tokens_buf[0, 1:] = 4
        err = io.StringIO()
        with redirect_stderr(err):
            patch.make_plan_verify(lambda self, **kw: "planned")(w, batch=batch, draft_input=None,
                                                                prefix_lens=torch.tensor([49]))
        self.assertIn("inert", err.getvalue())
        self.assertIsNone(w._inkling_pcopy)
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [4] * G)

    def test_failure_preparing_the_sample_is_stock(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        w._selector_sample = ("not", "tensors")             # fails while preparing, before the in-place copy
        with quiet():
            self.run_step(w, decode_batch([req], [0], greedy=False), [len(prompt) - 1], [prompt[-1]], [[0] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [0] * G)   # drafts untouched
        self.assertIsNone(w._inkling_pcopy)

    def test_failure_after_the_copy_raises(self):
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        real = patch._set_min_live
        calls = {"n": 0}

        def boom(t):
            calls["n"] += 1
            if t is not None:
                raise RuntimeError("after the copy")
            return real(t)

        patch._set_min_live = boom
        try:
            with self.assertRaises(RuntimeError):
                self.run_step(w, decode_batch([req], [0]), [len(prompt) - 1], [prompt[-1]], [[0] * G])
        finally:
            patch._set_min_live = real

    def test_padding_id_stops_the_copy(self):
        w = FakeWorker()
        d = doc(200)
        d[132] = 5000                                       # an id past the real vocabulary inside the continuation
        prompt = d + [5, 6] + d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        seen = self.run_step(w, decode_batch([req], [0]), [len(prompt) - 1], [prompt[-1]], [[9] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), d[130:132] + [9] * 5)
        self.assertEqual(seen["min_live"].tolist(), [3])

    def gate_case(self, min_ctx):
        patch.STATE["min_ctx"] = min_ctx
        w = FakeWorker()
        d = doc(200)
        prompt = d + [5, 6] + d[100:130]                    # 232 tokens; the last 30 repeat d[100:130]
        req = make_req(prompt, 0)
        self.prefill(w, req, 0, len(prompt) - 1)
        p0 = len(prompt) - 1
        seen = self.run_step(w, decode_batch([req], [0]), [p0], [prompt[-1]], [[0] * G])
        return w, seen, prompt, d

    def test_gate_below_min_ctx_is_stock(self):
        gated = patch.STATE["stats"]["gated"]
        w, seen, prompt, d = self.gate_case(len(prompt_of_case()) + 1)      # one token short of the minimum
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [0] * G)   # drafts untouched
        self.assertIsNone(seen["min_live"])                                    # no cap minimum
        self.assertIsNone(w._inkling_pcopy["forced_any"])
        self.assertEqual(int(w._inkling_pcopy["hist"].cov[0]), 0)             # its history was never seeded
        self.assertEqual(patch.STATE["stats"]["gated"], gated + 1)

    def test_gate_at_min_ctx_copies(self):
        w, seen, prompt, d = self.gate_case(len(prompt_of_case()))          # exactly the minimum
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), d[130:137])
        self.assertEqual(seen["min_live"].tolist(), [G + 1])

    def test_gate_after_a_forced_step_on_the_same_worker(self):
        patch.STATE["min_ctx"] = 100
        w = FakeWorker()
        d = doc(200)
        long_p = d + [5, 6] + d[100:130]
        req = make_req(long_p, 0)
        self.prefill(w, req, 0, len(long_p) - 1)
        seen = self.run_step(w, decode_batch([req], [0]), [len(long_p) - 1], [long_p[-1]], [[0] * G])
        self.assertEqual(seen["min_live"].tolist(), [G + 1])             # the first step forced a copy
        short = make_req([x + 3 for x in d[:40]] + [x + 3 for x in d[10:30]], 0)   # gated, reusing slot 0
        self.prefill(w, short, 0, 59)
        dec = decode_batch([short], [0])
        dec.forward_mode = types.SimpleNamespace(is_extend=lambda: False)
        with quiet():
            patch.make_forward(lambda self, b: "out")(w, dec)              # the next step's forward resets the outputs
        seen = self.run_step(w, dec, [59], [short.origin_input_ids[-1]], [[7] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), [7] * G)   # untouched
        self.assertIsNone(seen["min_live"])                                    # the planner sees no minimum
        self.assertIsNone(w._inkling_pcopy["forced_any"])                      # the estimator is not masked

    def test_gate_mixed_batch(self):
        patch.STATE["min_ctx"] = 100
        w = FakeWorker()
        d = doc(200)
        long_p = d + [5, 6] + d[100:130]
        short_p = [x + 1 for x in d[:40]] + [x + 1 for x in d[10:30]]    # 60 tokens (gated), with a repeat
        reqs = [make_req(long_p, 0), make_req(short_p, 1)]
        for i, r in enumerate(reqs):
            self.prefill(w, r, i, len(r.origin_input_ids) - 1)
        p = [len(long_p) - 1, len(short_p) - 1]
        seen = self.run_step(w, decode_batch(reqs, [0, 1]), p, [long_p[-1], short_p[-1]], [[0] * G, [0] * G])
        self.assertEqual(w._draft_block_tokens_buf[0, 1:].tolist(), d[130:137])   # the long prompt copies
        self.assertEqual(w._draft_block_tokens_buf[1, 1:].tolist(), [0] * G)      # the short one is untouched
        self.assertEqual(seen["min_live"].tolist(), [G + 1, 0])


@unittest.skipIf(torch is None, "needs torch")
class Install(unittest.TestCase):
    def test_min_ctx_env(self):
        self.assertEqual(pc.min_ctx_from_env({}), 8192)
        self.assertEqual(pc.min_ctx_from_env({pc.ENV_MIN_CTX: "0"}), 0)
        self.assertEqual(pc.min_ctx_from_env({pc.ENV_MIN_CTX: "16384"}), 16384)
        for raw in ("x", "-1", "8k", "1.5"):
            with self.assertRaises(RuntimeError):
                pc.min_ctx_from_env({pc.ENV_MIN_CTX: raw})

    def test_env(self):
        self.assertEqual(pc.key_from_env({}), 0)
        self.assertEqual(pc.key_from_env({pc.ENV: "0"}), 0)
        self.assertEqual(pc.key_from_env({pc.ENV: "12"}), 12)
        for raw in ("8", "x", "16"):
            with self.assertRaises(RuntimeError):
                pc.key_from_env({pc.ENV: raw})

    def test_install_registers_hooks(self):
        pc.STATE.update(installed=False, key=0)
        saved = _sources.check
        _sources.check = lambda *a, **k: None
        try:
            with quiet():
                self.assertEqual(pc.install({pc.ENV: "12"}), 12)
            for m in patch.MODULES:
                self.assertIn(m, _hook.pending())
        finally:
            _sources.check = saved
            for m in patch.MODULES:
                _hook._callbacks.pop(m, None)
            pc.STATE.update(installed=False, key=0)

    def test_rsfix_gate_reads_parts_and_targets(self):
        """the serving rsfix STATE: parts are part names, applied are target modules"""
        fake = types.ModuleType("inkling_rsfix")
        fake.STATE = {"parts": ("reject", "markov", "blockwise", "selector"),
                      "applied": ["blockwise", "markov", "reject", "selector_kernel", "selector_lilicorr",
                                  "selector_model"]}
        saved = {k: sys.modules.get(k) for k in ("inkling_rsfix", "sglang.srt.runtime_context")}
        rc = types.ModuleType("sglang.srt.runtime_context")
        rc.get_disagg = lambda: types.SimpleNamespace(disaggregation_mode="null", enable_pd_role_switch=False)
        rc.get_parallel = lambda: types.SimpleNamespace(attn_dp_enabled=False)
        rc.get_schedule = lambda: types.SimpleNamespace(enable_mixed_chunk=False)
        sys.modules["inkling_rsfix"], sys.modules["sglang.srt.runtime_context"] = fake, rc
        try:
            w = types.SimpleNamespace(selector=object(), draft_kv_ring=object(), _verify_planner=None,
                                      _megakernel_sampled=None)
            self.assertIsNone(patch.refusal(w))
            fake.STATE = {"parts": ("reject",), "applied": ["reject"]}
            self.assertIn("rsfix", patch.refusal(w))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_pins_list(self):
        self.assertEqual(sorted(_sources.EXPECTED[_sources.DFLASH_WORKER]),
                         ["DFlashWorkerV2._init_draft_kv_ring", "DFlashWorkerV2._observe_acceptance",
                          "DFlashWorkerV2._plan_verify", "DFlashWorkerV2.forward_batch_generation"])

    def test_outermost_over_the_verify_cap(self):
        """sitecustomize installs the verify cap first: its _plan_verify wrapper is applied first, pcopy's around it,
        so pcopy's step runs before the cap plans (and reads min_live)."""
        order = []
        cls = type("DFlashWorkerV2", (), {n: (lambda self, *a, **k: None) for n in
                                           ("_init_draft_kv_ring", "forward_batch_generation", "_observe_acceptance")})

        def stock_plan(self, *, batch, draft_input, prefix_lens):
            order.append("stock")
            return (None, None, None)

        def vcap_plan(self, *, batch, draft_input, prefix_lens):
            order.append("vcap")
            return stock_plan(self, batch=batch, draft_input=draft_input, prefix_lens=prefix_lens)

        cls._plan_verify = vcap_plan
        mod = types.SimpleNamespace(DFlashWorkerV2=cls)
        patch.patch_worker(mod)
        self.assertIs(cls._plan_verify.__wrapped__, vcap_plan)
        w = cls()
        w._inkling_pcopy = None
        cls._plan_verify(w, batch=None, draft_input=None, prefix_lens=None)
        self.assertEqual(order, ["vcap", "stock"])

    def test_patch_worker_once(self):
        cls = type("DFlashWorkerV2", (), {n: (lambda self, *a, **k: None) for n in
                                           ("_init_draft_kv_ring", "forward_batch_generation", "_plan_verify",
                                            "_observe_acceptance")})
        mod = types.SimpleNamespace(DFlashWorkerV2=cls)
        self.assertTrue(patch.patch_worker(mod))
        self.assertFalse(patch.patch_worker(mod))


if __name__ == "__main__":
    unittest.main()
