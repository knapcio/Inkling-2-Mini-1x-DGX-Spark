"""CPU tests of inkling_dfbytes: option parsing and the patch wiring on fake modules."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import inkling_dfbytes as dfb  # noqa: E402


class Parse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(dfb.parse({}), ())
        self.assertEqual(dfb.parse({dfb.ENV: "0"}), ())
        self.assertEqual(dfb.parse({dfb.ENV: "mx8, hot8"}), ("mx8", "hot8"))
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "mx8,bogus"})
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "hot8,hot16"})


@unittest.skipIf(torch is None, "needs torch")
class Wiring(unittest.TestCase):
    def setUp(self):
        from inkling_dfbytes import patches

        self.p = patches
        dfb.STATE.update(opts=("mx8", "hot8"), hot_file="x", hot_n=4)
        self.calls = []

    def test_patch_dflash(self):
        calls = self.calls

        class Base:
            def load_weights(self, weights):
                calls.append("load")
                return "loaded"

        class Sel(Base):
            lm_head = None
            merge_candidates_fused = False

            def compute_candidates(self, hidden):
                calls.append("stock")
                return "stock"

            def _transform_unary_logits(self, v):
                return v

        mod = types.SimpleNamespace(DFlashDraftModel=Base, DFlash2DraftModel=Sel, radix_topk=None)
        prepared = []
        orig_prep = self.p.prepare_layers
        self.p.prepare_layers = lambda m: prepared.append(m)
        try:
            self.p.patch_dflash(mod)
            self.p.patch_dflash(mod)  # idempotent
            s = Sel()
            self.assertEqual(s.load_weights([]), "loaded")
            self.assertEqual(prepared, [s])
            self.assertEqual(s.compute_candidates(torch.zeros(2, 4)), "stock")  # no lm_head yet: stock
        finally:
            self.p.prepare_layers = orig_prep

    def test_worker_builds_hot_before_capture(self):
        built = []
        orig = self.p._ensure_hot
        self.p._ensure_hot = lambda model, head: built.append((model, head))
        try:
            class W:
                def _maybe_build_draft_sampler(self):
                    return "built"

            mod = types.SimpleNamespace(DFlashWorkerV2=W, unwrap_lora_layer=lambda m: m)
            self.p.patch_worker(mod)
            w = W()
            head = object()
            w.draft_model = types.SimpleNamespace(candidate_selector=object())
            w._target_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
                model=types.SimpleNamespace(lm_head=head)))
            self.assertEqual(w._maybe_build_draft_sampler(), "built")
            self.assertEqual(built, [(w.draft_model, head)])
        finally:
            self.p._ensure_hot = orig


if __name__ == "__main__":
    unittest.main()
