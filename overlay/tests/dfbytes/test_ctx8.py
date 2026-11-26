"""CPU tests of inkling_dfbytes ctx8 (MXFP8 context projections): switch parsing, install wiring, source pins, and the
dispatch of the two patched functions on fake modules (the b12x GEMM and the MXFP8 build are replaced by CPU stand-ins;
tools/gpu_check_ctx8.py checks the real ones)."""
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
from inkling_dfbytes import _hook  # noqa: E402


class ParseCtx8(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(dfb.parse_ctx8({}), (False, 256))
        for off in ("0", "off", "no", ""):
            self.assertEqual(dfb.parse_ctx8({dfb.CTX8_ENV: off}), (False, 256))
        for on in ("1", "on", "yes", "TRUE"):
            self.assertEqual(dfb.parse_ctx8({dfb.CTX8_ENV: on}), (True, 256))
        self.assertEqual(dfb.parse_ctx8({dfb.CTX8_ENV: "1", dfb.CTX8_MAX_M_ENV: "64"}), (True, 64))
        for bad in ("2", "mx8", "x"):
            with self.assertRaises(RuntimeError):
                dfb.parse_ctx8({dfb.CTX8_ENV: bad})
        for bad in ("0", "4097", "x", "-1"):
            with self.assertRaises(RuntimeError):
                dfb.parse_ctx8({dfb.CTX8_ENV: "1", dfb.CTX8_MAX_M_ENV: bad})
            self.assertEqual(dfb.parse_ctx8({dfb.CTX8_ENV: "0", dfb.CTX8_MAX_M_ENV: bad}), (False, 256))  # off: ignored

    def test_aux_graph_requested(self):
        self.assertFalse(dfb.aux_graph_requested({}))
        self.assertTrue(dfb.aux_graph_requested({"SGLANG_OPT_DFLASH_AUX_GRAPH": "1"}))
        self.assertTrue(dfb.aux_graph_requested({"SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": "true"}))
        self.assertFalse(dfb.aux_graph_requested({"SGLANG_OPT_DFLASH_AUX_GRAPH": "0",
                                                  "SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": "1"}))
        # SGLang's EnvBool: no strip, an invalid value falls back to the default (the mega flag here)
        mega = {"SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": "1"}
        for bad in ("off", " 0 ", "disabled"):
            self.assertTrue(dfb.aux_graph_requested(dict(mega, SGLANG_OPT_DFLASH_AUX_GRAPH=bad)), bad)
            self.assertFalse(dfb.aux_graph_requested({"SGLANG_OPT_DFLASH_AUX_GRAPH": bad}), bad)
        for v, want in (("Y", True), ("No", False), ("TRUE", True)):
            self.assertEqual(dfb.aux_graph_requested({"SGLANG_OPT_DFLASH_AUX_GRAPH": v}), want)
        self.assertFalse(dfb.aux_graph_requested({"SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION": "on"}))


class InstallWiring(unittest.TestCase):
    def setUp(self):
        self.saved = dict(dfb.STATE)
        self.saved_cbs = {k: list(v) for k, v in _hook._callbacks.items()}

    def tearDown(self):
        dfb.STATE.clear()
        dfb.STATE.update(self.saved)
        _hook._callbacks.clear()
        _hook._callbacks.update(self.saved_cbs)

    def _install(self, env):
        dfb.STATE["installed"] = False
        _hook._callbacks.clear()
        return dfb.install(env)

    @unittest.skipIf(torch is None, "ctx8 imports torch")
    def test_ctx8_alone_hooks_both_modules(self):
        self.assertEqual(self._install({dfb.CTX8_ENV: "1", dfb.CTX8_MAX_M_ENV: "128"}), ())
        self.assertTrue(dfb.STATE["ctx8"])
        self.assertEqual(dfb.STATE["ctx8_max_m"], 128)
        self.assertEqual(sorted(_hook.pending()), sorted([dfb.DFLASH_MODULE, dfb.FUSED_KV_MODULE, dfb.WORKER_MODULE]))
        from inkling_dfbytes import ctx8

        names = {cb.__name__ for cbs in _hook._callbacks.values() for cb in cbs}
        self.assertEqual(names, {ctx8.patch_dflash.__name__, ctx8.patch_fused_kv.__name__, ctx8.patch_worker.__name__})

    def test_off_hooks_nothing(self):
        self.assertEqual(self._install({dfb.CTX8_ENV: "0"}), ())
        self.assertFalse(dfb.STATE["ctx8"])
        self.assertEqual(_hook.pending(), [])

    def test_aux_graph_leaves_ctx8_off(self):
        self._install({dfb.CTX8_ENV: "1", "SGLANG_OPT_DFLASH_AUX_GRAPH": "1"})
        self.assertFalse(dfb.STATE["ctx8"])
        self.assertEqual(_hook.pending(), [])

    def test_bad_value_refuses(self):
        with self.assertRaises(RuntimeError):
            self._install({dfb.CTX8_ENV: "2"})


@unittest.skipIf(torch is None, "needs torch")
class Pins(unittest.TestCase):
    def test_digest_and_mismatch(self):
        from inkling_dfbytes import ctx8

        text = "class A:\n    def f(self):\n        return 1\n\n\ndef g():\n    pass\n"
        d = ctx8.digest(text, "A.f")
        self.assertEqual(len(d), 64)
        self.assertIsNone(ctx8.digest(text, "A.h"))
        self.assertEqual(ctx8.mismatches(text, {"A.f": d}), [])
        self.assertEqual(ctx8.mismatches(text.replace("return 1", "return 2"), {"A.f": d})[0][0], "A.f")
        self.assertIn("missing", ctx8.mismatches(text, {"A.h": d})[0][1])
        dup = text + "\ndef g():\n    pass\n"
        self.assertIn("missing", ctx8.mismatches(dup, {"g": ctx8.digest(text, "g")})[0][1])

    def test_pins_match_image_source_when_present(self):
        """With INKLING_SGLANG_SRC=<the pinned image's python/sglang directory>, the pins must match it."""
        from inkling_dfbytes import ctx8

        root = os.environ.get("INKLING_SGLANG_SRC", "")
        if not root:
            self.skipTest("INKLING_SGLANG_SRC not set")
        for module, pins in ctx8.PINS.items():
            rel = module.split(".", 1)[1].replace(".", "/") + ".py"
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                self.assertEqual(ctx8.mismatches(fh.read(), pins), [], module)

    def test_check_module_refuses_a_changed_file(self):
        import tempfile

        from inkling_dfbytes import ctx8

        name = "sglang.srt.models.dflash"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("class DFlashDraftModel:\n    def project_target_hidden(self, x):\n        return x\n")
        try:
            with self.assertRaises(dfb.DfbytesFatal):
                ctx8.check_module(types.SimpleNamespace(__name__=name, __file__=fh.name))
            with self.assertRaises(dfb.DfbytesFatal):
                ctx8.check_module(types.SimpleNamespace(__name__=name, __file__=None))
        finally:
            os.unlink(fh.name)


class _FakeW:
    """A Ctx8Weight stand-in: the BF16 weight itself; fake_gemm multiplies by it."""

    def __init__(self, weight):
        self.weight = weight
        self.q = weight
        self.n, self.k = weight.shape

    def nbytes(self):
        return self.weight.numel()


@unittest.skipIf(torch is None, "needs torch")
class Ctx8Dispatch(unittest.TestCase):
    def setUp(self):
        from inkling_dfbytes import ctx8

        self.c = ctx8
        self.saved = dict(dfb.STATE)
        dfb.STATE.update(ctx8=True, ctx8_max_m=256, ctx8_fc=0, ctx8_kv=0, ctx8_fc_calls=0, ctx8_kv_calls=0)
        self.gemm_calls = []
        self.orig = (ctx8.gemm, ctx8.build, ctx8.check_module, torch.Tensor.is_cuda)
        self.orig_warmup = ctx8.warmup
        self.warmed = []
        ctx8.warmup = lambda w: (self.warmed.append(w), (0.0, [1]))[1]

        def fake_gemm(x2, w, out=None):
            self.gemm_calls.append((x2.shape[0], out is not None))
            y = (x2.float() @ w.weight.float().t()).to(torch.bfloat16)
            if out is not None:
                out.copy_(y)
                return out
            return y

        ctx8.gemm = fake_gemm
        ctx8.build = lambda weight, old=None: old if old is not None else _FakeW(weight)
        ctx8.check_module = lambda mod: None
        torch.Tensor.is_cuda = property(lambda self: True)

    def tearDown(self):
        self.c.gemm, self.c.build, self.c.check_module, torch.Tensor.is_cuda = self.orig
        self.c.warmup = self.orig_warmup
        dfb.STATE.clear()
        dfb.STATE.update(self.saved)

    # ---------------------------------------------------------------- fc
    def _dflash_mod(self):
        calls = []

        class Model(torch.nn.Module):
            quantizable = False
            fc_tp_columns = None

            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.fc = torch.nn.Linear(256, 128, bias=False).to(torch.bfloat16)
                self.hidden_norm = lambda y: ("norm", y)

            def load_weights(self, weights):
                calls.append("load")
                return "loaded"

            def project_target_hidden(self, target_hidden):
                calls.append(("stock", target_hidden.shape[0]))
                return self.hidden_norm(self.fc(target_hidden.to(self.fc.weight.dtype)))

        return types.SimpleNamespace(__name__="sglang.srt.models.dflash", DFlashDraftModel=Model), Model, calls

    def test_fc_dispatch(self):
        mod, Model, calls = self._dflash_mod()
        self.assertTrue(self.c.patch_dflash(mod))
        self.assertFalse(self.c.patch_dflash(mod))  # idempotent
        m = Model()
        x = torch.randn(8, 256).to(torch.bfloat16)
        self.assertEqual(m.project_target_hidden(x)[0], "norm")
        self.assertEqual(calls, [("stock", 8)])  # not prepared yet: stock
        self.assertEqual(m.load_weights([]), "loaded")
        self.assertEqual(dfb.STATE["ctx8_fc"], 1)
        self.assertEqual(m._derived_weight_cache_error, self.c.UPDATE_ERROR)  # online updates refused
        tag, y = m.project_target_hidden(x)
        self.assertEqual(tag, "norm")  # hidden_norm is applied to the b12x output
        torch.testing.assert_close(y.float(), (x.float() @ m.fc.weight.float().t()), atol=0.05, rtol=0.02)
        self.assertEqual(self.gemm_calls, [(8, False)])
        self.assertEqual(dfb.STATE["ctx8_fc_calls"], 1)
        # rows outside 1..max_m, a strided view, a K mismatch, FP32, 3-D: stock
        dfb.STATE["ctx8_max_m"] = 16
        calls.clear()
        m.project_target_hidden(torch.zeros(17, 256, dtype=torch.bfloat16))
        m.project_target_hidden(torch.zeros(4, 512, dtype=torch.bfloat16)[:, :256])
        m.project_target_hidden(torch.zeros(4, 256, dtype=torch.float32))
        m.project_target_hidden(torch.zeros(0, 256, dtype=torch.bfloat16))
        with self.assertRaises(RuntimeError):  # K mismatch reaches the stock function (which raises in the image)
            m.project_target_hidden(torch.zeros(4, 128, dtype=torch.bfloat16))
        self.assertEqual([c[1] for c in calls if c != "load"], [17, 4, 4, 0, 4])
        m.project_target_hidden(torch.zeros(16, 256, dtype=torch.bfloat16))
        self.assertEqual(self.gemm_calls[-1], (16, False))
        # a TP-split or quantized fc: stock
        m.fc_tp_columns = (0, 64)
        calls.clear()
        m.project_target_hidden(torch.zeros(4, 256, dtype=torch.bfloat16))
        self.assertEqual(calls, [("stock", 4)])

    def test_fc_reload_refreshes_in_place(self):
        mod, Model, _ = self._dflash_mod()
        self.c.patch_dflash(mod)
        m = Model()
        m.load_weights([])
        first = m.fc.__dict__["_idf_c8"]
        m.load_weights([])
        self.assertIs(m.fc.__dict__["_idf_c8"], first)
        self.assertEqual(self.warmed, [first])  # warmed once, at the first load
        self.assertEqual(dfb.STATE["ctx8_fc"], 2)

    def test_fc_unsupported_stays_stock(self):
        mod, Model, calls = self._dflash_mod()
        self.c.patch_dflash(mod)
        m = Model()
        m.fc_tp_columns = (0, 64)
        self.assertEqual(m.load_weights([]), "loaded")
        m2 = Model()
        m2.fc = torch.nn.Linear(256, 128, bias=True).to(torch.bfloat16)
        m2.load_weights([])

        def boom(weight, old=None):
            raise ValueError("no b12x tactic for this shape")

        self.c.build = boom
        m3 = Model()
        m3.load_weights([])
        for mm in (m, m2, m3):
            self.assertNotIn("_idf_c8", mm.fc.__dict__)
            self.assertIsNone(getattr(mm, "_derived_weight_cache_error", None))
        self.assertEqual(dfb.STATE["ctx8_fc"], 0)
        calls.clear()
        m3.project_target_hidden(torch.zeros(4, 256, dtype=torch.bfloat16))
        self.assertEqual(calls, [("stock", 4)])

    def test_fc_serving_error_falls_back_for_good(self):
        mod, Model, calls = self._dflash_mod()
        self.c.patch_dflash(mod)
        m = Model()
        m.load_weights([])

        def boom(x2, w, out=None):
            raise NotImplementedError("b12x: unsupported")

        self.c.gemm = boom
        calls.clear()
        m.project_target_hidden(torch.zeros(4, 256, dtype=torch.bfloat16))
        m.project_target_hidden(torch.zeros(4, 256, dtype=torch.bfloat16))
        self.assertEqual(calls, [("stock", 4), ("stock", 4)])
        self.assertNotIn("_idf_c8", m.fc.__dict__)

    def test_load_wrappers_chain_with_mx8(self):
        """ctx8 and mx8 both wrap load_weights: each prepare runs once per load, in either order."""
        from inkling_dfbytes import patches

        for ctx8_first in (True, False):
            mod, Model, _ = self._dflash_mod()
            mod.DFlash2DraftModel = type("Sel", (Model,), {"compute_candidates": lambda s, h: None,
                                                           "_transform_unary_logits": lambda s, v: v})
            mod.radix_topk = None
            prepared = []
            orig_prep = patches.prepare_layers
            patches.prepare_layers = lambda m: prepared.append(m)
            saved_opts = dfb.STATE["opts"]
            dfb.STATE["opts"] = ("mx8",)
            try:
                if ctx8_first:
                    self.c.patch_dflash(mod)
                    patches.patch_dflash(mod)
                else:
                    patches.patch_dflash(mod)
                    self.c.patch_dflash(mod)
                self.assertFalse(self.c.patch_dflash(mod))
                m = Model()
                dfb.STATE["ctx8_fc"] = 0
                m.load_weights([])
                self.assertEqual(prepared, [m])
                self.assertEqual(dfb.STATE["ctx8_fc"], 1)
            finally:
                patches.prepare_layers = orig_prep
                dfb.STATE["opts"] = saved_opts

    # ---------------------------------------------------------------- stacked K/V
    def _fused_mod(self, n_layers=2, kv=64, hidden=128):
        norm_calls = []

        def _fused_norm_rope_stacked(proj, k_norm_weights, eps, cos_sin, positions, num_kv_heads, head_dim,
                                     rotary_dim, k_out=None, v_out=None):
            norm_calls.append(proj.shape)
            total, nl, two_kv = proj.shape
            k_out.copy_(proj[:, :, : two_kv // 2].permute(1, 0, 2).reshape(k_out.shape))
            v_out.copy_(proj[:, :, two_kv // 2:].permute(1, 0, 2).reshape(v_out.shape))
            return k_out, v_out

        class Helper:
            def __init__(self, layers=(), **kwargs):
                torch.manual_seed(1)
                self.n_layers, self.num_kv_heads, self.head_dim = n_layers, 2, kv // 2
                self.layer_out_dim = 2 * kv
                self.flat_kv_weight = torch.randn(n_layers * 2 * kv, hidden).to(torch.bfloat16)
                self.k_norm_weights = None
                self.eps_values = None
                self.rotary_dim = kv // 2
                self.device = torch.device("cpu")
                self.max_position_hint = 100
                self._proj_workspace = self._k_workspace = self._v_workspace = None
                self.stock_calls = 0

            def _ensure_rope_cache(self, max_position):
                return None

            def _ensure_workspace(self, total_ctx, dtype):
                if self._proj_workspace is None or self._proj_workspace.shape[0] < total_ctx:
                    self._proj_workspace = torch.empty(total_ctx, self.n_layers * self.layer_out_dim, dtype=dtype)
                    self._k_workspace = torch.empty(self.n_layers, total_ctx, self.num_kv_heads, self.head_dim,
                                                    dtype=dtype)
                    self._v_workspace = torch.empty_like(self._k_workspace)

            def materialize(self, ctx_hidden, positions, write_layer_kv):
                # the stock body with the BF16 mm (enough of it for the comparison below)
                self.stock_calls += 1
                total = ctx_hidden.shape[0]
                if total == 0:
                    return
                self._ensure_workspace(total, ctx_hidden.dtype)
                proj = self._proj_workspace[:total]
                proj.copy_((ctx_hidden.float() @ self.flat_kv_weight.float().t()).to(torch.bfloat16))
                k, v = _fused_norm_rope_stacked(proj.view(total, self.n_layers, self.layer_out_dim), None, None, None,
                                                positions, 0, 0, 0, k_out=self._k_workspace[:, :total],
                                                v_out=self._v_workspace[:, :total])
                for i in range(self.n_layers):
                    write_layer_kv(i, k[i], v[i])

        mod = types.SimpleNamespace(__name__="sglang.kernels.ops.speculative.fused_kv_materialize",
                                    FusedKVMaterializeHelper=Helper,
                                    _fused_norm_rope_stacked=_fused_norm_rope_stacked)
        return mod, Helper, norm_calls

    @staticmethod
    def _collect():
        got = []
        return got, lambda i, k, v: got.append((i, k.clone(), v.clone()))

    def test_kv_dispatch_and_same_writes(self):
        mod, Helper, norm_calls = self._fused_mod()
        self.assertTrue(self.c.patch_fused_kv(mod))
        self.assertFalse(self.c.patch_fused_kv(mod))
        h = Helper()  # wrapped __init__ builds the copy
        self.assertEqual(dfb.STATE["ctx8_kv"], 1)
        x = torch.randn(8, 128).to(torch.bfloat16)
        pos = torch.arange(8)
        got, write = self._collect()
        h.materialize(x, pos, write)
        self.assertEqual(h.stock_calls, 0)
        self.assertEqual(self.gemm_calls, [(8, True)])  # into the workspace rows
        self.assertEqual(dfb.STATE["ctx8_kv_calls"], 1)
        ref, write_ref = self._collect()
        Helper.materialize.__wrapped__(h, x, pos, write_ref)  # the stock function on the same helper
        self.assertEqual([g[0] for g in got], [0, 1])  # layer order
        for (i, k, v), (j, kr, vr) in zip(got, ref):  # fake gemm == stock mm: the copied body writes the same bytes
            self.assertEqual(i, j)
            self.assertTrue(torch.equal(k, kr) and torch.equal(v, vr))
        # 2-D positions are flattened as in the stock body; a count mismatch raises
        got2, write2 = self._collect()
        h.materialize(x, pos.view(2, 4), write2)
        self.assertTrue(all(torch.equal(a[1], b[1]) for a, b in zip(got, got2)))
        with self.assertRaises(ValueError):
            h.materialize(x, torch.arange(7), write2)

    def test_kv_fallbacks(self):
        mod, Helper, _ = self._fused_mod()
        self.c.patch_fused_kv(mod)
        h = Helper()
        dfb.STATE["ctx8_max_m"] = 16
        _, write = self._collect()
        for x in (torch.zeros(17, 128, dtype=torch.bfloat16),  # rows > max_m
                  torch.zeros(0, 128, dtype=torch.bfloat16),  # empty
                  torch.zeros(4, 256, dtype=torch.bfloat16)[:, :128],  # strided
                  torch.zeros(4, 128, dtype=torch.float32)):  # FP32
            h.materialize(x, torch.arange(x.shape[0]), write)
        self.assertEqual(h.stock_calls, 4)
        self.assertEqual(self.gemm_calls, [])
        h.materialize(torch.zeros(16, 128, dtype=torch.bfloat16), torch.arange(16), write)
        self.assertEqual(self.gemm_calls, [(16, True)])
        h.flat_kv_weight = h.flat_kv_weight.float()  # a non-BF16 stacked weight: stock
        h.materialize(torch.zeros(4, 128, dtype=torch.bfloat16), torch.arange(4), write)
        self.assertEqual(h.stock_calls, 5)

    def test_kv_unsupported_stays_stock(self):
        mod, Helper, _ = self._fused_mod()
        self.c.patch_fused_kv(mod)
        self.c.build = self.orig[1]  # real eligibility check runs before any flashinfer import
        torch.Tensor.is_cuda = self.orig[3]
        h = Helper()  # a CPU weight: no copy, the helper itself is intact
        self.assertNotIn("_idf_c8", h.__dict__)
        torch.Tensor.is_cuda = property(lambda self: True)

        def boom(weight, old=None):
            raise ValueError("no b12x tactic for this shape")

        self.c.build = boom
        h = Helper()
        self.assertNotIn("_idf_c8", h.__dict__)
        _, write = self._collect()
        h.materialize(torch.zeros(4, 128, dtype=torch.bfloat16), torch.arange(4), write)
        self.assertEqual(h.stock_calls, 1)
        self.assertEqual(dfb.STATE["ctx8_kv"], 0)

    def test_kv_serving_error_reruns_stock(self):
        mod, Helper, _ = self._fused_mod()
        self.c.patch_fused_kv(mod)
        h = Helper()
        layer = torch.nn.Linear(1, 1)
        h2 = Helper.__new__(Helper)
        Helper.__init__(h2, layers=[layer])  # the first layer module declares the derived copy
        self.assertEqual(layer._derived_weight_cache_error, self.c.UPDATE_ERROR)

        def boom(x2, w, out=None):
            raise NotImplementedError("b12x: unsupported")

        self.c.gemm = boom
        got, write = self._collect()
        h.materialize(torch.randn(8, 128).to(torch.bfloat16), torch.arange(8), write)
        self.assertEqual(h.stock_calls, 1)
        self.assertEqual(len(got), 2)  # written once, by the stock rerun
        self.assertNotIn("_idf_c8", h.__dict__)
        self.assertEqual(dfb.STATE["ctx8_kv_calls"], 0)

    def test_kv_partial_workspace_write_then_recoverable_error(self):
        """The GEMM writes part of the workspace, then fails recoverably: the stock rerun writes each layer once, with
        the stock bytes."""
        mod, Helper, _ = self._fused_mod()
        self.c.patch_fused_kv(mod)
        h = Helper()
        x = torch.randn(8, 128).to(torch.bfloat16)
        ref, write_ref = self._collect()
        Helper.materialize.__wrapped__(h, x, torch.arange(8), write_ref)

        def partial(x2, w, out=None):
            out[: x2.shape[0] // 2].fill_(7.0)
            raise ValueError("b12x: no valid tactic")

        self.c.gemm = partial
        got, write = self._collect()
        h.materialize(x, torch.arange(8), write)
        self.assertEqual([g[0] for g in got], [0, 1])
        self.assertTrue(all(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) for a, b in zip(got, ref)))
        self.assertEqual(dfb.STATE["ctx8_disabled"], 1)

    def test_cuda_errors_are_fatal(self):
        mod, Helper, _ = self._fused_mod()
        self.c.patch_fused_kv(mod)
        h = Helper()

        def cuda_err(x2, w, out=None):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        self.c.gemm = cuda_err
        _, write = self._collect()
        with self.assertRaises(dfb.DfbytesFatal):
            h.materialize(torch.randn(8, 128).to(torch.bfloat16), torch.arange(8), write)
        self.assertIn("_idf_c8", h.__dict__)  # not silently disabled
        self.c.build = lambda weight, old=None: (_ for _ in ()).throw(RuntimeError("CUDA error: launch failure"))
        with self.assertRaises(dfb.DfbytesFatal):  # at set-up too (passes the worker's except Exception)
            Helper()
        self.assertTrue(self.c.fatal(RuntimeError("CUDA out of memory. Tried to allocate")))
        self.assertFalse(self.c.fatal(ValueError("b12x requires K % 128 == 0")))
        for text in ("CUDA_ERROR_MISALIGNED_ADDRESS", "CUDA_ERROR_LAUNCH_TIMEOUT", "anything"):
            self.assertTrue(self.c.fatal(RuntimeError(text)), text)  # RuntimeError is never recoverable
        self.assertTrue(self.c.fatal(ValueError("cudaErrorLaunchFailure")))
        try:
            try:
                raise RuntimeError("CUDA_ERROR_MISALIGNED_ADDRESS")
            except RuntimeError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            self.assertTrue(self.c.fatal(outer))  # a CUDA fault chained under a recoverable type
        self.assertFalse(self.c.fatal(NotImplementedError("tile 16x128 unsupported")))

    def test_worker_marks_both_models(self):
        class W:
            def _init_fused_kv_helper(self):
                return "init"

        mod = types.SimpleNamespace(__name__="sglang.srt.speculative.dflash_worker_v2", DFlashWorkerV2=W)
        self.assertTrue(self.c.patch_worker(mod))
        self.assertFalse(self.c.patch_worker(mod))
        draft, target = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        draft.fc = torch.nn.Linear(1, 1)
        w = W()
        w.draft_model = draft
        w.target_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=target))
        w._fused_kv_helper = types.SimpleNamespace()
        self.assertEqual(w._init_fused_kv_helper(), "init")
        self.assertIsNone(getattr(target, "_derived_weight_cache_error", None))  # no copy built: no refusal
        w._fused_kv_helper.__dict__["_idf_c8"] = object()
        w._init_fused_kv_helper()
        self.assertEqual(target._derived_weight_cache_error, self.c.UPDATE_ERROR)
        self.assertEqual(draft._derived_weight_cache_error, self.c.UPDATE_ERROR)

    def test_why_not(self):
        w = _FakeW(torch.zeros(128, 64, dtype=torch.bfloat16))
        ok = torch.zeros(8, 64, dtype=torch.bfloat16)
        self.assertIsNone(self.c.why_not(ok, w))
        self.assertEqual(self.c.why_not(torch.zeros(300, 64, dtype=torch.bfloat16), w), "rows")
        self.assertEqual(self.c.why_not(torch.zeros(0, 64, dtype=torch.bfloat16), w), "rows")
        self.assertIn("K", self.c.why_not(torch.zeros(8, 32, dtype=torch.bfloat16), w))
        self.assertIn("contiguous", self.c.why_not(torch.zeros(8, 128, dtype=torch.bfloat16)[:, ::2], w))
        self.assertIn("aligned", self.c.why_not(torch.zeros(8 * 64 + 1, dtype=torch.bfloat16)[1:].view(8, 64), w))
        self.assertEqual(self.c.why_not(torch.zeros(2, 8, 64, dtype=torch.bfloat16), w), "not 2-D")

    def test_ineligible_weight(self):
        torch.Tensor.is_cuda = self.orig[3]
        self.assertIn("CUDA", self.c.ineligible_weight(torch.zeros(128, 128, dtype=torch.bfloat16)))
        torch.Tensor.is_cuda = property(lambda self: True)
        self.assertIsNone(self.c.ineligible_weight(torch.zeros(256, 128, dtype=torch.bfloat16)))
        self.assertIn("multiples", self.c.ineligible_weight(torch.zeros(576, 2304, dtype=torch.bfloat16)))
        self.assertIn("CUDA BF16", self.c.ineligible_weight(torch.zeros(128, 128, dtype=torch.float16)))


if __name__ == "__main__":
    unittest.main()
