#!/usr/bin/env python3
"""GPU check of inkling_dfbytes ctx8 (the DFlash2 drafter's context fc + stacked K/V projections on b12x MXFP8).
Run in the pinned SGLang image with no server (about 2 GB of GPU memory):

  cd overlay && TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \\
    python3 tools/gpu_check_ctx8.py --draft <DFlash2 drafter dir> [--hidden <dir of captured *.npz>] [--json out.json]

It uses the image's own DFlashDraftModel.project_target_hidden and FusedKVMaterializeHelper (patched by ctx8 exactly as
in serving; the source pins are checked) on the drafter's real weights (fc, hidden_norm, the 5 layers' q/k/v and k_norm),
rows 1 / 8 / 32 / 64 / 192 / 256 / 257:

1. faithful copy: the ctx8 materialize with its GEMM swapped back to the stock torch.mm writes K/V bytes identical to
   the stock materialize (so the only difference in serving is the GEMM), and 257 rows (> max_m) take the stock path
   bit for bit;
2. the b12x GEMM into the workspace equals the standalone b12x GEMM fed to the stock norm + RoPE (bitwise);
3. error vs BF16: relative Frobenius error and the worst row cosine of the fc output (after hidden_norm), and of K / V
   per layer, (a) K/V GEMM alone on the stock fc output, (b) end to end from the target features;
4. no compile after the load-time warm-up (any row count runs in < 50 ms eagerly); eager determinism; CUDA graph capture: a captured fc + K/V append replayed on new inputs equals eager (bitwise);
5. timing: one CUDA graph of fc + K/V append per variant (stock BF16 vs ctx8), L2 flushed before every replay.
Inputs: N(0, 1) target features per 2,304-wide block, scaled 1 / 2 / 4 / 8 (the four captured target layers grow in
magnitude); with --hidden, captured target post-norm rows fill every block (a proxy; the fc input in serving is the
four captured layers' features).
The drafter's token agreement and acceptance are measured in serving (tools/ctx8_agree.py on two steplog captures).
Prints PASS / FAIL lines; exit 1 on any FAIL."""
import argparse
import glob
import json
import os
import statistics
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

FAILED = []
REPORT = {}


def log(*a):
    print(*a, flush=True)


def check(name, ok, detail=""):
    log(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def relerr(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def min_cos(a, b):
    a, b = a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())


def load_draft(path, dev):
    from safetensors import safe_open

    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    n_layers = int(cfg["num_hidden_layers"])
    want = ["fc.weight", "hidden_norm.weight"] + [f"layers.{i}.self_attn.{t}" for i in range(n_layers)
                                                 for t in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                                           "k_norm.weight")]
    out = {}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as s:
        keys = set(s.keys())
        missing = [k for k in want if k not in keys]
        if missing:
            raise SystemExit(f"drafter checkpoint lacks {missing[:4]}... (has e.g. {sorted(keys)[:6]})")
        for k in want:
            out[k] = s.get_tensor(k).to(dev).contiguous()
    return cfg, out


def rotary(cfg, dev, max_pos):
    head_dim = int(cfg["head_dim"])
    theta = float(cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 10000.0)))
    try:
        from sglang.srt.layers.rotary_embedding import get_rope

        rope = get_rope(head_dim, rotary_dim=head_dim, max_position=max_pos, base=theta, is_neox_style=True)
        rope = rope.to(dev)
        if getattr(rope, "cos_sin_cache", None) is None:
            raise RuntimeError("no cos_sin_cache")
        return rope, "sglang get_rope"
    except Exception as exc:  # noqa: BLE001 - both arms read the same cache, so a plain one is as good for the check
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        f = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=dev), inv)
        cache = torch.cat([f.cos(), f.sin()], dim=-1)
        return types.SimpleNamespace(rotary_dim=head_dim, is_neox_style=True, cos_sin_cache=cache), \
            f"plain table ({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--hidden", default="")
    ap.add_argument("--rows", default="1,8,32,64,192,256,257")
    ap.add_argument("--time-rows", default="8,32,64,192")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    os.environ["INKLING_DFLASH_CTX8"] = "1"
    import inkling_dfbytes as dfb
    from inkling_dfbytes import ctx8

    dfb.STATE.update(ctx8=True, ctx8_max_m=256)
    import sglang.kernels.ops.speculative.fused_kv_materialize as fkm
    from sglang.srt.models import dflash as dmod

    ctx8.patch_fused_kv(fkm)  # checks the source pins
    ctx8.patch_dflash(dmod)
    Helper = fkm.FusedKVMaterializeHelper
    stock_materialize = Helper.materialize.__wrapped__
    stock_project = dmod.DFlashDraftModel.project_target_hidden.__wrapped__
    patched_project = dmod.DFlashDraftModel.project_target_hidden
    import flashinfer

    log(f"torch {torch.__version__}, flashinfer {flashinfer.__version__}, "
        f"SMs {torch.cuda.get_device_properties(dev).multi_processor_count}")

    cfg, W = load_draft(a.draft, dev)
    hidden = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    nh, nkv, hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    feat = W["fc.weight"].shape[1]
    max_pos = 65536
    rope, rope_kind = rotary(cfg, dev, max_pos)
    log(f"drafter: hidden {hidden}, {n_layers} layers, fc {tuple(W['fc.weight'].shape)}, rotary: {rope_kind}")

    # ---- the drafter pieces the two functions read
    class TorchRMSNorm(torch.nn.Module):  # used only if the image's RMSNorm cannot run outside a server
        def __init__(self, n, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(n))
            self.variance_epsilon = eps

        def forward(self, x):
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return (y * self.weight.float()).to(x.dtype)

    try:
        from sglang.srt.layers.layernorm import RMSNorm

        hidden_norm = RMSNorm(hidden, eps=eps).to(dev, torch.bfloat16)
        with torch.inference_mode():
            hidden_norm(torch.zeros(2, hidden, device=dev, dtype=torch.bfloat16))
        norm_kind = "sglang RMSNorm"
    except Exception as exc:  # noqa: BLE001 - both arms run the same norm
        RMSNorm = TorchRMSNorm
        hidden_norm = RMSNorm(hidden, eps=eps).to(dev, torch.bfloat16)
        norm_kind = f"torch RMSNorm ({type(exc).__name__}: {str(exc)[:80]})"
    k_norm_cls = RMSNorm
    log(f"hidden_norm: {norm_kind}")
    with torch.no_grad():
        hidden_norm.weight.copy_(W["hidden_norm.weight"])
    fc = torch.nn.Linear(feat, hidden, bias=False, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        fc.weight.copy_(W["fc.weight"])
    model = types.SimpleNamespace(fc=fc, hidden_norm=hidden_norm, quantizable=False, fc_tp_columns=None,
                                  shards_context_fc=True, num_context_features=feat // hidden,
                                  config=types.SimpleNamespace(hidden_size=hidden))
    ctx8.prepare_fc(model)
    layers = []
    for i in range(n_layers):
        p = f"layers.{i}.self_attn."
        k_norm = k_norm_cls(hd, eps=eps).to(dev, torch.bfloat16)
        with torch.no_grad():
            k_norm.weight.copy_(W[p + "k_norm.weight"])
        qkv = torch.cat([W[p + "q_proj.weight"], W[p + "k_proj.weight"], W[p + "v_proj.weight"]]).contiguous()
        attn = types.SimpleNamespace(num_kv_heads=nkv, head_dim=hd, rotary_emb=rope, q_size=nh * hd, kv_size=nkv * hd,
                                     qkv_proj=types.SimpleNamespace(weight=qkv), k_norm=k_norm)
        layers.append(types.SimpleNamespace(self_attn=attn))
    helper = Helper(layers=layers, rotary_emb=rope, num_kv_heads=nkv, head_dim=hd, device=dev,
                    max_position_hint=max_pos - 1)
    w_kv = helper.__dict__["_idf_c8"]
    w_fc = fc.__dict__["_idf_c8"]
    bf16_mb = (helper.flat_kv_weight.numel() + fc.weight.numel()) * 2 / 1e6
    log(f"MXFP8 copies: fc {w_fc.nbytes() / 1e6:.1f} MB + K/V {w_kv.nbytes() / 1e6:.1f} MB (BF16 {bf16_mb:.1f} MB)")
    REPORT["mxfp8_mb"] = (w_fc.nbytes() + w_kv.nbytes()) / 1e6

    # the load-time warm-up compiled every plan: no row count 1..max_m, in either production call shape, compiles more
    import time

    try:
        import flashinfer.gemm.gemm_mm_mxfp8_cute_dsl as fm

        kcache = fm._B12X_MM_MXFP8_KERNEL_CACHE
    except Exception:  # noqa: BLE001
        kcache = None
    before = None if kcache is None else len(kcache)
    worst = 0.0
    helper._ensure_workspace(dfb.STATE["ctx8_max_m"], torch.bfloat16)
    with torch.inference_mode():
        xs = {w: torch.randn((dfb.STATE["ctx8_max_m"], w.k), device=dev).to(torch.bfloat16) for w in (w_fc, w_kv)}
        for m in range(1, dfb.STATE["ctx8_max_m"] + 1):
            for w, out in ((w_fc, None), (w_kv, helper._proj_workspace[:m])):
                torch.cuda.synchronize()
                t0 = time.time()
                ctx8.gemm(xs[w][:m], w, out=out)
                torch.cuda.synchronize()
                worst = max(worst, time.time() - t0)
    if kcache is not None:
        check("no b12x compile after the load-time warm-up (rows 1..max_m, fc out=None and K/V out=workspace)",
              len(kcache) == before, f"kernel cache {before} -> {len(kcache)}; slowest call {worst * 1e3:.1f} ms")
    else:
        check("flashinfer's b12x MXFP8 kernel cache is readable (needed to prove no compile after warm-up)", False,
              f"slowest call {worst * 1e3:.1f} ms")

    hid_rows = None
    if a.hidden:
        import numpy as np

        caps = []
        for path in sorted(glob.glob(os.path.join(a.hidden, "*.npz"))):
            with np.load(path) as c:
                r = c["hidden"]
                caps.append(torch.from_numpy(r.astype(np.int16)).view(torch.bfloat16) if r.dtype.kind in "iu"
                            else torch.from_numpy(r).to(torch.bfloat16))
        if caps:
            hid_rows = torch.cat(caps).to(dev)
            log(f"captured rows: {hid_rows.shape[0]}")

    def features(m, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        if hid_rows is not None and hid_rows.shape[1] == hidden:
            idx = torch.randint(0, hid_rows.shape[0], (m, feat // hidden), generator=g, device=dev)
            return hid_rows[idx].reshape(m, feat).contiguous()
        blocks = [torch.randn((m, hidden), generator=g, device=dev) * s for s in (1.0, 2.0, 4.0, 8.0)]
        return torch.cat(blocks[: feat // hidden], dim=1).to(torch.bfloat16).contiguous()

    def run(materialize, project, x, pos):
        got = []
        ctx = project(model, x)
        materialize(helper, ctx, pos, lambda i, k, v: got.append((k.clone(), v.clone())))
        return ctx.clone(), got

    def same(a_, b_):
        return all(torch.equal(k1, k2) and torch.equal(v1, v2) for (k1, v1), (k2, v2) in zip(a_, b_)) \
            and len(a_) == len(b_) == n_layers

    patched_materialize = Helper.materialize
    rows_list = [int(r) for r in a.rows.split(",")]
    errs = {}
    with torch.inference_mode():
        for m in rows_list:
            x = features(m, 100 + m)
            pos = torch.randint(0, max_pos - 1, (m,), device=dev)
            ctx_ref, kv_ref = run(stock_materialize, stock_project, x, pos)
            before = (dfb.STATE["ctx8_fc_calls"], dfb.STATE["ctx8_kv_calls"])
            ctx_c8, kv_c8 = run(patched_materialize, patched_project, x, pos)
            served = (dfb.STATE["ctx8_fc_calls"] - before[0], dfb.STATE["ctx8_kv_calls"] - before[1])
            if m > dfb.STATE["ctx8_max_m"]:
                check(f"rows {m} > max_m take the stock path bit for bit",
                      served == (0, 0) and torch.equal(ctx_c8, ctx_ref) and same(kv_c8, kv_ref))
                continue
            check(f"rows {m}: both GEMMs on b12x", served == (1, 1), f"calls fc/kv {served}")
            # 1. faithful copy: swap the GEMM back to the stock torch.mm
            orig_gemm = ctx8.gemm

            def mm_gemm(x2, w, out=None):
                return torch.mm(x2, helper.flat_kv_weight.t(), out=out)

            ctx8.gemm = mm_gemm
            try:
                got = []
                patched_materialize(helper, ctx_ref, pos, lambda i, k, v: got.append((k.clone(), v.clone())))
            finally:
                ctx8.gemm = orig_gemm
            check(f"rows {m}: copied materialize body == stock (GEMM swapped back)", same(got, kv_ref))
            # 2. workspace GEMM == standalone b12x GEMM through the stock norm + RoPE
            proj = ctx8.gemm(ctx_c8, w_kv).view(m, helper.n_layers, helper.layer_out_dim)
            k2, v2 = fkm._fused_norm_rope_stacked(proj, helper.k_norm_weights, helper.eps_values,
                                                  helper._ensure_rope_cache(max_pos - 1), pos, nkv, hd,
                                                  helper.rotary_dim)
            check(f"rows {m}: b12x into the workspace == standalone b12x + stock norm/RoPE",
                  same([(k2[i], v2[i]) for i in range(n_layers)], kv_c8))
            # 3. error vs BF16
            got_kv = []
            patched_materialize(helper, ctx_ref, pos, lambda i, k, v: got_kv.append((k.clone(), v.clone())))
            e = {
                "fc_rel": relerr(ctx_c8, ctx_ref), "fc_cos_min": min_cos(ctx_c8, ctx_ref),
                "kv_only_k_rel": max(relerr(k, kr) for (k, _), (kr, _) in zip(got_kv, kv_ref)),
                "kv_only_v_rel": max(relerr(v, vr) for (_, v), (_, vr) in zip(got_kv, kv_ref)),
                "e2e_k_rel": max(relerr(k, kr) for (k, _), (kr, _) in zip(kv_c8, kv_ref)),
                "e2e_v_rel": max(relerr(v, vr) for (_, v), (_, vr) in zip(kv_c8, kv_ref)),
                "e2e_k_cos_min": min(min_cos(k, kr) for (k, _), (kr, _) in zip(kv_c8, kv_ref)),
                "e2e_v_cos_min": min(min_cos(v, vr) for (_, v), (_, vr) in zip(kv_c8, kv_ref)),
            }
            errs[m] = e
            log(f"  rows {m}: fc rel {e['fc_rel']:.3e} (min cos {e['fc_cos_min']:.5f}); K/V GEMM alone K "
                f"{e['kv_only_k_rel']:.3e} V {e['kv_only_v_rel']:.3e}; end to end K {e['e2e_k_rel']:.3e} V "
                f"{e['e2e_v_rel']:.3e} (min cos K {e['e2e_k_cos_min']:.5f} V {e['e2e_v_cos_min']:.5f})")
            check(f"rows {m}: MXFP8 error in the expected range (rel < 0.1, cos > 0.99)",
                  max(e["fc_rel"], e["e2e_k_rel"], e["e2e_v_rel"]) < 0.1
                  and min(e["fc_cos_min"], e["e2e_k_cos_min"], e["e2e_v_cos_min"]) > 0.99)
            # 4a. determinism
            ctx_c8b, kv_c8b = run(patched_materialize, patched_project, x, pos)
            check(f"rows {m}: eager repeat bitwise", torch.equal(ctx_c8b, ctx_c8) and same(kv_c8b, kv_c8))
        REPORT["errors"] = errs

        # 1b. the b12x GEMM writes the stock workspace rows in place (out=), at odd row counts too
        helper._ensure_workspace(dfb.STATE["ctx8_max_m"], torch.bfloat16)
        orig_gemm = ctx8.gemm
        for m in (1, 3, 127, 129, 255):
            seen = []

            def rec_gemm(x2, w, out=None):
                y = orig_gemm(x2, w, out=out)
                seen.append((None if out is None else out.data_ptr(), y.data_ptr()))
                return y

            x = features(m, 300 + m)
            pos = torch.randint(0, max_pos - 1, (m,), device=dev)
            ctx = stock_project(model, x)
            helper._proj_workspace.fill_(float("nan"))
            ctx8.gemm = rec_gemm
            try:
                patched_materialize(helper, ctx, pos, lambda i, k, v: None)
            finally:
                ctx8.gemm = orig_gemm
            ws = helper._proj_workspace
            ref = orig_gemm(ctx, w_kv)
            check(f"rows {m}: b12x wrote the workspace rows in place",
                  seen == [(ws.data_ptr(), ws.data_ptr())] and not torch.isnan(ws[:m]).any().item()
                  and torch.equal(ws[:m], ref) and torch.isnan(ws[m:m + 1]).all().item())

        # 4b. CUDA graph capture of the ctx8 append, replayed on new inputs
        for m in (8, 64):
            x_s = features(m, 7).clone()
            pos_s = torch.randint(0, max_pos - 1, (m,), device=dev)
            outs = [(torch.empty(m, nkv, hd, device=dev, dtype=torch.bfloat16),
                     torch.empty(m, nkv, hd, device=dev, dtype=torch.bfloat16)) for _ in range(n_layers)]

            def write(i, k, v):
                outs[i][0].copy_(k)
                outs[i][1].copy_(v)

            def step():
                patched_materialize(helper, patched_project(model, x_s), pos_s, write)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                step()
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                step()
            ok = True
            for seed in (11, 12):
                x_s.copy_(features(m, seed))
                pos_s.copy_(torch.randint(0, max_pos - 1, (m,), device=dev))
                gr.replay()
                torch.cuda.synchronize()
                ref = [(k.clone(), v.clone()) for k, v in outs]
                _, eager = run(patched_materialize, patched_project, x_s, pos_s)
                ok &= same(ref, eager)
            check(f"rows {m}: CUDA graph replay == eager (bitwise, 2 new inputs)", ok)
            del gr

        # 5. timing (cold L2)
        flush = torch.ones(64 << 20, device=dev)
        fo = torch.empty((), device=dev)
        timing = {}
        for m in [int(r) for r in a.time_rows.split(",")]:
            x_s = features(m, 5)
            pos_s = torch.randint(0, max_pos - 1, (m,), device=dev)
            res = {}
            for name, mat, proj in (("bf16", stock_materialize, stock_project),
                                    ("ctx8", patched_materialize, patched_project)):
                def step():
                    mat(helper, proj(model, x_s), pos_s, lambda i, k, v: None)

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    step()
                    step()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    step()
                ts = []
                for _ in range(25):
                    torch.sum(flush, 0, out=fo)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    gr.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1))
                res[name] = statistics.median(ts[3:])
                del gr
            timing[m] = res
            log(f"  time rows {m}: fc + K/V append BF16 {res['bf16']:.3f} ms -> ctx8 {res['ctx8']:.3f} ms "
                f"({res['ctx8'] - res['bf16']:+.3f} ms)")
        REPORT["timing_ms"] = timing

        # 5b. eager timing as served: fc + K/V append with real ring writes (index_copy_ into per-layer K/V rings)
        slots = 8192
        ring = [(torch.zeros(slots, nkv, hd, device=dev, dtype=torch.bfloat16),
                 torch.zeros(slots, nkv, hd, device=dev, dtype=torch.bfloat16)) for _ in range(n_layers)]
        eager = {}
        for m in [int(r) for r in a.time_rows.split(",")]:
            x_s = features(m, 6)
            pos_s = torch.randint(0, max_pos - 1, (m,), device=dev)
            loc = torch.randperm(slots, device=dev)[:m]

            def write(i, k, v):
                ring[i][0].index_copy_(0, loc, k)
                ring[i][1].index_copy_(0, loc, v)

            res = {}
            for name, mat, proj in (("bf16", stock_materialize, stock_project),
                                    ("ctx8", patched_materialize, patched_project)):
                for _ in range(5):
                    mat(helper, proj(model, x_s), pos_s, write)
                torch.cuda.synchronize()
                ts = []
                for _ in range(40):
                    torch.sum(flush, 0, out=fo)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    mat(helper, proj(model, x_s), pos_s, write)
                    e1.record()
                    torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1))
                res[name] = statistics.median(ts)
            eager[m] = res
            log(f"  eager rows {m} (ring writes, cold L2): BF16 {res['bf16']:.3f} ms -> ctx8 {res['ctx8']:.3f} ms "
                f"({res['ctx8'] - res['bf16']:+.3f} ms)")
        REPORT["eager_ms"] = eager
        check("ctx8 faster than BF16 eagerly (ring writes) at every timed row count",
              all(r["ctx8"] < r["bf16"] for r in eager.values()))
    check("no projection fell back to BF16 during the checks", dfb.STATE.get("ctx8_disabled", 0) == 0,
          f"disabled {dfb.STATE.get('ctx8_disabled', 0)}")

    REPORT["failed"] = FAILED
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(REPORT, fh, indent=1)
    log("ALL OK" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
