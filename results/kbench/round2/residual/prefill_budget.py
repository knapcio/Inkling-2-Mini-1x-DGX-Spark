#!/usr/bin/env python3
"""Rebuilt prefill budget for the SHIPPED stack (4 inkling_prefill levers) and two ceilings.

    python3 -I prefill_budget.py [path/to/docs/prefill/bench.json]

Labels: [B] GB10 microbenchmark of the stock kernel (docs/prefill/bench.json, box B), [E] measured end to end
(docs/experiments.md speedbench longctx cold; README published), [D] derived = closing term or byte arithmetic,
[A] assumption. Rates: NVFP4 grouped GEMM composite 164 TFLOP/s at T=8192 (w13 186 / w2 133), NOT the 300 that
roofline.log assumed; the rebuilt compute-only floor uses the measured per-stage kernel times directly.
"""
import json, os, sys

CHUNK = 8192
MB = 1e6
BW = 224.0                    # [B] GB/s, device copy / fp32*bf16 elementwise
# [E] cold TTFT (s): stock PF0, PFO2 (qkvr_split + partial_bf16), shipped (README published 4,266 / 3,941 / 2,797)
TTFT = {'stock': {8192: 8192 / 3408, 32768: 32768 / 3235, 131072: 131072 / 2409},
        'pfo2': {8192: 8192 / 3781, 32768: 32768 / 3558, 131072: 131072 / 2537},
        'shipped': {8192: 8192 / 4266, 32768: 32768 / 3941, 131072: 131072 / 2797}}
FIXED_S = 0.06                # [D] per-request fixed cost (PREFILL.md 2.6: 1K remainder minus per-token work)
QKVR_SPLIT_MS = 3.77          # [E/B] split call per stacked visit at 8K (experiments.md: 10.55 -> 3.77)
QUANT_ONCE_MS = 0.70          # [A] PREFILL C3a estimate: NVFP4 routed quantize 1.87 -> ~0.7 ms per visit
PARTIAL_BF16_SAVE_MS = 0.60   # [A] PREFILL C3d estimate per visit (48 visits)
EINSUM_OUT_MB = CHUNK * 24 * 1024 * 2 / MB          # 402.7 [D]
SHEARED_MB = CHUNK * 24 * 1280 * 2 / MB             # 503.3 [D] round_up(t,128) x 24 x 1280 bf16
SHEAR_MS = (EINSUM_OUT_MB + SHEARED_MB) / BW        # 4.04 per visit [D]: read einsum out, write sheared copy
PRESHEAR_MS = SHEARED_MB / BW                       # 2.25 per visit [D]: bias_layout Triton writes the sheared layout
FUSED_MMA_OVERHEAD = 0.08     # [A] inkling_attn README: +8 % MMA work for in-kernel bias (near-diagonal tiles)
ELEMWISE_MODEL_US_TOK = 48.9  # [D] roofline.py byte model of the unbenchmarked elementwise ops (10.95 MB/token @224)


def load(path):
    d = json.load(open(path))['records']
    g = {(x['name'], x['m']): x['med_us'] / 1e3 for x in d if x['section'] == 'gemm' and 'med_us' in x and 'm' in x}
    moe = {(x['precision'], x['tokens']): {k: v / 1e3 for k, v in x['stages_us'].items()}
           for x in d if x['section'] == 'moe' and 'stages_us' in x}
    att = {(x['kind'], x['prefix'], x['rel_bias']): x['med_us'] / 1e3
           for x in d if x['section'] == 'attn' and x['chunk'] == CHUNK and 'med_us' in x}
    return g, moe, att


def interp(att, kind, bias, p):
    pts = sorted((pp, t) for (k, pp, b), t in att.items() if k == kind and b == bias)
    for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
        if p0 <= p <= p1:
            return t0 + (t1 - t0) * (p - p0) / (p1 - p0)
    (p0, t0), (p1, t1) = pts[-2], pts[-1]
    return t1 + (t1 - t0) * (p - p1) / (p1 - p0)


def chunk_ms(g, moe, att, p):
    """Shipped per-chunk component times (ms) at prefix p, from stock kernels minus the 4 levers."""
    n4, m8 = moe[('nvfp4', CHUNK)], moe[('mxfp8', CHUNK)]
    r = {}
    r['MoE grouped GEMMs (NVFP4 x44 + MXFP8 x4) [B]'] = 44 * (n4['GEMM1 w13'] + n4['GEMM2 w2']) + 4 * (
        m8['GEMM1 w13'] + m8['GEMM2 w2'])
    glue = {
        'router + topk + sort metadata': 44 * (n4['router GEMM + route_topk'] + n4['sort-route metadata'])
                                         + 4 * (m8['router GEMM + route_topk'] + m8['sort-route metadata']),
        'routed quantize (quant_once on NVFP4)': 44 * QUANT_ONCE_MS + 4 * m8['routed quantize'],
        'swiglu-quantize': 44 * n4['swiglu-quantize'] + 4 * m8['swiglu-quantize'],
        'combine (partial_bf16)': 44 * n4['routed_partial combine + cast'] + 4 * m8['routed_partial combine + cast']
                                  - 48 * PARTIAL_BF16_SAVE_MS,
    }
    r['MoE glue [B - lever estimates]'] = sum(glue.values())
    r['dense GEMMs: qkvr 20 single + 28 split, wo, dyn-conv, draft ctx K/V [B]'] = (
        20 * g[('qkvr local (1 source)', CHUNK)] + 28 * QKVR_SPLIT_MS + 48 * g[('wo', CHUNK)]
        + 16 * g[('global dyn-conv weight_proj (32 x 2304)', CHUNK)] + g[('DSpark draft ctx K/V (2048 x 2304)', CHUNK)])
    swa0, swab = att[('swa', 0, False)], att[('swa', 0, True)]
    if p > 0:
        swa0, swab = interp(att, 'swa', False, p), interp(att, 'swa', True, p)
    gl0, glb = interp(att, 'global', False, p), interp(att, 'global', True, p)
    r['attention mainloop without bias (SWA x40 + global x8) [B]'] = 40 * swa0 + 8 * gl0
    # stock 'FA4 with bias' bench = mainloop + shear copy + bias-read penalty; bias_layout replaced einsum + shear
    # by one Triton kernel writing the sheared layout, so what remains is that write + the mainloop penalty.
    pen_swa = max(0.0, swab - swa0 - SHEAR_MS)
    pen_gl = max(0.0, glb - gl0 - SHEAR_MS)
    r['rel-bias left: presheared write x48 [D]'] = 48 * PRESHEAR_MS
    r['rel-bias left: SWA mainloop penalty x40 [B-D]'] = 40 * pen_swa
    r['rel-bias left: global mainloop penalty x8 [B-D]'] = 8 * pen_gl
    return r, glue, dict(swa0=swa0, gl0=gl0, pen_swa=pen_swa, pen_gl=pen_gl)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../../../Inkling-2-Mini-1x-DGX-Spark/docs/prefill/bench.json')
    g, moe, att = load(path)
    p = print
    # consistency: lever deltas predicted from kernels vs measured end to end at 8K (one chunk, prefix 0)
    n4 = moe[('nvfp4', CHUNK)]
    pred2 = 28 * (g[('qkvr stacked 3 sources (serving: 28 of 48 visits)', 3 * CHUNK)] - QKVR_SPLIT_MS) + 48 * PARTIAL_BF16_SAVE_MS
    pred_bias = 48 * (g[('rel-bias einsum [t,24,16]x[16,1024]', CHUNK)] + SHEAR_MS - PRESHEAR_MS)
    pred_q = 44 * (n4['routed quantize'] - QUANT_ONCE_MS)
    meas2 = (TTFT['stock'][8192] - TTFT['pfo2'][8192]) * 1e3
    meas3 = (TTFT['pfo2'][8192] - 8192 / 4253) * 1e3
    p("## Lever cross-check at 8K (ms per chunk)")
    p(f"qkvr_split + partial_bf16: predicted {pred2:.0f} [B+A], measured {meas2:.0f} [E]")
    p(f"bias_layout + quant_once: predicted {pred_bias:.0f} + {pred_q:.0f} = {pred_bias + pred_q:.0f} [D], "
      f"measured {meas3:.0f} [E] (PFO3 4,253 tok/s)")
    p(f"per-visit bytes: einsum out {EINSUM_OUT_MB:.0f} MB, sheared {SHEARED_MB:.0f} MB -> shear {SHEAR_MS:.2f} ms, "
      f"presheared write {PRESHEAR_MS:.2f} ms at {BW:.0f} GB/s")
    p("")
    for n in (8192, 32768, 131072):
        chunks = [i * CHUNK for i in range(n // CHUNK)]
        tot, glue_t, aux = {}, {}, []
        for pfx in chunks:
            r, glue, a = chunk_ms(g, moe, att, pfx)
            aux.append(a)
            for k, v in r.items():
                tot[k] = tot.get(k, 0.0) + v
            for k, v in glue.items():
                glue_t[k] = glue_t.get(k, 0.0) + v
        T = TTFT['shipped'][n] * 1e3
        known = sum(tot.values())
        fixed = FIXED_S * 1e3
        rem = T - known - fixed
        nch = len(chunks)
        p(f"## {n // 1024}K cold prefill, shipped: {T / 1e3:.2f} s ({n / T * 1e3:,.0f} tok/s) [E]; {nch} chunk(s); "
          f"per chunk {T / nch:.0f} ms")
        p("| component | ms total | ms / chunk | share |")
        p("|---|---:|---:|---:|")
        for k, v in tot.items():
            p(f"| {k} | {v:.0f} | {v / nch:.0f} | {100 * v / T:.1f} % |")
        p(f"| fixed per request (host, first chunk) [D] | {fixed:.0f} | | {100 * fixed / T:.1f} % |")
        p(f"| elementwise remainder (attn-res, norms, clips, convs, KV store, gate_sink, controllers, aux) "
          f"[D closing term] | {rem:.0f} | {rem / nch:.0f} | {100 * rem / T:.1f} % |")
        p("MoE glue split (ms total): " + ", ".join(f"{k} {v:.0f}" for k, v in glue_t.items()))
        model_rem = ELEMWISE_MODEL_US_TOK * n / 1e3
        p(f"remainder vs its byte model: {rem / n * 1e3:.1f} us/token measured-derived vs {ELEMWISE_MODEL_US_TOK} "
          f"modelled ({rem / model_rem:.2f}x)")
        # ceilings
        a_swa = sum(PRESHEAR_MS * 40 + 40 * x['pen_swa'] - FUSED_MMA_OVERHEAD * 40 * x['swa0'] for x in aux)
        a_gl = sum(PRESHEAR_MS * 8 + 8 * x['pen_gl'] - FUSED_MMA_OVERHEAD * 8 * x['gl0'] for x in aux)
        b_moe = glue_t['routed quantize (quant_once on NVFP4)'] + glue_t['swiglu-quantize'] + glue_t['combine (partial_bf16)']
        b_rem_model = max(0.0, rem - model_rem)
        b_rem_half = max(0.0, rem - 0.5 * model_rem)
        rows = [
            ('(a1) fused rel-bias attention, SWA only (in-kernel bias at bias-free FA4 rate +8 % MMA)', a_swa),
            ('(a2) (a1) + global layers', a_swa + a_gl),
            ('(b1) MoE glue into GEMM prologue/epilogues (quant, swiglu-quant, combine)', b_moe),
            ('(b2) elementwise remainder down to its byte model at 224 GB/s', b_rem_model),
            ('(b3) elementwise remainder fused to half its byte model', b_rem_half),
            ('(a2)+(b1)+(b2)', a_swa + a_gl + b_moe + b_rem_model),
            ('(a2)+(b1)+(b3)', a_swa + a_gl + b_moe + b_rem_half),
            ('ideal: tensor kernels only (MoE GEMMs, dense, bias-free attention +8 %) + fixed',
             T - (tot['MoE grouped GEMMs (NVFP4 x44 + MXFP8 x4) [B]']
                  + tot['dense GEMMs: qkvr 20 single + 28 split, wo, dyn-conv, draft ctx K/V [B]']
                  + (1 + FUSED_MMA_OVERHEAD) * tot['attention mainloop without bias (SWA x40 + global x8) [B]'] + fixed)),
        ]
        p("| ceiling | saves s | TTFT s | tok/s | speedup |")
        p("|---|---:|---:|---:|---:|")
        for name, s in rows:
            p(f"| {name} | {s / 1e3:.2f} | {(T - s) / 1e3:.2f} | {n / (T - s) * 1e3:,.0f} | {T / (T - s):.2f}x |")
        p("")


if __name__ == '__main__':
    main()
