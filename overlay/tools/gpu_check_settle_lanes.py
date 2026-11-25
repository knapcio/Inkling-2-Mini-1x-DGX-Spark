#!/usr/bin/env python3
"""GB10 check of the DFLASH verify settle over all packed global visits (INKLING_KV12S_DFLASH_SETTLE). No server.

  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/gpu_check_settle_lanes.py \
      [--reference <previous release>/overlay]

Copies of the static pool's packed store (8 visits in one envelope buffer, lock and event rows of shared buffers, as
static.py builds them) get the same verify steps, one copy per settle arm:
- "v294" (with --reference): the previous release's inkling_kv12s, loaded from that overlay: its writer, its
  kernels and its hook's loop (each visit settled on its own);
- "layer", "shared", "batch": this tree through patch._settle_lanes, the function the DFLASH hook calls.
Per batch size, two CUDA graphs per copy, captured once and replayed every step, do what the target verify graph
does: tentative K/V writes of every visit (different values per visit; one graph-padding row at the sink page), and
the packed attention of every visit over the prefix plus the step's rows. Every third step is ragged: each request
writes only its leading rows (the rest go to the sink page) and accepts at most those; such steps skip attention.
Then the eager settle with random accepted lengths. Checked at every step:
- attention outputs of every visit: equal bitwise across all arms, and the first arm equal to the stock BF16 kernel
  (same batch, same page table) on a BF16 oracle cache;
- plain values: every byte of the envelopes, staging and overflow arenas, lock and event words and failure marks
  equal across all arms (staging has a slot per page and no page-head spills, so no claim ever races);
- spill values (most words escape the exponent window; an overflow arena of one page of page-heads and one fallback
  page, so both run out and later pages stay raw in staging), one request: at most one page is promoted per step, so
  no claim races and every byte must be equal as above;
- spill values, four requests: pages promoted in the same step race for the last side slots, and which page wins
  depends on scheduling even inside one launch of the per-visit path (both outcomes exact). There the check is
  attention outputs and every accepted row's bits, no row lost in any arm (raw byte and page-status equality are
  reported); across the spill cases the fallback and kept events must have happened.
At the end, every accepted row of every visit must equal the written K/V. Then the eager settle cost per step,
8 visits, page-sealing commits, store refreshed outside the timed interval.
"""

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from statistics import median
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_check_kv12s as base  # noqa: E402  (also puts the overlay root on sys.path)

import torch  # noqa: E402

from inkling_kv12s.storage import (DIM, EVENT_FALLBACK, EVENT_KEPT, EVENT_LOST, EVENT_WORDS, PAGE,  # noqa: E402
                                   STAGE_MASK, STAGE_SHIFT, Layout)

VISITS = 8
WIDTH = 8
FAILED = []
COVERAGE = [0, 0]  # spill cases: fallback, kept events


def log(*a):
    print(*a, flush=True)


def load_reference(overlay):
    """The previous release's inkling_kv12s.device, as package kv12s_ref (it never touches this tree's modules)."""
    pkg = Path(overlay) / "inkling_kv12s"
    spec = importlib.util.spec_from_file_location("kv12s_ref", pkg / "__init__.py",
                                                  submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["kv12s_ref"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("kv12s_ref.device")


def arms(reference):
    from inkling_kv12s import device
    from inkling_kv12s.patch import _settle_lanes

    out = []
    if reference is not None:
        def old_hook(store, ids, commit):
            for lane in store.layers:  # the previous DFLASH hook
                lane.settle(ids, commit)
        out.append(("v294", reference.DeviceLayer, old_hook))
    for mode in ("layer", "shared", "batch"):
        out.append((mode, device.DeviceLayer,
                    lambda store, ids, commit, mode=mode: _settle_lanes(store.view, store.layers, ids, commit, mode)))
    return out


def packed_store(cls, pages, slots, overflow, fallback, device):
    heads = base.HEADS_K
    envelope = Layout(heads).size
    stride = envelope * VISITS
    raw = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
    locks = torch.zeros((VISITS, slots + fallback + overflow), dtype=torch.int32, device=device)
    events = torch.zeros((VISITS, EVENT_WORDS), dtype=torch.int32, device=device)
    layers = [cls(raw, heads, stride, i * envelope, slots, overflow, fallback, locks=locks[i], events=events[i])
              for i in range(VISITS)]
    for layer in layers:  # torch.empty side arenas: start every copy from the same bytes
        layer.stage.zero_()
        layer.overflow.zero_()
    return SimpleNamespace(raw=raw, locks=locks, events=events, layers=layers, view=SimpleNamespace())


def store_parts(store):
    parts = {"envelopes": store.raw, "locks": store.locks, "events": store.events}
    for i, layer in enumerate(store.layers):
        parts[f"stage{i}"] = layer.stage
        parts[f"overflow{i}"] = layer.overflow
        parts[f"fail{i}"] = layer.fail
    return parts


def differing(a, b):
    pa, pb = store_parts(a), store_parts(b)
    return [name for name in pa if not torch.equal(pa[name].contiguous().view(torch.uint8),
                                                   pb[name].contiguous().view(torch.uint8))]


def content(store, pages, history):
    """Per visit: status class and row count per page (slot ids dropped), event words, held lock counts, rows."""
    out = []
    for layer in store.layers:
        f = layer.flags[pages].long()
        staged = f >= 2
        fallback = staged & ((f >> STAGE_SHIFT) >= layer.live)
        status = torch.where(staged, 2 + (f & STAGE_MASK) * 4 + fallback.long() * 1024, f)
        out.append((status.tolist(), layer.events.tolist(), int(layer.stage_locks.ne(0).sum()),
                    int(layer.overflow_locks.ne(0).sum()), layer.read_rows(history).contiguous().view(torch.int16)))
    return out


def same_content(a, b):
    return len(a) == len(b) and all(x[:4] == y[:4] and torch.equal(x[4], y[4]) for x, y in zip(a, b))


def equal_bits(label, got, want):
    if got.shape != want.shape or got.dtype != want.dtype:
        raise AssertionError(f"{label}: shape/dtype {tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}")
    base.equal_bits(label, got, want)


def values(shape, device, generator, spill):
    if not spill:
        return base.normal(shape, device, generator)
    # Finite BF16 words with exponents 100..139: about 60 % escape the inline window, so every page-head spills.
    bits = torch.randint(0, 1 << 16, shape, device=device, generator=generator, dtype=torch.int32)
    exp = torch.randint(100, 140, shape, device=device, generator=generator, dtype=torch.int32)
    return ((bits & 0x807F) | (exp << 7)).to(torch.int16).view(torch.bfloat16)


def run_case(B, steps, device, generator, reference, *, spill, oracle_lanes):
    from inkling_kv12s.attention import partials
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    H, HQ = base.HEADS_K, base.HEADS_Q
    start = [PAGE - 5 - (3 * i) % 40 for i in range(B)]                 # prompts end near a page end
    per_req = (max(start) + steps * WIDTH + PAGE - 1) // PAGE
    npages = 1 + B * per_req
    # One staging slot per page (no staging claim ever probes). Spill: an overflow arena of one page's page-heads and
    # one fallback page, so both run out and later pages stay raw in their staging slot.
    slots, overflow, fallback = (npages + 1, 2 * H, 1) if spill else (npages + 1, 64, 4)
    arm_list = arms(reference)
    stores = [packed_store(cls, npages, slots, overflow, fallback, device) for _, cls, _ in arm_list]
    table = torch.zeros((B, per_req), dtype=torch.int32, device=device)
    rows = B + 1                                                           # + one graph-padding row (sink page)
    loc_w = torch.zeros((rows, WIDTH), dtype=torch.int64, device=device)  # the rows the verify writes
    seq = base.lengths(device, [WIDTH] * B)
    q = base.normal((B * WIDTH, HQ, DIM), device, generator)
    bias = base.normal((B * WIDTH, HQ, base.EXTENT), device, generator)
    kv = torch.zeros((VISITS, 2, rows * WIDTH, H, DIM), dtype=torch.bfloat16, device=device)

    def writes(store):
        def run():
            for lane, layer in enumerate(store.layers):
                layer.write(loc_w.view(-1), kv[lane, 0], kv[lane, 1], tentative=True)
        return run

    def attention(store):
        def run():
            return [partials(layer, q, table, seq, bias, base.SCALE, device_metadata=True) for layer in store.layers]
        return run

    # Captured with every location 0 and every page table entry 0: the warm-up writes nothing.
    write_graphs = [base.capture_forward(writes(store), device)[0] for store in stores]
    attn_graphs = [base.capture_forward(attention(store), device) for store in stores]
    order = torch.randperm(npages - 1, generator=torch.Generator().manual_seed(B))[:B * per_req] + 1
    table.copy_(order.view(B, per_req).to(device=device, dtype=torch.int32))
    tab = table.cpu()
    oracle = torch.zeros((VISITS, 2, npages * PAGE, H, DIM), dtype=torch.bfloat16, device=device)
    history = []
    cols = torch.arange(WIDTH)

    def phys(b, positions):
        return [int(tab[b, p // PAGE]) * PAGE + p % PAGE for p in positions]

    prefix = list(start)
    for b in range(B):                                                     # committed prompt rows
        ids = torch.tensor(phys(b, range(prefix[b])), device=device)
        for lane in range(VISITS):
            k, v = (values((ids.numel(), H, DIM), device, generator, spill) for _ in range(2))
            oracle[lane, 0, ids], oracle[lane, 1, ids] = k, v
            for store in stores:
                store.layers[lane].write(ids, k, v)
        history += ids.tolist()
    torch.cuda.synchronize()
    exact = not spill or B == 1                                            # no claim can race
    raw_equal = True
    rng = torch.Generator().manual_seed(1000 + B + 7 * spill)
    for step in range(steps):
        ragged = step % 3 == 2
        width = (torch.randint(1, WIDTH + 1, (B,), generator=rng) if ragged
                 else torch.full((B,), WIDTH, dtype=torch.int64))
        loc = torch.zeros((rows, WIDTH), dtype=torch.int64)
        for b in range(B):
            loc[b] = torch.tensor(phys(b, range(prefix[b], prefix[b] + WIDTH)))
        written = torch.zeros((rows, WIDTH), dtype=torch.bool)
        written[:B] = cols[None, :] < width[:, None]
        loc_w.copy_(torch.where(written, loc, 0))
        ids = loc.to(device)                                               # the settle sees every verify location
        seq.copy_(torch.tensor([p + WIDTH for p in prefix], dtype=torch.int32))
        kv.copy_(values(kv.shape, device, generator, spill))
        q.copy_(base.normal(q.shape, device, generator))
        bias.copy_(base.normal(bias.shape, device, generator))
        live = loc[written].to(device)
        oracle[:, :, live] = kv[:, :, written.view(-1).to(device)]
        for graph in write_graphs:
            graph.replay()
        if not ragged:
            for graph, _ in attn_graphs:
                graph.replay()
            outs = [out for _, out in attn_graphs]
            for lane in range(VISITS):
                for (name, _, _), other in zip(arm_list[1:], outs[1:]):
                    for got, want in zip(other[lane], outs[0][lane]):
                        try:
                            equal_bits(f"{name} vs {arm_list[0][0]} attention", got, want)
                        except AssertionError as exc:
                            FAILED.append(f"B={B} step {step} visit {lane}: {exc}")
                if lane in oracle_lanes:
                    kc, vc = (oracle[lane, i].view(npages, PAGE, H, DIM) for i in range(2))
                    expected = decode_partials(q, kc, vc, table, seq, bias, base.SCALE)
                    for got, want in zip(outs[0][lane], expected):
                        try:
                            equal_bits("packed verify vs BF16 oracle", got, want)
                        except AssertionError as exc:
                            FAILED.append(f"B={B} step {step} visit {lane}: {exc}")
        accept = torch.zeros(rows, dtype=torch.int64)
        for b in range(B):
            accept[b] = int(torch.randint(1, int(width[b]) + 1, (1,), generator=rng))
        commit = accept.to(device=device, dtype=torch.int32)
        for store, (_, _, settle) in zip(stores, arm_list):
            settle(store, ids, commit)
        torch.cuda.synchronize()
        for b in range(B):
            history += loc[b, :int(accept[b])].tolist()
            prefix[b] += int(accept[b])
        for (name, _, _), store in zip(arm_list[1:], stores[1:]):
            differ = differing(stores[0], store)
            if differ:
                raw_equal = False
                if exact:
                    FAILED.append(f"B={B} step {step}: {name} store bytes differ from {arm_list[0][0]}: {differ[:6]}")
    batch_store = stores[[name for name, _, _ in arm_list].index("batch")]
    if batch_store.view._inkling_kv12s_settle_group[1] is None:
        FAILED.append(f"B={B}: the batched copy was not grouped")
    pages = list(range(1, npages))
    ref = content(stores[0], pages, history)
    status_equal = True
    for (name, _, _), store in zip(arm_list[1:], stores[1:]):
        got = content(store, pages, history)
        if not same_content(got, ref):
            status_equal = False
            if exact:
                FAILED.append(f"B={B}: {name} page status, counts or accepted rows differ from {arm_list[0][0]}")
        if not all(torch.equal(x[4], y[4]) for x, y in zip(got, ref)):
            FAILED.append(f"B={B}: {name} accepted rows differ from {arm_list[0][0]}")
    for (name, _, _), store in zip(arm_list, stores):
        lost = int(store.events[:, EVENT_LOST].sum())
        if lost:
            FAILED.append(f"B={B}: {name} lost {lost} rows")
    for lane in range(VISITS):
        want = torch.stack((oracle[lane, 0, history], oracle[lane, 1, history]), dim=1).view(torch.int16)
        if not torch.equal(ref[lane][4], want):
            FAILED.append(f"B={B} visit {lane}: accepted rows differ from the written K/V")
    sealed = sum(s == 1 for s in ref[0][0])
    events = [sum(r[1][i] for r in ref) for i in range(EVENT_WORDS)]
    if spill:
        COVERAGE[0] += events[EVENT_FALLBACK]
        COVERAGE[1] += events[EVENT_KEPT]
    log(f"  {'spill' if spill else 'plain'} B={B:>2}: {steps} steps ({steps // 3} ragged), {len(history)} rows/visit, "
        f"{sealed} pages sealed in visit 0, events (fallback, kept, lost) {events}; arms "
        f"{'/'.join(name for name, _, _ in arm_list)}: store bytes "
        f"{'identical' if raw_equal else 'differ (claim order)'}, page status "
        f"{'identical' if status_equal else 'differs (claim order)'}; {'ok' if not FAILED else 'FAILED'}")
    del write_graphs, attn_graphs, stores, oracle
    torch.cuda.empty_cache()


def time_settle(B, device, reps):
    from inkling_kv12s.device import DeviceLayer
    from inkling_kv12s.patch import _settle_lanes

    npages = 1 + 2 * B
    store = packed_store(DeviceLayer, npages, npages + 1, 64, 4, device)
    head = torch.cat([(1 + 2 * i) * PAGE + torch.arange(PAGE - 4) for i in range(B)]).to(device)
    pos = torch.arange(PAGE - 4, PAGE - 4 + WIDTH)
    loc = torch.stack([(1 + 2 * i) * PAGE + pos for i in range(B)]).to(device)  # crosses into page 2 + 2i
    pages = torch.arange(1, npages, device=device)
    commit = torch.full((B,), WIDTH, dtype=torch.int32, device=device)           # every request seals a page
    kv = base.normal((head.numel(), base.HEADS_K, DIM), device, torch.Generator(device=device).manual_seed(B))

    def refresh():
        for layer in store.layers:
            layer.release(pages)
            layer.write(head, kv, kv)
            layer.write(loc.view(-1), kv[:B * WIDTH], kv[:B * WIDTH], tentative=True)

    res = {}
    for mode in ("layer", "shared", "batch") * 2:
        ts = []
        for _ in range(reps):
            refresh()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record()
            _settle_lanes(store.view, store.layers, loc, commit, mode)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        res[mode] = median(ts[5:])
    sealed = int((store.layers[0].flags[1::2] == 1).sum()) if B else 0
    log(f"  settle B={B:>2} ({B * WIDTH} rows, {VISITS} visits, eager, {sealed} pages sealed per visit): per visit "
        f"{res['layer']:.1f} us, shared {res['shared']:.1f} us, batched {res['batch']:.1f} us -> "
        f"{(res['layer'] - res['batch']) / 1000:.3f} ms per step")


def run_checks(args):
    import triton

    device = torch.device("cuda", torch.cuda.current_device())
    log(f"Torch {torch.__version__}, Triton {triton.__version__}, {torch.cuda.get_device_name(device)}, "
        f"ptxas {os.environ.get('TRITON_PTXAS_BLACKWELL_PATH', 'bundled')}")
    reference = load_reference(args.reference) if args.reference else None
    log(f"arms: {'/'.join(name for name, _, _ in arms(reference))}"
        f"{'' if reference else ' (no --reference: the previous release is not compared)'}")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, 12 * base.GIB / total), device)
    generator = torch.Generator(device=device).manual_seed(295)
    with torch.inference_mode():
        for B in args.batches:
            run_case(B, args.steps if B < 48 else max(9, args.steps // 2), device, generator, reference,
                     spill=False, oracle_lanes=range(VISITS) if B <= 8 else (0, VISITS - 1))
        for B in (1, 4):
            run_case(B, args.steps, device, generator, reference, spill=True, oracle_lanes=(0, VISITS - 1))
        if not (COVERAGE[0] and COVERAGE[1]):
            FAILED.append(f"spill cases did not exhaust the side arenas (fallback, kept events {COVERAGE})")
        if args.reps:
            for B in (1, 8, 24, 48):
                time_settle(B, device, args.reps)
    peak = torch.cuda.max_memory_allocated(device) / base.GIB
    if FAILED:
        log("FAILED: " + "; ".join(FAILED[:12]))
        raise SystemExit(1)
    log(f"ALL OK: settle arms bit-identical. Peak allocated {peak:.2f} GiB.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", help="the previous release's overlay directory (holds inkling_kv12s/)")
    parser.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 8, 24, 48])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--reps", type=int, default=100, help="settle timing samples per mode (0: skip)")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--build-timeout-seconds", type=int, default=240)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        parser.exit(status=1, message="This check requires GB10 (SM121).\n")
    if args.worker:
        run_checks(args)
        return
    deadline = time.monotonic() + args.timeout_seconds
    with tempfile.TemporaryDirectory(prefix="kv12s-settle-") as cache:
        env = dict(os.environ, INKLING_ATTN_CACHE=cache, PYTHONUNBUFFERED="1")
        env.pop("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", None)   # Triton's strict default
        env.pop("TRITON_INTERPRET", None)
        root = base._ROOT if (base._ROOT / "inkling_kv12s").is_dir() else base._ROOT / "overlay"
        log("[extension prebuild] start (fresh cache)")
        base.run_process(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(root)!r}); "
             "from inkling_attn import build; sys.exit(build.main([]))"],
            env, min(args.build_timeout_seconds, deadline - time.monotonic()), "extension prebuild")
        log("[extension prebuild] done")
        base.run_process([sys.executable, __file__, *sys.argv[1:], "--worker"], env,
                         deadline - time.monotonic(), "GPU checker")


if __name__ == "__main__":
    main()
