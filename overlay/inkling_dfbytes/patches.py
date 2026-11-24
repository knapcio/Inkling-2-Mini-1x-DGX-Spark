"""inkling_dfbytes patches: the DFlash2 drafter's layers (mx8) and its candidate projection (hot8 / hot16).

Patch points (draft side only):
  sglang.srt.models.dflash
    DFlashDraftModel.load_weights        -> mx8: MXFP8 copies of the decoder layers' BF16 linears, their
                                            quant_method swapped for inkling_draftbytes' W8 method (instances)
    DFlash2DraftModel.compute_candidates -> hot8 / hot16: top-k over the hot rows of the target head
  sglang.srt.speculative.dflash_worker_v2
    DFlashWorkerV2._maybe_build_draft_sampler -> build the hot plane from the target head before the draft
                                                 graph is captured
"""

import functools
import os
import sys

import torch

from . import STATE, DfbytesFatal, _log

MARK = "_inkling_dfbytes"


def _require(owner, names, where):
    missing = [n for n in names if not hasattr(owner, n)]
    if missing:
        raise DfbytesFatal(f"inkling_dfbytes: {where} lacks {missing}")


def _draftbytes():
    from inkling_draftbytes import _opts, ops, patches as dbp, quant, w8gemv

    return _opts, ops, dbp, quant, w8gemv


def configure_draftbytes(opts):
    """inkling_draftbytes' kernels read their settings from its STATE.opts. Under SPEC=dflash2 that package is not
    installed (no DSpark draft); configure it here for the W8 kernels only (no DSpark patch is armed)."""
    _opts, ops, dbp, _, _ = _draftbytes()
    if dbp.STATE.opts is not None:
        raise DfbytesFatal("inkling_dfbytes: inkling_draftbytes is armed (DSpark draft); the two are exclusive")
    text = "draftmx8,wide8,w8tile,w8gemv" if "mx8" in opts else "head8,wide8"
    db_opts = _opts.parse(text, environ=os.environ)
    if "mx8" in opts and not os.environ.get("INKLING_DRAFT_W8_MAX_M", "").strip():
        # GB10 bench 2026-11-24 (tools/bench_dfbytes.py, real drafter weights, cold L2): the 20 drafter linears at
        # 8 rows 3.17 -> 2.04 ms (w8gemv); at 56 rows the Triton W8 path is slower than stock (3.47 vs 3.32 ms)
        import dataclasses

        db_opts = dataclasses.replace(db_opts, w8_max_m=16)
    dbp.STATE.opts = db_opts
    ops.configure(db_opts)
    STATE["db_opts"] = db_opts


class HotPlane:
    def __init__(self, ids64, pack, head_ptr):
        self.ids64 = ids64
        self.pack = pack
        self.n = int(ids64.numel())
        self.head_ptr = head_ptr


def build_hot(model, lm_head):
    _opts, ops, dbp, quant, _ = _draftbytes()
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise DfbytesFatal("inkling_dfbytes: hot plane needs the target's dense BF16 lm_head weight")
    limit = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    valid = getattr(model, "valid_vocab_size", None)
    if valid is not None:
        limit = min(limit, int(valid))
    ids = dbp.load_hot_ids(STATE["hot_file"], STATE["hot_n"], limit)
    top_k = int(getattr(getattr(model, "candidate_selector", None), "top_k", 16) or 16)
    if len(ids) < max(1024, top_k):
        raise DfbytesFatal(f"inkling_dfbytes: only {len(ids)} usable hot ids in {STATE['hot_file']}")
    ids64 = torch.tensor(ids, dtype=torch.int64, device=weight.device)
    rows = weight.index_select(0, ids64)
    if "hot16" in STATE["opts"]:
        pack = quant.bf16_pack(rows.contiguous())
    else:
        pack = quant.quantize_mode(rows, _opts.parse_mode("row", "row"))
    del rows
    hot = HotPlane(ids64, pack, weight.data_ptr())
    STATE["hot_built"] += 1
    _log(f"hot plane: {hot.n} of {limit} ids, {'BF16' if 'hot16' in STATE['opts'] else 'FP8 row'} "
         f"{pack.nbytes() / 1e6:.1f} MB")
    return hot


def hot_logits(h, pack):
    """[N, n_hot] BF16 logits of the hot plane. A BF16 pack runs in <= 64-row launches: the multi-row-block wide
    mode with a BF16 weight needs more shared memory than GB10 has (as in inkling_draftbytes.rnnhoist); rows are
    independent, so the chunking changes nothing numerically."""
    _, ops, _, _, _ = _draftbytes()
    if pack.scale_kind != 0 or h.shape[0] <= 64:
        return ops.w8_linear(h, pack, None, torch.bfloat16)
    out = torch.empty((h.shape[0], pack.n), dtype=torch.bfloat16, device=h.device)
    for i in range(0, h.shape[0], 64):
        out[i:i + 64] = ops.w8_linear(h[i:i + 64], pack, None, torch.bfloat16)
    return out


def _ensure_hot(model, lm_head):
    hot = getattr(model, "_idf_hot", None)
    weight = getattr(lm_head, "weight", None)
    if hot is not None and isinstance(weight, torch.Tensor) and hot.head_ptr == weight.data_ptr():
        return hot
    if hot is not None:
        # Captured draft graphs read the existing plane: a moved target head (online weight update) is unsupported.
        raise DfbytesFatal("inkling_dfbytes: the target lm_head moved after the hot plane was built")
    if torch.cuda.is_current_stream_capturing():
        raise DfbytesFatal("inkling_dfbytes: hot plane missing at capture (built before the draft graph)")
    model._idf_hot = build_hot(model, lm_head)
    return model._idf_hot


def prepare_layers(model):
    _, _, dbp, _, _ = _draftbytes()
    size = dbp._prepare_draftmx8(model, refresh=False)
    linears = dbp._draft_linears(model)
    STATE["prepared"] += 1
    shapes = sorted({tuple(m.weight.shape) for _, m in linears})
    _log(f"mx8: {len(linears)} drafter linears {shapes} -> {size / 1e6:.1f} MB MXFP8")


def patch_dflash(mod):
    _require(mod, ["DFlashDraftModel", "DFlash2DraftModel", "radix_topk"], "sglang.srt.models.dflash")
    base, sel = mod.DFlashDraftModel, mod.DFlash2DraftModel
    _require(base, ["load_weights"], "DFlashDraftModel")
    _require(sel, ["compute_candidates", "_transform_unary_logits"], "DFlash2DraftModel")
    opts = STATE["opts"]
    if "mx8" in opts and not getattr(base.load_weights, MARK, False):
        orig_load = base.load_weights

        @functools.wraps(orig_load)
        def load_weights(self, weights):
            out = orig_load(self, weights)
            prepare_layers(self)
            return out

        setattr(load_weights, MARK, True)
        base.load_weights = load_weights
    if any(p.startswith("hot") for p in opts) and not getattr(sel.compute_candidates, MARK, False):
        orig_cc = sel.compute_candidates
        radix_topk = mod.radix_topk

        @functools.wraps(orig_cc)
        def compute_candidates(self, hidden):
            lm_head = self.lm_head
            if (lm_head is None or hidden.dim() != 2 or not hidden.is_cuda
                    or int(getattr(lm_head, "tp_size", 1)) != 1 or getattr(self, "merge_candidates_fused", False)):
                return orig_cc(self, hidden)
            hot = _ensure_hot(self, lm_head)
            logits = hot_logits(hidden.to(torch.bfloat16), hot.pack)
            vals, idx = radix_topk(logits, self.candidate_selector.top_k)
            ids = hot.ids64.index_select(0, idx.reshape(-1).long()).view(idx.shape)
            return ids, self._transform_unary_logits(vals)

        setattr(compute_candidates, MARK, True)
        sel.compute_candidates = compute_candidates
    return True


def patch_worker(mod):
    _require(mod, ["DFlashWorkerV2"], "sglang.srt.speculative.dflash_worker_v2")
    cls = mod.DFlashWorkerV2
    _require(cls, ["_maybe_build_draft_sampler"], "DFlashWorkerV2")
    if not any(p.startswith("hot") for p in STATE["opts"]) or getattr(cls._maybe_build_draft_sampler, MARK, False):
        return False
    orig = cls._maybe_build_draft_sampler
    unwrap = getattr(mod, "unwrap_lora_layer", None)

    @functools.wraps(orig)
    def _maybe_build_draft_sampler(self, *args, **kwargs):
        # Build the hot plane eagerly, before the draft graph that calls compute_candidates is captured.
        model = getattr(self, "draft_model", None)
        target = getattr(getattr(getattr(self, "_target_worker", None), "model_runner", None), "model", None)
        lm_head = getattr(target, "lm_head", None)
        if lm_head is not None and unwrap is not None:
            lm_head = unwrap(lm_head)  # as the worker does
        megakernel = getattr(mod, "megakernel_enabled", None)
        if (model is not None and getattr(model, "candidate_selector", None) is not None and lm_head is not None
                and int(getattr(lm_head, "tp_size", 1)) == 1 and not getattr(model, "merge_candidates_fused", False)
                and not (callable(megakernel) and megakernel())):
            _ensure_hot(model, lm_head)
        return orig(self, *args, **kwargs)

    setattr(_maybe_build_draft_sampler, MARK, True)
    cls._maybe_build_draft_sampler = _maybe_build_draft_sampler
    return True
