"""CPU tests of inkling_attn's visit eligibility (torch CPU, no GPU, no sglang).

    python3 -m unittest discover -s inkling_attn/tests     (from the overlay directory)
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from inkling_attn import patch as attn_patch  # noqa: E402


class FakeCuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def setup(local, images, **mode):
    flags = dict(extend=True, decode=False, verify=False, draft=False) | mode
    forward_mode = types.SimpleNamespace(
        is_extend=lambda: flags["extend"], is_decode=lambda: flags["decode"],
        is_target_verify=lambda: flags["verify"], is_draft_extend=lambda: flags["draft"],
    )
    rel = 1024
    proj = torch.zeros((attn_patch.D_REL, rel), dtype=torch.bfloat16).as_subclass(FakeCuda)
    owner = types.SimpleNamespace(
        is_local=local, head_dim=attn_patch.HEAD_DIM, d_rel=attn_patch.D_REL, num_tp_heads=24, num_tp_kv_heads=8,
        rel_logits_proj=types.SimpleNamespace(proj=proj), rel_extent=rel,
    )
    meta = types.SimpleNamespace()
    backend = types.SimpleNamespace(forward_metadata=meta)
    batch = types.SimpleNamespace(forward_mode=forward_mode, contains_mm_inputs=lambda: images)
    attn = types.SimpleNamespace(pack_gqa=False, sliding_window_size=rel - 1 if local else -1)
    model = types.SimpleNamespace(kv_cache_dtype="auto")
    return owner, batch, attn, (lambda: backend), (lambda: model), meta


def visit(owner, batch, attn, get_backend, get_model):
    with mock.patch.object(attn_patch, "_is_sm12x", lambda device: True):
        return attn_patch._visit_for(owner, batch, attn, get_backend, get_model, None, ("swa", "global"))


class VisitEligibility(unittest.TestCase):
    def test_text_batches(self):
        for local in (True, False):
            v = visit(*setup(local, images=False)[:5])
            self.assertIsNotNone(v, local)
            self.assertEqual(v.local, local)
        for mode in ({"verify": True}, {"decode": True}, {"draft": True}, {"extend": False}):
            self.assertIsNone(visit(*setup(False, images=False, **mode)[:5]), mode)

    def test_image_batches_take_the_fused_readers(self):
        # v2.9.8: Inkling2 image tokens are causal positions (the vision tower only supplies their input
        # embeddings), so an image chunk gets the fused prefill reader (on KV12S, the packed reader / bounded
        # scratch) instead of stock FA4 on a BF16 gather of the packed store.
        for local in (True, False):
            v = visit(*setup(local, images=True)[:5])
            self.assertIsNotNone(v, local)
            self.assertEqual((v.local, v.window_left), (local, 1023 if local else None))

    def test_image_masks_keep_the_stock_path(self):
        for name in ("image_token_ranges", "swa_spec_metadata", "local_attn_metadata"):
            owner, batch, attn, get_backend, get_model, meta = setup(False, images=True)
            setattr(meta, name, object())
            self.assertIsNone(visit(owner, batch, attn, get_backend, get_model), name)


if __name__ == "__main__":
    unittest.main()
