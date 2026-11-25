"""CPU tests of the DFLASH verify settle over all packed visits (INKLING_KV12S_DFLASH_SETTLE).

The hook tests run anywhere. The kernel tests (LaneGroup against the per-visit settle, every byte of the store)
need Triton with TRITON_INTERPRET=1.
"""

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inkling_kv12s import IncompatibleImage
from inkling_kv12s import patch as hooks
from inkling_kv12s.storage import EVENT_WORDS, PAGE, Layout

INTERPRET = bool(importlib.util.find_spec("triton")) and os.getenv("TRITON_INTERPRET") == "1"


class Lane:
    """A packed visit stand-in that records how it was settled."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def settle_pages(self, ids):
        self.log.append(("pages", self.name))
        return ("prepared", ids.tolist())

    def settle(self, ids, commit, prepared=None):
        self.log.append(("settle", self.name, prepared))


def dflash_worker(view, block=4, bs=3):
    pool = SimpleNamespace(full_kv_pool=view)

    class DFlashWorkerV2:
        block_size = block
        _hosts_draft = True

        def __init__(self):
            self.target_worker = SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=pool))
            self._draft_verify_out_cache_loc_buf = torch.arange(bs * block).reshape(bs, block) + 256

        def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None):
            batch.forward_mode = SimpleNamespace(is_extend=lambda: True, is_idle=lambda: False)
            return SimpleNamespace(accept_lens=torch.tensor([2, 1], dtype=torch.int32))

    hooks._register_dflash(SimpleNamespace(DFlashWorkerV2=DFlashWorkerV2))
    return DFlashWorkerV2()


def decode_batch():
    mode = SimpleNamespace(is_extend=lambda: False, is_idle=lambda: False)
    return SimpleNamespace(forward_mode=mode, is_extend_in_batch=False, reqs=[])


class SettleHookTests(unittest.TestCase):
    ROWS = [[256, 257, 258, 259], [260, 261, 262, 263]]

    def _run(self, mode, steps=1, group_module=None):
        log = []
        view = SimpleNamespace(is_inkling_kv12s=True, owned_layers=(Lane(log, "a"), Lane(log, "b"), Lane(log, "c")))
        worker = dflash_worker(view)
        modules = {"inkling_kv12s.device": group_module} if group_module is not None else {}
        with patch.object(hooks, "_DFLASH_SETTLE", mode), patch("inkling_kv12s.pressure.GUARD", []), \
                patch.dict(sys.modules, modules):
            for _ in range(steps):
                worker.forward_batch_generation(decode_batch())
        return log, view

    def test_layer_mode_is_the_per_visit_settle(self):
        log, _ = self._run("layer")
        self.assertEqual(log, [("settle", "a", None), ("settle", "b", None), ("settle", "c", None)])

    def test_shared_mode_builds_the_candidates_once(self):
        log, _ = self._run("shared")
        prepared = ("prepared", self.ROWS)
        self.assertEqual(log, [("pages", "a"), ("settle", "a", prepared), ("settle", "b", prepared),
                               ("settle", "c", prepared)])

    def _group_module(self, build):
        module = ModuleType("inkling_kv12s.device")
        module.LaneGroup = SimpleNamespace(build=build)
        return module

    def test_batch_mode_settles_through_one_group_built_once(self):
        built, calls = [], []

        class Group:
            def settle(self, ids, commit, prepared=None):
                calls.append((ids.tolist(), commit.tolist(), prepared))

        def build(lanes):
            built.append(tuple(lane.name for lane in lanes))
            return Group()

        log, view = self._run("batch", steps=2, group_module=self._group_module(build))
        prepared = ("prepared", self.ROWS)
        self.assertEqual(built, [("a", "b", "c")])                      # cached on the view
        self.assertEqual(calls, [(self.ROWS, [2, 1], prepared)] * 2)
        self.assertEqual(log, [("pages", "a")] * 2)                     # no visit settled on its own
        self.assertIsNotNone(view._inkling_kv12s_settle_group[1])

    def test_batch_mode_falls_back_to_shared_candidates(self):
        built = []

        def build(lanes):
            built.append(len(lanes))
            return None

        with self.assertLogs("inkling_kv12s.patch", "WARNING"):
            log, _ = self._run("batch", steps=2, group_module=self._group_module(build))
        prepared = ("prepared", self.ROWS)
        step = [("pages", "a"), ("settle", "a", prepared), ("settle", "b", prepared), ("settle", "c", prepared)]
        self.assertEqual(built, [3])                                   # the refusal is cached too
        self.assertEqual(log, step * 2)

    def test_lanes_without_candidates_settle_on_their_own(self):
        settled = []
        lane = SimpleNamespace(settle=lambda ids, commit: settled.append(ids.shape))
        view = SimpleNamespace(is_inkling_kv12s=True, owned_layers=(lane, lane))
        worker = dflash_worker(view)
        with patch.object(hooks, "_DFLASH_SETTLE", "batch"), patch("inkling_kv12s.pressure.GUARD", []):
            worker.forward_batch_generation(decode_batch())
        self.assertEqual(settled, [(2, 4), (2, 4)])

    def test_dspark_keeps_the_shared_candidates(self):
        log = []
        lanes = (Lane(log, "a"), Lane(log, "b"))
        ids = torch.arange(8).view(2, 4)
        hooks._settle_lanes(SimpleNamespace(), lanes, ids, torch.tensor([1, 2]), "shared")
        prepared = ("prepared", ids.tolist())
        self.assertEqual(log, [("pages", "a"), ("settle", "a", prepared), ("settle", "b", prepared)])

    def test_switch_values(self):
        self.assertEqual({hooks._SETTLE_MODES[v] for v in ("1", "on", "batch")}, {"batch"})
        self.assertEqual(hooks._SETTLE_MODES["shared"], "shared")
        self.assertEqual({hooks._SETTLE_MODES[v] for v in ("0", "off", "layer")}, {"layer"})
        with patch.object(hooks, "_DFLASH_SETTLE", None), patch.object(hooks, "_DFLASH_SETTLE_RAW", "2"), \
                self.assertRaises(IncompatibleImage):
            hooks.register_static()


def packed_store(pages, heads, lanes=8, slots=6, overflow=4, fallback=2):
    """The static pool's binding (static.py): one envelope buffer, lock and event rows of shared buffers."""
    from inkling_kv12s.device import DeviceLayer

    envelope = Layout(heads).size
    stride = envelope * lanes
    raw = torch.zeros(pages * stride, dtype=torch.uint8)
    locks = torch.zeros((lanes, slots + fallback + overflow), dtype=torch.int32)
    events = torch.zeros((lanes, EVENT_WORDS), dtype=torch.int32)
    layers = [DeviceLayer(raw, heads, stride, i * envelope, slots, overflow, fallback, locks=locks[i],
                          events=events[i]) for i in range(lanes)]
    for layer in layers:  # torch.empty side arenas: start both stores from the same bytes
        layer.stage.zero_()
        layer.overflow.zero_()
    return SimpleNamespace(raw=raw, locks=locks, events=events, layers=layers)


def store_bytes(store):
    parts = {"envelopes": store.raw, "locks": store.locks, "events": store.events}
    for i, layer in enumerate(store.layers):
        parts[f"stage{i}"] = layer.stage
        parts[f"overflow{i}"] = layer.overflow
        parts[f"fail{i}"] = layer.fail
    return parts


@unittest.skipUnless(INTERPRET, "requires Triton and TRITON_INTERPRET=1")
class LaneGroupTests(unittest.TestCase):
    """LaneGroup.settle against DeviceLayer.settle per visit, from identical stores: every byte equal."""

    HEADS = 2
    WIDTH = 8

    def _steps(self, spill, requests=3, steps=10, seed=0, lanes=8, **sizes):
        from inkling_kv12s.device import DeviceLayer, LaneGroup

        g = torch.Generator().manual_seed(seed)
        per_request = 3
        pages = 1 + requests * per_request
        stores = [packed_store(pages, self.HEADS, lanes=lanes, **sizes) for _ in range(2)]
        group = LaneGroup.build(stores[1].layers)
        self.assertIsNotNone(group)
        prefix = [PAGE - 3 - 2 * i for i in range(requests)]       # every request crosses a page end early
        table = [list(range(1 + i * per_request, 1 + (i + 1) * per_request)) for i in range(requests)]

        def values(n):
            if spill:  # arbitrary BF16 bit patterns: most words escape the exponent window, page-heads spill
                return torch.randint(-32768, 32768, (n, self.HEADS, 128), generator=g, dtype=torch.int32).to(
                    torch.int16).view(torch.bfloat16)
            return torch.randn((n, self.HEADS, 128), generator=g).to(torch.bfloat16)

        for i in range(requests):  # committed prompt rows
            rows = torch.tensor([table[i][p // PAGE] * PAGE + p % PAGE for p in range(prefix[i])])
            for lane in range(lanes):
                k, v = values(rows.numel()), values(rows.numel())
                for store in stores:
                    store.layers[lane].write(rows, k, v)
        for step in range(steps):
            loc = torch.zeros((requests + 1, self.WIDTH), dtype=torch.int64)  # + one padding row at the sink
            for i in range(requests):
                loc[i] = torch.tensor([table[i][p // PAGE] * PAGE + p % PAGE
                                       for p in range(prefix[i], prefix[i] + self.WIDTH)])
            for lane in range(lanes):  # tentative verify writes, different values per visit
                k, v = values(loc.numel()), values(loc.numel())
                for store in stores:
                    store.layers[lane].write(loc.view(-1), k, v, tentative=True)
            accept = torch.randint(1, self.WIDTH + 1, (requests + 1,), generator=g)
            accept[-1] = 0
            for layer in stores[0].layers:
                layer.settle(loc, accept)
            group.settle(loc, accept, prepared=DeviceLayer.settle_pages(loc))
            for name, a in store_bytes(stores[0]).items():
                b = store_bytes(stores[1])[name]
                self.assertTrue(torch.equal(a.view(torch.uint8) if a.dtype != torch.uint8 else a,
                                            b.view(torch.uint8) if b.dtype != torch.uint8 else b),
                                f"step {step}: {name} differs")
            for i in range(requests):
                prefix[i] += int(accept[i])
        return stores

    def test_plain_values(self):
        stores = self._steps(spill=False)
        self.assertGreater(int((stores[1].layers[3].flags == 1).sum()), 0)   # pages were sealed

    def test_spills_and_arena_exhaustion(self):
        stores = self._steps(spill=True, steps=8, slots=12, overflow=3, fallback=1)
        events = stores[1].events
        self.assertGreater(int(events.sum()), 0)                       # fallback / kept / lost all exercised

    def test_one_visit(self):
        self._steps(spill=False, lanes=1, steps=4)

    def test_mixed_or_uneven_visits_are_not_grouped(self):
        from inkling_kv12s.device import LaneGroup, RawLane

        store = packed_store(3, self.HEADS)
        raw = RawLane(store.raw, self.HEADS, store.layers[0].page_stride, 0, 3)
        self.assertIsNone(LaneGroup.build(store.layers + [raw]))
        self.assertIsNone(LaneGroup.build([]))
        other = packed_store(3, self.HEADS, slots=7)
        self.assertIsNone(LaneGroup.build(store.layers[:4] + other.layers[4:]))
        self.assertIsNone(LaneGroup.build(store.layers[:2] + store.layers[:1]))     # one visit twice
        twin = packed_store(3, self.HEADS)
        twin.layers[1].stage_locks = store.layers[0].stage_locks                     # shared lock words
        self.assertIsNone(LaneGroup.build(store.layers[:1] + twin.layers[1:2]))
        self.assertEqual(LaneGroup.build(store.layers).deltas.shape, (8, 12))

    def test_geometry_is_checked(self):
        from inkling_kv12s.device import LaneGroup

        store = packed_store(3, self.HEADS)
        group = LaneGroup.build(store.layers)
        with self.assertRaises(ValueError):
            group.settle(torch.zeros(4, dtype=torch.int64), torch.zeros(1, dtype=torch.int32))
        with self.assertRaises(ValueError):
            group.settle(torch.zeros((2, 4), dtype=torch.int64), torch.zeros(3, dtype=torch.int32))
        group.settle(torch.zeros((0, 4), dtype=torch.int64), torch.zeros(0, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
