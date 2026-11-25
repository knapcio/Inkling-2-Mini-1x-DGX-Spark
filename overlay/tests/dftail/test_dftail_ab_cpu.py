"""CPU tests of (A defers to B) in inkling_dftail: the switch, the install scope, the reservation (covered / uncovered
snapshots), its check after the radix lookup (fallback to the prefix tail's bound), its check before the restore
(stale, flushed or replaced snapshot), the capture-eviction pin, and that the switch off is the v2.9.3 path. A stand-in
Req with the stock lookup order (bound -> match -> bound again) and a stand-in radix tree whose reusable endpoints are
explicit; the B rig (paged ring with the stock arithmetic, CPU KV buffers) does the restores. No GPU, no sglang.

    python3 -m unittest discover -s tests/dftail
"""
import os
import sys
import types
import unittest
import weakref
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_dftail as dt  # noqa: E402
from inkling_dftail import _hook, _sources, bpatch  # noqa: E402
from inkling_dftail import snapshot as sn  # noqa: E402
from test_dftail_b_cpu import PAGE, Rig, quiet, reset_stats, toks, torch  # noqa: E402

TAIL = 2048


class UnifiedRadixCache:
    """Radix stand-in (named as the device-only tree the deferral is allowed on): the reusable endpoints (state
    checkpoints) it holds; a match is the longest one within the limit. ``during`` runs inside the match."""
    cache_controller = linker = None

    def __init__(self, ends, during=None):
        self.ends, self.during, self.limits = sorted(ends), during, []

    def match(self, limit):
        self.limits.append(limit)
        if self.during is not None:
            self.during()
        return max([e for e in self.ends if e <= limit], default=0)


Tree = UnifiedRadixCache


def stock_module():
    class Req:
        session = None

        def __init__(self, tokens, salt=None, extra=None):
            self.origin_input_ids = array("q", tokens)
            self.full_untruncated_fill_ids = array("q", tokens)
            self.cache_salt, self.extra_key = salt, extra
            self.positional_embed_overrides = None
            self.extend_end = None
            self.prefix_len = 0
            self.kv = types.SimpleNamespace(req_pool_idx=None)

        def is_dllm(self):
            return False

        def _compute_max_prefix_len(self, input_len):
            return max(input_len - 1, 0)

        def init_next_round_input(self, tree_cache=None, cow_mamba=None, *, defer_cow_slot=False):
            input_len = len(self.full_untruncated_fill_ids)
            key_limit = self._compute_max_prefix_len(input_len)   # the bound the match uses
            if tree_cache is not None:
                self.prefix_len = tree_cache.match(key_limit)
                max_len = self._compute_max_prefix_len(input_len)  # match_kv_cache asks again
                self.num_matched_prefix_tokens = min(self.prefix_len, max_len)

    return types.SimpleNamespace(Req=Req)


def install(defer=True, tail=TAIL, slots=2):
    """dt.install's effect on a stand-in module, without the import hook or the source pin."""
    on = bool(defer and tail and slots)
    dt.STATE.update(installed=True, tail=tail, max_input=4096, bounded=0, defer=on, deferred=0, defer_dropped=0)
    sn.STATE.update(defer=on, solo=True)
    dt.SCOPE.update(dflash=True, why="", logged=False)
    mod = stock_module()
    dt.patch(mod)
    return mod


def uninstall():
    dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0, defer=False, deferred=0, defer_dropped=0)
    sn.STATE.update(slots=0, defer=False, solo=False)


def ab(name):
    return sn.STATE["ab_stats"][name]


@unittest.skipIf(torch is None, "needs torch (CPU)")
class Base(unittest.TestCase):
    N = 3000

    def setUp(self):
        reset_stats()
        sn.STATE["generation"] = 0
        for k in sn.STATE["ab_stats"]:
            sn.STATE["ab_stats"][k] = 0
        self.mod = install()                                  # before the store: its slots hold one page more
        self.rig = Rig(slots=2)
        sn.STATE["stores"].add(self.rig.store)
        self.t = toks(self.N)

    def tearDown(self):
        sn.STATE["stores"] = weakref.WeakSet()
        uninstall()

    def prime(self, tokens, slot=0, prefix=0):
        """A cold (or prefix-hit) prefill of a whole prompt in ring slot ``slot``: captured at its end."""
        req = self.mod.Req(tokens)
        self.rig.alloc(req, slot)
        with quiet():
            self.rig.extend(req, prefix, len(tokens))
        return req

    def lookup(self, req, tree):
        with quiet():
            req.init_next_round_input(tree)
        return req.prefix_len

    def run_prefill(self, req, slot):
        """The admitted request's first extend: record_floors (restore) -> append -> drain -> forward epoch."""
        self.rig.alloc(req, slot)
        with quiet():
            self.rig.extend(req, req.prefix_len, len(req.full_untruncated_fill_ids))
        self.rig.store.index.epoch += 1
        return int(self.rig.ring.min_pos[slot])


class Switch(unittest.TestCase):
    def tearDown(self):
        uninstall()

    def install_env(self, env):
        """dt.install with the source pin stubbed; returns schedule_batch's stand-in after its import hook ran."""
        saved = _sources.check
        _sources.check = lambda *a, **k: None
        try:
            with quiet():
                dt.install(env)
            mod = stock_module()
            _hook._run(dt.MODULE, mod)
            return mod
        finally:
            _sources.check = saved
            for name in (dt.MODULE,) + tuple(bpatch.MODULES):
                _hook._callbacks.pop(name, None)

    def test_install_arms_only_with_tail_and_snapshots(self):
        both = {dt.ENV: "2048", sn.ENV_SLOTS: "8"}
        for env, want in ((dict(both, **{dt.ENV_DEFER: "1"}), True), (dict(both, **{dt.ENV_DEFER: "0"}), False),
                          (dict(both), False), ({dt.ENV: "2048", dt.ENV_DEFER: "1"}, False)):
            uninstall()
            mod = self.install_env(env)
            self.assertEqual(dt.STATE["defer"], want, env)
            self.assertEqual(getattr(mod.Req.init_next_round_input, dt.MARK, False), want, env)

    def test_install_refuses_a_bad_value(self):
        with self.assertRaises(RuntimeError):
            self.install_env({dt.ENV: "2048", sn.ENV_SLOTS: "8", dt.ENV_DEFER: "yes"})
        self.assertFalse(dt.STATE["installed"])

    def test_parse(self):
        self.assertFalse(dt.defer_from_env({}))
        self.assertFalse(dt.defer_from_env({dt.ENV_DEFER: " 0 "}))
        self.assertTrue(dt.defer_from_env({dt.ENV_DEFER: "1"}))
        for raw in ("2", "on", "yes", "-1"):
            with self.assertRaises(RuntimeError, msg=raw):
                dt.defer_from_env({dt.ENV_DEFER: raw})

    def test_needs_both_tail_and_snapshots(self):
        for tail, slots, want in ((TAIL, 8, True), (0, 8, False), (TAIL, 0, False)):
            mod = install(defer=True, tail=tail, slots=slots)
            self.assertEqual(dt.STATE["defer"], want)
            wrapped = getattr(mod.Req.init_next_round_input, dt.MARK, False)
            self.assertEqual(wrapped, want and bool(tail))

    def test_off_is_the_v293_patch(self):
        plain = stock_module()
        orig_lookup = plain.Req.init_next_round_input
        dt.STATE.update(installed=True, tail=TAIL, max_input=4096, defer=False)
        dt.patch(plain)
        self.assertIs(plain.Req.init_next_round_input, orig_lookup)   # the lookup is not wrapped
        self.assertTrue(getattr(plain.Req._compute_max_prefix_len, dt.MARK, False))


class Covered(Base):
    def test_repeated_prompt_reuses_cache_and_restores(self):
        self.prime(self.t)                                   # snapshot: end 3000, rows from 896 (page-aligned)
        req = self.mod.Req(self.t)
        tree = Tree([0, 640, 2944])
        P = self.lookup(req, tree)
        self.assertEqual(tree.limits, [2999])                # the stock bound, not 3000 - 2048
        self.assertEqual(P, 2944)
        self.assertEqual(dt.STATE["deferred"], 1)
        self.assertEqual(dt.STATE["bounded"], 0)
        floor = self.run_prefill(req, 1)
        self.assertEqual(floor, 896)                        # restored [896, 2944) + extend [2944, 3000): A's floor
        self.assertTrue(self.rig.read_ok(1, 896, 3000))
        self.assertEqual(ab("reserved_restores"), 1)
        self.assertNotIn(sn.RES_ATTR, req.__dict__)          # consumed by the restore

    def test_bound_capped_at_snapshot_end(self):
        self.prime(self.t[:2900])                            # snapshot of a 2,900-token prefix of the prompt
        req = self.mod.Req(self.t)
        tree = Tree([0, 2880, 2960])                          # the tree reuses past the snapshot's end
        P = self.lookup(req, tree)
        self.assertEqual(tree.limits, [2900])
        self.assertEqual(P, 2880)                            # never above the snapshot: no gap in the ring
        floor = self.run_prefill(req, 1)
        self.assertEqual(floor, 896)
        self.assertTrue(self.rig.read_ok(1, 896, 3000))

    def test_short_prompt_full_window(self):
        t = toks(1500)
        self.prime(t)                                        # lo 0: the whole prompt
        req = self.mod.Req(t)
        P = self.lookup(req, Tree([0, 1408]))
        self.assertEqual(P, 1408)                            # A alone would re-prefill all 1,500 tokens
        self.assertEqual(self.run_prefill(req, 1), 0)
        self.assertTrue(self.rig.read_ok(1, 0, 1500))

    def test_endpoint_below_window_needs_no_restore(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        P = self.lookup(req, Tree([0, 512]))                 # the target can reuse only 512 anyway
        self.assertEqual(P, 512)
        self.assertEqual(self.run_prefill(req, 1), 512)      # the extend fills the whole window itself
        self.assertTrue(self.rig.read_ok(1, 952, 3000))

    def test_same_batch_two_reservations_of_one_slot(self):
        self.prime(self.t)
        a, b = self.mod.Req(self.t), self.mod.Req(self.t)
        tree = Tree([0, 2944])
        self.lookup(a, tree)
        self.lookup(b, tree)
        self.rig.alloc(a, 1)
        self.rig.alloc(b, 2)
        for r in (a, b):
            r.extend_end = 3000
        with quiet():
            self.rig.record(self.rig.ring, [a, b], [2944, 2944], self.rig.gen)
        self.assertEqual([int(self.rig.ring.min_pos[s]) for s in (1, 2)], [896, 896])
        self.assertEqual(ab("reserved_restores"), 2)


class Excluded(Base):
    def setUp(self):
        super().setUp()
        self.prime(self.t)

    def assert_tail(self, req, tree):
        self.lookup(req, tree)
        self.assertEqual(tree.limits, [3000 - TAIL])
        self.assertEqual(dt.STATE["deferred"], 0)

    def test_logprob_request_keeps_the_tail(self):
        req = self.mod.Req(self.t)
        req.return_logprob = True
        self.assert_tail(req, Tree([0, 2944]))

    def test_multi_rank_keeps_the_tail(self):
        sn.STATE["solo"] = False
        self.assert_tail(self.mod.Req(self.t), Tree([0, 2944]))

    def test_hierarchical_storage_keeps_the_tail(self):
        for attr in ("cache_controller", "linker"):
            tree = Tree([0, 2944])
            setattr(tree, attr, object())
            self.assert_tail(self.mod.Req(self.t), tree)

    def test_other_tree_types_keep_the_tail(self):
        class FlexKVRadixCache(Tree):                         # a storage-backed tree, even as a subclass
            pass

        class RadixCache:
            def __init__(self, ends):
                self.inner, self.limits = Tree(ends), []

            def match(self, limit):
                self.limits.append(limit)
                return self.inner.match(limit)

        self.assert_tail(self.mod.Req(self.t), FlexKVRadixCache([0, 2944]))
        self.assert_tail(self.mod.Req(self.t), RadixCache([0, 2944]))

    def test_host_hit_drops(self):
        req = self.mod.Req(self.t)
        req.host_hit_length = 128
        self.lookup(req, Tree([0, 2944]))
        self.assertEqual(dt.STATE["defer_dropped"], 1)


class Exact(Base):
    """The deferred window covers everything the tail recomputes: the floor is the page-aligned floor of n - L, the
    same page the tail's re-prefill starts at or below."""

    def test_floor_matches_the_tail_for_unaligned_lengths(self):
        for n in (2049, 2100, 2177, 2900, 3001, 3999, 4096):
            with self.subTest(n=n):
                self.setUp()
                t = toks(n, seed=n)
                self.prime(t)
                req = self.mod.Req(t)
                P = (n - 1) // PAGE * PAGE
                self.assertEqual(self.lookup(req, Tree([0, P])), P)
                floor = self.run_prefill(req, 1)
                self.assertLessEqual(floor, n - TAIL)         # at or below the tail's bound: nothing missing
                self.assertEqual(floor % PAGE, 0)
                self.assertTrue(self.rig.read_ok(1, floor, n))
                self.tearDown()

    def test_snapshot_slot_holds_one_page_more(self):
        self.assertEqual(self.rig.store.k_snap[0].shape[1], 2048 + PAGE)
        uninstall()
        self.assertEqual(sn.Store(self.rig.ring, self.rig.k, self.rig.v, 2).k_snap[0].shape[1], 2048)


class Uncovered(Base):
    def assert_tail(self, req, tree):
        P = self.lookup(req, tree)
        self.assertEqual(tree.limits, [3000 - TAIL])
        self.assertIsNone(req.__dict__.get(sn.RES_ATTR))
        return P

    def test_no_snapshot(self):
        self.assert_tail(self.mod.Req(self.t), Tree([0, 2944]))

    def test_window_not_covered(self):
        self.prime(self.t, prefix=2304)                       # captured with a ring floor of 2304: lo 2304 > 1024
        self.assert_tail(self.mod.Req(self.t), Tree([0, 2944]))

    def test_snapshot_ends_at_or_below_the_tail_bound(self):
        t = toks(4000)
        self.prime(t[:1952])                                  # ends exactly at the tail's bound 4000 - 2048
        req = self.mod.Req(t)
        tree = Tree([0, 1920, 3968])
        self.lookup(req, tree)
        self.assertEqual(tree.limits, [1952])                 # no reuse beyond the tail's bound to gain: the tail
        self.assertEqual(dt.STATE["deferred"], 0)

    def test_prefix_snapshot_above_the_tail_bound_defers(self):
        t = toks(3000, seed=3)
        self.prime(t[:1100])                                  # a 1,100-token prefix of the prompt, lo 0
        req = self.mod.Req(t)
        tree = Tree([0, 1024, 2944])
        self.assertEqual(self.lookup(req, tree), 1024)
        self.assertEqual(tree.limits, [1100])

    def test_salt_isolates(self):
        self.prime(self.t)
        req = self.mod.Req(self.t, salt="other")
        self.assert_tail(req, Tree([0, 2944]))

    def test_changed_token_misses(self):
        self.prime(self.t)
        t = list(self.t)
        t[100] += 1
        self.assert_tail(self.mod.Req(t), Tree([0, 2944]))

    def test_above_max_input_is_stock(self):
        t = toks(5000)
        self.prime(t)
        req = self.mod.Req(t)
        tree = Tree([0, 4992])
        self.lookup(req, tree)
        self.assertEqual(tree.limits, [4999])
        self.assertEqual(dt.STATE["deferred"], 0)

    def test_lookup_without_tree_keeps_state(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        res = req.__dict__[sn.RES_ATTR]
        req.init_next_round_input()                           # a chunk continuation: no match
        self.assertIs(req.__dict__[sn.RES_ATTR], res)


class Stale(Base):
    def test_changed_during_lookup_keeps_the_hit_and_drops(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        tree = Tree([0, 640, 2944], during=self.rig.store.clear)   # flush while the match runs (not a stock order)
        P = self.lookup(req, tree)
        self.assertEqual(tree.limits, [2999])                 # one match only: no repeated stateful lookup
        self.assertEqual(P, 2944)                             # a valid cache hit of the tree
        self.assertEqual(dt.STATE["defer_dropped"], 1)
        self.assertIsNone(req.__dict__.get(sn.RES_ATTR))
        self.assertEqual(self.run_prefill(req, 1), 2944)      # no restore: rows below the floor are never read
        self.assertTrue(self.rig.read_ok(1, 2944, 3000))

    def test_endpoint_above_reservation_drops(self):
        self.prime(self.t[:2900])
        req = self.mod.Req(self.t)

        class UnifiedRadixCache(Tree):                        # a tree that ignores the limit
            def match(self, limit):
                self.limits.append(limit)
                return 2944

        P = self.lookup(req, UnifiedRadixCache([]))
        self.assertEqual(P, 2944)
        self.assertEqual(dt.STATE["defer_dropped"], 1)
        self.assertEqual(self.run_prefill(req, 1), 2944)      # the snapshot ends at 2900: never restored past it

    def test_check_exception_drops(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        saved = sn.holds
        sn.holds = lambda *a: 1 / 0
        try:
            P = self.lookup(req, Tree([0, 2944]))
        finally:
            sn.holds = saved
        self.assertEqual(P, 2944)
        self.assertEqual(dt.STATE["defer_dropped"], 1)

    def test_claim_exception_falls_back_to_find(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        saved = sn.claim
        sn.claim = lambda *a: 1 / 0
        try:
            floor = self.run_prefill(req, 1)
        finally:
            sn.claim = saved
        self.assertEqual(floor, 1024)                         # the v2.9.3 find() restores the same window
        self.assertEqual(ab("reserve_lost"), 1)
        self.assertEqual(sn.STATE["inert"], "")               # snapshots stay on
        self.assertTrue(self.rig.read_ok(1, 1024, 3000))

    def test_flushed_before_restore(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.assertEqual(self.lookup(req, Tree([0, 2944])), 2944)
        self.rig.store.clear()                                # flush_cache between the lookup and the forward
        self.assertEqual(self.run_prefill(req, 1), 2944)      # no restore: the stock floor, nothing read below it
        self.assertEqual(ab("reserve_lost"), 1)
        self.assertEqual(sn.STATE["stats"]["restores"], 0)

    def test_generation_change_before_restore(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        sn.STATE["generation"] += 1                           # a weight update / inert switch
        self.assertIsNone(sn.claim(self.rig.store, req.__dict__[sn.RES_ATTR], 2944, 3000))

    def test_replaced_slot_with_same_prefix_is_not_the_reservation(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        res = req.__dict__[sn.RES_ATTR]
        self.rig.store.clear()
        self.prime(self.t, slot=3)                            # recaptured into the same snapshot slot
        self.assertEqual(self.rig.store.index.entries[res.slot].digest, res.digest)
        self.assertIsNone(sn.claim(self.rig.store, res, 2944, 3000))   # new version: not the reserved entry


class Eviction(Base):
    def other_capture(self, seed, slot):
        t = toks(3000, seed=seed)
        req = self.mod.Req(t)
        self.rig.alloc(req, slot)
        with quiet():
            self.rig.extend(req, 0, 3000)                     # record_floors -> append -> drain (place)

    def test_pin_survives_a_drain_between_lookup_and_restore(self):
        self.prime(self.t, slot=0)                            # snapshot slot 0
        self.other_capture(seed=5, slot=1)                    # snapshot slot 1 (newer)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))                     # reserves snapshot slot 0 (and touches it)
        res = req.__dict__[sn.RES_ATTR]
        self.rig.store.index.entries[res.slot].used = 0       # make it the LRU victim
        self.other_capture(seed=6, slot=2)                    # a capture drains before the restore
        self.assertEqual(self.rig.store.index.entries[res.slot].version, res.version)   # pinned: not evicted
        self.assertEqual(self.run_prefill(req, 3), 896)
        self.assertTrue(self.rig.read_ok(3, 896, 3000))

    def test_pin_expires_after_a_forward(self):
        self.prime(self.t, slot=0)
        self.other_capture(seed=5, slot=1)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        res = req.__dict__[sn.RES_ATTR]
        self.rig.store.index.entries[res.slot].used = 0
        self.rig.store.index.epoch += 1                       # a forward finished: the pin no longer holds
        self.other_capture(seed=6, slot=2)                    # evicts the reserved snapshot
        self.assertNotEqual(self.rig.store.index.entries[res.slot].version, res.version)
        self.assertEqual(self.run_prefill(req, 3), 2944)      # restore refused; stock floor, no foreign rows
        self.assertEqual(ab("reserve_lost"), 1)
        self.assertTrue(self.rig.read_ok(3, 2944, 3000))

    def test_all_slots_pinned_skips_the_capture(self):
        ix = sn.Index(1, 2048, PAGE)
        t = array("q", toks(3000))
        ix.place(sn.prefix_digest(t, 3000), 3000, 952, sn._tail(t, 3000), None, None)
        ix.pin(0)
        t2 = array("q", toks(3000, seed=9))
        self.assertIsNone(ix.place(sn.prefix_digest(t2, 3000), 3000, 952, sn._tail(t2, 3000), None, None))
        self.assertEqual(ab("pinned_skips"), 1)
        ix.epoch += 1
        self.assertEqual(ix.place(sn.prefix_digest(t2, 3000), 3000, 952, sn._tail(t2, 3000), None, None), 0)

    def test_restored_pin_does_not_block_the_same_forwards_capture(self):
        self.prime(self.t, slot=0)
        self.other_capture(seed=5, slot=1)                    # both snapshot slots full
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        res = req.__dict__[sn.RES_ATTR]
        self.rig.alloc(req, 3)
        req.extend_end = 3000
        with quiet():
            self.rig.record(self.rig.ring, [req], [2944], self.rig.gen)   # the restore, inside this forward
        self.assertFalse(self.rig.store.index.pinned(res.slot))          # released once enqueued
        self.rig.store.pending = []                           # (its own prompt-end capture would refresh the slot)
        self.rig.store.index.entries[res.slot].used = 0       # the LRU victim of this forward's capture
        self.other_capture(seed=6, slot=2)
        self.assertNotEqual(self.rig.store.index.entries[res.slot].version, res.version)
        self.assertEqual(ab("pinned_skips"), 0)

    def test_superseded_reservation_releases_its_pin(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        slot = req.__dict__[sn.RES_ATTR].slot
        self.lookup(req, Tree([0, 2944]))                     # the admission pass matches again
        self.assertEqual(self.rig.store.index.pins[slot][1], 1)

    def test_lapsed_reservation_never_unpins_a_current_one(self):
        self.prime(self.t)
        a, b = self.mod.Req(self.t), self.mod.Req(self.t)
        self.lookup(a, Tree([0, 2944]))                       # a waits through a forward ...
        slot = a.__dict__[sn.RES_ATTR].slot
        self.rig.store.index.epoch += 1
        self.lookup(b, Tree([0, 2944]))                       # ... b pins the same slot in the new epoch
        self.lookup(a, Tree([0, 2944]))                       # a matches again: drops its lapsed pin, pins anew
        self.assertEqual(self.rig.store.index.pins[slot][1], 2)
        sn.drop(a.__dict__[sn.RES_ATTR])
        sn.drop(a.__dict__[sn.RES_ATTR])                      # released once only
        self.assertTrue(self.rig.store.index.pinned(slot))    # b's pin stands

    def test_zero_prefix_releases_the_pin(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        self.lookup(req, Tree([0, 2944]))
        slot = req.__dict__[sn.RES_ATTR].slot
        req.prefix_len = 0                                    # the target cache lost the prefix before admission
        self.run_prefill(req, 1)
        self.assertEqual(self.rig.store.index.pins[slot][1], 0)

    def test_forward_bumps_the_epoch(self):
        worker = types.SimpleNamespace(draft_kv_ring=self.rig.ring, model_runner=types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_generation=self.rig.gen)))
        fwd = bpatch.make_forward(lambda self, batch: "out")
        before = self.rig.store.index.epoch
        self.assertEqual(fwd(worker, None), "out")
        self.assertEqual(self.rig.store.index.epoch, before + 1)


@unittest.skipIf(torch is None, "needs torch (CPU)")
class OffIsV293(Base):
    """DEFER=0: the bound and the floors of the same traffic equal the independent (A) + (B) path."""

    def setUp(self):
        super().setUp()
        self.mod = install(defer=False)

    def test_tail_applies_and_b_fills_below(self):
        self.prime(self.t)
        req = self.mod.Req(self.t)
        tree = Tree([0, 640, 2944])
        P = self.lookup(req, tree)
        self.assertEqual(tree.limits, [952])
        self.assertEqual(P, 640)
        self.assertNotIn(sn.RES_ATTR, req.__dict__)
        self.assertEqual(self.run_prefill(req, 1), 640)      # (B) has nothing to add: 640 - lo < MIN_RESTORE
        self.assertEqual(dt.STATE["deferred"], 0)
        self.assertEqual(ab("reserved"), 0)

    def test_versions_and_epoch_are_the_v293_ones(self):
        ix = sn.Index(1, 2048, PAGE)
        t = array("q", toks(3000))
        for k in range(3):
            tk = array("q", toks(3000, seed=20 + k))
            ix.place(sn.prefix_digest(tk, 3000), 3000, 952, sn._tail(tk, 3000), None, None)
            self.assertEqual(ix.entries[0].version, k)        # per-slot count, as released
        worker = types.SimpleNamespace(draft_kv_ring=self.rig.ring, model_runner=types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_generation=self.rig.gen)))
        bpatch.make_forward(lambda self, batch: None)(worker, None)
        self.assertEqual(self.rig.store.index.epoch, 0)
        self.assertNotIn("reserved", sn.STATE["stats"])       # the released stats line is unchanged


if __name__ == "__main__":
    unittest.main()
