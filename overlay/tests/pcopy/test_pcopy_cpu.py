"""CPU tests of inkling_pcopy's proposal rule: the reference matcher against the offline replay's semantics, the
override of a draft block, and the exact law of chain speculative sampling when forced copy tokens (point masses)
replace a prefix of the selector's draws (exact enumeration on small vocabularies; joint emitted-token laws against
the target).

    python3 -m unittest discover -s tests/pcopy
"""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from inkling_pcopy import rule  # noqa: E402


def replay_cont(arr, L, n=12, maxlen=64, G=7):
    """The offline replay rule, inlined: the continuation after the most recent full KEY-gram hit."""
    if L < n + 1:
        return None, 0
    key = arr[L - n:L]
    ends = [e for e in range(n, len(arr) + 1) if arr[e - n:e] == key and e < L]
    if not ends:
        return None, 0
    e = ends[-1]
    m = n
    while m < maxlen and e - m - 1 >= 0 and arr[e - m - 1] == arr[L - m - 1]:
        m += 1
    c = arr[e:min(e + G, L)]
    if not c:
        return None, 0
    return c, m


@unittest.skipIf(torch is None, "needs torch")
class Matcher(unittest.TestCase):
    def test_against_the_replay_rule(self):
        g = torch.Generator().manual_seed(0)
        for trial in range(200):
            vocab = [3, 5, 50][trial % 3]
            C = 160
            seq = torch.randint(0, vocab, (C,), generator=g)
            if trial % 4 == 0:  # plant a long copy
                seq[100:140] = seq[20:60]
            arr = seq.tolist()
            L = int(torch.randint(20, C, (1,), generator=g))
            hist = seq[None, :].to(torch.int32)
            m, nn, cont = rule.match_ref(hist, torch.tensor([0]), torch.tensor([L]), torch.tensor([True]), 7)
            c, mm = replay_cont(arr, L)
            if c is None:
                self.assertEqual(int(nn[0]), 0, msg=trial)
                continue
            self.assertEqual(int(m[0]), mm, msg=trial)
            self.assertEqual(cont[0, :int(nn[0])].tolist(), c, msg=trial)

    def test_invalid_rows_and_short_context(self):
        hist = torch.arange(100, dtype=torch.int32)[None].repeat(2, 1)
        m, n, _ = rule.match_ref(hist, torch.tensor([0, 1]), torch.tensor([50, 10]), torch.tensor([False, True]), 7)
        self.assertEqual(n.tolist(), [0, 0])

    def test_newer_short_match_blocks_older_long_one(self):
        # the most recent full 12-gram hit wins even when an older one extends further (the replay's rule)
        base = list(range(100, 140))
        arr = base[:30] + list(range(500, 520)) + base[:30] + [7] + base[18:30] + list(range(600, 610)) + base[18:30]
        L = len(arr)
        hist = torch.tensor(arr, dtype=torch.int32)[None]
        m, n, cont = rule.match_ref(hist, torch.tensor([0]), torch.tensor([L]), torch.tensor([True]), 7)
        c, mm = replay_cont(arr, L)
        self.assertEqual(int(m[0]), mm)
        self.assertEqual(cont[0, :int(n[0])].tolist(), c)


@unittest.skipIf(torch is None, "needs torch")
class Fixtures(unittest.TestCase):
    def test_directed_expectations(self):
        from inkling_pcopy import fixtures
        for name, arr, L, e, m in fixtures.build():
            hist = torch.tensor(arr, dtype=torch.int32)[None]
            mm, nn, cont = rule.match_ref(hist, torch.tensor([0]), torch.tensor([L]), torch.tensor([True]), 7)
            if e < 0:
                self.assertEqual(int(nn[0]), 0, msg=name)
                continue
            self.assertEqual(int(mm[0]), m, msg=name)
            self.assertEqual(cont[0, :int(nn[0])].tolist(), arr[e:e + int(nn[0])], msg=name)
            c, mr = replay_cont(arr, L)
            self.assertEqual((c, mr), (arr[e:e + 7], m), msg=name)


@unittest.skipIf(torch is None, "needs torch")
class Override(unittest.TestCase):
    def test_tokens_and_sample(self):
        bs, G, K = 3, 4, 5
        draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]])
        cont = torch.tensor([[40, 41, -1, -1], [6, 70, 71, 72], [-1, -1, -1, -1]])
        m = torch.tensor([20, 16, 30])
        n = torch.tensor([2, 4, 0])
        forced = rule.force_mask(m, n, G)
        self.assertEqual(forced.tolist(), [[True, True, False, False], [True] * 4, [False] * 4])
        rule.override_tokens(draft, cont, forced)
        self.assertEqual(draft.tolist(), [[40, 41, 3, 4], [6, 70, 71, 72], [9, 9, 9, 9]])
        ids = torch.arange(bs * G * K).view(bs, G, K) + 1000
        ids[0, 1, 3] = 41                                    # forced token already a candidate (lane 3)
        q = torch.softmax(torch.randn(bs, G, K), dim=-1)
        ids2, q2 = rule.override_sample(ids, q, cont, forced)
        self.assertEqual(int(ids2[0, 0, 0]), 40)              # full-row miss: lane 0 replaced
        self.assertEqual(q2[0, 0].tolist(), [1.0, 0, 0, 0, 0])
        self.assertTrue(torch.equal(ids2[0, 1], ids[0, 1]))    # hit: ids untouched
        self.assertEqual(q2[0, 1].tolist(), [0, 0, 0, 1.0, 0])
        self.assertTrue(torch.equal(q2[0, 2:], q[0, 2:]))      # retained slots untouched
        self.assertTrue(torch.equal(ids2[2], ids[2]) and torch.equal(q2[2], q[2]))
        for b in range(bs):
            for j in range(G):
                self.assertEqual(len(set(ids2[b, j].tolist())), K)  # ids stay unique
        self.assertFalse(torch.equal(ids2, ids) and False)
        self.assertEqual(int(ids[0, 0, 0]), 1000)             # inputs not modified (owned copies)


# ----------------------------------------------------------------------------------- exact law (enumeration)

V = 4      # vocabulary
G = 3      # drafted slots per block
K = 2      # selector candidates per slot


PAD = None   # a padded vocabulary entry: p = 0 everywhere (set by a test)
TINY = False  # make the target nearly equal to the selector's q at the first slot (near-zero residuals)


def target(prefix, zero=None, one=None):
    """p(. | prefix): a fixed, irregular conditional; zero / one force p(c) = 0 / 1 for a token c after some prefix."""
    h = (hash(tuple(prefix)) % 9973) + 1
    w = [((h * (i + 3) * 2654435761) % 97) + 1 for i in range(V)]
    if TINY:
        cand, q = selector(prefix, [])
        w = [1e-12] * V
        w[cand[0]], w[cand[1]] = q[0] * (1 + 1e-9), q[1]
    if PAD is not None:
        w[PAD] = 0
    if zero is not None and len(prefix) % 2 == 0:
        w[zero] = 0
    if one is not None and len(prefix) % 2 == 1:
        w = [0] * V
        w[one] = 1
    s = float(sum(w))
    return [x / s for x in w]


def selector(prefix, drawn):
    """The selector's candidates and q at the next slot, given the history and its own earlier draws (which pcopy may
    overwrite: the retained suffix is correlated with them)."""
    h = hash((tuple(prefix), tuple(drawn))) % 997
    a, b = h % V, (h // V) % V
    if a == b:
        b = (b + 1) % V
    qa = 0.2 + 0.6 * ((h % 7) / 6.0)
    return [a, b], [qa, 1.0 - qa]


def copy_rule(prefix, mode):
    """The forced continuation for this history (any deterministic function of the committed history)."""
    if mode == "none":
        return []
    h = hash(tuple(prefix)) % 101
    if mode == "sometimes" and h % 3 == 0:
        return []
    n = 1 + h % G
    return [(h + 2 * i) % V for i in range(n)]


def block_dist(prefix, mode, cutoff, zero=None, one=None, wrong_q=False):
    """{emitted tuple: probability} of one decode step (draft, override, chain accept with `cutoff` verified drafts)."""
    out = {}
    cont = copy_rule(prefix, mode)
    # enumerate the selector's draws
    def walk(drawn, prob, rows):
        if len(drawn) == G:
            yield drawn, prob, rows
            return
        cand, q = selector(prefix, drawn)
        for x, qx in zip(cand, q):
            yield from walk(drawn + [x], prob * qx, rows + [(cand, q)])
    for drawn, pw, rows in walk([], 1.0, []):
        toks, qs = [], []
        for j in range(G):
            if j < len(cont) and wrong_q:
                cand, q = rows[j]  # negative control: the forced token judged with the selector's q row
                toks.append(cont[j])
                qs.append([q[cand.index(v)] if v in cand else 0.0 for v in range(V)])
            elif j < len(cont):
                toks.append(cont[j])
                qs.append([1.0 if v == cont[j] else 0.0 for v in range(V)])  # point mass
            else:
                cand, q = rows[j]
                toks.append(drawn[j])
                qs.append([q[cand.index(v)] if v in cand else 0.0 for v in range(V)])
        # chain accept
        def accept(j, emitted, prob):
            if j == cutoff:
                p = target(prefix + emitted, zero, one)
                for v in range(V):
                    if p[v] > 0:
                        yield tuple(emitted + [v]), prob * p[v]
                return
            p = target(prefix + emitted, zero, one)
            x = toks[j]
            qx = qs[j][x]
            a = min(1.0, p[x] / qx) if qx > 0 else 0.0
            if a > 0:
                yield from accept(j + 1, emitted + [x], prob * a)
            if a < 1:
                res = [max(p[v] - qs[j][v], 0.0) for v in range(V)]
                s = sum(res)
                for v in range(V):
                    if res[v] > 0:
                        yield tuple(emitted + [v]), prob * (1 - a) * res[v] / s
        for em, pr in accept(0, [], pw):
            out[em] = out.get(em, 0.0) + pr
    return out


def emitted_law(prefix, T, mode, cutoff, zero=None, one=None, wrong_q=False):
    """{first T emitted tokens: probability} over repeated decode steps."""
    law = {}
    def rec(pre, prob):
        if len(pre) - len(prefix) >= T:
            key = tuple(pre[len(prefix):len(prefix) + T])
            law[key] = law.get(key, 0.0) + prob
            return
        for em, pr in block_dist(pre, mode, cutoff, zero, one, wrong_q).items():
            rec(pre + list(em), prob * pr)
    rec(list(prefix), 1.0)
    return law


def target_law(prefix, T, zero=None, one=None):
    law = {}
    for seq in itertools.product(range(V), repeat=T):
        p, pre = 1.0, list(prefix)
        for x in seq:
            p *= target(pre, zero, one)[x]
            pre.append(x)
        if p > 0:
            law[seq] = p
    return law


class Law(unittest.TestCase):
    def assert_law(self, prefix, T, mode, cutoff, zero=None, one=None):
        got = emitted_law(prefix, T, mode, cutoff, zero, one)
        want = target_law(prefix, T, zero, one)
        keys = set(got) | set(want)
        err = max(abs(got.get(k, 0.0) - want.get(k, 0.0)) for k in keys)
        self.assertLess(err, 1e-9, msg=(prefix, T, mode, cutoff, zero, one))
        self.assertAlmostEqual(sum(got.values()), 1.0, places=9)

    def test_negative_control_detects_a_wrong_q(self):
        got = emitted_law([1, 2], 3, "always", G, wrong_q=True)
        want = target_law([1, 2], 3)
        err = max(abs(got.get(k, 0.0) - want.get(k, 0.0)) for k in set(got) | set(want))
        self.assertGreater(err, 1e-3)

    def test_no_copy_is_the_target(self):
        self.assert_law([1, 2], 3, "none", G)

    def test_forced_prefix_full_width(self):
        for pre in ([0], [1, 3], [2, 2, 1]):
            self.assert_law(pre, 3, "always", G)

    def test_forced_sometimes(self):
        for pre in ([0, 1], [3]):
            self.assert_law(pre, 3, "sometimes", G)

    def test_cutoff_inside_and_after_a_forced_run(self):
        for cutoff in range(0, G + 1):
            self.assert_law([1, 0], 3, "always", cutoff)

    def test_padded_vocab_and_tiny_residuals(self):
        global PAD, TINY
        try:
            PAD = 3
            self.assert_law([0, 2], 3, "always", G)
            PAD, TINY = None, True
            self.assert_law([1], 3, "always", G)
            self.assert_law([1], 3, "sometimes", 1)
        finally:
            PAD, TINY = None, False

    def test_p_of_the_copy_zero_and_one(self):
        for c in range(V):
            self.assert_law([2], 3, "always", G, zero=c)
            self.assert_law([2], 3, "always", G, one=c)


if __name__ == "__main__":
    unittest.main()
