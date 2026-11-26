"""CPU test of ctx8_lcpair.py on synthetic long-context steplog captures."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx8_lcpair  # noqa: E402


def write(d, rows):
    """rows: (rid, step, prompt_hash, prompt_len, context, drafted[7], tokens list); one request row per step."""
    n = len(rows)
    com = np.zeros((n, 8), dtype=np.int32)
    cl = np.zeros(n, dtype=np.int8)
    for i, r in enumerate(rows):
        com[i, : len(r[6])] = r[6]
        cl[i] = len(r[6])
    np.savez_compressed(
        os.path.join(d, "dflash1-x-000000.npz"), format=ctx8_lcpair.FORMAT, bs=np.ones(n, dtype=np.int32),
        step=np.asarray([r[1] for r in rows], dtype=np.int64), walks=np.ones(n, dtype=np.int32),
        rid=np.asarray([r[0] for r in rows]), prompt_hash=np.asarray([r[2] for r in rows], dtype=np.int64),
        prompt_len=np.asarray([r[3] for r in rows], dtype=np.int32),
        context=np.asarray([r[4] for r in rows], dtype=np.int64), drafted=np.asarray([r[5] for r in rows], np.int32),
        committed=com, commit_len=cl, greedy=np.ones(n, dtype=bool))


D = list(range(7))
L32, L128 = 32000, 130000


def toks(a, b):
    return list(range(1000 + a, 1000 + b))


class LcPair(unittest.TestCase):
    def test_pairs_by_prompt_and_send_order(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            # A (off): cold request c1 takes 4 steps of 4 tokens, cached c2 takes 2 steps of 8; long prompt: 2 x 8
            write(a, [("c1", 1, 5, L32, L32 + 0, D, toks(0, 4)), ("c1", 2, 5, L32, L32 + 4, D, toks(4, 8)),
                      ("c1", 3, 5, L32, L32 + 8, D, toks(8, 12)), ("c1", 4, 5, L32, L32 + 12, D, toks(12, 16)),
                      ("c2", 9, 5, L32, L32, D, toks(0, 8)), ("c2", 10, 5, L32, L32 + 8, D, toks(8, 16)),
                      ("l1", 20, 6, L128, L128, D, toks(0, 8)), ("l1", 21, 6, L128, L128 + 8, D, toks(8, 16)),
                      ("s1", 30, 7, 100, 100, D, toks(0, 8))])  # short prompt: ignored
            # B (on): cold takes 2 steps of 8 (better), cached 2 steps of 8; long prompt diverges at token 12
            other = toks(8, 16)
            other[4] = 1
            write(b, [("x1", 1, 5, L32, L32, D, toks(0, 8)), ("x1", 2, 5, L32, L32 + 8, D, toks(8, 16)),
                      ("x2", 5, 5, L32, L32, [9] + D[1:], toks(0, 8)), ("x2", 6, 5, L32, L32 + 8, D, toks(8, 16)),
                      ("y1", 20, 6, L128, L128, D, toks(0, 8)), ("y1", 21, 6, L128, L128 + 8, D, other)])
            A, _ = ctx8_lcpair.load(a, 16000)
            B, _ = ctx8_lcpair.load(b, 16000)
            self.assertEqual(sorted(A), [(5, L32, 0), (5, L32, 1), (6, L128, 0)])
            r = ctx8_lcpair.compare(A, B, 65536)
            lt, ge = r["lt"], r["ge"]
            self.assertEqual((lt["requests"], lt["tokens"]), (2, 32))
            self.assertAlmostEqual(lt["tps_a"], 32 / 6)
            self.assertAlmostEqual(lt["tps_b"], 32 / 4)
            self.assertEqual(lt["requests_b_higher"], 1)
            # matched steps: cold L32+0, L32+8; cached L32+0 (slot 0 differs), L32+8
            self.assertEqual(lt["matched_steps"], 4)
            self.assertAlmostEqual(lt["slot_agreement"][0], 0.75)
            self.assertAlmostEqual(lt["block_agreement"], 0.75)
            self.assertEqual((ge["requests"], ge["diverged"], ge["tokens"]), (1, 1, 12))


if __name__ == "__main__":
    unittest.main()
