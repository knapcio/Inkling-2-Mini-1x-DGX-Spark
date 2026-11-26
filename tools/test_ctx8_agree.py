"""CPU test of ctx8_agree.py on synthetic steplog captures."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx8_agree  # noqa: E402

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctx8_agree.py")


def write(d, name, rows, walks=None):
    """rows: (rid, prompt_hash, context, drafted[7], accepted, committed[8], commit_len, greedy); one row per step."""
    n = len(rows)
    np.savez_compressed(
        os.path.join(d, name), format=ctx8_agree.FORMAT, bs=np.ones(n, dtype=np.int32),
        walks=np.asarray(walks if walks is not None else [1] * n, dtype=np.int32),
        rid=np.asarray([r[0] for r in rows]),
        prompt_hash=np.asarray([r[1] for r in rows], dtype=np.int64), prompt_len=np.full(n, 100, dtype=np.int32),
        context=np.asarray([r[2] for r in rows], dtype=np.int64),
        drafted=np.asarray([r[3] for r in rows], dtype=np.int32), accepted=np.asarray([r[4] for r in rows], np.int8),
        committed=np.asarray([r[5] for r in rows], dtype=np.int32),
        commit_len=np.asarray([r[6] for r in rows], dtype=np.int8), greedy=np.asarray([r[7] for r in rows]))


def c(start, n):
    """committed row: tokens start..start+n-1 (the target stream is token i at context 100 + i), padded to 8."""
    return [100 + start + j for j in range(n)] + [0] * (8 - n)


D7 = list(range(10, 17))


class Agree(unittest.TestCase):
    def test_compare(self):
        with tempfile.TemporaryDirectory() as on, tempfile.TemporaryDirectory() as off:
            # prompt 1: one request, steps at 100 / 108 / 111; prompt 7: sent twice (ambiguous step keys)
            write(on, "dflash1-a-000000.npz", [
                ("a", 1, 100, D7, 7, c(0, 8), 8, True),
                ("a", 1, 108, D7, 2, c(8, 3), 3, True),
                ("a", 1, 111, D7, 1, c(11, 2), 2, True),
                ("b", 7, 100, D7, 0, c(0, 1), 1, True),
                ("b2", 7, 100, [0] * 7, 0, c(0, 1), 1, True),  # same key twice: ambiguous
                ("s", 2, 100, D7, 1, c(0, 2), 2, False),  # sampled: ignored
                ("w", 3, 100, D7, 1, c(0, 2), 2, True),  # walks != 1: not a matched step
            ], walks=[1, 1, 1, 1, 1, 1, 2])
            write(off, "dflash1-b-000000.npz", [
                ("x", 1, 100, D7, 7, c(0, 8), 8, True),
                ("x", 1, 108, D7[:3] + [99, 98, 97, 96], 4, c(8, 5), 5, True),
                ("y", 7, 100, D7, 0, c(0, 1), 1, True),
                ("z", 5, 100, D7, 1, c(0, 2), 2, True),  # unmatched prompt
            ])
            a, sa = ctx8_agree.load(on)
            b, _ = ctx8_agree.load(off)
            self.assertEqual((sa["ambiguous_keys"], sa["ambiguous_steps"]), (1, 2))
            self.assertAlmostEqual(sa["tokens_per_step"], 3.0)  # every complete (walks == 1) greedy step counts
            r = ctx8_agree.compare(a, b)
            self.assertEqual(r["prompts_token_differ"], 0)
            # prompt 1: on 100..112 (13 tokens), off 100..112 (13 tokens): equal streams
            self.assertEqual(r["prompts_token_equal"], 2)
            self.assertEqual(r["matched_steps"], 2)  # (1,100) and (1,108); prompt 7 is ambiguous in the on arm
            self.assertEqual(r["slot_agreement"][:3], [1.0, 1.0, 1.0])
            self.assertEqual(r["slot_agreement"][3], 0.5)
            self.assertEqual(r["block_agreement"], 0.5)
            self.assertAlmostEqual(r["accepted_on_minus_off"], -1.0)
            self.assertEqual(r["target_consistent"], 1.0)

    def test_token_streams_detect_a_different_prefix(self):
        with tempfile.TemporaryDirectory() as on, tempfile.TemporaryDirectory() as off:
            write(on, "dflash1-a-000000.npz", [("a", 1, 100, D7, 2, c(0, 3), 3, True),
                                               ("a", 1, 103, D7, 2, c(3, 3), 3, True)])
            other = c(3, 3)
            other[0] = 555  # same step keys, a different earlier token
            write(off, "dflash1-b-000000.npz", [("x", 1, 100, D7, 2, c(0, 3), 3, True),
                                                ("x", 1, 103, D7, 2, other, 3, True)])
            r = ctx8_agree.compare(ctx8_agree.load(on)[0], ctx8_agree.load(off)[0])
            self.assertEqual(r["prompts_token_differ"], 1)
            self.assertNotEqual(self._cli(on, off), 0)

    def test_longer_stream_with_the_same_prefix_agrees(self):
        with tempfile.TemporaryDirectory() as on, tempfile.TemporaryDirectory() as off:
            write(on, "dflash1-a-000000.npz", [("a", 1, 100, D7, 2, c(0, 3), 3, True),
                                               ("a", 1, 103, D7, 4, c(3, 5), 5, True)])  # committed past max_tokens
            write(off, "dflash1-b-000000.npz", [("x", 1, 100, D7, 2, c(0, 3), 3, True),
                                                ("x", 1, 103, D7, 2, c(3, 3), 3, True)])
            r = ctx8_agree.compare(ctx8_agree.load(on)[0], ctx8_agree.load(off)[0])
            self.assertEqual((r["prompts_token_equal"], r["prompts_token_differ"], r["matched_steps"]), (1, 0, 2))

    def test_incomplete_request_is_unverifiable(self):
        with tempfile.TemporaryDirectory() as on, tempfile.TemporaryDirectory() as off:
            write(on, "dflash1-a-000000.npz", [("a", 1, 100, D7, 2, c(0, 3), 3, True),
                                               ("a", 1, 110, D7, 2, c(10, 3), 3, True)])  # step 103 missing
            write(off, "dflash1-b-000000.npz", [("x", 1, 100, D7, 2, c(0, 3), 3, True),
                                                ("x", 1, 103, D7, 2, c(3, 3), 3, True)])
            a, sa = ctx8_agree.load(on)
            self.assertEqual(sa["incomplete_requests"], 1)
            r = ctx8_agree.compare(a, ctx8_agree.load(off)[0])
            self.assertEqual((r["matched_steps"], r["prompts_unverifiable"]), (0, 1))  # nothing verifiable to match
            self.assertNotEqual(Agree._cli(on, off), 0)

    @staticmethod
    def _cli(on, off, extra=()):
        return subprocess.run([sys.executable, TOOL, on, off, *extra], capture_output=True, text=True).returncode


class WarmHashes(unittest.TestCase):
    def _run(self, warm_on, warm_off):
        with tempfile.TemporaryDirectory() as on, tempfile.TemporaryDirectory() as off, \
                tempfile.TemporaryDirectory() as w:
            write(on, "dflash1-a-000000.npz", [("a", 1, 100, D7, 2, c(0, 3), 3, True)])
            write(off, "dflash1-b-000000.npz", [("x", 1, 100, D7, 2, c(0, 3), 3, True)])
            extra = []
            for name, reqs in (("on", warm_on), ("off", warm_off)):
                if reqs is not None:
                    path = os.path.join(w, name + ".json")
                    with open(path, "w") as fh:
                        json.dump({"requests": [{"key": k, "sha256": h} for k, h in reqs]}, fh)
                    extra += ["--warm-" + name, path]
            return Agree._cli(on, off, extra)

    def test_hashes(self):
        self.assertEqual(self._run([("a", "1"), ("b", "2")], [("a", "1"), ("b", "2")]), 0)
        self.assertEqual(self._run([("a", "1"), ("b", "2")], [("a", "1")]), 1)  # a request missing in one arm
        self.assertEqual(self._run([("a", "1")], [("a", "9")]), 1)  # outputs differ
        self.assertNotEqual(self._run([("a", "1"), ("a", "1")], [("a", "1")]), 0)  # duplicate request
        self.assertNotEqual(self._run([("a", "1")], None), 0)  # only one file


if __name__ == "__main__":
    unittest.main()
