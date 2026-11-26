#!/usr/bin/env python3
"""ctx8_agree.py ON_DIR OFF_DIR [--json out.json]: the drafter's token agreement between two DFLASH steplog captures
("inkling-dflash-steplog/1", STEPLOG=<label>) of the same greedy prompts, one with a draft-side change on (e.g.
DFLASH_CTX8=1) and one with it off.

Greedy output is the target's argmax chain in both arms, so a request step that starts at the same context length of the
same prompt (key: prompt_hash, prompt_len, context) has the same target prefix in both arms; only the drafter's state
differs (here: its context K/V written through MXFP8 instead of BF16 projections, for every append so far). On those
matched steps it reports:
- per-slot agreement of the drafted tokens, and whole-block agreement;
- accepted drafts per step on vs off (and on the drafts both arms proposed identically);
- target consistency: the committed tokens of the two arms agree on their common length (must be 100 %; anything else
  means the keys did not match the same target prefix);
plus each arm's tokens per step over all its greedy steps and the match coverage. Only complete steps (walks == 1, i.e.
the drafted tokens are this step's) of greedy requests with valid ids (no -1 in the drafted tokens or the committed
ones) are used. A key seen more than once in one arm (a prompt sent twice, or two prompts with equal hashes) is
ambiguous and dropped from that arm.

The committed-token check above only covers each step's own commit. The whole target prefix is checked on token ids:
each greedy request's committed tokens are chained over its steps (by context; a request with a missing step or a -1
id is incomplete); per prompt (prompt_hash, prompt_len), every pair of complete requests (one per arm, same start
context) must agree token for token on their common length (lengths differ when a prompt is sent with different
max_tokens, or when the last step commits past max_tokens). Steps of prompts that differ make the run fail; steps of prompts without
a complete request in both arms are left out of the matching (counted). Separately, --warm-on / --warm-off (the arms'
speedbench output json) compares every request's output text sha256 and fails unless the request sets and all hashes
are equal."""
import argparse
import glob
import json
import os
import sys

import numpy as np

FORMAT = "inkling-dflash-steplog/1"


def load(d):
    files = sorted(glob.glob(os.path.join(d, "**", "dflash1-*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"no dflash1-*.npz under {d}")
    rows = {}
    seen = {}
    reqs = {}  # rid -> [prompt key, [(context, tokens or None)]]
    n_greedy = 0
    invalid = 0
    tok = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if str(z["format"]) != FORMAT:
            raise SystemExit(f"{f}: format {z['format']} != {FORMAT}")
        for k in ("drafted", "committed", "commit_len", "prompt_hash", "prompt_len"):
            if k not in z.files:
                raise SystemExit(f"{f}: no {k!r} (capture predates the lattice fields)")
        walks = np.repeat(z["walks"], z["bs"])
        for i in np.flatnonzero(z["greedy"] & (z["commit_len"] > 0)):  # every greedy step, for the token streams
            cl = int(z["commit_len"][i])
            t = z["committed"][i][:cl]
            r = reqs.setdefault(str(z["rid"][i]), [(int(z["prompt_hash"][i]), int(z["prompt_len"][i])), []])
            r[1].append((int(z["context"][i]), None if (t < 0).any() else t.astype(np.int32).copy()))
        ok = z["greedy"] & (walks == 1) & (z["commit_len"] > 0)
        tok.append(z["commit_len"][ok].astype(np.int64))
        for i in np.flatnonzero(ok):
            cl = int(z["commit_len"][i])
            if (z["drafted"][i] < 0).any() or (z["committed"][i][:cl] < 0).any():
                invalid += 1
                continue
            n_greedy += 1
            key = (int(z["prompt_hash"][i]), int(z["prompt_len"][i]), int(z["context"][i]))
            seen[key] = seen.get(key, 0) + 1
            rows[key] = (z["drafted"][i].copy(), int(z["accepted"][i]), z["committed"][i].copy(), cl)
    ambiguous = [k for k, n in seen.items() if n > 1]
    for k in ambiguous:
        del rows[k]
    tok = np.concatenate(tok) if tok else np.zeros(0, dtype=np.int64)
    streams = {}  # prompt key -> set of (start context, int32 token bytes) of complete requests
    incomplete = 0
    for prompt, steps in reqs.values():
        steps.sort(key=lambda x: x[0])
        complete = all(t is not None for _, t in steps) and all(
            steps[j + 1][0] == steps[j][0] + len(steps[j][1]) for j in range(len(steps) - 1))
        streams.setdefault(prompt, set())
        if not complete:
            incomplete += 1
            continue
        streams[prompt].add((steps[0][0], np.concatenate([t for _, t in steps]).tobytes()))
    rows = {"steps": rows, "streams": streams}
    return rows, {"files": len(files), "requests": len(reqs), "incomplete_requests": incomplete, "greedy_steps": n_greedy, "invalid_ids": invalid,
                  "ambiguous_keys": len(ambiguous), "ambiguous_steps": int(sum(seen[k] for k in ambiguous)),
                  "tokens_per_step": float(tok.mean()) if len(tok) else float("nan")}


def compare(on_all, off_all):
    s_on, s_off = on_all["streams"], off_all["streams"]
    common = set(s_on) & set(s_off)
    def agree(p):
        for st_a, ta in s_on[p]:
            for st_b, tb in s_off[p]:
                n = min(len(ta), len(tb))
                if st_a == st_b and ta[:n] != tb[:n]:
                    return False
        return True

    both = [p for p in common if s_on[p] and s_off[p]]
    differ = [p for p in both if not agree(p)]
    good = {p for p in both if agree(p)}
    on = {k: v for k, v in on_all["steps"].items() if k[:2] in good}
    off = {k: v for k, v in off_all["steps"].items() if k[:2] in good}
    excluded = sum(1 for k in on_all["steps"] if k[:2] not in good)
    keys = sorted(set(on) & set(off))
    head = {"prompts_token_equal": len(good), "prompts_token_differ": len(differ),
            "prompts_unverifiable": len(common) - len(good) - len(differ), "on_steps_excluded": excluded}
    if not keys:
        return dict(head, matched_steps=0, coverage_on=0.0, coverage_off=0.0, slot_agreement=[],
                    block_agreement=float("nan"), first_slot_agreement=float("nan"), accepted_on=float("nan"),
                    accepted_off=float("nan"), accepted_on_minus_off=float("nan"),
                    accepted_when_block_equal=float("nan"), target_consistent=float("nan"))
    d_on = np.stack([on[k][0] for k in keys])
    d_off = np.stack([off[k][0] for k in keys])
    g = d_on.shape[1]
    valid = (d_on >= 0) & (d_off >= 0)
    slot = [float((d_on[:, j] == d_off[:, j])[valid[:, j]].mean()) if valid[:, j].any() else float("nan")
            for j in range(g)]
    block = (d_on == d_off).all(axis=1)
    acc_on = np.array([on[k][1] for k in keys], dtype=np.float64)
    acc_off = np.array([off[k][1] for k in keys], dtype=np.float64)
    consistent = 0
    for k in keys:
        c_on, l_on = on[k][2], on[k][3]
        c_off, l_off = off[k][2], off[k][3]
        n = min(l_on, l_off)
        consistent += bool(np.array_equal(c_on[:n], c_off[:n]))
    return dict(head, **{
        "matched_steps": len(keys),
        "coverage_on": len(keys) / max(1, len(on_all["steps"])),
        "coverage_off": len(keys) / max(1, len(off_all["steps"])),
        "slot_agreement": slot,
        "block_agreement": float(block.mean()),
        "first_slot_agreement": slot[0],
        "accepted_on": float(acc_on.mean()), "accepted_off": float(acc_off.mean()),
        "accepted_on_minus_off": float((acc_on - acc_off).mean()),
        "accepted_when_block_equal": float(acc_on[block].mean()) if block.any() else float("nan"),
        "target_consistent": consistent / len(keys),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("on")
    ap.add_argument("off")
    ap.add_argument("--json", default="")
    ap.add_argument("--warm-on", default="", help="the on arm's speedbench output json (per-request sha256)")
    ap.add_argument("--warm-off", default="")
    a = ap.parse_args()
    on, s_on = load(a.on)
    off, s_off = load(a.off)
    r = compare(on, off)
    r.update(on_arm=s_on, off_arm=s_off)
    for name, st in (("on ", s_on), ("off", s_off)):
        print(f"{name}: {st['files']} files, {st['requests']} greedy requests ({st['incomplete_requests']} incomplete), "
              f"{st['greedy_steps']} valid greedy steps ({st['invalid_ids']} with -1 ids "
              f"skipped, {st['ambiguous_keys']} ambiguous keys = {st['ambiguous_steps']} steps dropped), tokens/step "
              f"{st['tokens_per_step']:.3f}")
    print(f"matched steps {r['matched_steps']} (coverage on {100 * r['coverage_on']:.1f} % / off "
          f"{100 * r['coverage_off']:.1f} %): committed tokens consistent {100 * r['target_consistent']:.2f} %")
    print(f"prompts: token streams equal {r['prompts_token_equal']}, DIFFER {r['prompts_token_differ']}, "
          f"unverifiable {r['prompts_unverifiable']} (no complete request in an arm; {r['on_steps_excluded']} on-arm "
          f"steps left out)")
    bad = r["matched_steps"] == 0 or not r["target_consistent"] >= 1.0 or r["prompts_token_differ"] > 0
    if bool(a.warm_on) != bool(a.warm_off):
        raise SystemExit("--warm-on and --warm-off go together")
    if a.warm_on:
        sha = []
        for path in (a.warm_on, a.warm_off):
            with open(path) as fh:
                q = json.load(fh)
            d = {}
            for x in q.get("requests", []):
                if x.get("kind", "measure") != "measure":
                    continue
                k = f"{x['key']}/{x.get('repeat', 0)}"
                if k in d:
                    raise SystemExit(f"{path}: request {k} appears twice")
                d[k] = x.get("sha256")
            sha.append(d)
        same_set = set(sha[0]) == set(sha[1])
        common = sorted(set(sha[0]) & set(sha[1]))
        equal = sum(sha[0][k] == sha[1][k] and sha[0][k] is not None for k in common)
        r["greedy_outputs_equal"] = [equal, len(common)]
        r["request_sets_equal"] = same_set
        print(f"greedy outputs equal {equal}/{len(common)}; request sets "
              f"{'equal' if same_set else 'DIFFER (%d vs %d)' % (len(sha[0]), len(sha[1]))}")
        bad |= equal != len(common) or not common or not same_set
    else:
        print("note: without --warm-on/--warm-off the target prefix is only checked on each step's own commit")
    print("drafted-token agreement per slot: " + " ".join(f"{100 * x:.1f}" for x in r["slot_agreement"])
          + f" %; whole block {100 * r['block_agreement']:.1f} %")
    print(f"accepted drafts per matched step: on {r['accepted_on']:.3f}, off {r['accepted_off']:.3f} "
          f"(on - off {r['accepted_on_minus_off']:+.3f})")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1)
    if bad:
        print("WARNING: the arms' target outputs differ (matched keys may not share the target prefix)")
        sys.exit(1)


if __name__ == "__main__":
    main()
