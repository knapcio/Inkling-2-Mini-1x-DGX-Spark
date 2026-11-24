#!/usr/bin/env python3
"""benchok.py FILE: exit 0 only when a speedbench result has no failed request.
  speedbench-result/1 : >= 1 request and every request ok
  speedbench-longctx/1: every context row n_ok >= 1 and n_failed == 0; every decode context has a decode rate
Prints one line with the reason."""
import json
import sys


def main(path):
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {path}: unreadable ({exc!r})")
        return 1
    sch = doc.get("schema")
    if sch == "speedbench-result/1":
        reqs = doc.get("requests") or []
        bad = [r for r in reqs if not r.get("ok")]
        if not reqs or bad:
            print(f"FAIL {path}: {len(bad)} failed of {len(reqs)} requests")
            return 1
        print(f"OK {path}: {len(reqs)} requests")
        return 0
    if sch == "speedbench-longctx/1":
        rows = doc.get("table") or []
        dec = set((doc.get("settings") or {}).get("decode_contexts") or [])
        bad = [r.get("context") for r in rows if not r.get("n_ok") or r.get("n_failed")]
        nodec = [r.get("context") for r in rows if r.get("context") in dec and r.get("decode_tok_s") is None]
        if not rows or bad or nodec:
            print(f"FAIL {path}: contexts with failures {bad}, decode contexts without a rate {nodec}")
            return 1
        print(f"OK {path}: contexts {[r.get('context') for r in rows]}")
        return 0
    print(f"FAIL {path}: unknown schema {sch!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
