"""Source pinning for inkling_dftail: the stock prefix-match bound it wraps and the lookup that calls it are hashed
(sha256 of their exact source lines, decorators included) and compared with the text of the pinned image
(lmsysorg/sglang@sha256:efac853b...). Any difference refuses the overlay at start-up.

Pure Python (ast + hashlib): runs before sglang is imported, and on a machine without it (the CPU tests)."""

import ast
import hashlib
import importlib.util
import os

SCHEDULE_BATCH = "srt/managers/schedule_batch.py"

# qualified name -> sha256 of the stock source segment in the pinned image.
EXPECTED = {
    # wrapped: the bound it returns is the one the tail tightens
    "Req._compute_max_prefix_len": "28ac28ced09b6cb7bbf0f69febbcba4103cd96b90c362a9ae354093a58686a74",
    # the radix lookup: passes the bound as max_prefix_len (and the streaming-session logprob rule before it)
    "Req.init_next_round_input": "fb0049c4f3441b785757230a32cd3dcca7eaef1b67bee49fbbc7109d223649af",
}


def _find(tree, qualname):
    node = tree
    for part in qualname.split("."):
        hits = [
            n for n in getattr(node, "body", [])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part
        ]
        if len(hits) != 1:
            return None
        node = hits[0]
    return node


def segment(text, qualname):
    """Exact source lines of a top-level function / class method, decorators included, or None when the name is
    missing or defined more than once."""
    node = _find(ast.parse(text), qualname)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines(keepends=True)
    return "".join(lines[start - 1: node.end_lineno])


def digest(text, qualname):
    seg = segment(text, qualname)
    return None if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()


def mismatches(text):
    """[(qualname, why)] for every pinned segment that differs in ``text``."""
    bad = []
    for name, want in EXPECTED.items():
        got = digest(text, name)
        if got is None:
            bad.append((name, "missing or defined twice"))
        elif got != want:
            bad.append((name, f"sha256 {got[:12]} != pinned {want[:12]}"))
    return bad


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # a top-level find_spec imports nothing
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_dftail: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check(root=None, off_hint=""):
    """Raise RuntimeError unless the installed schedule_batch.py matches the pinned text."""
    path = os.path.join(root or sglang_root(), SCHEDULE_BATCH)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise RuntimeError(f"inkling_dftail: patch target {path} is missing{off_hint}") from exc
    bad = mismatches(text)
    if bad:
        raise RuntimeError(f"inkling_dftail: {path} differs from the pinned image: {bad}{off_hint}")
