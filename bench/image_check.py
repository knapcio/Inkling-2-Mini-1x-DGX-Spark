#!/usr/bin/env python3
"""Image input checks for an Inkling-2-Mini endpoint started with MM=1 (needs Pillow and a TrueType font, e.g. DejaVu).

  python3 bench/image_check.py run --out A.jsonl          eight greedy image questions with known answers (one and two
                                                           images per request, 1 to ~5,000 vision tokens per image)
  python3 bench/image_check.py compare A.jsonl B.jsonl     greedy parity of two runs (identical text per question)
  python3 bench/image_check.py load --conc 8 --context 60000 --images 2 --rounds 2
                                                           concurrent requests, each a unique long text prefix plus
                                                           two 8,160-token images (the default 16,384-token request
                                                           bound; memory worst case, run it next to kvstress)
  python3 bench/image_check.py oversize                    a request above MM_MAX_IMAGE_TOKENS is refused (HTTP 4xx
                                                           with the size message), the server keeps answering

The images are drawn here (deterministic), so two runs send identical bytes. Temperature 0, reasoning effort low.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def canvas(width, height, color="white"):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), color)
    return image, ImageDraw.Draw(image)


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def img_number():
    image, draw = canvas(640, 320)
    draw.text((60, 90), "4729", fill="black", font=font(150))
    return png(image)


def img_shapes():
    image, draw = canvas(900, 400)
    draw.ellipse((40, 100, 240, 300), fill=(220, 30, 30))
    draw.rectangle((350, 100, 550, 300), fill=(30, 60, 220))
    draw.polygon([(760, 90), (660, 310), (860, 310)], fill=(30, 170, 60))
    return png(image)


def img_chart():
    image, draw = canvas(1000, 700)
    bars = (("Mon", 3), ("Tue", 7), ("Wed", 2), ("Thu", 9), ("Fri", 5))
    for i, (day, value) in enumerate(bars):
        x = 80 + i * 180
        draw.rectangle((x, 620 - value * 60, x + 110, 620), fill=(70, 110, 200))
        draw.text((x + 10, 630), day, fill="black", font=font(40))
        draw.text((x + 35, 560 - value * 60), str(value), fill="black", font=font(44))
    return png(image)


def img_words():
    image, draw = canvas(1440, 800, (250, 246, 236))  # v2.9.8 parity runs used 1200 px: two lines were cut at the edge
    lines = ("The meeting moves to Thursday at 15:30.", "Room: Aurora, third floor.",
             "Bring the signed budget form (code KX-81).", "Contact: Maria Lindqvist")
    for i, line in enumerate(lines):
        draw.text((60, 80 + i * 160), line, fill=(20, 20, 20), font=font(52))
    return png(image)


def img_large():
    """3,600 x 2,200 px: 90 x 55 = 4,950 vision tokens."""
    image, draw = canvas(3600, 2200, (235, 242, 250))
    rng = random.Random(7)
    for _ in range(60):
        x, y = rng.randrange(0, 3400), rng.randrange(0, 2000)
        draw.rectangle((x, y, x + 180, y + 180), outline=(150, 170, 200), width=6)
    draw.text((900, 850), "ORBIT 58", fill=(180, 20, 20), font=font(420))
    return png(image)


def img_sized(tokens_w, tokens_h, label, seed):
    """A tokens_w x tokens_h-token image (40 px per token side) with varied content and a label."""
    image, draw = canvas(tokens_w * 40, tokens_h * 40, (245, 245, 240))
    rng = random.Random(seed)
    for _ in range(tokens_w * tokens_h // 8):
        x, y = rng.randrange(0, tokens_w * 40), rng.randrange(0, tokens_h * 40)
        draw.rectangle((x, y, x + 60, y + 60), fill=tuple(rng.randrange(256) for _ in range(3)))
    draw.text((80, 80), label, fill="black", font=font(min(tokens_w, tokens_h) * 6))
    return png(image)


def part(data):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def questions():
    return [
        ("number", [img_number()], "What number is written in the image? Answer with the number only.", ["4729"]),
        ("shapes", [img_shapes()], "List the shapes in the image from left to right with their colors.",
         ["circle", "triangle"]),
        ("chart_max", [img_chart()], "This is a bar chart. Which day has the highest bar, and what is its value?",
         ["thu", "9"]),
        ("chart_sum", [img_chart()], "Add up the values of all bars in the chart. Give the total.", ["26"]),
        ("ocr", [img_words()], "Read the note. When and where is the meeting, and what is the form code?",
         ["thursday", "15:30", "aurora", "kx-81"]),
        ("large", [img_large()], "What text is written in large red letters?", ["orbit", "58"]),
        ("two_images", [img_number(), img_shapes()],
         "The first image shows a number, the second shapes. What is the number, and how many shapes are there?",
         ["4729", "3"]),
        ("text_only", [], "What is 17 times 23? Answer with the number only.", ["391"]),
    ]


def chat(a, content, max_tokens, timeout=1800):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    return text, m.get("content") or "", d.get("usage") or {}, round(time.monotonic() - t0, 2)


def cmd_run(a):
    out = open(a.out, "w") if a.out else None
    ok = 0
    qs = questions()
    for qid, images, prompt, expect in qs:
        content = [*map(part, images), {"type": "text", "text": prompt}] if images else prompt
        text, answer, usage, secs = chat(a, content, a.max_tokens)
        hit = all(e in answer.lower() for e in expect)
        ok += hit
        row = {"id": qid, "pass": hit, "sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "seconds": secs, "answer": answer.strip()[:300], "text": text}
        print(f"{qid:11s} {'PASS' if hit else 'FAIL'} prompt {row['prompt_tokens']:>6} tok, {secs:6.2f} s: "
              f"{row['answer'][:110]!r}", flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
    print(f"{ok}/{len(qs)} answers contain the expected content")
    return 0 if ok == len(qs) else 1


def cmd_compare(a):
    rows = []
    for path in (a.a, a.b):
        with open(path) as fh:
            rows.append({r["id"]: r for r in map(json.loads, fh)})
    keys = sorted(set(rows[0]) | set(rows[1]))
    expected = {q[0] for q in questions()}
    if set(keys) != expected or set(rows[0]) != set(rows[1]):
        print(f"incomplete runs: A {sorted(rows[0])}, B {sorted(rows[1])}, expected {sorted(expected)}")
        return 1
    same = [k for k in keys if rows[0][k]["sha256"] == rows[1][k]["sha256"]]
    for k in keys:
        if k not in same:
            ra, rb = rows[0].get(k, {}), rows[1].get(k, {})
            ta, tb = ra.get("text", ""), rb.get("text", "")
            n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            print(f"{k}: differ at char {n}: A {ta[max(0, n - 40):n + 60]!r}\n{'':{len(k) + 2}}B {tb[max(0, n - 40):n + 60]!r}")
    print(f"greedy identical {len(same)}/{len(keys)}")
    return 0 if len(same) == len(keys) else 1


def filler(tokens, seed):
    rng = random.Random(seed)
    words = ("river", "copper", "lantern", "harvest", "orbit", "meadow", "signal", "granite", "velvet", "summit",
             "archive", "pepper", "falcon", "tundra", "beacon", "cobalt", "willow", "ember", "quartz", "harbor")
    parts, n = [f"Document {seed}."], 0
    while n < tokens:
        parts.append(" ".join(rng.choice(words) for _ in range(12)) + f" ({rng.randrange(10**6)}).")
        n += 18
    return " ".join(parts)


def cmd_load(a):
    t0 = time.monotonic()

    def one(i):
        images = [img_sized(a.tokens_w, a.tokens_h, f"R{i}-{j}", seed=1000 * i + j) for j in range(a.images)]
        content = [{"type": "text", "text": filler(a.context, seed=i) if a.context else f"Request {i}."},
                   *map(part, images),
                   {"type": "text", "text": "Describe the images in two sentences, then quote their labels."}]
        try:
            _, answer, usage, secs = chat(a, content, a.max_tokens, timeout=3600)
            return {"i": i, "ok": True, "prompt_tokens": usage.get("prompt_tokens"), "seconds": secs,
                    "answer": answer.strip()[:120]}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"i": i, "ok": False, "error": repr(exc)[:300]}

    results = []
    with concurrent.futures.ThreadPoolExecutor(a.conc) as pool:
        for r in pool.map(one, range(a.conc * a.rounds)):
            results.append(r)
            print(json.dumps(r), flush=True)
    failed = sum(not r["ok"] for r in results)
    print(f"load: {len(results)} requests ({a.conc} concurrent, ~{a.context} text tokens + {a.images} x "
          f"{a.tokens_w * a.tokens_h} image tokens each), {failed} failed, {time.monotonic() - t0:.0f} s")
    return 1 if failed else 0


def cmd_oversize(a):
    big = img_sized(a.tokens_w, a.tokens_h, "BIG", seed=5)
    try:
        chat(a, [part(big), {"type": "text", "text": "Describe the image."}], 32)
        refused = False
        print(f"oversize: a {a.tokens_w * a.tokens_h}-token image was ACCEPTED")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        refused = 400 <= exc.code < 500 and "vision tokens" in body
        print(f"oversize: HTTP {exc.code} ({'the size refusal' if refused else 'NOT the size refusal'}): {body[:300]!r}")
    _, answer, _, _ = chat(a, "What is 2 + 2? Answer with the number only.", 32)
    alive = "4" in answer
    print(f"oversize: server answers afterwards: {alive} ({answer.strip()!r})")
    return 0 if refused and alive else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out")
    r.add_argument("--max-tokens", type=int, default=384)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    load = sub.add_parser("load")
    load.add_argument("--conc", type=int, default=8)
    load.add_argument("--rounds", type=int, default=1)
    load.add_argument("--context", type=int, default=60000, help="approximate text tokens before the images")
    load.add_argument("--images", type=int, default=2)
    load.add_argument("--tokens-w", type=int, default=96, help="image width in vision tokens (40 px each)")
    load.add_argument("--tokens-h", type=int, default=85, help="image height in vision tokens (96 x 85 = 8,160)")
    load.add_argument("--max-tokens", type=int, default=256)
    o = sub.add_parser("oversize")
    o.add_argument("--tokens-w", type=int, default=130, help="130 x 130 = 16,900 tokens, above the default 16,384")
    o.add_argument("--tokens-h", type=int, default=130)
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "load": cmd_load, "oversize": cmd_oversize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
