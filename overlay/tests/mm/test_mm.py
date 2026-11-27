"""CPU tests of inkling_mm (the per-image size bound) and of the launcher's MM switch.

    python3 -m unittest discover -s tests/mm -v        (from overlay/; the launcher tests need Linux /proc/meminfo)
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

import inkling_mm  # noqa: E402


def png(width, height):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class Bound(unittest.TestCase):
    def test_env(self):
        for raw, cap in (("", 0), ("0", 0), ("0000", 0), (" 16384 ", 16384), (str(1 << 20), 1 << 20)):
            self.assertEqual(inkling_mm.cap_from_env({inkling_mm.ENV: raw}), cap, raw)
        self.assertEqual(inkling_mm.cap_from_env({}), 0)
        for raw in ("-1", "x", "1.5", str((1 << 20) + 1)):
            with self.assertRaises(RuntimeError, msg=raw):
                inkling_mm.cap_from_env({inkling_mm.ENV: raw})

    def test_token_count_matches_the_processor(self):
        # Inkling2ImageProcessor.count_tokens_for_image: ((w + 39) // 40) * ((h + 39) // 40)
        for w, h in ((1, 1), (40, 40), (41, 40), (4032, 3024), (5120, 5120), (0, 5)):
            expect = ((w + 39) // 40) * ((h + 39) // 40) if w > 0 and h > 0 else 0
            self.assertEqual(inkling_mm.image_tokens(w, h), expect, (w, h))

    def test_check_sums_the_request_from_headers(self):
        from PIL import Image

        load = lambda image: image  # noqa: E731  (bytes in, bytes out, like _load_image_bytes)
        self.assertEqual(inkling_mm.check([png(80, 40)], 2, load), 2)
        self.assertEqual(inkling_mm.check([png(40, 40), Image.new("RGB", (40, 40))], 2, load), 2)
        with self.assertRaisesRegex(ValueError, "1 image.*81 x 40 pixels.*3 vision tokens"):
            inkling_mm.check([png(81, 40)], 2, load)
        with self.assertRaisesRegex(ValueError, "2 image.*are 3 vision tokens"):  # each fits, together they do not
            inkling_mm.check([png(80, 40), Image.new("RGB", (40, 40))], 2, load)

    def processor_module(self, encoded):
        class Processor:
            def count_tokens_for_image(self, width, height):
                return ((width + 39) // 40) * ((height + 39) // 40)

            def preprocess(self, images, return_tensors="pt"):
                encoded.append(images)
                return "FEATURES"

            __call__ = preprocess

        return types.SimpleNamespace(Inkling2ImageProcessor=Processor, _load_image_bytes=lambda image: image)

    def test_patch_refuses_before_decoding(self):
        encoded = []
        module = self.processor_module(encoded)
        saved = dict(inkling_mm.STATE)
        inkling_mm.STATE["cap"] = 4
        try:
            inkling_mm._patch(module)
        finally:
            inkling_mm.STATE.update(saved)
        processor = module.Inkling2ImageProcessor()
        small, large = png(80, 80), png(120, 80)
        self.assertEqual(processor.preprocess([small]), "FEATURES")
        self.assertEqual(processor(small), "FEATURES")  # __call__ is patched too
        for images in ([large], [small, small]):
            with self.assertRaises(ValueError):
                processor.preprocess(images)
        self.assertEqual(encoded, [[small], small])

    def test_patch_refuses_another_processor(self):
        with self.assertRaises(RuntimeError):
            inkling_mm._patch(types.SimpleNamespace(_decode_rgb=lambda data: data))
        module = self.processor_module([])
        module.Inkling2ImageProcessor.__call__ = lambda self, images: images  # preprocess no longer the entry point
        with self.assertRaises(RuntimeError):
            inkling_mm._patch(module)


class InFlight(unittest.TestCase):
    def test_budget_waits_and_wakes(self):
        import asyncio

        async def scenario():
            budget = inkling_mm.Budget(10)
            await budget.acquire("a", 6)
            await budget.acquire("big", 0)  # nothing to hold
            second = asyncio.ensure_future(budget.acquire("b", 6))  # 12 > 10: waits
            await asyncio.sleep(0.05)
            self.assertFalse(second.done())
            self.assertEqual(budget.release("a"), 6)
            self.assertEqual(budget.release("a"), 0)  # idempotent
            await asyncio.wait_for(second, 1)
            self.assertEqual((budget.used, budget.waited, budget.peak), (6, 1, 6))
            budget.release("b")
            await budget.acquire("lone", 25)  # above the bound, nothing in flight: admitted
            self.assertEqual(budget.used, 25)
            budget.release("lone")
            self.assertEqual(budget.used, 0)

        asyncio.run(scenario())

    def test_processor_and_tokenizer_hooks(self):
        import asyncio
        import concurrent.futures

        calls = []

        class Processor:
            io_executor = concurrent.futures.ThreadPoolExecutor(1)

            async def process_mm_data_async(self, image_data=None, audio_data=None, video_data=None, input_text="",
                                            request_obj=None, **kwargs):
                calls.append((request_obj.rid, image_data))
                return "MM"

        class TokenizerManager:
            async def generate_request(self, obj, request=None):
                for i in range(3):
                    yield i

        fake_image = types.SimpleNamespace(_load_image_bytes=lambda image: image)
        processor_module = types.SimpleNamespace(Inkling2MultimodalProcessor=Processor,
                                                 _resolve_image=lambda image: image)
        saved = (dict(inkling_mm.STATE), inkling_mm.BUDGET)
        inkling_mm.STATE["cap"], inkling_mm.BUDGET = 8, inkling_mm.Budget(8)
        modules = {"sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
                   "sglang.srt.multimodal": types.ModuleType("sglang.srt.multimodal"),
                   "sglang.srt.multimodal.inkling2": types.SimpleNamespace(image_processing=fake_image)}
        try:
            with unittest.mock.patch.dict(sys.modules, modules):
                inkling_mm._patch_processor(processor_module)
            inkling_mm._patch_tokenizer(types.SimpleNamespace(TokenizerManager=TokenizerManager))
            budget = inkling_mm.BUDGET

            class Req:
                def __init__(self, rid):
                    self.rid = rid

            async def scenario():
                p, tm = Processor(), TokenizerManager()
                one, two = Req("r1"), Req("r2")
                self.assertEqual(await p.process_mm_data_async([png(160, 80)], request_obj=one), "MM")  # 8 tokens
                self.assertEqual(budget.used, 8)
                waiting = asyncio.ensure_future(p.process_mm_data_async([png(40, 40)], request_obj=two))
                await asyncio.sleep(0.1)
                self.assertFalse(waiting.done())  # 8 + 1 > 8: waits before its image is processed
                self.assertEqual([c[0] for c in calls], ["r1"])
                self.assertEqual([x async for x in tm.generate_request(one)], [0, 1, 2])  # r1 finishes
                await asyncio.wait_for(waiting, 1)
                self.assertEqual(budget.used, 1)
                stream = tm.generate_request(two)
                await stream.__anext__()
                await stream.aclose()  # client went away mid-stream
                self.assertEqual(budget.used, 0)
                with self.assertRaisesRegex(ValueError, "vision tokens"):  # above the per-request bound
                    await p.process_mm_data_async([png(160, 120)], request_obj=Req("r3"))
                self.assertEqual(budget.used, 0)
                self.assertEqual(await p.process_mm_data_async(None, request_obj=Req("t")), "MM")  # text: no wait

            asyncio.run(scenario())
        finally:
            inkling_mm.STATE.update(saved[0])
            inkling_mm.BUDGET = saved[1]


@unittest.skipUnless(os.path.exists("/proc/meminfo") and shutil.which("bash"), "launcher tests need Linux")
class Launcher(unittest.TestCase):
    """scripts/launch.sh up against a fake docker that records the command line."""

    def run_launch(self, env_extra, *args):
        repo = os.path.dirname(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            bindir, model, draft, log = (os.path.join(tmp, d) for d in ("bin", "model", "draft", "docker.log"))
            for d in (bindir, model, draft):
                os.makedirs(d)
            for f in ("config.json", "mtp.safetensors"):
                open(os.path.join(model, f), "w").close()
            with open(os.path.join(bindir, "docker"), "w") as fh:
                fh.write(f'#!/bin/sh\n[ "$1" = run ] && printf "%s\\n" "$@" > {log}\nexit 0\n')
            os.chmod(os.path.join(bindir, "docker"), 0o755)
            with open(os.path.join(bindir, "nohup"), "w") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(os.path.join(bindir, "nohup"), 0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("MM", "SGLANG_"))}
            env.update(PATH=bindir + ":" + os.environ["PATH"], MODEL_DIR=model, CACHE_DIR=tmp, SPEC="dspark",
                       MIN_AVAIL_GIB="-100", **env_extra)
            proc = subprocess.run(["bash", os.path.join(repo, "scripts", "launch.sh"), "up", *args], env=env,
                                  capture_output=True, text=True, timeout=60)
            argv = []
            if os.path.exists(log):
                with open(log) as fh:
                    argv = fh.read().split("\n")
            return proc, argv

    def test_default_is_text_only(self):
        proc, argv = self.run_launch({})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("--enable-multimodal", argv)
        self.assertFalse(any(a.startswith("INKLING_MM_") for a in argv))
        self.assertEqual(argv[argv.index("--mem-fraction-static") + 1], "0.89")  # SPEC=dspark + KV12S=1

    def test_images_with_the_bound_and_margin(self):
        proc, argv = self.run_launch({"MM": "1"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--enable-multimodal", argv)
        self.assertEqual(argv[argv.index("--mm-io-worker-num") + 1], "1")
        self.assertIn("INKLING_MM_MAX_IMAGE_TOKENS=16384", argv)
        self.assertIn("INKLING_MM_INFLIGHT_TOKENS=16384", argv)
        self.assertEqual(argv[argv.index("--mem-fraction-static") + 1], "0.86")  # 0.89 - 0.03
        proc, argv = self.run_launch({"MM": "1", "MEM_FRACTION": "0.60"})  # an explicit fraction is kept
        self.assertEqual(argv[argv.index("--mem-fraction-static") + 1], "0.60")

    def test_refuses_the_raw_flags(self):
        for args in (("--enable-multimodal",), ("--disable-multimodal",)):
            proc, argv = self.run_launch({"MM": "1"}, *args)
            self.assertEqual(proc.returncode, 2, args)
            self.assertIn("MM=1", proc.stderr)
            self.assertEqual(argv, [])
        for env in ({"MM": "2"}, {"MM_MAX_IMAGE_TOKENS": "1048577"}, {"MM_MAX_IMAGE_TOKENS": "-5"},
                    {"MM_MAX_IMAGE_TOKENS": "8k"}, {"MM_INFLIGHT_TOKENS": "8192"}, {"MM_INFLIGHT_TOKENS": "x"}):
            proc, argv = self.run_launch(env)
            self.assertEqual(proc.returncode, 2, env)
            self.assertEqual(argv, [])

    def test_bound_is_normalized(self):
        proc, argv = self.run_launch({"MM": "1", "MM_MAX_IMAGE_TOKENS": "0008192"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("INKLING_MM_MAX_IMAGE_TOKENS=8192", argv)


if __name__ == "__main__":
    unittest.main()
