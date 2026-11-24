"""Runs the toy DFLASH loop on the CPU stand-in package with the overlay armed the way the container arms it
(sitecustomize on PYTHONPATH). Saves the worker's reference history (torch.save) and prints one JSON line.

Usage (env): INKLING_VERIFY_CAP / INKLING_VERIFY_CAP_STEPLOG as in the container, FAKE_TEMPS=0,0.8,..(one per
request; 0 = greedy), FAKE_STEPS=n, FAKE_SKIP=i,j (steps that bypass _accept_block), FAKE_CAPACITY=n,
FAKE_HISTORY=path.
"""

import json
import os
from types import SimpleNamespace

import torch

torch.use_deterministic_algorithms(True)


def main() -> None:
    from inkling_vcap.state import STATE

    if STATE.config is not None:
        from inkling_vcap import dflash

        cap = int(os.environ.get("FAKE_CAPACITY", "8"))
        dflash._graph_capacity = lambda: cap
        if os.environ.get("FAKE_FAIL_FLUSH"):
            from inkling_vcap import dfsteplog

            def broken(self, records):
                raise OSError(28, "No space left on device (test)")

            dfsteplog.DfStepLog._flush = broken
    from sglang.srt.speculative import dflash_worker_v2 as w

    temps = [float(t) for t in os.environ.get("FAKE_TEMPS", "0,0.8,0").split(",")]
    bs = len(temps)
    worker = w.DFlashWorkerV2(seed=int(os.environ.get("FAKE_SEED", "0")))
    worker.skip_accept_steps = {int(s) for s in os.environ.get("FAKE_SKIP", "").split(",") if s}
    batch = SimpleNamespace(
        seq_lens=torch.tensor([17 + 5 * i for i in range(bs)], dtype=torch.int64),
        anchor=torch.tensor([3 + i for i in range(bs)], dtype=torch.int64),
        reqs=[SimpleNamespace(rid=f"req-{i}", origin_input_ids=list(range(5 + i, 22 + 6 * i))) for i in range(bs)],
        sampling_info=SimpleNamespace(temperatures=torch.tensor(temps, dtype=torch.float32)[:, None],
                                      has_custom_logit_processor=bool(os.environ.get("FAKE_CUSTOM_LP"))),
    )
    emitted = []
    for _ in range(int(os.environ.get("FAKE_STEPS", "40"))):
        emitted.append(worker.forward_batch_generation(batch).tolist())
        if os.environ.get("FAKE_STEP_SLEEP"):
            import time

            time.sleep(float(os.environ["FAKE_STEP_SLEEP"]))
    summary = {"emitted": emitted, "enabled": STATE.enabled, "applied": sorted(STATE.applied),
               "algorithm": STATE.algorithm}
    if STATE.steplog is not None:
        STATE.steplog.close()
        summary.update(steps=STATE.steplog.step, chunks=STATE.steplog.chunk, dropped=STATE.steplog.dropped,
                       written=STATE.steplog.written, failed=STATE.steplog.failed,
                       writer_error=repr(STATE.steplog._writer_error) if STATE.steplog._writer_error else None,
                       missing_routes=STATE.steplog.missing_routes, stale=STATE.steplog.stale_features,
                       run=STATE.steplog.run, skipped_walks=STATE.df.skipped)
    path = os.environ.get("FAKE_HISTORY")
    if path:
        torch.save(worker.history, path)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
