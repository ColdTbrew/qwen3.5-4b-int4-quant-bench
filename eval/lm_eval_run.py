"""lm-evaluation-harness (vLLM backend) with identical settings for all checkpoints.

Each task runs with its configured num_fewshot / limit from configs/eval.yaml.
One vLLM engine is created per checkpoint and reused for all tasks."""
import argparse
import json
import time

from common import ROOT, eval_cfg, model_entry, versions, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tasks", default=None, help="comma list (default: all in config)")
    ap.add_argument("--limit", type=float, default=None, help="override limit (smoke)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = eval_cfg()
    vc, lc = cfg["vllm"], cfg["lm_eval"]
    me = model_entry(a.model)

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    lm = VLLM(
        pretrained=me["resolved_path"],
        gpu_memory_utilization=vc["gpu_memory_utilization"],
        max_model_len=vc["max_model_len"],
        max_num_seqs=vc["max_num_seqs"],
        language_model_only=vc["language_model_only"],
        seed=vc["seed"],
        batch_size=lc["batch_size"],
    )
    tasks = a.tasks.split(",") if a.tasks else list(lc["tasks"])
    all_res = {}
    for task in tasks:
        tcfg = lc["tasks"][task]
        limit = a.limit if a.limit is not None else tcfg.get("limit")
        out = ROOT / "results" / "raw" / "lm_eval" / f"{a.model}{a.tag}" / f"{task}.json"
        if out.exists() and json.load(open(out)).get("limit") == limit:
            print(task, "exists, skipping"); continue
        t0 = time.perf_counter()
        r = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=tcfg["num_fewshot"],
            limit=limit,
            random_seed=lc["seed"],
            numpy_random_seed=lc["seed"],
            torch_random_seed=lc["seed"],
            fewshot_random_seed=lc["seed"],
            log_samples=False,
        )
        entry = {
            "results": r["results"],
            "n_samples": r.get("n-samples"),
            "num_fewshot": tcfg["num_fewshot"],
            "limit": limit,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        all_res[task] = entry
        print(task, json.dumps(r["results"].get(task, {}), default=str))
        write_json(out, {"model": a.model, "path": me["resolved_path"], "task": task, **entry, "versions": versions()})


if __name__ == "__main__":
    main()
    # vLLM 0.30 engine teardown can abort (SIGABRT) after results are written; exit cleanly.
    import os, sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
