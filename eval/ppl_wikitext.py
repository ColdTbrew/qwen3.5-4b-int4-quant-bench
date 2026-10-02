"""WikiText-2 (raw, test) token-level perplexity with vLLM prompt logprobs.

Same tokenizer, context length and stride for every checkpoint
(configs/eval.yaml: ppl.context_length / ppl.stride)."""
import argparse
import math
import time

from common import ROOT, eval_cfg, model_entry, versions, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-windows", type=int, default=None, help="smoke test only")
    a = ap.parse_args()
    cfg = eval_cfg()
    pc, vc = cfg["ppl"], cfg["vllm"]
    me = model_entry(a.model)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(me["resolved_path"])
    text = pc["join"].join(load_dataset(pc["dataset"], pc["config"], split=pc["split"])["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    L, S = pc["context_length"], pc["stride"]
    windows = [ids[i : i + L] for i in range(0, len(ids) - L + 1, S)]
    if a.max_windows:
        windows = windows[: a.max_windows]

    llm = LLM(
        model=me["resolved_path"],
        gpu_memory_utilization=vc["gpu_memory_utilization"],
        max_model_len=max(vc["max_model_len"], L + 8),
        max_num_seqs=4,  # prompt logprobs over a 248k vocab are memory hungry
        language_model_only=vc["language_model_only"],
        seed=vc["seed"],
        enable_prefix_caching=False,
    )
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows], sp)
    nll, n = 0.0, 0
    for w, o in zip(windows, outs):
        for j in range(1, len(w)):
            nll -= o.prompt_logprobs[j][w[j]].logprob
            n += 1
    ppl = math.exp(nll / n)
    res = {
        "model": a.model,
        "path": me["resolved_path"],
        "ppl": ppl,
        "tokens_scored": n,
        "windows": len(windows),
        "dataset_tokens": len(ids),
        "settings": pc,
        "seconds": round(time.perf_counter() - t0, 1),
        "versions": versions(),
    }
    print(res)
    out = ROOT / "results" / "raw" / "ppl" / f"{a.model}{'_smoke' if a.max_windows else ''}.json"
    write_json(out, res)


if __name__ == "__main__":
    main()
