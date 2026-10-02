"""Supplementary HQQ study (HF transformers, fake-quant, PPL only).

vLLM can only execute integer zero points, so the main-table HQQ checkpoint
projects HQQ's zero point to integers. This script measures, in plain
transformers with dequantised weights, what that constraint costs:

  bf16        : original weights
  rtn_asym    : min-max asymmetric int4, g128 (integer zero)
  hqq_float   : official `hqq` library (Quantizer.quantize optimize=True), float zero, g128
  hqq_int     : our integer-zero HQQ solver (quant/hqq_quant.py)

Same WikiText-2 windows (ctx 2048, stride 2048) as eval/ppl_wikitext.py.
Run with the quant venv:  python eval/hf_fakequant_ppl.py --variants bf16,rtn_asym,hqq_float,hqq_int
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quant"))


def ignored(name: str, patterns) -> bool:
    import re

    return any(re.match(p[3:], name) if p.startswith("re:") else p == name for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bf16,rtn_asym,hqq_float,hqq_int")
    ap.add_argument("--max-windows", type=int, default=None)
    a = ap.parse_args()
    qcfg = yaml.safe_load(open(ROOT / "configs/quant.yaml"))
    ecfg = yaml.safe_load(open(ROOT / "configs/eval.yaml"))["ppl"]
    hcfg = qcfg["methods"]["hqq"]
    g = qcfg["w4"]["group_size"]

    from datasets import load_dataset
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from hqq.core.quantize import Quantizer
    from hqq_quant import hqq_quantize

    tok = AutoTokenizer.from_pretrained(qcfg["model_id"])
    text = ecfg["join"].join(load_dataset(ecfg["dataset"], ecfg["config"], split=ecfg["split"])["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    L, S = ecfg["context_length"], ecfg["stride"]
    windows = [ids[i : i + L] for i in range(0, len(ids) - L + 1, S)]
    if a.max_windows:
        windows = windows[: a.max_windows]

    results = {}
    out = ROOT / "results" / "raw" / "hqq_supplementary.json"
    if out.exists():
        results = json.loads(out.read_text()).get("ppl", {})
    for variant in a.variants.split(","):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(qcfg["model_id"], dtype=torch.bfloat16).cuda().eval()
        t0 = time.perf_counter()
        n_q = 0
        if variant != "bf16":
            for name, mod in model.named_modules():
                if not isinstance(mod, torch.nn.Linear) or ignored(name, qcfg["ignore"]):
                    continue
                W = mod.weight.data.float()
                if variant == "rtn_asym":
                    Wq, meta = Quantizer.quantize(W, nbits=4, channel_wise=True, group_size=g, optimize=False,
                                                  axis=1, bitpack=False, compute_dtype=torch.float32, device="cuda")
                    Wd = Quantizer.dequantize(Wq, meta)
                    # hqq rounds nothing here; force integer zero for a true RTN-asym baseline
                    Wg = W.reshape(-1, g)
                    mn, mx = Wg.min(1, keepdim=True).values, Wg.max(1, keepdim=True).values
                    s = (mx - mn) / 15
                    z = (-mn / s).round().clamp(0, 15)
                    Wd = (((Wg / s + z).round().clamp(0, 15) - z) * s).reshape(W.shape)
                elif variant == "hqq_float":
                    Wq, meta = Quantizer.quantize(W, nbits=4, channel_wise=True, group_size=g, optimize=True,
                                                  round_zero=False, axis=1, bitpack=False,
                                                  compute_dtype=torch.float32, device="cuda")
                    Wd = Quantizer.dequantize(Wq, meta).reshape(W.shape)
                elif variant == "hqq_int":
                    scale, zero, _ = hqq_quantize(W, g, hcfg["iters"], hcfg["lp_norm"], hcfg["beta"], hcfg["kappa"], True)
                    Wg = W.reshape(-1, g)
                    s, z = scale.reshape(-1, 1), zero.reshape(-1, 1)
                    Wd = (((Wg / s + z).round().clamp(0, 15) - z) * s).reshape(W.shape)
                else:
                    raise ValueError(variant)
                mod.weight.data.copy_(Wd.to(mod.weight.dtype))
                n_q += 1
        qs = time.perf_counter() - t0
        nll, n = 0.0, 0
        with torch.no_grad():
            for w in windows:
                x = torch.tensor([w], device="cuda")
                logits = model(input_ids=x).logits.float()
                lp = torch.log_softmax(logits[0, :-1], -1)
                nll -= lp.gather(1, x[0, 1:, None]).sum().item()
                n += len(w) - 1
        ppl = math.exp(nll / n)
        results[variant] = {"ppl": ppl, "layers_quantized": n_q, "quant_seconds": round(qs, 1), "windows": len(windows)}
        print(variant, results[variant], flush=True)
        del model
        torch.cuda.empty_cache()
        out.write_text(json.dumps({"note": __doc__, "settings": ecfg, "group_size": g, "hqq_params": hcfg, "ppl": results}, indent=2))


if __name__ == "__main__":
    main()
