"""Collect raw results into results/summary.csv (+ quality.csv, speed.csv, summary.md)."""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


TASK_METRIC = {
    "mmlu": ("acc,none", "mmlu_acc"),
    "gsm8k": ("exact_match,flexible-extract", "gsm8k_em_flex"),
    "arc_challenge": ("acc_norm,none", "arc_c_acc_norm"),
    "hellaswag": ("acc_norm,none", "hellaswag_acc_norm"),
}


def main():
    rows = []
    for m in yaml.safe_load(open(ROOT / "configs/models.yaml"))["models"]:
        n = m["name"]
        row = {"method": m["label"], "name": n, "bits": m["bits"]}
        p = jload(R / "raw/ppl" / f"{n}.json")
        row["wikitext2_ppl"] = p["ppl"] if p else None
        for task, (key, col) in TASK_METRIC.items():
            t = jload(R / "raw/lm_eval" / n / f"{task}.json")
            row[col] = t["results"][task].get(key) if t else None
            if task == "gsm8k" and t:
                row["gsm8k_em_strict"] = t["results"][task].get("exact_match,strict-match")
        scores = [row[c] for _, c in TASK_METRIC.values() if row.get(c) is not None]
        row["lm_eval_avg"] = sum(scores) / len(scores) if len(scores) == len(TASK_METRIC) else None
        b = jload(R / "raw/bench" / n / "summary.json")
        if b:
            bc = {int(k): v for k, v in b["by_concurrency"].items()}
            row["ttft_ms_b1"] = bc.get(1, {}).get("mean_ttft_ms")
            row["decode_tok_s_b1"] = bc.get(1, {}).get("decode_tok_s_per_req")
            for c in (1, 8, 32):
                row[f"throughput_tok_s_b{c}"] = bc.get(c, {}).get("output_throughput_tok_s")
            for c in (8, 32):
                row[f"ttft_ms_b{c}"] = bc.get(c, {}).get("mean_ttft_ms")
                row[f"decode_tok_s_b{c}"] = bc.get(c, {}).get("decode_tok_s_per_req")
            row["weights_mem_gib"] = b.get("weights_gib")
            row["gpu_mem_peak_gib"] = round(b["gpu_mem_peak_mib"] / 1024, 2) if b.get("gpu_mem_peak_mib") else None
        q = jload(R / "quant" / f"{n}.json")
        row["quant_seconds"] = q["quant_seconds"] if q else (0.0 if n == "bf16" else None)
        row["checkpoint_gib"] = q["checkpoint_gib"] if q else None
        row["calib_samples"] = (q.get("calibration") or {}).get("num_samples") if q else None
        rows.append(row)
    df = pd.DataFrame(rows)
    bf = df[df.name == "bf16"].iloc[0] if (df.name == "bf16").any() else None
    if bf is not None and bf["wikitext2_ppl"]:
        df["ppl_delta_pct"] = (df["wikitext2_ppl"] / bf["wikitext2_ppl"] - 1) * 100
    df.to_csv(R / "summary.csv", index=False, float_format="%.4f")
    qcols = ["method", "wikitext2_ppl", "ppl_delta_pct", "mmlu_acc", "gsm8k_em_flex", "gsm8k_em_strict",
             "arc_c_acc_norm", "hellaswag_acc_norm", "lm_eval_avg"]
    df[[c for c in qcols if c in df]].to_csv(R / "quality.csv", index=False, float_format="%.4f")
    scols = ["method", "ttft_ms_b1", "ttft_ms_b8", "ttft_ms_b32", "decode_tok_s_b1", "decode_tok_s_b8", "decode_tok_s_b32",
             "throughput_tok_s_b1", "throughput_tok_s_b8", "throughput_tok_s_b32", "weights_mem_gib", "gpu_mem_peak_gib",
             "checkpoint_gib", "quant_seconds"]
    df[[c for c in scols if c in df]].to_csv(R / "speed.csv", index=False, float_format="%.2f")

    def f(v, fmt):
        return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else format(v, fmt)

    lines = ["| Method | WikiText-2 PPL ↓ | MMLU (0-shot) | GSM8K (5-shot, flex) | ARC-C (acc_norm) | HellaSwag (acc_norm) | TTFT b1 (ms) | Decode tok/s b1 | Throughput b1 / b8 / b32 (tok/s) | Weights mem (GiB) | Quant time |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        qt = r["quant_seconds"]
        qt_s = "–" if qt is None or pd.isna(qt) else ("n/a" if r["name"] == "bf16" else (f"{qt:.0f} s" if qt < 120 else f"{qt/60:.1f} min"))
        lines.append(
            f"| {r['method']} | {f(r['wikitext2_ppl'], '.3f')} | {f(r['mmlu_acc'] * 100 if r['mmlu_acc'] is not None and not pd.isna(r['mmlu_acc']) else None, '.2f')} | "
            f"{f(r['gsm8k_em_flex'] * 100 if r['gsm8k_em_flex'] is not None and not pd.isna(r['gsm8k_em_flex']) else None, '.2f')} | "
            f"{f(r['arc_c_acc_norm'] * 100 if r['arc_c_acc_norm'] is not None and not pd.isna(r['arc_c_acc_norm']) else None, '.2f')} | "
            f"{f(r['hellaswag_acc_norm'] * 100 if r['hellaswag_acc_norm'] is not None and not pd.isna(r['hellaswag_acc_norm']) else None, '.2f')} | "
            f"{f(r.get('ttft_ms_b1'), '.1f')} | {f(r.get('decode_tok_s_b1'), '.1f')} | "
            f"{f(r.get('throughput_tok_s_b1'), '.0f')} / {f(r.get('throughput_tok_s_b8'), '.0f')} / {f(r.get('throughput_tok_s_b32'), '.0f')} | "
            f"{f(r.get('weights_mem_gib'), '.2f')} | {qt_s} |"
        )
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
