"""Plots from results/summary.csv -> plots/*.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "plots"
P.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "results/summary.csv")
COLORS = {"BF16": "#111111", "INT8 W8A8 (SQ+GPTQ)": "#7a7a7a", "RTN": "#e5484d", "GPTQ (act-order)": "#0070f3",
          "AWQ": "#f5a623", "HQQ": "#8e4ec6", "AutoRound": "#29a383"}
col = [COLORS.get(m, "#999") for m in df["method"]]


def scatter(x, y, xl, yl, fname, invert_y=False):
    d = df.dropna(subset=[x, y])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for _, r in d.iterrows():
        ax.scatter(r[x], r[y], s=90, color=COLORS.get(r["method"], "#999"), zorder=3)
        ax.annotate(r["method"], (r[x], r[y]), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.set_title("Qwen3.5-4B on DGX Spark (GB10): quality vs speed")
    fig.tight_layout()
    fig.savefig(P / fname, dpi=150)
    plt.close(fig)


def bar(cols, labels, title, ylabel, fname, scale=1.0, ylim_zero=True):
    d = df.dropna(subset=cols, how="all")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(d)), 4.6))
    n = len(cols)
    w = 0.8 / n
    xs = range(len(d))
    for i, (c, lab) in enumerate(zip(cols, labels)):
        vals = d[c] * scale
        bars = ax.bar([x + (i - (n - 1) / 2) * w for x in xs], vals, w,
                      color=[COLORS.get(m, "#999") for m in d["method"]] if n == 1 else None, label=lab if n > 1 else None)
        for b_, v in zip(bars, vals):
            if pd.notna(v):
                ax.annotate(f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}", (b_.get_x() + b_.get_width() / 2, b_.get_height()),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(d["method"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n > 1:
        ax.legend()
    if not ylim_zero:
        lo = (d[cols].min().min() * scale)
        hi = (d[cols].max().max() * scale)
        ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(P / fname, dpi=150)
    plt.close(fig)


scatter("decode_tok_s_b1", "lm_eval_avg", "decode tok/s per request (concurrency 1)", "mean of 4 lm-eval scores", "scatter_quality_vs_decode.png")
scatter("throughput_tok_s_b32", "wikitext2_ppl", "output throughput tok/s (concurrency 32)", "WikiText-2 PPL (lower is better)", "scatter_ppl_vs_throughput.png", invert_y=True)
bar(["wikitext2_ppl"], ["PPL"], "WikiText-2 perplexity (ctx 2048, stride 2048)", "PPL ↓", "bar_ppl.png", ylim_zero=False)
for c, t in [("mmlu_acc", "MMLU (0-shot, acc)"), ("gsm8k_em_flex", "GSM8K (5-shot, exact match flexible)"),
             ("arc_c_acc_norm", "ARC-Challenge (0-shot, acc_norm)"), ("hellaswag_acc_norm", "HellaSwag (0-shot, acc_norm)"),
             ("lm_eval_avg", "Mean of 4 lm-eval tasks")]:
    bar([c], [c], t, "score (%)", f"bar_{c}.png", scale=100, ylim_zero=False)
bar(["ttft_ms_b1", "ttft_ms_b8", "ttft_ms_b32"], ["c=1", "c=8", "c=32"], "Mean TTFT (1024-token prompts)", "ms ↓", "bar_ttft.png")
bar(["decode_tok_s_b1", "decode_tok_s_b8", "decode_tok_s_b32"], ["c=1", "c=8", "c=32"], "Per-request decode speed (1/TPOT)", "tok/s ↑", "bar_decode.png")
bar(["throughput_tok_s_b1", "throughput_tok_s_b8", "throughput_tok_s_b32"], ["c=1", "c=8", "c=32"], "Output throughput", "tok/s ↑", "bar_throughput.png")
bar(["weights_mem_gib"], ["weights"], "Model weight memory in vLLM", "GiB ↓", "bar_memory.png")
bar(["quant_seconds"], ["quant time"], "Quantization wall time (GB10)", "seconds ↓", "bar_quant_time.png")
print(sorted(p.name for p in P.glob("*.png")))
