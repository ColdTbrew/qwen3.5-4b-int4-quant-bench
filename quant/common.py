"""Shared helpers for all quantization scripts.

Every method loads the same BF16 checkpoint, uses the same ignore list and
(when it needs data) the exact same calibration token blocks, then saves a
compressed-tensors checkpoint that vLLM loads with its Marlin W4A16 kernels.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: str | os.PathLike | None = None) -> dict:
    with open(path or ROOT / "configs" / "quant.yaml") as f:
        return yaml.safe_load(f)


def parse_args(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "quant.yaml"))
    p.add_argument("--out", default=None, help="override output dir")
    p.add_argument("--num-samples", type=int, default=None, help="override calibration sample count (smoke tests)")
    return p.parse_args()


def resolve_revision(model_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:  # offline
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, revision=revision, local_files_only=True)).name


def load_model(cfg: dict, offloaded: bool = True):
    """Load Qwen3.5-4B (Qwen3_5ForConditionalGeneration) in BF16.

    offloaded=True uses llm-compressor's load_context so the sequential
    calibration pipeline can onload one decoder layer at a time to the GPU.
    """
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    rev = cfg.get("_resolved_revision") or cfg["model_revision"]
    ctx = contextlib.nullcontext()
    if offloaded:
        from llmcompressor.utils import load_context

        ctx = load_context(Qwen3_5ForConditionalGeneration)
    with ctx:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg["model_id"], revision=rev, dtype=torch.bfloat16
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=rev)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=rev)
    return model, processor, tokenizer


def build_calibration(tokenizer, cfg: dict, num_samples: int | None = None):
    """Concat-and-chunk calibration set: shuffled pile-10k docs joined by EOS,
    cut into `seq_len` token blocks; first `num_samples` blocks are used."""
    from datasets import Dataset, load_dataset

    c = cfg["calibration"]
    n = num_samples or c["num_samples"]
    L = c["seq_len"]
    ds = load_dataset(c["dataset"], split=c["split"]).shuffle(seed=c["seed"])
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    blocks: list[list[int]] = []
    for text in ds["text"]:
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= L and len(blocks) < n:
            blocks.append(buf[:L])
            buf = buf[L:]
        if len(blocks) >= n:
            break
    assert len(blocks) == n, f"only {len(blocks)} calibration blocks"
    return Dataset.from_dict({"input_ids": blocks, "attention_mask": [[1] * L for _ in blocks]})


def versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine(), "host": socket.gethostname()}
    for pkg in ["torch", "transformers", "llmcompressor", "compressed-tensors", "auto-round", "hqq", "accelerate", "datasets"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cuda"] = torch.version.cuda
    return out


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_gib = torch.cuda.max_memory_allocated() / 2**30
        else:
            self.peak_gib = None
        self.seconds = time.perf_counter() - self.t0


def out_dir(cfg: dict, method: str, override: str | None) -> Path:
    p = Path(override) if override else ROOT / cfg["output_root"] / cfg["methods"][method]["name"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(model, processor, tokenizer, path: Path, meta: dict) -> None:
    t0 = time.perf_counter()
    model.save_pretrained(path, save_compressed=True)
    processor.save_pretrained(path)
    tokenizer.save_pretrained(path)
    meta["save_seconds"] = round(time.perf_counter() - t0, 1)
    meta["checkpoint_gib"] = round(
        sum(f.stat().st_size for f in path.glob("*.safetensors")) / 2**30, 3
    )
    meta["versions"] = versions()
    (path / "quant_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    res = ROOT / "results" / "quant"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{meta['name']}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


def base_meta(cfg: dict, method: str, calibrated: bool, num_samples: int | None) -> dict:
    m = cfg["methods"][method]
    meta = {
        "method": method,
        "name": m["name"],
        "model_id": cfg["model_id"],
        "model_revision": cfg.get("_resolved_revision"),
        "settings": m,
        "group_size": cfg["w4"]["group_size"] if method != "int8" else None,
        "ignore": cfg["ignore"],
    }
    if calibrated:
        c = dict(cfg["calibration"])
        if num_samples:
            c["num_samples"] = num_samples
        meta["calibration"] = c
    else:
        meta["calibration"] = None
    return meta


def prepare(method: str, desc: str):
    """Common prologue: args, config, revision pin, model + tokenizer load."""
    args = parse_args(desc)
    cfg = load_cfg(args.config)
    cfg["_resolved_revision"] = resolve_revision(cfg["model_id"], cfg["model_revision"])
    torch.manual_seed(0)
    return args, cfg


def run_oneshot(model, recipe, tokenizer, calib=None, seq_len: int | None = None, **extra) -> Timer:
    """Run llm-compressor oneshot and time it (quantization wall time only;
    model load and checkpoint save are timed separately)."""
    from llmcompressor import oneshot

    kw = dict(extra)
    if calib is not None:
        kw.update(
            dataset=calib,
            max_seq_length=seq_len,
            num_calibration_samples=len(calib),
            shuffle_calibration_samples=False,
        )
    with Timer() as t:
        oneshot(model=model, recipe=recipe, processor=tokenizer, **kw)
    return t


def standard_main(method: str, desc: str, make_recipe, calibrated: bool, offloaded: bool = True, **oneshot_kw):
    args, cfg = prepare(method, desc)
    t0 = time.perf_counter()
    model, processor, tokenizer = load_model(cfg, offloaded=offloaded)
    load_s = time.perf_counter() - t0
    calib = None
    calib_s = 0.0
    if calibrated:
        t0 = time.perf_counter()
        calib = build_calibration(tokenizer, cfg, args.num_samples)
        calib_s = time.perf_counter() - t0
    recipe = make_recipe(cfg)
    t = run_oneshot(model, recipe, tokenizer, calib, cfg["calibration"]["seq_len"], **oneshot_kw)
    meta = base_meta(cfg, method, calibrated, args.num_samples)
    meta.update(
        quant_seconds=round(t.seconds, 1),
        peak_gpu_alloc_gib=round(t.peak_gib, 2) if t.peak_gib else None,
        load_seconds=round(load_s, 1),
        calib_build_seconds=round(calib_s, 1),
        recipe=str(recipe),
    )
    save(model, processor, tokenizer, out_dir(cfg, method, args.out), meta)
