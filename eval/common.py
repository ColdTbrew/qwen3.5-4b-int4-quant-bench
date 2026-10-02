from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# GB10 (sm_121): vLLM's CUTLASS int8 scaled_mm raises "Int8 not supported on SM121"
# for W8A8-INT8 checkpoints. Disabling it makes vLLM fall back to its Triton int8
# scaled_mm kernel (next in priority). Only affects int8 W8A8 layers; W4A16 models
# use Marlin/mixed-precision kernels and bf16 uses plain GEMMs. Inherited by the
# vllm serve subprocess launched from bench/serve_bench.py.
os.environ.setdefault("VLLM_DISABLED_KERNELS", "CutlassInt8ScaledMMLinearKernel")


def eval_cfg() -> dict:
    return yaml.safe_load(open(ROOT / "configs" / "eval.yaml"))


def model_entry(name: str) -> dict:
    for m in yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["models"]:
        if m["name"] == name:
            m = dict(m)
            p = ROOT / m["path"]
            m["resolved_path"] = str(p) if p.exists() else m["path"]
            return m
    raise KeyError(name)


def versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for pkg in ["vllm", "torch", "transformers", "lm_eval", "compressed-tensors", "flashinfer-python"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
