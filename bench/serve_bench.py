"""vLLM online-serving benchmark: TTFT, per-request decode speed, throughput at
concurrency 1/8/32 (random 1024-in / 256-out prompts, ignore_eos), plus memory.

Starts `vllm serve` for one checkpoint, runs `vllm bench serve` for each
concurrency level, records GPU memory of the server's own processes, then
stops only the server it started."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import ROOT, eval_cfg, model_entry, versions, write_json  # noqa: E402

VENV_BIN = Path(sys.executable).parent


def descendants(pid: int) -> set[int]:
    import psutil

    try:
        p = psutil.Process(pid)
        return {pid} | {c.pid for c in p.children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def gpu_mem_mib(pids: set[int]) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    tot = 0
    for line in out.strip().splitlines():
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) in pids:
                tot += int(mem)
        except ValueError:
            pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--concurrency", default=None, help="comma list override")
    a = ap.parse_args()
    cfg = eval_cfg()
    vc, bc = cfg["vllm"], cfg["bench"]
    me = model_entry(a.model)
    port = bc["port"]
    outdir = ROOT / "results" / "raw" / "bench" / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = ROOT / "logs" / "bench"
    logdir.mkdir(parents=True, exist_ok=True)
    slog = open(logdir / f"{a.model}_server.log", "w")
    cmd = [
        str(VENV_BIN / "vllm"), "serve", me["resolved_path"],
        "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", a.model,
        "--gpu-memory-utilization", str(vc["gpu_memory_utilization"]),
        "--max-model-len", str(vc["max_model_len"]),
        "--max-num-seqs", str(vc["max_num_seqs"]),
        "--seed", str(vc["seed"]),
    ]
    if vc["language_model_only"]:
        cmd.append("--language-model-only")
    print(" ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited rc={srv.returncode}; see {slog.name}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                if time.time() - t0 > 1200:
                    raise TimeoutError("server not healthy after 20 min")
                time.sleep(3)
        startup_s = time.time() - t0
        pids = descendants(srv.pid)
        idle_mem = gpu_mem_mib(pids)

        peak = {"mib": idle_mem}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                peak["mib"] = max(peak["mib"], gpu_mem_mib(descendants(srv.pid)))
                time.sleep(1.0)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        def bench(c: int, n: int, fname: str | None):
            bcmd = [
                str(VENV_BIN / "vllm"), "bench", "serve",
                "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
                "--model", a.model, "--tokenizer", me["resolved_path"],
                "--dataset-name", "random",
                "--random-input-len", str(bc["input_len"]), "--random-output-len", str(bc["output_len"]),
                "--num-prompts", str(n), "--max-concurrency", str(c),
                "--ignore-eos", "--seed", "0",
                "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
            ]
            if fname:
                bcmd += ["--save-result", "--result-dir", str(outdir), "--result-filename", fname]
            r = subprocess.run(bcmd, capture_output=True, text=True)
            (logdir / f"{a.model}_bench_c{c}{'' if fname else '_warmup'}.log").write_text(r.stdout + r.stderr)
            if r.returncode != 0:
                raise RuntimeError(f"bench failed c={c}: {r.stderr[-2000:]}")

        bench(8, bc["warmup_prompts"], None)
        concs = [int(x) for x in a.concurrency.split(",")] if a.concurrency else bc["concurrency"]
        summary = {}
        for c in concs:
            n = bc["num_prompts"][c]
            bench(c, n, f"c{c}.json")
            d = json.loads((outdir / f"c{c}.json").read_text())
            summary[c] = {
                "num_prompts": n,
                "mean_ttft_ms": d["mean_ttft_ms"], "median_ttft_ms": d["median_ttft_ms"], "p99_ttft_ms": d.get("p99_ttft_ms"),
                "mean_tpot_ms": d["mean_tpot_ms"], "median_tpot_ms": d["median_tpot_ms"],
                "decode_tok_s_per_req": 1000.0 / d["mean_tpot_ms"],
                "output_throughput_tok_s": d["output_throughput"],
                "total_token_throughput_tok_s": d.get("total_token_throughput"),
                "request_throughput": d["request_throughput"],
            }
            print(c, summary[c], flush=True)
        stop.set()
        th.join(timeout=5)
        slog.flush()
        log = Path(slog.name).read_text(errors="ignore")
        m_load = re.findall(r"Model loading took ([\d.]+) GiB", log)
        m_kv = re.findall(r"Available KV cache memory: ([\d.]+) GiB", log)
        res = {
            "model": a.model,
            "path": me["resolved_path"],
            "settings": {**bc, "vllm": vc},
            "startup_seconds": round(startup_s, 1),
            "weights_gib": float(m_load[-1]) if m_load else None,
            "kv_cache_gib": float(m_kv[-1]) if m_kv else None,
            "gpu_mem_idle_mib": idle_mem,
            "gpu_mem_peak_mib": peak["mib"],
            "by_concurrency": summary,
            "versions": versions(),
        }
        write_json(outdir / "summary.json", res)
        print(json.dumps(res, indent=2))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)  # only the process group we started
            srv.wait(timeout=60)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        slog.close()


if __name__ == "__main__":
    main()
