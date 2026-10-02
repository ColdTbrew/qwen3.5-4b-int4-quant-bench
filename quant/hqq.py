"""HQQ: calibration-free half-quadratic optimisation of the int4 zero-point
(sparsity-promoting l_p<1 error, proximal solver), asymmetric, group 128.

Implementation notes
--------------------
* The solver follows hqq.core.optimize.optimize_weights_proximal_legacy
  (same lp_norm / beta / kappa / iters defaults, early stop on error increase).
* vLLM 0.30 removed its HQQ-Marlin backend and its compressed-tensors/GPTQ/AWQ
  loaders only accept *integer* zero points. We therefore project the zero
  point onto the integer grid after every proximal update ("integer_zero"),
  so the optimum is found inside the format vLLM can execute. The resulting
  scale/zero are written into a compressed-tensors W4A16_ASYM checkpoint
  (same Marlin kernel as AWQ). With integer_zero=false the float-zero optimum
  is rounded once at the end instead.
* For transparency the script also reports the mean |W - W_hat| reconstruction
  error of RTN-asym vs HQQ (float zero) vs HQQ (integer zero).
"""
import time

import torch

from common import base_meta, load_model, out_dir, prepare, run_oneshot, save, Timer

QMAX = 15  # uint4 grid used during optimisation; stored as signed int4 (-8..7)


def shrink_lp(x: torch.Tensor, beta: float, p: float) -> torch.Tensor:
    if p == 1:
        return torch.sign(x) * torch.relu(x.abs() - 1.0 / beta)
    return torch.sign(x) * torch.relu(x.abs() - (1.0 / beta) * x.abs().pow(p - 1))


@torch.no_grad()
def hqq_quantize(W: torch.Tensor, group: int, iters: int, p: float, beta: float, kappa: float, integer_zero: bool):
    """W: [out, in] float32 (cuda). Returns (scale [out, in/g], zero [out, in/g] float, err)."""
    out_f, in_f = W.shape
    Wg = W.reshape(-1, group)
    wmin = Wg.min(dim=1, keepdim=True).values
    wmax = Wg.max(dim=1, keepdim=True).values
    inv_s = (QMAX / (wmax - wmin)).clamp(max=2e4)  # hqq convention: q = W*inv_s + z
    z = -wmin * inv_s
    if integer_zero:
        z = z.round().clamp(0, QMAX)

    def err_of(z_):
        Wq = (Wg * inv_s + z_).round().clamp(0, QMAX)
        return ((Wq - z_) / inv_s - Wg).abs().mean().item(), Wq

    best_err, _ = err_of(z)
    best_z = z.clone()
    for _ in range(iters):
        Wq = (Wg * inv_s + z).round().clamp(0, QMAX)
        Wr = (Wq - z) / inv_s
        We = shrink_lp(Wg - Wr, beta, p)
        z = torch.mean(Wq - (Wg - We) * inv_s, dim=1, keepdim=True)
        if integer_zero:
            z = z.round().clamp(0, QMAX)
        beta *= kappa
        e, _ = err_of(z)
        if e < best_err:
            best_err, best_z = e, z.clone()
        else:
            break
    if not integer_zero:
        best_z = best_z.round().clamp(0, QMAX)
    final_err, _ = err_of(best_z)
    scale = (1.0 / inv_s).reshape(out_f, in_f // group)
    zero = best_z.reshape(out_f, in_f // group)
    return scale, zero, final_err


def rtn_asym_err(W: torch.Tensor, group: int) -> float:
    Wg = W.reshape(-1, group)
    wmin, wmax = Wg.min(1, keepdim=True).values, Wg.max(1, keepdim=True).values
    s = (wmax - wmin) / QMAX
    z = (-wmin / s).round().clamp(0, QMAX)
    Wq = (Wg / s + z).round().clamp(0, QMAX)
    return ((Wq - z) * s - Wg).abs().mean().item()


def main():
    args, cfg = prepare("hqq", __doc__)
    m = cfg["methods"]["hqq"]
    g = cfg["w4"]["group_size"]
    t0 = time.perf_counter()
    # plain (non-offloaded) load: weights stay resident on CPU and are copied to GPU per layer
    model, processor, tokenizer = load_model(cfg, offloaded=False)
    load_s = time.perf_counter() - t0

    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(targets="Linear", scheme=m["scheme"], ignore=cfg["ignore"])
    # 1) attach W4A16_ASYM quantization params (min-max init) to every target Linear
    t_init = run_oneshot(model, recipe, tokenizer, None)

    # 2) replace scale / zero-point with the HQQ optimum
    errs = {"rtn_asym": [], "hqq_float_zero": [], "hqq_int_zero": []}
    n = 0
    with Timer() as t_hqq:
        for name, mod in model.named_modules():
            if not (hasattr(mod, "weight_scale") and hasattr(mod, "weight_zero_point")):
                continue
            W = mod.weight.detach().to("cuda", torch.float32)
            scale, zero, e_int = hqq_quantize(W, g, m["iters"], m["lp_norm"], m["beta"], m["kappa"], m["integer_zero"])
            if n % 16 == 0:  # diagnostic on a subset of layers
                errs["rtn_asym"].append(rtn_asym_err(W, g))
                _, _, e_f = hqq_quantize(W, g, m["iters"], m["lp_norm"], m["beta"], m["kappa"], False)
                errs["hqq_float_zero"].append(e_f)
                errs["hqq_int_zero"].append(e_int)
            ws, wz = mod.weight_scale, mod.weight_zero_point
            assert ws.shape == scale.shape, (name, ws.shape, scale.shape)
            ws.data.copy_(scale.to(ws.dtype).to(ws.device))
            wz.data.copy_((zero - 8).to(wz.dtype).to(wz.device))  # uint4 zero -> signed int4 zero
            n += 1
    print(f"HQQ optimised {n} Linear layers in {t_hqq.seconds:.1f}s")
    diag = {k: (sum(v) / len(v) if v else None) for k, v in errs.items()}
    print("mean |W-W_hat| on sampled layers:", diag)

    meta = base_meta(cfg, "hqq", calibrated=False, num_samples=None)
    meta.update(
        quant_seconds=round(t_init.seconds + t_hqq.seconds, 1),
        quant_seconds_breakdown={"param_init": round(t_init.seconds, 1), "hqq_solver": round(t_hqq.seconds, 1)},
        peak_gpu_alloc_gib=round(t_hqq.peak_gib, 2) if t_hqq.peak_gib else None,
        load_seconds=round(load_s, 1),
        layers_quantized=n,
        reconstruction_mae_sampled=diag,
        recipe=str(recipe) + " + HQQ zero-point optimisation",
    )
    save(model, processor, tokenizer, out_dir(cfg, "hqq", args.out), meta)


if __name__ == "__main__":
    main()
