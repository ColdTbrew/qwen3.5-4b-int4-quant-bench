"""AutoRound (Intel): signed-gradient-descent tuning of rounding offsets + clipping per block.

Uses llm-compressor's AutoRoundModifier, which calls Intel's `auto-round`
library for the per-decoder-block optimisation but shares our calibration
pipeline and exports compressed-tensors (same vLLM kernel as other methods).
"""
from common import standard_main


def recipe(cfg):
    from llmcompressor.modifiers.autoround import AutoRoundModifier

    m = cfg["methods"]["autoround"]
    return AutoRoundModifier(
        targets="Linear",
        scheme=m["scheme"],
        ignore=cfg["ignore"],
        iters=m["iters"],
        batch_size=m["batch_size"],
        enable_torch_compile=m.get("enable_torch_compile", True),
    )


if __name__ == "__main__":
    standard_main("autoround", __doc__, recipe, calibrated=True)
