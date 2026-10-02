"""GPTQ: second-order (Hessian-based) error-compensating int4 rounding, act-order on."""
from common import standard_main


def recipe(cfg):
    from llmcompressor.modifiers.gptq import GPTQModifier

    m = cfg["methods"]["gptq"]
    return GPTQModifier(
        targets="Linear",
        scheme=m["scheme"],
        ignore=cfg["ignore"],
        actorder=m["actorder"],
        dampening_frac=m["dampening_frac"],
        block_size=m["block_size"],
    )


if __name__ == "__main__":
    standard_main("gptq", __doc__, recipe, calibrated=True)
