"""INT8 W8A8 comparator: SmoothQuant (migrate activation outliers into weights) + GPTQ int8 weights,
dynamic per-token int8 activations."""
from common import standard_main


def recipe(cfg):
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    m = cfg["methods"]["int8"]
    return [
        SmoothQuantModifier(smoothing_strength=m["smoothing_strength"]),
        GPTQModifier(targets="Linear", scheme=m["scheme"], ignore=cfg["ignore"]),
    ]


if __name__ == "__main__":
    standard_main("int8", __doc__, recipe, calibrated=True)
