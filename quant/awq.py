"""AWQ: activation-aware per-channel scaling (grid-searched) before int4 RTN with zero point."""
from common import standard_main


def recipe(cfg):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    m = cfg["methods"]["awq"]
    return [
        AWQModifier(duo_scaling=m["duo_scaling"], n_grid=m["n_grid"]),
        QuantizationModifier(targets="Linear", scheme=m["scheme"], ignore=cfg["ignore"]),
    ]


if __name__ == "__main__":
    standard_main("awq", __doc__, recipe, calibrated=True)
