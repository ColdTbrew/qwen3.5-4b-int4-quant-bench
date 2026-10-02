"""RTN baseline: round-to-nearest int4 (symmetric, group 128, absmax scale). No data."""
from common import standard_main


def recipe(cfg):
    from llmcompressor.modifiers.quantization import QuantizationModifier

    m = cfg["methods"]["rtn"]
    return QuantizationModifier(targets="Linear", scheme=m["scheme"], ignore=cfg["ignore"])


if __name__ == "__main__":
    standard_main("rtn", __doc__, recipe, calibrated=False)
