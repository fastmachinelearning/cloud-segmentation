from torch import nn
import brevitas.nn as qnn
from utils.quant_utils.quantizer.fix_bit_width_quantizer import (
    Int8WeightPerTensorPoTMSE,
    Int8BiasPerTensorPoTMSE,
    Int8ActPerTensorPoTMSE,
    Uint8ActPerTensorPoTMSE,
)
from utils.quant_utils.quant_activation import QuantLeakyReLU


def get_config():
    COMPUTE_LAYER_MAP = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "weight_quant": Int8WeightPerTensorPoTMSE,
                "bias_quant": Int8BiasPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "weight_quant": Int8WeightPerTensorPoTMSE,
                "bias_quant": Int8BiasPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
    }

    # unused
    LAYERWISE_COMPUTE_LAYER_MAP = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "weight_quant": Int8WeightPerTensorPoTMSE,
                "bias_quant": Int8BiasPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "weight_quant": Int8WeightPerTensorPoTMSE,
                "bias_quant": Int8BiasPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
    }

    UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6)

    QUANT_ACT_MAP = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
        nn.ReLU6: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "max_val": 6.0,
                "return_quant_tensor": True,
            },
        ),
        nn.LeakyReLU: (
            QuantLeakyReLU,
            {
                "act_quant": Int8ActPerTensorPoTMSE,
                "input_quant": Int8ActPerTensorPoTMSE,
                "negative_slope": 26 / 256,
                "return_quant_tensor": True,
            },
        ),
    }

    QUANT_IDENTITY_MAP = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorPoTMSE,
                "return_quant_tensor": True,
            },
        ),
    }

    DATA_FUSION_LAYERS = {
        "add": (
            None,  # no replacement
            {
                "act_quant": Int8ActPerTensorPoTMSE,
            },
        ),
        "concat": (
            None,  # no replacement
            {
                "act_quant": Int8ActPerTensorPoTMSE,
            },
        ),
    }

    return {
        "COMPUTE_LAYER_MAP": COMPUTE_LAYER_MAP,
        "LAYERWISE_COMPUTE_LAYER_MAP": LAYERWISE_COMPUTE_LAYER_MAP,
        "QUANT_ACT_MAP": QUANT_ACT_MAP,
        "UNSIGNED_ACT_TUPLE": UNSIGNED_ACT_TUPLE,
        "QUANT_IDENTITY_MAP": QUANT_IDENTITY_MAP,
        "DATA_FUSION_LAYERS": DATA_FUSION_LAYERS,
    }
