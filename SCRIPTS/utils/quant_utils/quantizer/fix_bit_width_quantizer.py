from brevitas.inject.enum import RestrictValueType
from brevitas.quant import (
    Int8ActPerTensorFixedPointMSE,
    Int8WeightPerTensorFixedPointMSE,
    Int16Bias,
    Uint8ActPerTensorFixedPointMSE,
)
from brevitas.quant.base import (
    MSESymmetricScale,
    MSEWeightZeroPoint,
)


class Int8WeightPerTensorPoTMSE(Int8WeightPerTensorFixedPointMSE):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    
    
class Int8BiasPerTensorPoTMSE(
    MSESymmetricScale, MSEWeightZeroPoint, Int16Bias
):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO


class Int8ActPerChannelPoTMSE(Int8ActPerTensorFixedPointMSE):
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (
        1,
        0,
        2,
        3,
    )  # assuming (N, C, H, W), put channel dimension first
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO

class Uint8ActPerChannelPoTMSE(Uint8ActPerTensorFixedPointMSE):
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (
        1,
        0,
        2,
        3,
    )  # assuming (N, C, H, W), put channel dimension first
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO


class Int8ActPerTensorPoTMSE(Int8ActPerTensorFixedPointMSE):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO


class Uint8ActPerTensorPoTMSE(Uint8ActPerTensorFixedPointMSE):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    