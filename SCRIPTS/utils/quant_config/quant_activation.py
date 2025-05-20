from typing import Optional

from torch import nn

from brevitas.inject.defaults import Int8ActPerTensorFloat

from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL


class QuantLeakyReLU(QuantNLAL):

    def __init__(
        self,
        act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
        input_quant: Optional[ActQuantType] = None,
        return_quant_tensor: bool = False,
        **kwargs,
    ):
        QuantNLAL.__init__(
            self,
            act_impl=nn.LeakyReLU,
            passthrough_act=False,  # NEED! to be False for leakyReLU, because ?? -> passthrough = "# preserve scale/zp/bit/sign even without output quant"
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs,
        )
