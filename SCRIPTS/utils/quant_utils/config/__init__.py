from utils.quant_utils.config import quant_config_pow2_8_bit_fix


def get_quant_config(quant_config_str):
    if quant_config_str.lower() == "8bit_fix":
        return quant_config_pow2_8_bit_fix.get_config()
    else:
        raise ValueError(f"quant config : {quant_config_str} not supported")
