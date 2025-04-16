import pytest
from pytorch_model_man.unet_model import UNetModel50K, UNetModel100k, save_quantized_model

@pytest.mark.parametrize(
    "model, path",
    [
        (UNetModel50K(),  "MODEL/onnx_models_dut/qonnx_unet_model_50k.onnx"),
        (UNetModel100k(), "MODEL/onnx_models_dut/qonnx_unet_model_100k.onnx"),
    ],
)
def test_export_to_onnx(model, path):
    save_quantized_model(
        model,
        path,
        input_shape=(1, 3, 128, 128),
    )