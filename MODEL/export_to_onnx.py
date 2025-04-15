import pytorch_model_man.unet_model as unet_model
import torch
import onnx


dummy_input = torch.randn(1, 3, 128, 128)  # adjust input size as needed
model = unet_model.UNetModel50K()
torch.onnx.export(
    model,
    dummy_input,
    "onnx_models_dut/pytorch_manual_50k.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

onnx.save(onnx.shape_inference.infer_shapes(onnx.load("onnx_models_dut/pytorch_manual_50k.onnx")), "onnx_models_dut/pytorch_manual_50k.onnx")

dummy_input = torch.randn(1, 3, 512, 512)
model = unet_model.UNetModel100k()
torch.onnx.export(
    model,
    dummy_input,
    "onnx_models_dut/pytorch_manual_100k.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

onnx.save(onnx.shape_inference.infer_shapes(onnx.load("onnx_models_dut/pytorch_manual_100k.onnx")), "onnx_models_dut/pytorch_manual_100k.onnx")