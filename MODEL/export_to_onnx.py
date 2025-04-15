import pytorch_model_man.unet_model as UNetModel
import torch
import onnx

dummy_input = torch.randn(1, 3, 128, 128)  # adjust input size as needed
model = UNetModel.UNetModel()
torch.onnx.export(
    model,
    dummy_input,
    "pytorch_model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

onnx.save(onnx.shape_inference.infer_shapes(onnx.load("pytorch_model.onnx")), "pytorch_model.onnx")
