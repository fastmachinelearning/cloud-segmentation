import sys

from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.export import export_qonnx


def export_model_qonnx(model, device, inp, export_path):

    return export_qonnx(
        model.to(device), args=inp.to(device), export_path=export_path
    )
