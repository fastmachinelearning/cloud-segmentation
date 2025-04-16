import onnx
import pytest

def get_tensor_shapes(model):
    """Returns a dict mapping tensor name to shape"""
    tensor_shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        shape = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)  # dynamic dimension
        tensor_shapes[vi.name] = shape
    return tensor_shapes

def load_model_graph(onnx_path):
    model = onnx.load(onnx_path)
    shape_dict = get_tensor_shapes(model)
    graph = model.graph
    node_descriptions = []
    for node in graph.node:
        node_descriptions.append({
            'op_type': node.op_type,
            'num_inputs': len(node.input),
            'input_shapes': [shape_dict.get(name, None) for name in node.input],
            'num_outputs': len(node.output),
            'output_shapes': [shape_dict.get(name, None) for name in node.output],
            'attributes': {
                attr.name: onnx.helper.get_attribute_value(attr)
                for attr in node.attribute
            }
        })
    return node_descriptions

@pytest.mark.parametrize(
    "model_dut_path, model_ref_path",
    [
        ("MODEL/onnx_models_dut/pytorch_manual_50k.onnx", "MODEL/onnx_models_ref/ags_tiny_unet_50k.onnx"),
        ("MODEL/onnx_models_dut/pytorch_manual_100k.onnx", "MODEL/onnx_models_ref/ags_tiny_unet_100k.onnx"),
    ]
)
def test_models_are_structurally_identical(model_dut_path, model_ref_path):
    
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_dut_path)), model_dut_path)
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_ref_path)), model_ref_path)

    pytorch_model_graph = load_model_graph(model_dut_path)
    reference_model_graph = load_model_graph(model_ref_path)

    assert len(pytorch_model_graph) == len(reference_model_graph), \
        f"Node count mismatch: {len(pytorch_model_graph)} vs {len(reference_model_graph)}"

    for idx, (node1, node2) in enumerate(zip(pytorch_model_graph, reference_model_graph)):
        assert node1['op_type'] == node2['op_type'], f"Op type mismatch at node {idx}: {node1['op_type']} vs {node2['op_type']}"
        assert node1['num_inputs'] == node2['num_inputs'], f"Num inputs mismatch at node {idx}"
        assert node1['num_outputs'] == node2['num_outputs'], f"Num outputs mismatch at node {idx}"
        assert node1['input_shapes'] == node2['input_shapes'], f"Input shapes mismatch at node {idx}"
        assert node1['output_shapes'] == node2['output_shapes'], f"Output shapes mismatch at node {idx}"
        assert node1['attributes'] == node2['attributes'], f"Attributes mismatch at node {idx}"