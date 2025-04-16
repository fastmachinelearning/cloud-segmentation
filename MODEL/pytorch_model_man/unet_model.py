import torch
import torch.nn as nn

from brevitas.export import export_qonnx
from brevitas.graph.quantize import COMPUTE_LAYER_MAP, QUANT_ACT_MAP, QUANT_IDENTITY_MAP, UNSIGNED_ACT_TUPLE, preprocess_for_quantize, quantize

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch_depth, out_ch_depth, out_ch_point, out_relu=True): # assuming output channels of first block are same as input channels for the first block
        super().__init__()
        self.out_relu = out_relu
        self.depthwise = nn.Conv2d(
            in_channels=in_ch_depth,
            out_channels=out_ch_depth,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_ch_depth,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=out_ch_depth,
            out_channels=out_ch_point,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True
        )
        if out_relu:
            self.act = nn.LeakyReLU(negative_slope=0.1015625, inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x) if self.out_relu else x

class EncoderBlock(nn.Module):
    def __init__(self, b0_inp, b0_int, b0_out, b1_int, b1_out, in_maxpool=True, duplicate_output=True):
        super().__init__()
        self.duplicate_output = duplicate_output
        self.in_maxpool = in_maxpool
        if self.in_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2)
        self.block1 = DepthwiseSeparableBlock(b0_inp, b0_int, b0_out)
        self.block2 = DepthwiseSeparableBlock(b0_out, b1_int, b1_out)

    def forward(self, x):
        if self.in_maxpool:
            x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        return (x, x) if self.duplicate_output else (x, None)

class DecoderBlock(nn.Module):
    def __init__(self, b0_inp, b0_int, b0_out, b1_int, b1_out, in_concat=True, out_resize=True, out_relu=True):
        super().__init__()
        self.in_concat = in_concat
        self.out_resize = out_resize
        # if self.in_concat:
        #     self.concat = torch.cat(dim=1)
        self.block1 = DepthwiseSeparableBlock(b0_inp, b0_int, b0_out)
        self.block2 = DepthwiseSeparableBlock(b0_out, b1_int, b1_out, out_relu=out_relu)
        if self.out_resize:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skip):
        if self.in_concat:
            x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return self.up(x) if self.out_resize else x


class UNetModel50K(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 3, 16, 16, 16, in_maxpool=False)
        self.enc2 = EncoderBlock(16, 16, 32, 32, 32)
        self.enc3 = EncoderBlock(32, 32, 32, 32, 48)
        self.enc4 = EncoderBlock(48, 48, 48, 48, 48)

        self.enc_middle = EncoderBlock(48, 48, 64, 64, 64, duplicate_output=False)
        self.dec_middle = DecoderBlock(64, 64, 64, 64, 64, in_concat=False)

        self.dec1 = DecoderBlock(112, 112, 48, 48, 48)
        self.dec2 = DecoderBlock(96, 96, 48, 48, 32)
        self.dec3 = DecoderBlock(64, 64, 32, 32, 32)
        self.dec4 = DecoderBlock(48, 48, 16, 16, 2, out_resize=False, out_relu=False)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x, _ = self.enc_middle(x)
        x = self.dec_middle(x, None)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        return self.dec4(x, skip1)

class UNetModel100k(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 3, 32, 32, 32, in_maxpool=False)
        self.enc2 = EncoderBlock(32, 32, 32, 32, 64)
        self.enc3 = EncoderBlock(64, 64, 64, 64, 64)
        self.enc4 = EncoderBlock(64, 64, 64, 64, 64)

        self.enc_middle = EncoderBlock(64, 64, 64, 64, 96, duplicate_output=False)
        self.dec_middle = DecoderBlock(96, 96, 96, 96, 64, in_concat=False)

        self.dec1 = DecoderBlock(128, 128, 64, 64, 64)
        self.dec2 = DecoderBlock(128, 128, 64, 64, 64)
        self.dec3 = DecoderBlock(128, 128, 64, 64, 32)
        self.dec4 = DecoderBlock(64, 64, 32, 32, 2, out_resize=False, out_relu=False)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x, _ = self.enc_middle(x)
        x = self.dec_middle(x, None)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        return self.dec4(x, skip1)
    
def get_quantized_model(model):
    model = preprocess_for_quantize(
        model,
        trace_model=True,
        relu6_to_relu=True,
        equalize_iters=0,
        equalize_merge_bias=True,
        merge_bn=True,
    )

    qmodel = quantize(
        model,
        quant_identity_map=QUANT_IDENTITY_MAP,
        compute_layer_map=COMPUTE_LAYER_MAP,
        quant_act_map=QUANT_ACT_MAP,
        unsigned_act_tuple=UNSIGNED_ACT_TUPLE,
        requantize_layer_handler_output=False,
    )

    return qmodel

def save_quantized_model(model, path, input_shape=(1, 3, 256, 256)):
    qmodel = get_quantized_model(model)
    inp = torch.randn(input_shape)
    qmodel(inp)  # collect scale factors
    qmodel.eval()
    export_qonnx(
        qmodel,
        inp,
        path,
    )
    return qmodel