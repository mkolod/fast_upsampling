import torch
import nv_upsampling_cuda

class InterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        outputs = nv_upsampling.bilinear_forward(input)
#        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, dgrad):
        outputs = nv_upsampling.bilinear_backward(dgrad)
        return outputs

class Interpolation(torch.nn.Module):
    def __init__(self):
        super(Interpolation, self).__init__()

    def forward(self, input):
        return InterpolationFunction.apply(input)
