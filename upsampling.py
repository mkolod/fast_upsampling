import torch
import nv_upsampling_cuda

class InterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, new_h, new_w):
        outputs = nv_upsampling_cuda.bilinear_forward(input, new_h, new_w)
#        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, dgrad):
        outputs = nv_upsampling_cuda.bilinear_backward(dgrad)
        return outputs

class Interpolation(torch.nn.Module):
    def __init__(self):
        super(Interpolation, self).__init__()

    def forward(self, input, new_h, new_w):
        return InterpolationFunction.apply(input, new_h, new_w)
