import torch
import nv_upsampling_cuda

class InterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, new_h=None, new_w=None, scale_factor=None):
        dims = new_h is not None and new_w is not None
        assert dims or scale_factor, "either new dims (new_h, new_w) or scale factor need to be given"
        if scale_factor is not None:
            new_h, new_w = int(input.shape[-2] * scale_factor), int(input.shape[-1] * scale_factor)
        outputs = nv_upsampling_cuda.bilinear_forward(input, new_h, new_w)
        variables = [torch.as_tensor(input.size(-2)), torch.as_tensor(input.size(-1))]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, grad_o):
        outputs = nv_upsampling_cuda.bilinear_backward(grad_o.contiguous(), *ctx.saved_variables)
        return outputs, None, None, None

class Interpolation(torch.nn.Module):
    def __init__(self):
        super(Interpolation, self).__init__()

    def forward(self, input, new_h=None, new_w=None, scale_factor=None):
        return InterpolationFunction.apply(input, new_h, new_w, scale_factor)
