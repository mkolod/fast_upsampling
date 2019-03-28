import torch
import nv_upsampling

class Interpolation(torch.nn.Module):
    def __init__(self):
        super(Interpolation, self).__init__()

    def forward(self, input):
        return nv_upsampling.bilinear_forward(input)
