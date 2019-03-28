import torch
import upsampling
foo = torch.randn(128, 3, 224, 224)
interp = upsampling.Interpolation()
bar = interp(foo)

