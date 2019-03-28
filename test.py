import torch
import upsampling
foo = torch.randn(2, 1, 1, 1)
interp = upsampling.Interpolation()
bar = interp(foo)
print(foo)
print(bar)

