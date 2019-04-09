import torch
import torch.nn.functional as F

from upsampling import Interpolation

fp16 = False

foo = torch.randn(1, 1, 5, 5).cuda()
if fp16:
    foo = foo.half()

#foo = torch.randn(16, 3, 224, 224, requires_grad=True).cuda().half()

print("Reference implementation")

bar = F.interpolate(foo, scale_factor=2, mode='bilinear', align_corners=False)

print(foo)

print("Reference implementation")

print(bar)

interp = Interpolation()

baz = interp(foo, 10, 10)

print("My implementation")

print(baz)

print("Diff")

print(bar - baz)

#baz = bar.sum()
# baz.backward()

#loss_val.backward()

#print(loss_val)


