import torch
import torch.nn.functional as F

from upsampling import Interpolation

fp16 = True

num_runs = 20

log = False

foo = torch.randn(1, 1, 5, 5, requires_grad=True).cuda()
if fp16:
    foo = foo.half()

print(foo)

#foo = torch.randn(16, 3, 224, 224, requires_grad=True).cuda().half()

for i in range(num_runs):
    bar = F.interpolate(foo, scale_factor=2, mode='bilinear', align_corners=True) # False

if log:
    print("Original data")
    print(foo)
    print("Reference implementation")
    print(bar)

interp = Interpolation()

for i in range(num_runs):
    baz = interp(foo, scale_factor=2)

if log:
    print("My implementation")

    print(baz)
    
    print("Diff")
    
    print(bar - baz)
    metric = torch.abs(bar - baz) / torch.abs(bar) * 100
    print("Mean percentage error: {}".format(torch.mean(metric)))
    print("Max percentage error: {}".format(torch.max(metric)))


bar2 = bar.sum()
bar2.backward()

baz2 = baz.sum()
baz2.backward()

print(bar2.grad)
print(baz2.grad)

print(foo.grad)
