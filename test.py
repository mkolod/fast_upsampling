import torch
import torch.nn.functional as F

from upsampling import Interpolation

fp16 = False

num_runs = 10

log = False

foo = torch.randn(1, 1, 5, 5).cuda()
if fp16:
    foo = foo.half()

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
    baz = interp(foo, 10, 10)

if log:
    print("My implementation")

    print(baz)
    
    print("Diff")
    
    print(bar - baz)
    metric = torch.abs(bar - baz) / torch.abs(bar) * 100
    print("Mean percentage error: {}".format(torch.mean(metric)))
    print("Max percentage error: {}".format(torch.max(metric)))

#baz = bar.sum()
# baz.backward()

#loss_val.backward()

#print(loss_val)


