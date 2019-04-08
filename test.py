import torch
import torch.nn.functional as F

foo = torch.randn(2, 3, 28, 28, requires_grad=True).cuda().half()
bar = F.interpolate(foo, scale_factor=2, mode='bilinear', align_corners=False)
baz = bar.sum()

baz.backward()
#loss_val.backward()

#print(loss_val)


