import torch
import torch.nn.functional as F

import numpy as np

from matplotlib.image import imread, imsave

from upsampling import Interpolation

fp16 = True

num_runs = 20

log = False # True

# foo = torch.randn(1, 1, 5, 5, requires_grad=True).cuda()

x = imread("starry_small.jpg")
#foo = x.swapaxes(1, 2).swapaxes(0, 1)
foo = np.expand_dims(x, axis=0)

# NWHC to NCHW
foo = torch.from_numpy(foo).cuda().float().permute(0, 3, 1, 2)

foo.requires_grad = True

if fp16:
    foo = foo.half()

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
    
    print("Max fprop difference:")
    
    print(torch.max(torch.abs(bar - baz)))

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

bar2 = bar.sum()
baz2 = baz.sum()

#bar.register_hook(set_grad(bar))
#baz.register_hook(set_grad(baz))
#
#bar2.register_hook(set_grad(bar2))
#baz2.register_hook(set_grad(baz2))

for i in range(num_runs):
    bar2.backward(retain_graph=True)
    baz2.backward(retain_graph=True)


#print("bar2: {}".format(bar2))
#print("baz2: {}".format(baz2))


#spam = bar.detach().squeeze().permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)

#ham = baz.detach().squeeze().permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)

#print(spam.shape)
#print(ham.shape)

#print(spam)
#print(ham)
#print(spam - ham)

#pic = np.squeeze(spam).swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8)
#pic = ham
#print(pic.shape)
#print(pic.dtype)

#print(x)
#print("\n\n")
#print(pic)

#imsave('resized.jpg', pic)

#print(bar.grad, baz.grad)
#print(bar2.grad, baz2.grad)


#print("Maximum backprop difference for [bar, baz]:")
#print(torch.max(torch.abs(bar.grad - baz.grad)))
#
#print("Maximum backprop difference for [bar2, baz2]:")
#print(torch.max(torch.abs(bar2.grad - baz2.grad)))

# print(bar - baz)
