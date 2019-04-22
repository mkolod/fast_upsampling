## Fast Bilinear Upsampling for PyTorch

### What is this?

This implementation of bilinear upsampling is considerably faster than the native PyTorch one in half precision (fp16). For example, for a tensor of size (128, 3, 224, 224), on a Titan V and Core i7-7800X CPU @ 3.50GHz, this implementation was 1.5x faster for fprop and 3.5x faster in backprop. Single precision (fp32) performance is essentially the same as the original implementation.

### Requirements
* PyTorch 1.0.0+
* CUDA 10.0+
* For fp16: compute capability 6.0, and preferably 7.0+ (Tesla P100, Tesla V100, Titan V, GeForce RTX 2070/2080/2080Ti, etc.)

### Caveats

Currently this implementation is equivalent to PyTorch's bilinear upsampling with `align_corners=True`. The case of `align_corners=False` hasn't been implemented yet.

### Installation

`$python setup.py install`

### Sample execution

The script `test.py` demonstrates use. The TL;DR is that the following calls are equivalent:
* PyTorch: 
```result = torch.nn.functional.interpolate(data, scale_factor=2, mode='bilinear', align_corners=True)```
* This library: 
```
from bilinear_upsampling import Interpolation
interp = Interpolation()
result = interp(data, scale_factor=2)
```
