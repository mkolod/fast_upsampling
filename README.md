## Fast Bilinear Upsampling for PyTorch

### What is this?

This implementation of bilinear upsampling is considerably faster than the native PyTorch one in half precision (fp16). It is also slightly faster for single precision (fp32). See the "Performance" section below.

### Requirements
* PyTorch 1.0.0+
* CUDA 10.0+
* GPU with compute capability 7.0+ (Tesla V100, Titan V, GeForce RTX 2070/2080/2080Ti, etc.)

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

### Performance

Tensor dimensions: (128, 3, 224, 224)

**fp16**

| Direction | PyTorch  | This Implementation |
|-----------|----------|---------------------|
| forward   | 685 us   | 482 us              |
| backward  | 15.11 ms | 4.17 ms             |

**fp32**

| Direction | PyTorch  | This Implementation |
|-----------|----------|---------------------|
| forward   | 788 us   | 629 us              |
| backward  | 1.92 ms  | 1.49 ms             |


