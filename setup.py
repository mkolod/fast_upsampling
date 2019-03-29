from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='nv_upsampling_cuda',
      ext_modules=[
          CUDAExtension('nv_upsampling_cuda', ['nv_upsampling.cpp', 'nv_upsampling_cuda_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})

#          CppExtension('nv_upsampling_cpu', ['nv_upsampling.cpp']),

