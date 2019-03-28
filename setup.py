from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='upsampling',
      ext_modules=[
          CppExtension('nv_upsampling_cpu', ['nv_upsampling.cpp']),
          CUDAExtension('nv_upsampling_cuda', ['nv_upsampling.cu'])],
      cmdclass={'build_ext': BuildExtension})
