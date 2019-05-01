from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='nv_bilinear_upsampling',

      ext_modules=[
          CUDAExtension('nv_bilinear_upsampling_cuda', ['nv_upsampling.cpp', 'nv_upsampling_cuda_kernel.cu'],

      extra_compile_args={
        'cxx':  ['-std=c++14', '-O3', '-Wall'],
        'nvcc': [
            '-gencode',   'arch=compute_35,code=sm_35',
            '-gencode',   'arch=compute_50,code=sm_50',
            '-gencode',   'arch=compute_52,code=sm_52',
            '-gencode',   'arch=compute_60,code=sm_60',
            '-gencode',   'arch=compute_61,code=sm_61',
            '-gencode',   'arch=compute_70,code=sm_70',
            '-gencode',   'arch=compute_75,code=sm_75',
            '-gencode',   'arch=compute_70,code=compute_70',
            '-Xcompiler', '-Wall',
            '-std=c++14',
            '-O3',
            '--use_fast_math'
            ]
        })],
      version='0.1.0',
      py_modules=['bilinear_upsampling'],
      cmdclass={'build_ext': BuildExtension})
