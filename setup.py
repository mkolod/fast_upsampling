from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='nv_upsampling_cuda',

      ext_modules=[
          CUDAExtension('nv_upsampling_cuda', ['nv_upsampling.cpp', 'nv_upsampling_cuda_kernel.cu'],

      extra_compile_args={
        'cxx':  ['-std=c++14', '-O3', '-Wall'],
        'nvcc': [
            '-gencode',   'arch=compute_70,code=sm_70',
            '-gencode',   'arch=compute_75,code=sm_75',
            '-gencode',   'arch=compute_70,code=compute_70',
            '-Xcompiler', '-Wall',
            '-std=c++14',
            '-O3',
            '--use_fast_math'
            ]
        })],

      cmdclass={'build_ext': BuildExtension})

#          CppExtension('nv_upsampling_cpu', ['nv_upsampling.cpp']),

#from setuptools import setup
#from torch.utils.cpp_extension import CUDAExtension, BuildExtension
#
#package = 'adlr-interpolation'
#interpolation_module = CUDAExtension('interpolation_cuda', [
#    './src/interpolation.cpp',
#    './src/interpolation_kernels.cu'
#    ],
#    extra_compile_args={
#        'cxx':  ['-Wall', '-std=c++14'],
#        'nvcc': [
#            '-gencode',   'arch=compute_61,code=compute_61',
#            '-gencode',   'arch=compute_70,code=sm_70',
#            '-Xcompiler', '-Wall',
#            '-std=c++14'
#            ]
#        }
#    )
#
#setup(name=package,
#    version='0.1.0.0',
#    author="Alexey Kamenev",
#    description="Native (CUDA) implementation of interpolation",
#    py_modules=['interpolation'],
#    ext_modules=[interpolation_module],
#    cmdclass={'build_ext': BuildExtension})
#
#
