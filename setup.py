from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='upsampling',
      ext_modules=[CppExtension('nv_upsampling', ['nv_upsampling.cpp'])],
      cmdclass={'build_ext': BuildExtension})
