# setup.py

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# Detect platform
is_mac = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

# Build flags
if is_mac:
    # macOS: use clang with libomp
    # Locate Homebrew or Conda libomp
    possible = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        possible.append(os.path.join(conda_prefix, 'opt', 'llvm-openmp'))
    possible.extend([
        '/opt/homebrew/opt/libomp',
        '/usr/local/opt/libomp'
    ])
    libomp_prefix = next((p for p in possible if p and os.path.isdir(p)), None)
    if not libomp_prefix:
        raise RuntimeError('libomp not found; install via `brew install libomp` or `conda install -c conda-forge llvm-openmp`')
    extra_compile_args = ['-O3', '-Xpreprocessor', '-fopenmp']
    extra_link_args = [f'-L{os.path.join(libomp_prefix, "lib")}', '-lomp']
    include_dirs = [pybind11.get_include(), os.path.join(libomp_prefix, 'include')]
    library_dirs = [os.path.join(libomp_prefix, 'lib')]
elif is_linux:
    # Linux: use GCC/Clang with libgomp
    extra_compile_args = ['-O3', '-fopenmp']
    extra_link_args = ['-fopenmp']
    include_dirs = [pybind11.get_include()]
    library_dirs = []
else:
    raise RuntimeError(f'Unsupported platform: {sys.platform}')

ext_modules = [
    Extension(
        'force_kernel',
        sources=['force_kernel.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    name='force_kernel',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)

