#!/usr/bin/env python3
"""
setup.py - Build both force_kernel and fmm_kernel as pybind11 extensions with OpenMP.

Usage:
    python3.12 setup.py build_ext --inplace

This will produce:
    - force_kernel.cpython-<ver>-<arch>.so 
    - fmm_kernel.cpython-<ver>-<arch>.so

which can be imported in Python via:
    import force_kernel
    import fmm_kernel
"""

import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Detect system (for logging)
system = platform.system()
print(f"Building for {system}")

# Common compiler arguments: C++17, O3, native, fast‐math, and link with OpenMP
base_compile_args = [
    "-std=c++17",
    "-O3",
    "-march=native",
    "-ffast-math",
    "-fopenmp"
]
base_link_args = [
    "-fopenmp"
]

ext_modules = [
    # 1) Direct‐and‐Barnes‐Hut force kernel
    Pybind11Extension(
        name="force_kernel",                  # Module name: import force_kernel
        sources=["force_kernel_full.cpp"],    # Source file (must exist in same folder)
        cxx_std=17,
        extra_compile_args=base_compile_args,
        extra_link_args=base_link_args,
    ),

    # 2) Fast Multipole Method kernel
    Pybind11Extension(
        name="fmm_kernel",                    # Module name: import fmm_kernel
        sources=["fmm_kernel_full.cpp"],      # Source file (must exist in same folder)
        cxx_std=17,
        extra_compile_args=base_compile_args,
        extra_link_args=base_link_args,
    ),
]

setup(
    name="nbody_kernels",
    version="1.0.0",
    description="High-precision N-body simulation kernels (Direct, Barnes‐Hut, FMM) with OpenMP",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

