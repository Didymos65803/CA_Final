#!/usr/bin/env python3
"""
setup.py - Optimized build script for high-precision N-body kernels
Usage: python setup.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11
import platform
import os

# Detect system and set compiler flags
system = platform.system()
print(f"Building for {system}")

# Base compiler arguments
base_args = ["-std=c++17", "-O3", "-DNDEBUG"]
base_link_args = []

# System-specific optimizations
if system == "Linux":
    base_args.extend(["-march=native", "-ffast-math", "-fopenmp"])
    base_link_args.extend(["-fopenmp"])
elif system == "Darwin":  # macOS
    base_args.extend(["-march=native", "-ffast-math"])
    # Try to find OpenMP
    omp_paths = ["/opt/homebrew", "/usr/local", "/opt/local"]
    for path in omp_paths:
        if os.path.exists(f"{path}/include/omp.h"):
            base_args.extend(["-Xpreprocessor", "-fopenmp", f"-I{path}/include"])
            base_link_args.extend([f"-L{path}/lib", "-lomp"])
            print(f"Found OpenMP at {path}")
            break
    else:
        print("OpenMP not found - compiling without parallel support")
elif system == "Windows":
    base_args.extend(["/O2", "/openmp"])

print(f"Compiler flags: {base_args}")

# Define extensions
ext_modules = [
    Pybind11Extension(
        "force_kernel",
        ["force_kernel_full.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
        extra_compile_args=base_args,
        extra_link_args=base_link_args,
    ),
    Pybind11Extension(
        "fmm_kernel", 
        ["fmm_kernel_full.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
        extra_compile_args=base_args,
        extra_link_args=base_link_args,
    ),
]

setup(
    name="nbody_kernels",
    version="1.0.0",
    description="High-precision N-body simulation kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
