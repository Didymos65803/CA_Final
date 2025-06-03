#!/usr/bin/env python3
"""
setup_fixed.py - Robust build script for N-body simulation C++ kernels
Usage:
    python setup_fixed.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11
import platform
import os

# Detect system and set appropriate compiler flags
system = platform.system()
print(f"Detected system: {system}")

# Base compiler args
base_args = ["-std=c++17", "-O3"]
base_link_args = []

# System-specific flags
if system == "Linux":
    base_args.extend(["-fopenmp", "-fPIC"])
    base_link_args.extend(["-fopenmp"])
elif system == "Darwin":  # macOS
    # Try to use homebrew's libomp if available
    if os.path.exists("/opt/homebrew/include/omp.h"):
        base_args.extend(["-Xpreprocessor", "-fopenmp", 
                         "-I/opt/homebrew/include"])
        base_link_args.extend(["-L/opt/homebrew/lib", "-lomp"])
    elif os.path.exists("/usr/local/include/omp.h"):
        base_args.extend(["-Xpreprocessor", "-fopenmp", 
                         "-I/usr/local/include"])
        base_link_args.extend(["-L/usr/local/lib", "-lomp"])
    else:
        print("Warning: OpenMP not found, compiling without parallel support")
elif system == "Windows":
    base_args.extend(["/openmp"])

print(f"Compiler args: {base_args}")
print(f"Linker args: {base_link_args}")

# Define the extensions
ext_modules = [
    Pybind11Extension(
        "force_kernel",
        ["force_kernel.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../include",
            pybind11.get_include()
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=base_args,
        extra_link_args=base_link_args,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
    Pybind11Extension(
        "fmm_kernel", 
        ["fmm_kernel.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../include",
            pybind11.get_include()
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=base_args,
        extra_link_args=base_link_args,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="nbody_kernels",
    version="0.1.0",
    author="N-body Simulation",
    description="Fast N-body simulation kernels",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0",
    ],
)
