"""Compile the optimised kernels with OpenMP."""
from setuptools import setup, Extension
import pybind11, sys

compile_args = [
    "-std=c++17", "-O3", "-ffast-math", "-funroll-loops",
    "-fopenmp", "-DNDEBUG", "-fPIC"
]
link_args = ["-fopenmp"]

exts = [
    Extension(
        "force_kernel_opt",
        ["force_kernel_optimized.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    ),
    Extension(
        "fmm_kernel_opt",
        ["fmm_kernel_optimized.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    ),
]

setup(
    name="nbody_kernels_opt",
    version="3.0",
    description="Optimised N‑body kernels (direct + Barnes–Hut) with proper parallel scaling",
    ext_modules=exts,
    zip_safe=False,
    python_requires=">=3.8",
)

print("\nBuild:  python setup_optimized.py build_ext --inplace")
