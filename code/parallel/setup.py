# setup.py
# --------------------------------------------------
# Build script for PyBind11 + OpenMP:
#   • force_kernel (direct O(N^2) kernel)
#   • fmm_kernel   (Fast Multipole Method, O(N))
#
# Usage:
#    python3 setup.py build_ext --inplace
# --------------------------------------------------

from setuptools import setup, Extension
import pybind11

include_dirs = [pybind11.get_include()]

# Common compiler / linker flags:
extra_compile_args = ["-std=c++17", "-O3", "-march=native", "-ffast-math", "-fopenmp"]
extra_link_args    = ["-fopenmp"]

force_ext = Extension(
    name="force_kernel",
    sources=["force_kernel_full.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

fmm_ext = Extension(
    name="fmm_kernel",
    sources=["fmm_kernel_full.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

setup(
    name="nbody_kernels",
    version="1.0",
    author="(Your Name)",
    description="PyBind11 + OpenMP kernels for 2D N-Body: direct & FMM",
    ext_modules=[force_ext, fmm_ext],
    zip_safe=False,
)

