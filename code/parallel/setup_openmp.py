# setup_openmp.py
# --------------------------------------------------------------------------------
# Build two pybind11 extensions using setuptools:
#   1) force_openmp   (O(N^2) direct solver)
#   2) fmm_openmp     (θ-aware Barnes–Hut FMM solver)
#
# Usage: python3 setup_openmp.py build_ext --inplace
# --------------------------------------------------------------------------------
from setuptools import setup, Extension
import pybind11

# 1) Compile the direct solver module
force_mod = Extension(
    name="force_openmp",
    sources=["force_openmp.cpp"],
    include_dirs=[
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"]
)

# 2) Compile the FMM solver module
fmm_mod = Extension(
    name="fmm_openmp",
    sources=["fmm_openmp.cpp"],
    include_dirs=[
        pybind11.get_include(),
        pybind11.get_include(user=True)
    ],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"]
)

setup(
    name="openmp_fmm_example",
    version="0.1",
    author="Meiji",
    description="Direct & FMM N-body with OpenMP",
    ext_modules=[force_mod, fmm_mod],
)

