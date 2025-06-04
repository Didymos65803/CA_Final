# setup_openmp.py  –  build both kernels with OpenMP
from setuptools import setup, Extension
import pybind11

compile_flags = [
    "-std=c++17", "-O3", "-ffast-math", "-funroll-loops",
    "-march=native", "-fopenmp", "-fPIC"
]
link_flags = ["-fopenmp"]

ext_modules = [
    Extension(
        "fmm_openmp",
        ["fmm_openmp.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        language="c++",
    ),
    Extension(
        "force_openmp",
        ["force_openmp.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        language="c++",
    ),
]

setup(
    name="nbody_openmp",
    version="1.0",
    description="Direct + Barnes–Hut FMM kernels (OpenMP)",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)


