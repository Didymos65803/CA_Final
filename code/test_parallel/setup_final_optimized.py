# setup_final_optimized.py
from setuptools import setup, Extension
import pybind11

print("Building Final Optimized Kernel for Intel Xeon Cascadelake")

compile_flags = [
    "-std=c++17",
    "-O3", 
    "-DNDEBUG",
    "-ffast-math",
    "-funroll-loops",
    "-fopenmp",
    "-fPIC",
    "-falign-functions=32",
    "-falign-loops=32",
    "-fprefetch-loop-arrays",
    "-ftree-vectorize"
]

link_flags = ["-fopenmp"]

final_optimized_ext = Extension(
    "final_optimized_kernel",
    sources=["final_optimized_kernel.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=compile_flags,
    extra_link_args=link_flags
)

setup(
    name="final_optimized_nbody",
    version="3.0",
    description="Final optimized N-Body kernel",
    ext_modules=[final_optimized_ext],
    zip_safe=False,
    python_requires=">=3.7"
)
