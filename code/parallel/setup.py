# setup.py
# Optimized version with better compiler flags

from setuptools import setup, Extension
import pybind11
import sys
import os

def has_openmp():
    try:
        import subprocess
        result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True)
        return result.returncode == 0
    except:
        return False

include_dirs = [pybind11.get_include()]

# Optimized compiler flags for better performance
base_compile_args = [
    "-std=c++17", 
    "-O3", 
    "-DNDEBUG", 
    "-ffast-math",
    "-funroll-loops",
    "-fno-signed-zeros",
    "-fno-trapping-math"
]

base_link_args = []

# Add OpenMP with optimized settings
if has_openmp():
    base_compile_args.extend(["-fopenmp", "-DOMP_DYNAMIC=false"])
    base_link_args.append("-fopenmp")
    print("OpenMP support detected with optimizations")
else:
    print("Warning: OpenMP not available")

# Platform-specific optimizations
if sys.platform != "win32":
    base_compile_args.extend(["-march=native", "-mtune=native"])

# Add vectorization flags
base_compile_args.extend([
    "-ftree-vectorize",
    "-fopt-info-vec-optimized" if sys.platform != "win32" else ""
])

force_ext = Extension(
    name="force_kernel",
    sources=["force_kernel_full.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[arg for arg in base_compile_args if arg],
    extra_link_args=base_link_args
)

fmm_ext = Extension(
    name="fmm_kernel",
    sources=["fmm_kernel_full.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[arg for arg in base_compile_args if arg],
    extra_link_args=base_link_args
)

setup(
    name="nbody_kernels",
    version="2.1",
    author="Optimized Version",
    description="Optimized PyBind11 + OpenMP kernels for 2D N-Body",
    ext_modules=[force_ext, fmm_ext],
    zip_safe=False,
)

