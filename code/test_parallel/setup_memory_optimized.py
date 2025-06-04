# setup_memory_optimized.py
from setuptools import setup, Extension
import pybind11

print("Building Memory-Optimized Kernel")
print("=" * 40)

# Optimized flags for memory-bound problems
compile_flags = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-march=native",
    "-mtune=native",
    "-ffast-math",
    "-fopenmp",
    "-fPIC",
    # Memory optimization flags
    "-falign-functions=32",
    "-falign-loops=32",
    "-fprefetch-loop-arrays",
    "-funroll-loops",
    # Cache optimization
    "-mcx16",
    "-msse4.2",
    "-mavx",
    "-mavx2" if "avx2" in open("/proc/cpuinfo").read() else ""
]

# Remove empty flags
compile_flags = [f for f in compile_flags if f]

link_flags = ["-fopenmp"]

print(f"Compile flags: {' '.join(compile_flags)}")

# Create extension
memory_ext = Extension(
    "memory_optimized_kernel",
    sources=["memory_optimized_kernel.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=compile_flags,
    extra_link_args=link_flags
)

setup(
    name="memory_optimized_nbody",
    version="1.0",
    description="Memory-optimized N-Body kernel for high-latency systems",
    ext_modules=[memory_ext],
    zip_safe=False,
    python_requires=">=3.7"
)

print("\nBuild commands:")
print("python setup_memory_optimized.py build_ext --inplace")
print("python test_memory_optimized.py")
