# setup_final.py - Final working setup for the N-Body kernels
from setuptools import setup, Extension
import pybind11
import sys
import os

def check_openmp():
    """Check if OpenMP is available"""
    try:
        import subprocess
        result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# Compiler flags
compile_flags = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-ffast-math",
    "-funroll-loops",
    "-fPIC"
]

link_flags = []

# Add OpenMP if available
if check_openmp():
    compile_flags.extend(["-fopenmp"])
    link_flags.append("-fopenmp")
    print("✓ OpenMP enabled")
else:
    print("✗ OpenMP not available")

print("Compilation flags:", compile_flags)

# Include directories
include_dirs = [pybind11.get_include()]

# Extensions
extensions = []

# Force kernel
if os.path.exists("force_kernel_fixed_final.cpp"):
    force_ext = Extension(
        "force_kernel",
        sources=["force_kernel_fixed_final.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags
    )
    extensions.append(force_ext)
    print("✓ force_kernel prepared")

# FMM kernel
if os.path.exists("fmm_kernel_fixed_final.cpp"):
    fmm_ext = Extension(
        "fmm_kernel",
        sources=["fmm_kernel_fixed_final.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags
    )
    extensions.append(fmm_ext)
    print("✓ fmm_kernel prepared")

if not extensions:
    print("Error: No source files found!")
    print("Expected files:")
    print("  - force_kernel_fixed_final.cpp")
    print("  - fmm_kernel_fixed_final.cpp")
    sys.exit(1)

# Setup
setup(
    name="nbody_kernels_final",
    version="2.0",
    description="Working N-Body simulation kernels with proper parallelization",
    ext_modules=extensions,
    zip_safe=False,
    python_requires=">=3.7"
)

print("\nSetup completed!")
print("Next steps:")
print("1. Compile: python setup_final.py build_ext --inplace")
print("2. Test: python test_final.py")
print("3. Run: python main_program_final.py")
