# setup_minimal.py - Setup for minimal kernel testing
from setuptools import setup, Extension
import pybind11
import subprocess
import sys

def check_openmp():
    """Check if OpenMP is available and working"""
    try:
        # Test compilation with OpenMP
        result = subprocess.run([
            'gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'
        ], input='#include <omp.h>\nint main(){return omp_get_max_threads();}', 
           text=True, capture_output=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def get_compiler_info():
    """Get compiler information"""
    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
        print("GCC Version:")
        print(result.stdout.split('\n')[0])
    except:
        pass
    
    try:
        result = subprocess.run(['gcc', '-fopenmp', '-dM', '-E', '-'], 
                              input='', text=True, capture_output=True)
        if '_OPENMP' in result.stdout:
            for line in result.stdout.split('\n'):
                if '_OPENMP' in line:
                    print(f"OpenMP Version: {line}")
                    break
    except:
        pass

# Check system
print("System Check:")
print("=" * 40)
get_compiler_info()

openmp_available = check_openmp()
print(f"OpenMP available: {openmp_available}")

if not openmp_available:
    print("WARNING: OpenMP not available. Parallel scaling will not work.")
    print("Try installing: sudo apt-get install libomp-dev")

# Compilation flags
compile_flags = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-ffast-math",
    "-march=native",
    "-fPIC"
]

link_flags = []

# Add OpenMP if available
if openmp_available:
    compile_flags.extend(["-fopenmp"])
    link_flags.append("-fopenmp")
    print("✓ OpenMP flags added")
else:
    print("✗ Compiling without OpenMP")

print(f"Compile flags: {' '.join(compile_flags)}")
print(f"Link flags: {' '.join(link_flags)}")

# Create minimal extension
minimal_ext = Extension(
    "minimal_force_kernel",
    sources=["minimal_force_kernel.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=compile_flags,
    extra_link_args=link_flags
)

# Setup
setup(
    name="minimal_nbody_test",
    version="1.0",
    description="Minimal N-Body kernel for parallel testing",
    ext_modules=[minimal_ext],
    zip_safe=False,
    python_requires=">=3.7"
)

print("\nTo build:")
print("python setup_minimal.py build_ext --inplace")
print("\nTo test:")
print("python simple_parallel_test.py")
