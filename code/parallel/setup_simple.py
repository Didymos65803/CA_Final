# setup_simple.py - 簡化版編譯設定，確保成功編譯
from setuptools import setup, Extension
import pybind11
import sys
import os

# 基本編譯標誌
base_flags = [
    "-std=c++17",
    "-O3", 
    "-DNDEBUG",
    "-ffast-math"
]

link_flags = []

# 檢查OpenMP支援
try:
    import subprocess
    result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                          input='int main(){return 0;}', text=True,
                          capture_output=True, timeout=5)
    if result.returncode == 0:
        base_flags.append("-fopenmp")
        link_flags.append("-fopenmp")
        print("✓ OpenMP enabled")
    else:
        print("✗ OpenMP not available")
except:
    print("✗ OpenMP check failed")

# 平台特定設定
if sys.platform.startswith('linux'):
    base_flags.append("-fPIC")

print("Compiling with flags:", base_flags)

# 創建擴展
include_dirs = [pybind11.get_include()]

extensions = []

# Force kernel
if os.path.exists("force_kernel_optimized.cpp"):
    force_ext = Extension(
        "force_kernel",
        sources=["force_kernel_optimized.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=base_flags,
        extra_link_args=link_flags
    )
    extensions.append(force_ext)
    print("✓ force_kernel prepared")

# FMM kernel  
if os.path.exists("fmm_kernel_optimized.cpp"):
    fmm_ext = Extension(
        "fmm_kernel",
        sources=["fmm_kernel_optimized.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=base_flags,
        extra_link_args=link_flags
    )
    extensions.append(fmm_ext)
    print("✓ fmm_kernel prepared")

if not extensions:
    print("Error: No source files found!")
    sys.exit(1)

# 設定
setup(
    name="nbody_kernels_simple",
    version="1.0",
    description="Simple N-Body kernels",
    ext_modules=extensions,
    zip_safe=False
)

print("Setup completed. Run: python setup_simple.py build_ext --inplace")
