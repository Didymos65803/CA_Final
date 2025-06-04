# setup_fixed.py - 修正編譯錯誤的版本
from setuptools import setup, Extension
import pybind11
import sys
import os
import subprocess

def check_openmp():
    """檢查OpenMP支援"""
    try:
        result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def get_compile_flags():
    """獲取安全的編譯標誌"""
    
    # 基礎優化標誌（保守設定）
    base_flags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG", 
        "-ffast-math",
        "-funroll-loops"
    ]
    
    link_flags = []
    
    # 檢查並添加OpenMP支援
    if check_openmp():
        base_flags.extend(["-fopenmp"])
        link_flags.append("-fopenmp")
        print("✓ OpenMP support enabled")
    else:
        print("✗ OpenMP not available")
    
    # 平台特定的安全優化
    if sys.platform != "win32":
        base_flags.extend(["-fPIC"])
        
    # 添加向量化相關標誌（如果支援）
    try:
        result = subprocess.run(['gcc', '-mavx', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            base_flags.append("-mavx")
            print("✓ AVX support enabled")
    except:
        pass
    
    return base_flags, link_flags

# 主要設定
print("Fixed N-Body Kernels Compilation")
print("=" * 40)

# 獲取編譯標誌
compile_args, link_args = get_compile_flags()

include_dirs = [pybind11.get_include()]

# 檢查源文件
source_files = {
    "force_kernel": "force_kernel_optimized.cpp",
    "fmm_kernel": "fmm_kernel_optimized.cpp"
}

extensions = []
for name, source in source_files.items():
    if os.path.exists(source):
        ext = Extension(
            name=name,
            sources=[source],
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
        extensions.append(ext)
        print(f"✓ {name} extension prepared")
    else:
        print(f"✗ {source} not found")

if not extensions:
    print("Error: No source files found!")
    sys.exit(1)

setup(
    name="nbody_kernels_fixed",
    version="5.1",
    author="Fixed Version",
    description="Fixed PyBind11 + OpenMP kernels for 2D N-Body simulation",
    ext_modules=extensions,
    zip_safe=False,
    python_requires=">=3.7"
)

print("\nCompilation setup complete!")
print("Run: python setup_fixed.py build_ext --inplace")
