# setup_optimized.py - 最優化的編譯設定（修正版）
from setuptools import setup, Extension
import pybind11
import sys
import os
import subprocess

def check_compiler_support():
    """檢查編譯器支持的功能"""
    features = {
        'openmp': False,
        'avx': False,
        'native': False
    }
    
    try:
        # 檢查OpenMP支援
        result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True, timeout=10)
        features['openmp'] = (result.returncode == 0)
        
        # 檢查AVX支援
        result = subprocess.run(['gcc', '-mavx', '-x', 'c', '-', '-o', '/dev/null'],
                              input='int main(){return 0;}', text=True,
                              capture_output=True, timeout=10)
        features['avx'] = (result.returncode == 0)
        
        # 檢查native支援
        if sys.platform != "win32":
            result = subprocess.run(['gcc', '-march=native', '-x', 'c', '-', '-o', '/dev/null'],
                                  input='int main(){return 0;}', text=True,
                                  capture_output=True, timeout=10)
            features['native'] = (result.returncode == 0)
        
    except Exception as e:
        print(f"Warning: Compiler feature detection failed: {e}")
    
    return features

def get_optimization_flags():
    """根據編譯器支持獲取最優編譯標誌"""
    features = check_compiler_support()
    
    # 基礎優化標誌（保守且安全）
    base_flags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-ffast-math",
        "-funroll-loops"
    ]
    
    link_flags = []
    
    # 根據支持的功能添加標誌
    if features['openmp']:
        base_flags.extend([
            "-fopenmp",
            "-DOMP_DYNAMIC=false"
        ])
        link_flags.append("-fopenmp")
        print("✓ OpenMP support enabled")
    else:
        print("✗ OpenMP not supported")
    
    if features['avx']:
        base_flags.append("-mavx")
        print("✓ AVX support enabled")
    
    if features['native'] and sys.platform != "win32":
        base_flags.extend(["-mtune=native"])
        print("✓ Native tuning enabled")
    
    # 平台特定優化
    if sys.platform.startswith('linux'):
        base_flags.extend(["-fPIC"])
    elif sys.platform == "darwin":
        base_flags.extend(["-fPIC"])
    
    return base_flags, link_flags

def create_extension(name, source_file):
    """創建優化的擴展模組"""
    compile_args, link_args = get_optimization_flags()
    
    include_dirs = [pybind11.get_include()]
    
    # 添加額外的包含目錄
    if os.path.exists('/usr/local/include'):
        include_dirs.append('/usr/local/include')
    
    ext = Extension(
        name=name,
        sources=[source_file],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
    
    return ext

# 主要設定
if __name__ == "__main__":
    print("Optimized N-Body Kernels Compilation (Fixed Version)")
    print("=" * 55)
    
    # 檢查必要文件
    required_files = [
        "fmm_kernel_optimized.cpp",
        "force_kernel_optimized.cpp"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing source files: {missing_files}")
        print("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.cpp'):
                print(f"  {f}")
        sys.exit(1)
    
    # 創建擴展模組
    extensions = [
        create_extension("force_kernel", "force_kernel_optimized.cpp"),
        create_extension("fmm_kernel", "fmm_kernel_optimized.cpp")
    ]
    
    # 設定包信息
    setup(
        name="nbody_kernels_optimized_fixed",
        version="5.2",
        author="Fixed Optimized Version",
        description="Fixed PyBind11 + OpenMP kernels for 2D N-Body simulation",
        long_description="""
        修正版高度優化的2D N-body模擬核心，特點：
        - 解決false sharing問題
        - 改善cache locality
        - 移除problematic aligned clauses
        - 工作竊取式並行化
        - 塊式算法優化
        - 多級並行化策略
        """,
        ext_modules=extensions,
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "pybind11>=2.6.0",
            "numpy>=1.18.0"
        ]
    )
    
    print("\n" + "=" * 55)
    print("Compilation setup completed!")
    print("Optimizations applied:")
    
    features = check_compiler_support()
    for feature, supported in features.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {feature.upper()}")
    
    print("\nNext steps:")
    print("1. Compile: python setup_optimized.py build_ext --inplace")
    print("2. Test: python test_compilation.py")
    print("3. Run: python main_program_simple.py")
