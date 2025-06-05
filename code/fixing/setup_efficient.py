from setuptools import setup, Extension
import pybind11
import os

# 檢查檔案是否存在
cpp_file = "fmm_efficient_on.cpp"
if not os.path.exists(cpp_file):
    print(f"Error: {cpp_file} not found!")
    print("Please save the optimized code as fmm_efficient_on.cpp")
    exit(1)

# 獲取 pybind11 的 include 路徑
include_dirs = [
    pybind11.get_include(),
]

fmm_efficient_module = Extension(
    name="fmm_true_on",  # 保持模組名稱一樣，這樣測試代碼不用改
    sources=[cpp_file],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[
        "-std=c++14",
        "-fopenmp",         # 啟用 OpenMP
        "-O3",              # 高度優化
        "-march=native",    # 針對本機架構優化
        "-ffast-math",      # 數學函數優化
        "-funroll-loops",   # 循環展開
        "-DWITH_OPENMP"
    ],
    extra_link_args=[
        "-fopenmp"
    ],
)

setup(
    name="fmm_efficient",
    version="0.2",
    author="FMM Developer",
    description="Optimized O(N) Fast Multipole Method with enhanced parallel performance",
    ext_modules=[fmm_efficient_module],
    zip_safe=False,
)

print(f"Optimized FMM compilation completed! ({cpp_file} -> fmm_true_on.so)")
