from setuptools import setup, Extension
import pybind11
import os

# 檢查檔案是否存在
if not os.path.exists("fmm_true_on.cpp"):
    print("Error: fmm_true_on.cpp not found!")
    exit(1)

# 獲取 pybind11 的 include 路徑
include_dirs = [
    pybind11.get_include(),
]

fmm_true_on_module = Extension(
    name="fmm_true_on",
    sources=["fmm_true_on.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[
        "-std=c++14",
        "-fopenmp",     # 啟用 OpenMP
        "-O3",          # 優化
        "-march=native", # 本機架構優化
        "-DWITH_OPENMP"
    ],
    extra_link_args=[
        "-fopenmp"
    ],
)

setup(
    name="fmm_true_on",
    version="0.1",
    author="FMM Developer",
    description="True O(N) Fast Multipole Method with complete M2L implementation",
    ext_modules=[fmm_true_on_module],
    zip_safe=False,
)

print("fmm_true_on compilation completed!")
