from setuptools import setup, Extension
import pybind11

# 获取 pybind11 的 include 路径
include_dirs = [
    pybind11.get_include(),
]

fmm_module = Extension(
    name="fmm_omp",
    sources=["fmm_omp.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[
        "-std=c++11",
        "-fopenmp",     # 启用 OpenMP
        "-O3",          # 优化
        "-march=native" # 本机架构优化
    ],
    extra_link_args=[
        "-fopenmp"
    ],
)

setup(
    name="fmm_omp",
    version="0.1",
    author="(Your Name)",
    description="2D Barnes–Hut FMM (monopole only) with fully parallel OpenMP (O(N) parallel).",
    ext_modules=[fmm_module],
    zip_safe=False,
)

