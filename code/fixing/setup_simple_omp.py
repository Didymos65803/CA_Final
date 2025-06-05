from setuptools import setup, Extension
import pybind11

# 如果你已經有 fmm_omp.cpp 從其他地方，可以用 cp 複製
# cp ../path/to/fmm_omp.cpp .

fmm_module = Extension(
    name="fmm_omp",
    sources=["fmm_omp.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++14", "-fopenmp", "-O3"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="fmm_omp",
    ext_modules=[fmm_module],
    zip_safe=False,
)
