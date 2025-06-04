# setup_improved.py – Enhanced build configuration for optimal OpenMP performance
from setuptools import setup, Extension
import pybind11
import platform
import os

def get_optimal_compile_flags():
    """Get optimal compilation flags based on platform and compiler."""
    base_flags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",          # Disable debug assertions
        "-ffast-math",       # Aggressive floating point optimizations
        "-funroll-loops",    # Loop unrolling
        "-fPIC"              # Position independent code
    ]
    
    # Platform-specific optimizations
    if platform.system() == "Linux":
        base_flags.extend([
            "-march=native",     # Use all available CPU instructions
            "-mtune=native",     # Optimize for current CPU
            "-fopenmp",          # OpenMP support
        ])
    elif platform.system() == "Darwin":  # macOS
        base_flags.extend([
            "-march=native",
            "-Xpreprocessor", "-fopenmp",  # macOS OpenMP syntax
        ])
    elif platform.system() == "Windows":
        base_flags.extend([
            "/openmp",           # MSVC OpenMP flag
            "/O2",              # MSVC optimization
        ])
    else:
        base_flags.append("-fopenmp")
    
    # Additional performance flags
    performance_flags = [
        "-fomit-frame-pointer",   # Remove frame pointer for better performance
        "-fno-strict-aliasing",   # Safer optimizations
        "-mfpmath=sse",          # Use SSE for floating point math (x86/x64)
        "-msse2",                # Enable SSE2 instructions
    ]
    
    # Only add x86-specific flags if we're on x86/x64
    import subprocess
    try:
        arch = subprocess.check_output(['uname', '-m'], text=True).strip()
        if arch in ['x86_64', 'amd64', 'i386', 'i686']:
            base_flags.extend(performance_flags)
    except:
        pass  # Skip if we can't determine architecture
    
    return base_flags

def get_link_flags():
    """Get optimal linking flags."""
    if platform.system() == "Linux":
        return ["-fopenmp"]
    elif platform.system() == "Darwin":
        # macOS requires explicit OpenMP library linking
        return ["-lomp"]
    elif platform.system() == "Windows":
        return []  # MSVC handles this automatically
    else:
        return ["-fopenmp"]

def get_libraries():
    """Get required libraries."""
    libs = []
    if platform.system() == "Darwin":
        libs.append("omp")  # OpenMP library on macOS
    return libs

def get_library_dirs():
    """Get library directories."""
    lib_dirs = []
    
    # Common locations for OpenMP libraries
    potential_dirs = [
        "/usr/local/lib",
        "/opt/homebrew/lib",  # Homebrew on Apple Silicon
        "/usr/lib/gcc/x86_64-linux-gnu/9",  # Ubuntu/Debian
        "/usr/lib64",         # CentOS/RHEL
    ]
    
    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            lib_dirs.append(dir_path)
    
    return lib_dirs

# Get optimized compilation settings
compile_flags = get_optimal_compile_flags()
link_flags = get_link_flags()
libraries = get_libraries()
library_dirs = get_library_dirs()

print("Compilation configuration:")
print(f"  Platform: {platform.system()}")
print(f"  Compile flags: {' '.join(compile_flags)}")
print(f"  Link flags: {' '.join(link_flags)}")
print(f"  Libraries: {libraries}")
print(f"  Library dirs: {library_dirs}")

ext_modules = [
    Extension(
        "fmm_openmp",
        ["fmm_openmp.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
    ),
]

setup(
    name="fmm_openmp_optimized",
    version="2.0",
    description="Highly optimized Barnes–Hut FMM with OpenMP",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)
