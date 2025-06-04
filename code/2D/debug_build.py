#!/usr/bin/env python3
"""
debug_build.py
=============
Debug compilation issues and try multiple compilation methods
"""

import subprocess
import sys
import os
import platform
import glob
import shutil

def run_cmd(cmd, description):
    """Run command with detailed output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

def check_requirements():
    """Check if all requirements are available"""
    print("Checking requirements...")
    
    try:
        import pybind11
        print(f"✓ pybind11 version: {pybind11.__version__}")
        print(f"✓ pybind11 includes: {pybind11.get_include()}")
    except ImportError:
        print("✗ pybind11 not found. Install with: pip install pybind11")
        return False
    
    try:
        import numpy
        print(f"✓ numpy version: {numpy.__version__}")
    except ImportError:
        print("✗ numpy not found. Install with: pip install numpy")
        return False
    
    # Check for source files
    required_files = ["force_kernel.cpp", "fmm_kernel.cpp"]
    for f in required_files:
        if os.path.exists(f):
            print(f"✓ Found {f}")
        else:
            print(f"✗ Missing {f}")
            return False
    
    return True

def clean_old_builds():
    """Remove old build artifacts"""
    print("\nCleaning old build artifacts...")
    
    patterns = [
        "*.so", "*.pyd", "*.dll",
        "force_kernel*.so", "fmm_kernel*.so",
        "build/", "*.egg-info/"
    ]
    
    for pattern in patterns:
        for item in glob.glob(pattern):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"Removed file: {item}")
            except Exception as e:
                print(f"Could not remove {item}: {e}")

def method_1_setup_py():
    """Method 1: Use setup.py"""
    print("\nMethod 1: Using setup.py")
    return run_cmd("python setup_fixed.py build_ext --inplace", "Setup.py build")

def method_2_manual_gcc():
    """Method 2: Manual GCC compilation"""
    print("\nMethod 2: Manual GCC compilation")
    
    system = platform.system()
    
    # Get python config
    python_includes = subprocess.check_output([
        sys.executable, "-c", 
        "import pybind11; print(pybind11.get_include())"
    ], text=True).strip()
    
    extension_suffix = subprocess.check_output([
        sys.executable, "-c",
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    ], text=True).strip()
    
    print(f"Python includes: {python_includes}")
    print(f"Extension suffix: {extension_suffix}")
    
    # Base compile command
    base_cmd = f"g++ -O3 -Wall -shared -std=c++17 -fPIC -I{python_includes}"
    
    # Add system-specific flags
    if system == "Linux":
        base_cmd += " -fopenmp"
        link_flags = " -fopenmp"
    elif system == "Darwin":
        # Try different OpenMP locations for macOS
        if os.path.exists("/opt/homebrew/include/omp.h"):
            base_cmd += " -Xpreprocessor -fopenmp -I/opt/homebrew/include"
            link_flags = " -L/opt/homebrew/lib -lomp"
        elif os.path.exists("/usr/local/include/omp.h"):
            base_cmd += " -Xpreprocessor -fopenmp -I/usr/local/include"
            link_flags = " -L/usr/local/lib -lomp"
        else:
            print("Warning: OpenMP not found, compiling without parallel support")
            link_flags = ""
    else:
        link_flags = ""
    
    # Compile force_kernel
    cmd1 = f"{base_cmd} force_kernel.cpp -o force_kernel{extension_suffix}{link_flags}"
    success1 = run_cmd(cmd1, "Compile force_kernel")
    
    # Compile fmm_kernel
    cmd2 = f"{base_cmd} fmm_kernel.cpp -o fmm_kernel{extension_suffix}{link_flags}"
    success2 = run_cmd(cmd2, "Compile fmm_kernel")
    
    return success1 and success2

def method_3_no_openmp():
    """Method 3: Compile without OpenMP as fallback"""
    print("\nMethod 3: Fallback without OpenMP")
    
    # Get python config
    python_includes = subprocess.check_output([
        sys.executable, "-c", 
        "import pybind11; print(pybind11.get_include())"
    ], text=True).strip()
    
    extension_suffix = subprocess.check_output([
        sys.executable, "-c",
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    ], text=True).strip()
    
    base_cmd = f"g++ -O3 -Wall -shared -std=c++17 -fPIC -I{python_includes}"
    
    # Compile without OpenMP
    cmd1 = f"{base_cmd} force_kernel.cpp -o force_kernel{extension_suffix}"
    success1 = run_cmd(cmd1, "Compile force_kernel (no OpenMP)")
    
    cmd2 = f"{base_cmd} fmm_kernel.cpp -o fmm_kernel{extension_suffix}"
    success2 = run_cmd(cmd2, "Compile fmm_kernel (no OpenMP)")
    
    return success1 and success2

def test_import():
    """Test importing the compiled modules"""
    print("\nTesting imports...")
    
    try:
        import force_kernel
        print("✓ force_kernel imported successfully")
        
        import fmm_kernel
        print("✓ fmm_kernel imported successfully")
        
        # Quick functionality test
        import numpy as np
        x = np.array([1.0, 2.0])
        y = np.array([0.0, 1.0])
        m = np.array([1.0, 1.0])
        
        ax, ay = force_kernel.direct_omp(x, y, m)
        print(f"✓ Direct solver test: output shape {ax.shape}")
        
        ax, ay = force_kernel.bh_omp(x, y, m, 10.0)
        print(f"✓ BH solver test: output shape {ax.shape}")
        
        ax, ay = fmm_kernel.fmm_omp(x, y, m, 10.0)
        print(f"✓ FMM solver test: output shape {ax.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import/test failed: {e}")
        return False

def main():
    """Main debug procedure"""
    print("N-body Kernel Debug Compilation")
    print("="*60)
    
    if not check_requirements():
        print("✗ Requirements check failed")
        return False
    
    clean_old_builds()
    
    # Try different compilation methods
    methods = [
        ("Setup.py with fixed configuration", method_1_setup_py),
        ("Manual GCC compilation", method_2_manual_gcc),
        ("Fallback without OpenMP", method_3_no_openmp)
    ]
    
    for name, method in methods:
        print(f"\n{'='*60}")
        print(f"TRYING: {name}")
        print('='*60)
        
        if method():
            print(f"✓ {name} succeeded!")
            if test_import():
                print("\n" + "="*60)
                print("✓ SUCCESS: Compilation and testing complete!")
                print("="*60)
                return True
            else:
                print(f"✗ {name} compiled but testing failed")
        else:
            print(f"✗ {name} failed")
    
    print("\n" + "="*60)
    print("✗ ALL METHODS FAILED")
    print("Please check:")
    print("1. C++ compiler installation (g++ or clang++)")
    print("2. Python development headers")
    print("3. pybind11 installation")
    print("="*60)
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
