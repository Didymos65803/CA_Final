#!/usr/bin/env python3
"""
rebuild_and_test.py
==================
Script to clean, rebuild, and test the N-body simulation kernels
"""

import os
import subprocess
import sys
import shutil
import glob

def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} successful")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"✗ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception during {description}: {e}")
        return False
    return True

def clean_build_artifacts():
    """Remove old build artifacts"""
    print("\nCleaning old build artifacts...")
    
    # Patterns to remove
    patterns = [
        "*.so",
        "*.pyd", 
        "*.dll",
        "force_kernel*.so",
        "fmm_kernel*.so",
        "build/",
        "*.egg-info/",
        "__pycache__/",
        "*.pyc"
    ]
    
    removed_count = 0
    for pattern in patterns:
        for item in glob.glob(pattern):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {item}: {e}")
    
    print(f"Cleaned {removed_count} items")

def test_import():
    """Test importing the compiled modules"""
    print("\nTesting module imports...")
    
    try:
        print("Testing force_kernel...")
        import force_kernel
        print("✓ force_kernel imported successfully")
        
        print("Testing fmm_kernel...")
        import fmm_kernel  
        print("✓ fmm_kernel imported successfully")
        
        # Test basic functionality
        print("Testing basic functionality...")
        import numpy as np
        
        # Create small test case
        N = 10
        x = np.random.randn(N)
        y = np.random.randn(N)
        m = np.ones(N)
        
        # Test direct solver
        ax, ay = force_kernel.direct_omp(x, y, m)
        print(f"✓ Direct solver works, output shape: {ax.shape}")
        
        # Test BH solver
        ax, ay = force_kernel.bh_omp(x, y, m, 10.0)
        print(f"✓ BH solver works, output shape: {ax.shape}")
        
        # Test FMM solver
        ax, ay = fmm_kernel.fmm_omp(x, y, m, 10.0)
        print(f"✓ FMM solver works, output shape: {ax.shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Main rebuild and test procedure"""
    print("="*60)
    print("N-body Simulation Kernel Rebuild and Test")
    print("="*60)
    
    # Step 1: Clean
    clean_build_artifacts()
    
    # Step 2: Check if files exist
    required_files = ["force_kernel_fixed.cpp", "fmm_kernel_fixed.cpp", "setup.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n✗ Missing required files: {missing_files}")
        print("Please make sure all required files are in the current directory:")
        for f in required_files:
            print(f"  - {f}")
        return False
    
    # Step 3: Rename fixed files to replace originals
    print("\nUsing fixed versions of the files...")
    try:
        if os.path.exists("force_kernel_fixed.cpp"):
            shutil.copy2("force_kernel_fixed.cpp", "force_kernel.cpp")
            print("✓ Updated force_kernel.cpp")
            
        if os.path.exists("fmm_kernel_fixed.cpp"):
            shutil.copy2("fmm_kernel_fixed.cpp", "fmm_kernel.cpp")
            print("✓ Updated fmm_kernel.cpp")
    except Exception as e:
        print(f"✗ Error updating files: {e}")
        return False
    
    # Step 4: Build
    success = run_command("python setup.py build_ext --inplace", "Building extensions")
    if not success:
        print("\n✗ Build failed. Trying alternative compilation...")
        
        # Try manual g++ compilation
        cmds = [
            "g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) force_kernel.cpp -o force_kernel$(python3-config --extension-suffix) -fopenmp",
            "g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) fmm_kernel.cpp -o fmm_kernel$(python3-config --extension-suffix) -fopenmp"
        ]
        
        for cmd in cmds:
            if not run_command(cmd, "Manual compilation"):
                print("Manual compilation also failed")
                return False
    
    # Step 5: Test
    if test_import():
        print("\n" + "="*60)
        print("✓ SUCCESS: All kernels built and tested successfully!")
        print("You can now run: python main_program_parallel_fixed.py")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("✗ FAILED: Kernels built but testing failed")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
