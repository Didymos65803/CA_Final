#!/usr/bin/env python3
"""
diagnose_imports.py - Fix import issues
"""
import os
import sys

def diagnose_imports():
    print("=== Import Diagnosis ===")
    
    # Check for compiled modules
    import glob
    so_files = glob.glob("*.so")
    print(f"Found .so files: {so_files}")
    
    # Test different import possibilities
    modules_to_test = ['force_kernel', 'force_kernel_full', 'fmm_kernel', 'fmm_kernel_full']
    
    working_modules = {}
    
    for module in modules_to_test:
        try:
            exec(f"import {module}")
            print(f"✓ {module} imports successfully")
            
            # Test if it has the expected functions
            mod = sys.modules[module]
            if hasattr(mod, 'direct_omp'):
                print(f"  ✓ {module}.direct_omp found")
            if hasattr(mod, 'bh_omp'):
                print(f"  ✓ {module}.bh_omp found")  
            if hasattr(mod, 'fmm_omp'):
                print(f"  ✓ {module}.fmm_omp found")
                
            working_modules[module] = mod
            
        except ImportError:
            print(f"❌ {module} not found")
    
    print(f"\nWorking modules: {list(working_modules.keys())}")
    
    # Generate fix suggestions
    if 'force_kernel_full' in working_modules and 'force_kernel' not in working_modules:
        print("\n=== SOLUTION ===")
        print("Your module is named 'force_kernel_full' but tests expect 'force_kernel'")
        print("\nOption 1 - Quick fix (change test imports):")
        print("  sed -i 's/from force_kernel import/from force_kernel_full import/g' comprehensive_test.py")
        print("  sed -i 's/from force_kernel import/from force_kernel_full import/g' main_program_parallel_final.py")
        
        print("\nOption 2 - Proper fix (rename module in setup.py):")
        print("  Edit setup.py and change 'force_kernel_full' to 'force_kernel'")
        print("  Then: python setup.py build_ext --inplace")
        
    elif 'force_kernel' in working_modules:
        print("\n✓ force_kernel imports correctly!")
        
    # Test a simple function call
    if 'force_kernel_full' in working_modules:
        print("\n=== Testing force_kernel_full functions ===")
        try:
            import numpy as np
            mod = working_modules['force_kernel_full']
            
            x = np.array([0.0, 1.0])
            y = np.array([0.0, 0.0])
            m = np.array([1.0, 1.0])
            
            if hasattr(mod, 'direct_omp'):
                ax, ay = mod.direct_omp(x, y, m, G=1.0, soft=0.01)
                print(f"✓ direct_omp works: force = ({ax[0]:.3e}, {ay[0]:.3e})")
                
            if hasattr(mod, 'bh_omp'):
                ax, ay = mod.bh_omp(x, y, m, domain=10.0, theta=0.5, G=1.0, soft=0.01)
                print(f"✓ bh_omp works: force = ({ax[0]:.3e}, {ay[0]:.3e})")
                
        except Exception as e:
            print(f"❌ Function test failed: {e}")

if __name__ == "__main__":
    diagnose_imports()
