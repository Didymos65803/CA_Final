#!/usr/bin/env python3
"""diagnostic_test.py - Comprehensive OpenMP diagnostic and testing"""

import os
import sys
import time
import numpy as np
import subprocess

def compile_diagnostic_version():
    """Compile the diagnostic version of the FMM code."""
    print("=== Compiling Diagnostic Version ===")
    
    # First, backup the original file
    if os.path.exists("fmm_openmp.cpp"):
        os.rename("fmm_openmp.cpp", "fmm_openmp_original.cpp")
        print("Backed up original fmm_openmp.cpp")
    
    # Copy diagnostic version
    if os.path.exists("fmm_openmp_diagnostic.cpp"):
        os.rename("fmm_openmp_diagnostic.cpp", "fmm_openmp.cpp")
        print("Using diagnostic version")
    
    # Compile
    try:
        result = subprocess.run([
            sys.executable, "setup_improved.py", "build_ext", "--inplace"
        ], capture_output=True, text=True, check=True)
        print("✓ Compilation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def test_openmp_environment():
    """Test OpenMP environment variables and settings."""
    print("\n=== OpenMP Environment Test ===")
    
    # Check environment variables
    openmp_vars = ['OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_DYNAMIC']
    print("Environment variables:")
    for var in openmp_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Set optimal values if not set
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '4'
        print("  Set OMP_NUM_THREADS=4")
    
    if 'OMP_PROC_BIND' not in os.environ:
        os.environ['OMP_PROC_BIND'] = 'true'
        print("  Set OMP_PROC_BIND=true")

def run_basic_openmp_test():
    """Run basic OpenMP functionality tests."""
    print("\n=== Basic OpenMP Test ===")
    
    try:
        import fmm_openmp
        
        print("Module imported successfully")
        print(f"Max threads: {fmm_openmp.get_max_threads()}")
        
        # Run diagnostic
        print("\nRunning OpenMP diagnostic:")
        fmm_openmp.openmp_diagnostic()
        
        # Run simple test
        print("\nRunning simple OpenMP test:")
        fmm_openmp.test_openmp_simple()
        
        return True
        
    except ImportError as e:
        print(f"Failed to import module: {e}")
        return False
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def run_performance_comparison():
    """Run performance comparison with different thread counts."""
    print("\n=== Performance Comparison ===")
    
    try:
        import fmm_openmp
        
        # Test parameters
        sizes = [1000, 4000, 8000]  # Different problem sizes
        thread_counts = [1, 2, 4]
        
        for N in sizes:
            print(f"\nTesting N={N:,} particles:")
            
            # Generate test data
            np.random.seed(42)
            domain = 50.0
            x = np.random.uniform(-domain, domain, N).astype(np.float64)
            y = np.random.uniform(-domain, domain, N).astype(np.float64)
            m = np.ones(N, dtype=np.float64)
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            eps2 = 0.01**2
            
            direct_times = []
            fmm_times = []
            
            for threads in thread_counts:
                os.environ['OMP_NUM_THREADS'] = str(threads)
                print(f"\n  Testing {threads} threads:")
                
                # Test direct method
                start = time.perf_counter()
                fmm_openmp.direct_force(x, y, m, eps2, ax, ay)
                direct_time = time.perf_counter() - start
                direct_times.append(direct_time)
                
                # Test FMM method
                start = time.perf_counter()
                fmm_openmp.fmm_force_theta(x, y, m, eps2, domain, 0.6, ax, ay)
                fmm_time = time.perf_counter() - start
                fmm_times.append(fmm_time)
                
                print(f"    Direct: {direct_time:.4f}s")
                print(f"    FMM:    {fmm_time:.4f}s")
                
            # Calculate speedups
            print(f"\n  Speedups for N={N:,}:")
            for i, threads in enumerate(thread_counts):
                direct_speedup = direct_times[0] / direct_times[i] if direct_times[i] > 0 else 0
                fmm_speedup = fmm_times[0] / fmm_times[i] if fmm_times[i] > 0 else 0
                print(f"    {threads} threads: Direct={direct_speedup:.2f}×, FMM={fmm_speedup:.2f}×")
                
        return True
        
    except Exception as e:
        print(f"Error during performance test: {e}")
        return False

def check_compiler_openmp():
    """Check if the compiler properly supports OpenMP."""
    print("\n=== Compiler OpenMP Check ===")
    
    # Create a simple test program
    test_program = """
#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "Thread " << tid << " of " << nthreads << std::endl;
        }
    }
    
    return 0;
}
"""
    
    # Write test program
    with open("openmp_test.cpp", "w") as f:
        f.write(test_program)
    
    # Try to compile and run
    try:
        # Compile
        compile_cmd = ["g++", "-fopenmp", "-o", "openmp_test", "openmp_test.cpp"]
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print("✓ Test program compiled successfully")
        
        # Run
        result = subprocess.run(["./openmp_test"], capture_output=True, text=True, check=True)
        print("✓ Test program output:")
        print(result.stdout)
        
        # Clean up
        os.remove("openmp_test.cpp")
        os.remove("openmp_test")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Compiler test failed: {e}")
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"✗ Error during compiler test: {e}")
        return False

def restore_original():
    """Restore the original file."""
    if os.path.exists("fmm_openmp_original.cpp"):
        if os.path.exists("fmm_openmp.cpp"):
            os.remove("fmm_openmp.cpp")
        os.rename("fmm_openmp_original.cpp", "fmm_openmp.cpp")
        print("Restored original fmm_openmp.cpp")

def main():
    """Main diagnostic routine."""
    print("=== OpenMP Diagnostic Suite ===")
    print("This will help identify why OpenMP isn't providing speedup.")
    
    # Step 1: Test basic compiler OpenMP support
    compiler_ok = check_compiler_openmp()
    
    # Step 2: Test environment
    test_openmp_environment()
    
    # Step 3: Compile diagnostic version
    if not compile_diagnostic_version():
        print("Failed to compile diagnostic version!")
        return
    
    try:
        # Step 4: Test basic OpenMP functionality
        basic_ok = run_basic_openmp_test()
        
        # Step 5: Run performance tests
        if basic_ok:
            run_performance_comparison()
        
    finally:
        # Step 6: Restore original
        restore_original()
    
    print("\n=== Diagnostic Summary ===")
    print(f"Compiler OpenMP support: {'✓' if compiler_ok else '✗'}")
    print(f"Module OpenMP functionality: {'✓' if basic_ok else '✗'}")
    
    if not compiler_ok:
        print("\nRecommendations:")
        print("1. Install OpenMP development libraries:")
        print("   - Ubuntu/Debian: sudo apt-get install libomp-dev")
        print("   - CentOS/RHEL: sudo yum install libgomp-devel")
        print("   - macOS: brew install libomp")
        print("2. Check your GCC version supports OpenMP")
        print("3. Try using a different compiler (clang, icc)")
    
    elif not basic_ok:
        print("\nRecommendations:")
        print("1. Check Python OpenMP library conflicts")
        print("2. Try setting OMP_NUM_THREADS explicitly")
        print("3. Check if running in a container or restricted environment")
        print("4. Try different scheduling strategies")

if __name__ == "__main__":
    main()
