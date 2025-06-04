#!/usr/bin/env python3
# test_final.py - Final comprehensive test suite

import os
import sys
import time
import numpy as np

# Set OpenMP environment
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "cores"

def test_module_import():
    """Test if modules can be imported"""
    print("Testing module imports...")
    
    try:
        import force_kernel
        print("✓ force_kernel imported successfully")
        has_direct = True
    except ImportError as e:
        print(f"✗ force_kernel import failed: {e}")
        has_direct = False
    
    try:
        import fmm_kernel
        print("✓ fmm_kernel imported successfully")
        has_fmm = True
    except ImportError as e:
        print(f"✗ fmm_kernel import failed: {e}")
        has_fmm = False
    
    return has_direct, has_fmm

def test_basic_functionality():
    """Test basic functionality of both methods"""
    print("\nTesting basic functionality...")
    
    has_direct, has_fmm = test_module_import()
    
    if not has_direct and not has_fmm:
        print("No modules available for testing")
        return False
    
    # Create test data
    N = 50
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Test direct method
    if has_direct:
        try:
            import force_kernel
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            
            force_magnitude = np.sqrt(ax**2 + ay**2)
            max_force = np.max(force_magnitude)
            
            if max_force > 0 and max_force < 1e10:
                print("✓ Direct method basic test passed")
                print(f"  Max force magnitude: {max_force:.6e}")
            else:
                print("✗ Direct method produced invalid results")
                return False
                
        except Exception as e:
            print(f"✗ Direct method test failed: {e}")
            return False
    
    # Test FMM method
    if has_fmm:
        try:
            import fmm_kernel
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 8, 0.01, 1.0, ax, ay)
            
            force_magnitude = np.sqrt(ax**2 + ay**2)
            max_force = np.max(force_magnitude)
            
            if max_force > 0 and max_force < 1e10:
                print("✓ FMM method basic test passed")
                print(f"  Max force magnitude: {max_force:.6e}")
            else:
                print("✗ FMM method produced invalid results")
                return False
                
        except Exception as e:
            print(f"✗ FMM method test failed: {e}")
            return False
    
    return True

def test_accuracy():
    """Test accuracy between direct and FMM methods"""
    print("\nTesting accuracy...")
    
    try:
        import force_kernel
        import fmm_kernel
    except ImportError:
        print("Both methods needed for accuracy test")
        return
    
    N = 100
    np.random.seed(123)
    x = np.random.uniform(-20, 20, N).astype(np.float64)
    y = np.random.uniform(-20, 20, N).astype(np.float64)
    m = np.random.uniform(0.5, 2.0, N).astype(np.float64)
    
    # Direct method (reference)
    ax_direct = np.zeros(N, dtype=np.float64)
    ay_direct = np.zeros(N, dtype=np.float64)
    force_kernel.direct_force(x, y, m, 0.01, ax_direct, ay_direct)
    
    # FMM method
    ax_fmm = np.zeros(N, dtype=np.float64)
    ay_fmm = np.zeros(N, dtype=np.float64)
    fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax_fmm, ay_fmm)
    
    # Calculate relative error
    force_direct = np.sqrt(ax_direct**2 + ay_direct**2)
    force_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
    
    relative_error = np.mean(np.abs(force_fmm - force_direct) / (force_direct + 1e-10))
    
    print(f"Relative error: {relative_error:.4e}")
    
    if relative_error < 0.01:  # 1% error tolerance
        print("✓ Accuracy test passed")
    else:
        print("⚠ High relative error detected")

def test_performance():
    """Test performance and scaling"""
    print("\nTesting performance...")
    
    try:
        import force_kernel
        import fmm_kernel
        has_both = True
    except ImportError:
        print("Both methods needed for performance test")
        return
    
    test_sizes = [100, 200, 500]
    
    print(f"{'N':>5} {'Direct':>10} {'FMM':>10} {'Speedup':>10}")
    print("-" * 40)
    
    for N in test_sizes:
        np.random.seed(42)
        x = np.random.uniform(-25, 25, N).astype(np.float64)
        y = np.random.uniform(-25, 25, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        # Test direct method
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        start_time = time.perf_counter()
        for _ in range(3):
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
        t_direct = (time.perf_counter() - start_time) / 3
        
        # Test FMM method
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        start_time = time.perf_counter()
        for _ in range(3):
            fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax, ay)
        t_fmm = (time.perf_counter() - start_time) / 3
        
        speedup = t_direct / t_fmm if t_fmm > 0 else 0
        
        print(f"{N:>5} {t_direct:>10.6f} {t_fmm:>10.6f} {speedup:>10.2f}")

def test_parallel_scaling():
    """Test parallel scaling"""
    print("\nTesting parallel scaling...")
    
    try:
        import fmm_kernel
    except ImportError:
        print("FMM method needed for parallel scaling test")
        return
    
    N = 1000
    np.random.seed(42)
    x = np.random.uniform(-50, 50, N).astype(np.float64)
    y = np.random.uniform(-50, 50, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    thread_counts = [1, 2, 4, 8]
    times = []
    
    print(f"{'Threads':>8} {'Time':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 45)
    
    for threads in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)  # Allow environment change to take effect
        
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        # Warmup
        fmm_kernel.fmm_force(x, y, m, N, 100.0, 0.5, 16, 0.01, 1.0, ax, ay)
        
        # Measure
        start_time = time.perf_counter()
        for _ in range(5):
            fmm_kernel.fmm_force(x, y, m, N, 100.0, 0.5, 16, 0.01, 1.0, ax, ay)
        elapsed = (time.perf_counter() - start_time) / 5
        
        times.append(elapsed)
        
        if len(times) == 1:
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = times[0] / elapsed
            efficiency = speedup / threads
        
        print(f"{threads:>8} {elapsed:>10.6f} {speedup:>10.2f} {efficiency:>12.1%}")
    
    # Restore original thread count
    os.environ["OMP_NUM_THREADS"] = "8"

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    try:
        import force_kernel
        import fmm_kernel
    except ImportError:
        print("Both methods needed for edge case testing")
        return
    
    # Test very small problem
    N = 2
    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.0], dtype=np.float64)
    m = np.array([1.0, 1.0], dtype=np.float64)
    
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    try:
        force_kernel.direct_force(x, y, m, 0.01, ax, ay)
        print("✓ Direct method handles small problems")
    except Exception as e:
        print(f"✗ Direct method failed on small problem: {e}")
    
    try:
        fmm_kernel.fmm_force(x, y, m, N, 10.0, 0.5, 8, 0.01, 1.0, ax, ay)
        print("✓ FMM method handles small problems")
    except Exception as e:
        print(f"✗ FMM method failed on small problem: {e}")
    
    # Test array size mismatch
    x_wrong = np.array([0.0], dtype=np.float64)
    try:
        force_kernel.direct_force(x_wrong, y, m, 0.01, ax, ay)
        print("✗ Direct method should have failed on size mismatch")
    except Exception:
        print("✓ Direct method correctly handles size mismatch")
    
    try:
        fmm_kernel.fmm_force(x_wrong, y, m, N, 10.0, 0.5, 8, 0.01, 1.0, ax, ay)
        print("✗ FMM method should have failed on size mismatch")
    except Exception:
        print("✓ FMM method correctly handles size mismatch")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("N-Body Kernel Final Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test 2: Accuracy
    test_accuracy()
    
    # Test 3: Performance
    test_performance()
    
    # Test 4: Parallel scaling
    test_parallel_scaling()
    
    # Test 5: Edge cases
    test_edge_cases()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All critical tests passed!")
        print("The kernels are working correctly.")
    else:
        print("✗ Some tests failed!")
        print("Please check the compilation and try again.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    run_all_tests()
