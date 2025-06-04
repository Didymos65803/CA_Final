#!/usr/bin/env python3
# quick_test.py - Quick test of final optimized kernel

import os
import time
import numpy as np

# Set optimal environment
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OMP_PROC_BIND"] = "true"

def quick_test():
    print("Quick Test of Final Optimized Kernel")
    print("=" * 40)
    
    try:
        import final_optimized_kernel
        print("✓ Final optimized kernel loaded")
    except ImportError:
        print("✗ Kernel not available. Build with:")
        print("  python setup_final_optimized.py build_ext --inplace")
        return
    
    # Test different sizes
    for N in [500, 1000, 2000]:
        print(f"\nTesting N={N}:")
        
        np.random.seed(42)
        x = np.random.uniform(-10, 10, N).astype(np.float64)
        y = np.random.uniform(-10, 10, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        algorithms = [
            ("Single Thread", final_optimized_kernel.single_thread_optimized),
            ("Cache Blocked", final_optimized_kernel.optimized_cache_blocked_force),
            ("Auto-Choose", final_optimized_kernel.benchmark_and_choose)
        ]
        
        times = []
        for name, func in algorithms:
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            start = time.perf_counter()
            func(x, y, m, 0.01, ax, ay)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            speedup = times[0] / elapsed if len(times) > 1 else 1.0
            
            print(f"  {name:<15}: {elapsed:.6f}s ({speedup:.2f}x)")
    
    print(f"\n✓ Test completed!")
    print(f"Optimal threads: {final_optimized_kernel.get_optimal_threads_for_system()}")
    print(f"Current threads: {final_optimized_kernel.get_current_threads()}")

if __name__ == "__main__":
    quick_test()
