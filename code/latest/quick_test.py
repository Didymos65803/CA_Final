#!/usr/bin/env python3
"""quick_test.py - Quick performance test to verify OpenMP improvements"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import fmm_openmp as fm
except ImportError:
    print("ERROR: fmm_openmp module not found!")
    print("Please run: python setup_improved.py build_ext --inplace")
    exit(1)

def quick_performance_test():
    """Run a quick test to demonstrate OpenMP speedup."""
    print("=== Quick FMM OpenMP Performance Test ===")
    print(f"Maximum OpenMP threads available: {fm.get_max_threads()}")
    
    # Test parameters - use larger problem size for better OpenMP scaling
    N = 16000  # Increased from 4000 to see better OpenMP effects
    domain = 50.0
    eps2 = 0.01**2
    theta = 0.6
    
    # Generate test data
    np.random.seed(42)
    x = np.random.uniform(-domain, domain, N).astype(np.float64)
    y = np.random.uniform(-domain, domain, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    print(f"Testing with N={N:,} particles")
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8] if fm.get_max_threads() >= 8 else [1, 2, 4]
    direct_times = []
    fmm_times = []
    
    for threads in thread_counts:
        os.environ['OMP_NUM_THREADS'] = str(threads)
        print(f"\nTesting with {threads} threads:")
        
        # Warmup
        fm.direct_force(x, y, m, eps2, ax, ay)
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        
        # Time direct method
        start = time.perf_counter()
        for _ in range(3):  # Multiple runs for accuracy
            fm.direct_force(x, y, m, eps2, ax, ay)
        direct_time = (time.perf_counter() - start) / 3
        direct_times.append(direct_time)
        
        # Time FMM method
        start = time.perf_counter()
        for _ in range(5):  # More runs since FMM is faster
            fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        fmm_time = (time.perf_counter() - start) / 5
        fmm_times.append(fmm_time)
        
        # Calculate metrics
        direct_speedup = direct_times[0] / direct_time
        fmm_speedup = fmm_times[0] / fmm_time
        algorithmic_speedup = direct_time / fmm_time
        
        print(f"  Direct: {direct_time:.4f}s (speedup: {direct_speedup:.2f}×)")
        print(f"  FMM:    {fmm_time:.4f}s (speedup: {fmm_speedup:.2f}×)")
        print(f"  FMM vs Direct: {algorithmic_speedup:.1f}× faster")
    
    # Create quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    ax1.plot(thread_counts, direct_times, 'o-', label='Direct O(N²)', linewidth=2, markersize=8)
    ax1.plot(thread_counts, fmm_times, 's-', label='FMM O(N log N)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Time [s]')
    ax1.set_title(f'Performance vs Thread Count (N={N:,})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup comparison
    direct_speedups = [direct_times[0] / t for t in direct_times]
    fmm_speedups = [fmm_times[0] / t for t in fmm_times]
    
    ax2.plot(thread_counts, direct_speedups, 'o-', label='Direct', linewidth=2, markersize=8)
    ax2.plot(thread_counts, fmm_speedups, 's-', label='FMM', linewidth=2, markersize=8)
    ax2.plot(thread_counts, thread_counts, '--k', alpha=0.5, label='Ideal Linear')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup vs Serial')
    ax2.set_title('OpenMP Scaling Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: quick_test_results.png")
    
    # Summary
    print(f"\n=== Summary ===")
    max_direct_speedup = max(direct_speedups)
    max_fmm_speedup = max(fmm_speedups)
    max_algorithmic_speedup = max([dt/ft for dt, ft in zip(direct_times, fmm_times)])
    
    print(f"Maximum Direct speedup: {max_direct_speedup:.2f}× ({thread_counts[-1]} threads)")
    print(f"Maximum FMM speedup: {max_fmm_speedup:.2f}× ({thread_counts[-1]} threads)")
    print(f"Maximum algorithmic advantage: {max_algorithmic_speedup:.1f}× (FMM vs Direct)")
    
    # Check if OpenMP is working
    if max(max_direct_speedup, max_fmm_speedup) > 1.5:
        print("✓ OpenMP parallelization is working effectively!")
    else:
        print("⚠ Warning: Limited speedup observed. Check OpenMP configuration.")
    
    return thread_counts, direct_times, fmm_times

if __name__ == "__main__":
    quick_performance_test()
    plt.show()  # Display the plot
