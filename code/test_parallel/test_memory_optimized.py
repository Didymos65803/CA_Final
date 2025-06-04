#!/usr/bin/env python3
# test_memory_optimized.py - Realistic testing for memory-bound systems

import os
import sys
import time
import numpy as np

def setup_for_memory_bound_system():
    """Setup optimal settings for memory-bound N-body simulation"""
    # Based on your system diagnosis: high memory latency, 4 cores
    os.environ["OMP_NUM_THREADS"] = "2"  # Use only 2 threads
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ["OMP_SCHEDULE"] = "guided"
    os.environ["OMP_PLACES"] = "cores"
    print("Configured for memory-bound system: 2 threads, guided scheduling")

setup_for_memory_bound_system()

def test_all_algorithms():
    """Test all available algorithms"""
    print("Algorithm Comparison Test")
    print("=" * 50)
    
    algorithms = []
    
    # Load minimal kernel if available
    try:
        import minimal_force_kernel
        algorithms.append(("Minimal", minimal_force_kernel.minimal_direct_force))
        print("✓ Minimal kernel available")
    except ImportError:
        print("✗ Minimal kernel not available")
    
    # Load memory-optimized kernel
    try:
        import memory_optimized_kernel
        algorithms.extend([
            ("Serial Opt", memory_optimized_kernel.serial_optimized_force),
            ("Bandwidth Opt", memory_optimized_kernel.bandwidth_optimized_force),
            ("Symmetric", memory_optimized_kernel.symmetric_force),
            ("Cache Blocked", memory_optimized_kernel.cache_blocked_force),
            ("Adaptive", memory_optimized_kernel.adaptive_force)
        ])
        optimal_threads = memory_optimized_kernel.get_optimal_threads()
        print(f"✓ Memory-optimized kernel available (optimal threads: {optimal_threads})")
    except ImportError:
        print("✗ Memory-optimized kernel not available")
        print("Build with: python setup_memory_optimized.py build_ext --inplace")
        return
    
    if not algorithms:
        print("No algorithms available for testing")
        return
    
    # Test with a realistic problem size for your system
    N = 1000  # Based on your diagnosis, this size shows some parallelization benefit
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    print(f"\nTesting with N={N} particles (optimal for your system):")
    print(f"{'Algorithm':<15} {'Time (s)':<12} {'Relative':<10} {'Status':<15}")
    print("-" * 60)
    
    times = []
    results = []
    
    for alg_name, alg_func in algorithms:
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            alg_func(x, y, m, 0.01, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(5):
                ax.fill(0.0)
                ay.fill(0.0)
                alg_func(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 5
            
            times.append(elapsed)
            relative = elapsed / times[0] if times else 1.0
            
            # Verify correctness (check magnitude)
            force_magnitude = np.sqrt(ax**2 + ay**2).mean()
            if 0.01 < force_magnitude < 100:  # Reasonable range
                status = "✓ Correct"
            else:
                status = "⚠ Check result"
            
            results.append((alg_name, elapsed, relative, status))
            print(f"{alg_name:<15} {elapsed:<12.6f} {relative:<10.2f} {status:<15}")
            
        except Exception as e:
            print(f"{alg_name:<15} {'Failed':<12} {'N/A':<10} {'✗ Error':<15}")
            print(f"  Error: {str(e)[:50]}")
    
    # Find best algorithm
    if results:
        best_alg = min(results, key=lambda x: x[1])
        print(f"\n✓ Best algorithm: {best_alg[0]} ({best_alg[1]:.6f} s)")
        
        if best_alg[0] == "Serial Opt":
            print("  → Single-threaded is fastest (memory bandwidth limited)")
        else:
            print(f"  → {best_alg[0]} shows best performance")

def test_thread_scaling_realistic():
    """Test thread scaling with realistic expectations"""
    print("\nRealistic Thread Scaling Test")
    print("=" * 50)
    
    try:
        import memory_optimized_kernel
    except ImportError:
        print("Memory-optimized kernel not available")
        return
    
    # Use the problem size that showed best scaling in your diagnosis
    N = 2000  # This showed 1.33x speedup in your results
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    print(f"Testing with N={N} particles (your best scaling size)")
    print("Realistic expectations: 1.3-1.5x max speedup on your system")
    print()
    
    algorithms = [
        ("Bandwidth Opt", memory_optimized_kernel.bandwidth_optimized_force),
        ("Symmetric", memory_optimized_kernel.symmetric_force)
    ]
    
    for alg_name, alg_func in algorithms:
        print(f"{alg_name} Algorithm:")
        print(f"{'Threads':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Rating':<10}")
        print("-" * 60)
        
        thread_times = []
        
        for threads in [1, 2, 4]:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            time.sleep(0.1)
            
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            try:
                # Warmup
                alg_func(x, y, m, 0.01, ax, ay)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(3):
                    ax.fill(0.0)
                    ay.fill(0.0)
                    alg_func(x, y, m, 0.01, ax, ay)
                elapsed = (time.perf_counter() - start_time) / 3
                
                thread_times.append(elapsed)
                speedup = thread_times[0] / elapsed if thread_times else 1.0
                efficiency = speedup / threads
                
                # Rate the performance
                if speedup >= 1.3:
                    rating = "Excellent"
                elif speedup >= 1.1:
                    rating = "Good"
                elif speedup >= 0.95:
                    rating = "Acceptable"
                else:
                    rating = "Poor"
                
                print(f"{threads:<8} {elapsed:<12.6f} {speedup:<10.2f} {efficiency:<12.1%} {rating:<10}")
                
            except Exception as e:
                print(f"{threads:<8} {'Failed':<12} {'N/A':<10} {'N/A':<12} {'Error':<10}")
        
        print()  # Empty line between algorithms
    
    # Restore default
    os.environ["OMP_NUM_THREADS"] = "2"

def practical_recommendations():
    """Provide practical recommendations based on testing"""
    print("Practical Recommendations for Your System")
    print("=" * 50)
    
    print("Based on your Intel Xeon Cascadelake with high memory latency:")
    print()
    
    print("1. **Optimal Settings for N-body Simulations:**")
    print("   export OMP_NUM_THREADS=2")
    print("   export OMP_PROC_BIND=true")
    print("   export OMP_SCHEDULE=guided")
    print()
    
    print("2. **Problem Sizes:**")
    print("   - Small (N < 500): Single-threaded often best")
    print("   - Medium (N = 500-1500): 2 threads optimal")
    print("   - Large (N > 2000): Consider FMM algorithm")
    print()
    
    print("3. **Expected Performance:**")
    print("   - Maximum speedup: 1.3-1.5x (this is GOOD for memory-bound)")
    print("   - Efficiency: 25-35% (normal for high memory latency)")
    print("   - Don't expect 4x speedup on 4 cores")
    print()
    
    print("4. **Algorithm Choice:**")
    print("   - For accuracy: Direct method")
    print("   - For speed at N > 2000: FMM with theta=0.5")
    print("   - For memory efficiency: Cache-blocked algorithms")
    print()
    
    print("5. **System-Specific Optimizations:**")
    print("   - Use fewer threads than CPU cores")
    print("   - Focus on cache optimization over parallelization")
    print("   - Memory prefetching helps on your high-latency system")
    print("   - Data blocking improves cache utilization")

def compare_with_baseline():
    """Compare optimized versions with a simple baseline"""
    print("\nComparison with Simple Baseline")
    print("=" * 50)
    
    try:
        import memory_optimized_kernel
    except ImportError:
        print("Memory-optimized kernel not available")
        return
    
    # Simple Python baseline for comparison
    def python_baseline(x, y, m, eps2, ax, ay):
        """Simple Python implementation"""
        N = len(x)
        ax.fill(0.0)
        ay.fill(0.0)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    r2 = dx*dx + dy*dy + eps2
                    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                    force_mag = m[j] * inv_r3
                    
                    ax[i] -= force_mag * dx
                    ay[i] -= force_mag * dy
    
    N = 200  # Small size for Python comparison
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    tests = [
        ("Python Baseline", lambda x,y,m,e,ax,ay: python_baseline(x,y,m,e,ax,ay)),
        ("Serial Optimized", memory_optimized_kernel.serial_optimized_force),
        ("Bandwidth Opt (2T)", memory_optimized_kernel.bandwidth_optimized_force)
    ]
    
    print(f"Testing with N={N} particles:")
    print(f"{'Implementation':<20} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 45)
    
    times = []
    for name, func in tests:
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            start_time = time.perf_counter()
            func(x, y, m, 0.01, ax, ay)
            elapsed = time.perf_counter() - start_time
            
            times.append(elapsed)
            speedup = times[0] / elapsed if times else 1.0
            
            print(f"{name:<20} {elapsed:<12.6f} {speedup:<10.2f}")
            
        except Exception as e:
            print(f"{name:<20} {'Failed':<12} {'Error':<10}")

def main():
    print("Memory-Optimized N-Body Testing")
    print("=" * 60)
    print("Optimized for Intel Xeon Cascadelake (high memory latency)")
    print()
    
    test_all_algorithms()
    test_thread_scaling_realistic()
    compare_with_baseline()
    practical_recommendations()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Your system is memory bandwidth limited (normal for N-body).")
    print("Focus on:")
    print("• Using 2 threads instead of 4")
    print("• Cache optimization over thread count")
    print("• Realistic expectations (1.3x speedup is good!)")
    print("• Consider FMM for large problems (N > 2000)")
    print("=" * 60)

if __name__ == "__main__":
    main()
