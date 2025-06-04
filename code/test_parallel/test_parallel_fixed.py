#!/usr/bin/env python3
# test_parallel_fixed.py - Enhanced test suite focusing on parallel performance

import os
import sys
import time
import numpy as np

def setup_aggressive_openmp():
    """Setup aggressive OpenMP configuration for maximum parallelization"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # Use all available cores
    num_threads = cpu_count
    
    # Aggressive OpenMP settings
    env_settings = {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": "spread",  # Changed from "close" to "spread"
        "OMP_PLACES": "threads",    # Changed from "cores" to "threads"
        "OMP_SCHEDULE": "dynamic",  # Changed from "guided" to "dynamic"
        "OMP_DYNAMIC": "true",      # Changed to "true" for adaptive scheduling
        "OMP_WAIT_POLICY": "active", # Changed to "active" for lower latency
        "OMP_NESTED": "true",       # Enable nested parallelism
        "OMP_MAX_ACTIVE_LEVELS": "2",
        "OMP_THREAD_LIMIT": str(num_threads * 2)
    }
    
    for key, value in env_settings.items():
        os.environ[key] = value
    
    print(f"Aggressive OpenMP configured: {num_threads} threads")
    print(f"Available CPU cores: {cpu_count}")
    
    return num_threads

# Setup before importing modules
num_threads = setup_aggressive_openmp()

def test_parallel_scaling_detailed():
    """Detailed parallel scaling test with multiple algorithms"""
    print("\nDetailed Parallel Scaling Test")
    print("=" * 60)
    
    try:
        import force_kernel
        import fmm_kernel
    except ImportError as e:
        print(f"Module import failed: {e}")
        return
    
    N = 1500  # Increased problem size
    np.random.seed(42)
    x = np.random.uniform(-50, 50, N).astype(np.float64)
    y = np.random.uniform(-50, 50, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 6, 8, 12, 16] if num_threads >= 16 else [1, 2, 4, 8]
    
    print(f"Testing with {N} particles")
    print("Thread scaling analysis:")
    print(f"{'Method':<12} {'Threads':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 70)
    
    # Test Direct method
    direct_times = []
    for threads in thread_counts:
        if threads > num_threads:
            continue
            
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)  # Allow environment change
        
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(3):
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 3
            
            direct_times.append(elapsed)
            speedup = direct_times[0] / elapsed if direct_times else 1.0
            efficiency = speedup / threads
            
            print(f"{'Direct':<12} {threads:<8} {elapsed:<12.6f} {speedup:<10.2f} {efficiency:<12.1%}")
            
        except Exception as e:
            print(f"{'Direct':<12} {threads:<8} {'Failed':<12} {'N/A':<10} {'N/A':<12}")
            direct_times.append(float('nan'))
    
    # Test FMM method
    fmm_times = []
    for threads in thread_counts:
        if threads > num_threads:
            continue
            
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)
        
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            fmm_kernel.fmm_force(x, y, m, N, 100.0, 0.5, 16, 0.01, 1.0, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(5):
                fmm_kernel.fmm_force(x, y, m, N, 100.0, 0.5, 16, 0.01, 1.0, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 5
            
            fmm_times.append(elapsed)
            speedup = fmm_times[0] / elapsed if fmm_times else 1.0
            efficiency = speedup / threads
            
            print(f"{'FMM':<12} {threads:<8} {elapsed:<12.6f} {speedup:<10.2f} {efficiency:<12.1%}")
            
        except Exception as e:
            print(f"{'FMM':<12} {threads:<8} {'Failed':<12} {'N/A':<10} {'N/A':<12}")
            fmm_times.append(float('nan'))
    
    # Restore original settings
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    # Analysis
    print("\nParallel Scaling Analysis:")
    
    # Direct method analysis
    valid_direct = [(i, t) for i, t in enumerate(direct_times) if not np.isnan(t)]
    if len(valid_direct) > 1:
        max_speedup_direct = max([valid_direct[0][1] / t for _, t in valid_direct])
        print(f"Direct method max speedup: {max_speedup_direct:.2f}x")
        
        # Find optimal thread count
        best_efficiency = 0
        best_threads = 1
        for i, (idx, t) in enumerate(valid_direct):
            threads = thread_counts[idx]
            speedup = valid_direct[0][1] / t
            efficiency = speedup / threads
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_threads = threads
        print(f"Direct method best efficiency: {best_efficiency:.1%} at {best_threads} threads")
    
    # FMM method analysis
    valid_fmm = [(i, t) for i, t in enumerate(fmm_times) if not np.isnan(t)]
    if len(valid_fmm) > 1:
        max_speedup_fmm = max([valid_fmm[0][1] / t for _, t in valid_fmm])
        print(f"FMM method max speedup: {max_speedup_fmm:.2f}x")
        
        # Find optimal thread count
        best_efficiency = 0
        best_threads = 1
        for i, (idx, t) in enumerate(valid_fmm):
            threads = thread_counts[idx]
            speedup = valid_fmm[0][1] / t
            efficiency = speedup / threads
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_threads = threads
        print(f"FMM method best efficiency: {best_efficiency:.1%} at {best_threads} threads")

def test_algorithm_verification():
    """Verify that the new algorithms are working correctly"""
    print("\nAlgorithm Verification Test")
    print("=" * 50)
    
    try:
        import force_kernel
        import fmm_kernel
    except ImportError:
        print("Modules not available")
        return
    
    # Test with a simple 2-particle system where we know the answer
    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.0], dtype=np.float64)
    m = np.array([1.0, 1.0], dtype=np.float64)
    
    # Direct method
    ax_direct = np.zeros(2, dtype=np.float64)
    ay_direct = np.zeros(2, dtype=np.float64)
    force_kernel.direct_force(x, y, m, 0.01, ax_direct, ay_direct)
    
    # FMM method
    ax_fmm = np.zeros(2, dtype=np.float64)
    ay_fmm = np.zeros(2, dtype=np.float64)
    fmm_kernel.fmm_force(x, y, m, 2, 10.0, 0.5, 8, 0.01, 1.0, ax_fmm, ay_fmm)
    
    # Analytical solution for two unit masses 1 unit apart
    # F = G*m1*m2/r^2 = 1*1*1/(1^2 + 0.01) ≈ 0.99
    expected_force = 1.0 / (1.0 + 0.01)
    
    print(f"Two-particle test:")
    print(f"Expected force magnitude: {expected_force:.6f}")
    print(f"Direct method force on particle 0: ({ax_direct[0]:.6f}, {ay_direct[0]:.6f})")
    print(f"FMM method force on particle 0: ({ax_fmm[0]:.6f}, {ay_fmm[0]:.6f})")
    
    direct_magnitude = np.sqrt(ax_direct[0]**2 + ay_direct[0]**2)
    fmm_magnitude = np.sqrt(ax_fmm[0]**2 + ay_fmm[0]**2)
    
    print(f"Direct force magnitude: {direct_magnitude:.6f}")
    print(f"FMM force magnitude: {fmm_magnitude:.6f}")
    
    direct_error = abs(direct_magnitude - expected_force) / expected_force
    fmm_error = abs(fmm_magnitude - expected_force) / expected_force
    
    print(f"Direct relative error: {direct_error:.4e}")
    print(f"FMM relative error: {fmm_error:.4e}")
    
    if direct_error < 0.01:
        print("✓ Direct method verification passed")
    else:
        print("✗ Direct method verification failed")
        
    if fmm_error < 0.1:  # FMM has larger tolerance due to approximations
        print("✓ FMM method verification passed")
    else:
        print("✗ FMM method verification failed")

def test_large_scale_performance():
    """Test performance on larger problem sizes"""
    print("\nLarge Scale Performance Test")
    print("=" * 50)
    
    try:
        import force_kernel
        import fmm_kernel
    except ImportError:
        print("Modules not available")
        return
    
    test_sizes = [500, 1000, 2000, 4000]
    
    print(f"{'N':<6} {'Direct (s)':<12} {'FMM (s)':<12} {'Speedup':<10} {'FMM Accuracy':<15}")
    print("-" * 70)
    
    for N in test_sizes:
        if N > 2000 and num_threads < 4:
            print(f"{N:<6} {'Skipped':<12} {'Skipped':<12} {'N/A':<10} {'N/A':<15}")
            continue
            
        # Generate test data
        np.random.seed(42)
        x = np.random.uniform(-25, 25, N).astype(np.float64)
        y = np.random.uniform(-25, 25, N).astype(np.float64)
        m = np.random.uniform(0.5, 2.0, N).astype(np.float64)
        
        # Test direct method (skip for very large N)
        t_direct = None
        ax_direct = None
        ay_direct = None
        
        if N <= 2000:
            ax_direct = np.zeros(N, dtype=np.float64)
            ay_direct = np.zeros(N, dtype=np.float64)
            
            try:
                start_time = time.perf_counter()
                force_kernel.direct_force(x, y, m, 0.01, ax_direct, ay_direct)
                t_direct = time.perf_counter() - start_time
            except Exception:
                t_direct = None
        
        # Test FMM method
        ax_fmm = np.zeros(N, dtype=np.float64)
        ay_fmm = np.zeros(N, dtype=np.float64)
        
        try:
            start_time = time.perf_counter()
            fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax_fmm, ay_fmm)
            t_fmm = time.perf_counter() - start_time
            
            # Calculate accuracy if we have direct result
            accuracy_str = "N/A"
            if ax_direct is not None:
                force_direct = np.sqrt(ax_direct**2 + ay_direct**2)
                force_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
                rel_error = np.mean(np.abs(force_fmm - force_direct) / (force_direct + 1e-10))
                accuracy_str = f"{rel_error:.3e}"
            
            # Calculate speedup
            speedup_str = f"{t_direct/t_fmm:.2f}" if t_direct else "N/A"
            direct_str = f"{t_direct:.6f}" if t_direct else "Skipped"
            
            print(f"{N:<6} {direct_str:<12} {t_fmm:<12.6f} {speedup_str:<10} {accuracy_str:<15}")
            
        except Exception as e:
            print(f"{N:<6} {'N/A':<12} {'Failed':<12} {'N/A':<10} {'N/A':<15}")

def test_memory_usage():
    """Test memory efficiency"""
    print("\nMemory Usage Test")
    print("=" * 50)
    
    try:
        import psutil
        import force_kernel
        import fmm_kernel
    except ImportError:
        print("psutil or modules not available")
        return
    
    process = psutil.Process()
    
    N = 2000
    np.random.seed(42)
    x = np.random.uniform(-25, 25, N).astype(np.float64)
    y = np.random.uniform(-25, 25, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Test Direct method
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    try:
        force_kernel.direct_force(x, y, m, 0.01, ax, ay)
        direct_memory = process.memory_info().rss / 1024 / 1024  # MB
        direct_usage = direct_memory - baseline_memory
    except Exception:
        direct_usage = float('nan')
    
    # Test FMM method
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    try:
        fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax, ay)
        fmm_memory = process.memory_info().rss / 1024 / 1024  # MB
        fmm_usage = fmm_memory - baseline_memory
    except Exception:
        fmm_usage = float('nan')
    
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    print(f"Direct method additional memory: {direct_usage:.1f} MB")
    print(f"FMM method additional memory: {fmm_usage:.1f} MB")
    
    theoretical_direct = N * N * 8 / 1024 / 1024  # Rough estimate for O(N^2) temp storage
    print(f"Theoretical direct O(N^2) memory: {theoretical_direct:.1f} MB")

def run_comprehensive_tests():
    """Run all enhanced tests"""
    print("=" * 80)
    print("Enhanced N-Body Parallel Performance Test Suite")
    print("=" * 80)
    
    # System info
    print(f"Number of CPU cores: {num_threads}")
    print(f"OpenMP settings:")
    for var in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE"]:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Run tests
    test_algorithm_verification()
    test_parallel_scaling_detailed()
    test_large_scale_performance()
    test_memory_usage()
    
    print("\n" + "=" * 80)
    print("Enhanced test suite completed!")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_tests()
