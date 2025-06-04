#!/usr/bin/env python3
# simple_parallel_test.py - Comprehensive parallel diagnosis

import os
import sys
import time
import numpy as np

def setup_minimal_openmp():
    """Minimal OpenMP setup"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ["OMP_DYNAMIC"] = "false"
    
    print(f"OpenMP threads: {cpu_count}")
    return cpu_count

# Setup
num_threads = setup_minimal_openmp()

def test_openmp_detection():
    """Test if OpenMP is actually detected and working"""
    print("OpenMP Detection Test")
    print("=" * 40)
    
    try:
        import minimal_force_kernel
        max_threads = minimal_force_kernel.get_max_threads()
        current_threads = minimal_force_kernel.get_current_threads()
        
        print(f"Max threads available: {max_threads}")
        print(f"Current threads in use: {current_threads}")
        
        if max_threads == 1:
            print("❌ OpenMP not detected or not compiled in")
            return False
        else:
            print("✓ OpenMP detected and working")
            return True
            
    except ImportError:
        print("❌ minimal_force_kernel not available")
        print("Run: python setup_minimal.py build_ext --inplace")
        return False
    except Exception as e:
        print(f"❌ Error testing OpenMP: {e}")
        return False

def test_scheduling_strategies():
    """Test different OpenMP scheduling strategies"""
    print("\nScheduling Strategy Comparison")
    print("=" * 50)
    
    try:
        import minimal_force_kernel
    except ImportError:
        print("minimal_force_kernel not available")
        return
    
    N = 800
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    strategies = [
        ("Static", minimal_force_kernel.test_schedule_static),
        ("Dynamic", minimal_force_kernel.test_schedule_dynamic),
        ("Guided", minimal_force_kernel.test_schedule_guided),
        ("Manual", minimal_force_kernel.test_manual_chunking),
    ]
    
    print(f"Testing with N={N} particles")
    print(f"{'Strategy':<12} {'Time (s)':<12} {'Relative':<10}")
    print("-" * 40)
    
    times = []
    for name, func in strategies:
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            func(x, y, m, 0.01, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(3):
                ax.fill(0.0)
                ay.fill(0.0)
                func(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 3
            
            times.append(elapsed)
            relative = elapsed / times[0] if times else 1.0
            
            print(f"{name:<12} {elapsed:<12.6f} {relative:<10.2f}")
            
        except Exception as e:
            print(f"{name:<12} {'Failed':<12} {'N/A':<10}")
            print(f"  Error: {e}")

def test_minimal_vs_original():
    """Compare minimal kernel with original optimized kernel"""
    print("\nMinimal vs Original Kernel Comparison")
    print("=" * 50)
    
    kernels = []
    
    # Try to load minimal kernel
    try:
        import minimal_force_kernel
        kernels.append(("Minimal", minimal_force_kernel.minimal_direct_force))
    except ImportError:
        print("minimal_force_kernel not available")
    
    # Try to load original kernel
    try:
        import force_kernel
        kernels.append(("Original", force_kernel.direct_force))
    except ImportError:
        print("force_kernel not available")
    
    if not kernels:
        print("No kernels available for testing")
        return
    
    N = 600
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    print(f"Testing with N={N} particles")
    print(f"{'Kernel':<12} {'Threads':<8} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for kernel_name, kernel_func in kernels:
        thread_times = []
        
        for threads in [1, 2, 4, 8]:
            if threads > num_threads:
                continue
                
            os.environ["OMP_NUM_THREADS"] = str(threads)
            time.sleep(0.1)
            
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            try:
                # Warmup
                kernel_func(x, y, m, 0.01, ax, ay)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(5):
                    ax.fill(0.0)
                    ay.fill(0.0)
                    kernel_func(x, y, m, 0.01, ax, ay)
                elapsed = (time.perf_counter() - start_time) / 5
                
                thread_times.append(elapsed)
                speedup = thread_times[0] / elapsed if thread_times else 1.0
                
                print(f"{kernel_name:<12} {threads:<8} {elapsed:<12.6f} {speedup:<10.2f}")
                
            except Exception as e:
                print(f"{kernel_name:<12} {threads:<8} {'Failed':<12} {'N/A':<10}")
        
        print()  # Empty line between kernels
    
    # Restore
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

def test_problem_size_threshold():
    """Find the problem size where parallelization becomes beneficial"""
    print("\nProblem Size Threshold Analysis")
    print("=" * 50)
    
    try:
        import minimal_force_kernel
    except ImportError:
        print("minimal_force_kernel not available")
        return
    
    test_sizes = [50, 100, 200, 400, 600, 800, 1000, 1200]
    
    print(f"{'N':<6} {'1 Thread':<12} {'8 Threads':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 60)
    
    for N in test_sizes:
        if N > 1200 and num_threads < 4:
            continue
            
        np.random.seed(42)
        x = np.random.uniform(-10, 10, N).astype(np.float64)
        y = np.random.uniform(-10, 10, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        times = []
        
        # Test 1 thread
        os.environ["OMP_NUM_THREADS"] = "1"
        time.sleep(0.05)
        
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            start_time = time.perf_counter()
            for _ in range(3):
                ax.fill(0.0)
                ay.fill(0.0)
                minimal_force_kernel.minimal_direct_force(x, y, m, 0.01, ax, ay)
            t1 = (time.perf_counter() - start_time) / 3
            times.append(t1)
        except:
            times.append(float('nan'))
        
        # Test max threads
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        time.sleep(0.05)
        
        try:
            start_time = time.perf_counter()
            for _ in range(3):
                ax.fill(0.0)
                ay.fill(0.0)
                minimal_force_kernel.minimal_direct_force(x, y, m, 0.01, ax, ay)
            t_max = (time.perf_counter() - start_time) / 3
            times.append(t_max)
        except:
            times.append(float('nan'))
        
        if len(times) == 2 and not (np.isnan(times[0]) or np.isnan(times[1])):
            speedup = times[0] / times[1]
            efficiency = speedup / num_threads
            
            print(f"{N:<6} {times[0]:<12.6f} {times[1]:<12.6f} {speedup:<10.2f} {efficiency:<12.1%}")
        else:
            print(f"{N:<6} {'Failed':<12} {'Failed':<12} {'N/A':<10} {'N/A':<12}")
    
    # Restore
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

def test_system_characteristics():
    """Test system characteristics that affect parallel performance"""
    print("\nSystem Characteristics Test")
    print("=" * 50)
    
    # Memory bandwidth test
    print("Memory Bandwidth Test:")
    N = 1000000  # 1M elements
    
    # Sequential memory access
    data = np.arange(N, dtype=np.float64)
    start_time = time.perf_counter()
    result = np.sum(data)
    seq_time = time.perf_counter() - start_time
    
    # Random memory access
    indices = np.random.permutation(N)
    start_time = time.perf_counter()
    result = np.sum(data[indices])
    rand_time = time.perf_counter() - start_time
    
    print(f"  Sequential access: {seq_time:.6f} s")
    print(f"  Random access: {rand_time:.6f} s")
    print(f"  Random/Sequential ratio: {rand_time/seq_time:.2f}")
    
    if rand_time/seq_time > 2.0:
        print("  ⚠ High memory latency - cache optimization important")
    else:
        print("  ✓ Good memory performance")
    
    # CPU characteristics
    print(f"\nCPU Information:")
    try:
        import multiprocessing
        print(f"  Logical cores: {multiprocessing.cpu_count()}")
        
        # Try to get more detailed info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
                model_line = next((line for line in lines if 'model name' in line), None)
                if model_line:
                    print(f"  CPU model: {model_line.split(':')[1].strip()}")
        except:
            pass
            
    except:
        pass

def comprehensive_diagnosis():
    """Run comprehensive parallel performance diagnosis"""
    print("=" * 80)
    print("Comprehensive Parallel Performance Diagnosis")
    print("=" * 80)
    
    # Step 1: Basic OpenMP detection
    openmp_works = test_openmp_detection()
    
    if not openmp_works:
        print("\n❌ CRITICAL: OpenMP not working!")
        print("Solutions:")
        print("1. Install OpenMP: sudo apt-get install libomp-dev")
        print("2. Recompile with: python setup_minimal.py build_ext --inplace")
        print("3. Check compiler flags include -fopenmp")
        return
    
    # Step 2: Test scheduling strategies
    test_scheduling_strategies()
    
    # Step 3: Compare kernels
    test_minimal_vs_original()
    
    # Step 4: Find size threshold
    test_problem_size_threshold()
    
    # Step 5: System characteristics
    test_system_characteristics()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    print("Next steps:")
    print("1. If no speedup: OpenMP compilation issue")
    print("2. If speedup only at large N: Increase problem size or reduce overhead")
    print("3. If poor efficiency: Memory bandwidth limited or false sharing")
    print("4. If minimal works but original doesn't: Over-optimization issues")
    
    print("\nRecommended fixes:")
    print("- Use problem sizes N > 800 for meaningful parallel testing")
    print("- Try different scheduling strategies (static vs dynamic)")
    print("- Check memory access patterns")
    print("- Reduce OpenMP overhead with larger chunks")

def main():
    print("Comprehensive Parallel Performance Diagnosis")
    print("=" * 60)
    print(f"System: {num_threads} cores available")
    print()
    
    comprehensive_diagnosis()

if __name__ == "__main__":
    main()#!/usr/bin/env python3
# simple_parallel_test.py - Minimal test to diagnose parallel issues

import os
import sys
import time
import numpy as np

def setup_minimal_openmp():
    """Minimal OpenMP setup"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ["OMP_DYNAMIC"] = "false"
    
    print(f"OpenMP threads: {cpu_count}")
    return cpu_count

# Setup
num_threads = setup_minimal_openmp()

def test_direct_scaling_only():
    """Test only direct method scaling to isolate issues"""
    print("Direct Method Scaling Test")
    print("=" * 40)
    
    try:
        import force_kernel
    except ImportError:
        print("force_kernel not available")
        return
    
    # Use a problem size that should show scaling
    N = 1000
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    thread_counts = [1, 2, 4, 8]
    times = []
    
    print(f"Testing with N={N} particles")
    print(f"{'Threads':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 50)
    
    for threads in thread_counts:
        if threads > num_threads:
            continue
            
        # Set thread count
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)
        
        # Create fresh arrays for each test
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup run
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            
            # Actual benchmark - multiple runs for accuracy
            start_time = time.perf_counter()
            for _ in range(5):
                ax.fill(0.0)
                ay.fill(0.0)
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 5
            
            times.append(elapsed)
            
            if len(times) == 1:
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = times[0] / elapsed
                efficiency = speedup / threads
            
            print(f"{threads:<8} {elapsed:<12.6f} {speedup:<10.2f} {efficiency:<12.1%}")
            
        except Exception as e:
            print(f"{threads:<8} {'Failed':<12} {'N/A':<10} {'N/A':<12}")
            print(f"Error: {e}")
            times.append(float('nan'))
    
    # Restore
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    # Analysis
    valid_times = [t for t in times if not np.isnan(t)]
    if len(valid_times) > 1:
        max_speedup = max([valid_times[0] / t for t in valid_times[1:]], default=1.0)
        print(f"\nMax speedup achieved: {max_speedup:.2f}x")
        
        if max_speedup < 1.5:
            print("❌ Poor parallel scaling detected")
            print("Possible causes:")
            print("- Problem size too small for overhead")
            print("- Memory bandwidth limitations")
            print("- OpenMP overhead")
            print("- False sharing or contention")
        else:
            print("✓ Reasonable parallel scaling")

def test_problem_size_scaling():
    """Test how performance scales with problem size"""
    print("\nProblem Size Scaling Test")
    print("=" * 40)
    
    try:
        import force_kernel
    except ImportError:
        print("force_kernel not available")
        return
    
    # Use maximum threads
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    test_sizes = [100, 200, 500, 1000, 1500]
    
    print(f"Using {num_threads} threads")
    print(f"{'N':<6} {'Time (s)':<12} {'Time/N²':<12} {'Scaling':<10}")
    print("-" * 50)
    
    times = []
    for N in test_sizes:
        np.random.seed(42)
        x = np.random.uniform(-10, 10, N).astype(np.float64)
        y = np.random.uniform(-10, 10, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(3):
                ax.fill(0.0)
                ay.fill(0.0)
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 3
            
            times.append(elapsed)
            time_per_n2 = elapsed / (N * N) * 1e6  # microseconds per interaction
            
            if len(times) == 1:
                scaling = "Reference"
            else:
                expected_time = times[0] * (N / test_sizes[0])**2
                scaling = f"{elapsed/expected_time:.2f}"
            
            print(f"{N:<6} {elapsed:<12.6f} {time_per_n2:<12.2f} {scaling:<10}")
            
        except Exception as e:
            print(f"{N:<6} {'Failed':<12} {'N/A':<12} {'N/A':<10}")

def test_memory_access_pattern():
    """Test if memory access patterns are causing issues"""
    print("\nMemory Access Pattern Test")
    print("=" * 40)
    
    try:
        import force_kernel
    except ImportError:
        print("force_kernel not available")
        return
    
    N = 1000
    
    # Test 1: Contiguous arrays (good for cache)
    x_contig = np.arange(N, dtype=np.float64)
    y_contig = np.arange(N, dtype=np.float64)
    m_contig = np.ones(N, dtype=np.float64)
    
    # Test 2: Random arrays (potentially bad for cache)
    np.random.seed(42)
    x_random = np.random.uniform(-10, 10, N).astype(np.float64)
    y_random = np.random.uniform(-10, 10, N).astype(np.float64)
    m_random = np.random.uniform(0.5, 2.0, N).astype(np.float64)
    
    tests = [
        ("Contiguous", x_contig, y_contig, m_contig),
        ("Random", x_random, y_random, m_random)
    ]
    
    print(f"{'Pattern':<12} {'Time (s)':<12} {'Relative':<10}")
    print("-" * 35)
    
    times = []
    for name, x, y, m in tests:
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        try:
            # Warmup
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(3):
                ax.fill(0.0)
                ay.fill(0.0)
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            elapsed = (time.perf_counter() - start_time) / 3
            
            times.append(elapsed)
            relative = elapsed / times[0] if times else 1.0
            
            print(f"{name:<12} {elapsed:<12.6f} {relative:<10.2f}")
            
        except Exception as e:
            print(f"{name:<12} {'Failed':<12} {'N/A':<10}")

def diagnose_openmp_overhead():
    """Test OpenMP overhead"""
    print("\nOpenMP Overhead Diagnosis")
    print("=" * 40)
    
    try:
        import force_kernel
    except ImportError:
        print("force_kernel not available")
        return
    
    # Very small problem - should show pure overhead
    N_small = 50
    # Medium problem - should show some benefit
    N_medium = 500
    
    problems = [("Small (N=50)", N_small), ("Medium (N=500)", N_medium)]
    
    for name, N in problems:
        print(f"\n{name}:")
        print(f"{'Threads':<8} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 35)
        
        np.random.seed(42)
        x = np.random.uniform(-10, 10, N).astype(np.float64)
        y = np.random.uniform(-10, 10, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        thread_times = []
        for threads in [1, 2, 4, 8]:
            if threads > num_threads:
                continue
                
            os.environ["OMP_NUM_THREADS"] = str(threads)
            time.sleep(0.05)
            
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            try:
                # More iterations for small problems
                iterations = 20 if N < 100 else 5
                
                start_time = time.perf_counter()
                for _ in range(iterations):
                    ax.fill(0.0)
                    ay.fill(0.0)
                    force_kernel.direct_force(x, y, m, 0.01, ax, ay)
                elapsed = (time.perf_counter() - start_time) / iterations
                
                thread_times.append(elapsed)
                speedup = thread_times[0] / elapsed if thread_times else 1.0
                
                print(f"{threads:<8} {elapsed:<12.6f} {speedup:<10.2f}")
                
            except Exception as e:
                print(f"{threads:<8} {'Failed':<12} {'N/A':<10}")
    
    # Restore
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

def main():
    print("Simple Parallel Diagnosis Tool")
    print("=" * 50)
    print(f"System: {num_threads} cores available")
    
    test_direct_scaling_only()
    test_problem_size_scaling()
    test_memory_access_pattern()
    diagnose_openmp_overhead()
    
    print("\n" + "=" * 50)
    print("Diagnosis completed!")
    print("\nRecommendations:")
    print("1. If no speedup at all: Check OpenMP compilation")
    print("2. If speedup only at large N: Increase problem size")
    print("3. If poor efficiency: Reduce threads or optimize algorithm")
    print("4. If memory pattern matters: Optimize data layout")
    print("=" * 50)

if __name__ == "__main__":
    main()
