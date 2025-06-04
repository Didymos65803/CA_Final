#!/usr/bin/env python3
"""force_openmp_test.py - Force test OpenMP with larger computational work"""

import os
import sys
import time
import numpy as np
import subprocess

def run_diagnostics():
    """Run comprehensive OpenMP diagnostics."""
    print("=== OpenMP Force Test ===")
    
    # Set environment aggressively
    env_settings = {
        'OMP_NUM_THREADS': '4',
        'OMP_PROC_BIND': 'spread',
        'OMP_PLACES': 'threads',
        'OMP_DYNAMIC': 'false',
        'OMP_NESTED': 'false',
        'OMP_WAIT_POLICY': 'active',
        'OMP_MAX_ACTIVE_LEVELS': '1',
        # Disable NumPy threading to avoid conflicts
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
    }
    
    print("Setting aggressive OpenMP environment:")
    for key, value in env_settings.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    # Test basic compilation first
    test_basic_compilation()
    
    # Run the simple test
    print("\nRunning simple OpenMP test...")
    subprocess.run([sys.executable, "simple_openmp_test.py"])

def test_basic_compilation():
    """Test if OpenMP compilation works at all."""
    print("\n=== Basic OpenMP Compilation Test ===")
    
    # Create simple test
    simple_test = '''
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    const int N = 10000000;
    double sum = 0.0;
    
    std::cout << "OpenMP available: " << 
#ifdef _OPENMP
    "Yes, version " << _OPENMP << std::endl;
#else
    "No" << std::endl;
#endif
    
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Force parallel execution with more work
    #pragma omp parallel for reduction(+:sum) schedule(static,1000)
    for (int i = 0; i < N; ++i) {
        // Add more computational work to make parallelization worthwhile
        double x = i * 0.0001;
        for (int j = 0; j < 10; ++j) {
            x = sin(x) + cos(x);
        }
        sum += x;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    return 0;
}
'''
    
    with open('openmp_force_test.cpp', 'w') as f:
        f.write(simple_test)
    
    try:
        # Compile with verbose OpenMP
        compile_cmd = [
            'g++', '-fopenmp', '-O3', '-march=native', 
            '-o', 'openmp_force_test', 'openmp_force_test.cpp', '-lm'
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print("✓ Basic compilation successful")
        
        # Run with different thread counts
        for threads in [1, 2, 4]:
            print(f"\nTesting with {threads} threads:")
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(threads)
            
            result = subprocess.run(['./openmp_force_test'], 
                                  capture_output=True, text=True, env=env)
            print(result.stdout.strip())
        
        # Cleanup
        os.remove('openmp_force_test.cpp')
        os.remove('openmp_force_test')
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e}")
        print("STDERR:", e.stderr)
        return False

def test_current_fmm():
    """Test current FMM with forced larger problems."""
    print("\n=== Testing Current FMM with Larger Problems ===")
    
    try:
        import fmm_openmp
        
        # Test with much larger problem sizes and more computational work
        sizes = [50000, 100000]  # Much larger
        thread_counts = [1, 2, 4]
        
        for N in sizes:
            print(f"\nTesting N={N:,} particles:")
            
            # Generate data
            np.random.seed(42)
            domain = 100.0
            x = np.random.uniform(-domain, domain, N).astype(np.float64)
            y = np.random.uniform(-domain, domain, N).astype(np.float64)
            m = np.ones(N, dtype=np.float64)
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            eps2 = 0.01**2
            
            print("  Direct method scaling:")
            direct_times = []
            
            for threads in thread_counts:
                os.environ['OMP_NUM_THREADS'] = str(threads)
                
                # Warmup
                fmm_openmp.direct_force(x[:1000], y[:1000], m[:1000], eps2, 
                                      ax[:1000], ay[:1000])
                
                # Time measurement with multiple runs
                times = []
                for _ in range(3):
                    start = time.perf_counter()
                    fmm_openmp.direct_force(x, y, m, eps2, ax, ay)
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times)
                direct_times.append(avg_time)
                
                speedup = direct_times[0] / avg_time if avg_time > 0 else 0
                efficiency = speedup / threads
                
                print(f"    {threads} threads: {avg_time:.4f}s (speedup: {speedup:.2f}×, efficiency: {efficiency:.2f})")
            
            print("  FMM method scaling:")
            fmm_times = []
            
            for threads in thread_counts:
                os.environ['OMP_NUM_THREADS'] = str(threads)
                
                # Warmup
                fmm_openmp.fmm_force_theta(x[:1000], y[:1000], m[:1000], eps2, 
                                          domain, 0.6, ax[:1000], ay[:1000])
                
                # Time measurement
                times = []
                for _ in range(5):  # More runs since FMM is faster
                    start = time.perf_counter()
                    fmm_openmp.fmm_force_theta(x, y, m, eps2, domain, 0.6, ax, ay)
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times)
                fmm_times.append(avg_time)
                
                speedup = fmm_times[0] / avg_time if avg_time > 0 else 0
                efficiency = speedup / threads
                
                print(f"    {threads} threads: {avg_time:.4f}s (speedup: {speedup:.2f}×, efficiency: {efficiency:.2f})")
            
            # Summary for this size
            max_direct_speedup = max([direct_times[0]/t for t in direct_times])
            max_fmm_speedup = max([fmm_times[0]/t for t in fmm_times])
            
            print(f"  Summary for N={N:,}:")
            print(f"    Best direct speedup: {max_direct_speedup:.2f}×")
            print(f"    Best FMM speedup: {max_fmm_speedup:.2f}×")
            print(f"    Algorithmic advantage: {direct_times[0]/fmm_times[0]:.1f}×")
        
        return True
        
    except ImportError:
        print("FMM module not available")
        return False
    except Exception as e:
        print(f"Error testing FMM: {e}")
        return False

def analyze_system():
    """Analyze system for OpenMP compatibility issues."""
    print("\n=== System Analysis ===")
    
    # Check CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        # Extract relevant info
        lines = cpuinfo.split('\n')
        cpu_count = len([l for l in lines if l.startswith('processor')])
        cpu_model = [l for l in lines if l.startswith('model name')]
        
        print(f"CPU cores detected: {cpu_count}")
        if cpu_model:
            print(f"CPU model: {cpu_model[0].split(':')[1].strip()}")
        
    except:
        print("Could not read CPU info")
    
    # Check if we're in a container or restricted environment
    cgroup_limit = None
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
            period = int(f.read().strip())
        
        if quota > 0:
            cgroup_limit = quota / period
            print(f"Container CPU limit: {cgroup_limit:.2f} cores")
    except:
        pass
    
    # Check memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / (1024**2)
                print(f"Total memory: {mem_gb:.1f} GB")
                break
    except:
        print("Could not read memory info")
    
    # Check if running on cluster/SLURM
    if 'SLURM_JOB_ID' in os.environ:
        print("Running under SLURM scheduler")
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', 'not set')
        print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    
    # Check current limits
    import resource
    max_procs = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"Process limit: {max_procs}")

def main():
    """Main diagnostic and test routine."""
    print("=== Comprehensive OpenMP Force Test ===")
    print("This will aggressively test OpenMP functionality and identify issues.")
    
    # Step 1: Analyze system
    analyze_system()
    
    # Step 2: Run diagnostics
    run_diagnostics()
    
    # Step 3: Test with larger problems
    success = test_current_fmm()
    
    # Step 4: Provide recommendations
    print("\n=== Recommendations ===")
    if success:
        print("✓ OpenMP appears to be working")
        print("If you still see limited speedup:")
        print("1. The problem size might still be too small")
        print("2. Memory bandwidth could be the bottleneck")
        print("3. Try N > 100,000 for direct method")
        print("4. Consider that FMM reduces work so much that parallelization overhead dominates")
    else:
        print("✗ OpenMP issues detected")
        print("1. Check if running in a restricted environment (container, SLURM)")
        print("2. Try: export OMP_PROC_BIND=false")
        print("3. Check: ldd fmm_openmp*.so | grep omp")
        print("4. Install: sudo apt-get install libomp5 libomp-dev")

if __name__ == "__main__":
    main()
