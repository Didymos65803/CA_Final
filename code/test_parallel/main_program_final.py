#!/usr/bin/env python3
# main_program_final.py - Final working N-Body simulation program

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

def setup_openmp():
    """Setup optimal OpenMP environment"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    num_threads = min(cpu_count, 8)
    
    env_settings = {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_SCHEDULE": "guided",
        "OMP_DYNAMIC": "false",
        "OMP_WAIT_POLICY": "passive"
    }
    
    for key, value in env_settings.items():
        os.environ[key] = value
    
    print(f"OpenMP configured: {num_threads} threads")
    return num_threads

# Setup environment before importing modules
num_threads = setup_openmp()

# Import simulation modules
try:
    import force_kernel
    HAS_DIRECT = True
    print("✓ Direct force kernel loaded")
except ImportError as e:
    HAS_DIRECT = False
    print(f"✗ Direct force kernel failed: {e}")

try:
    import fmm_kernel
    HAS_FMM = True
    print("✓ FMM kernel loaded")
except ImportError as e:
    HAS_FMM = False
    print(f"✗ FMM kernel failed: {e}")

# Output directory
OUTPUT_DIR = "results_final"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class ParticleSystem:
    """Optimized particle system for N-body simulation"""
    
    def __init__(self, N, domain_size=50.0, seed=42):
        self.N = N
        self.domain_size = domain_size
        
        # Initialize with reproducible random seed
        rng = np.random.default_rng(seed)
        
        # Generate particles in circular distribution
        angles = rng.uniform(0, 2*math.pi, N)
        radii = domain_size * np.sqrt(rng.uniform(0, 1, N))
        
        # Ensure memory-aligned arrays
        self.x = np.ascontiguousarray(radii * np.cos(angles), dtype=np.float64)
        self.y = np.ascontiguousarray(radii * np.sin(angles), dtype=np.float64)
        self.m = np.ascontiguousarray(rng.uniform(0.8, 1.2, N), dtype=np.float64)
        
    def get_arrays(self):
        """Get particle arrays"""
        return self.x, self.y, self.m

def compute_direct_forces(x, y, m, eps=0.01):
    """Compute forces using direct method"""
    if not HAS_DIRECT:
        return None, None
    
    N = len(x)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    try:
        force_kernel.direct_force(x, y, m, eps*eps, ax, ay)
        return ax, ay
    except Exception as e:
        print(f"Direct force computation failed: {e}")
        return None, None

def compute_fmm_forces(x, y, m, domain_size=50.0, theta=0.5, maxLeaf=16, eps=0.01, G=1.0):
    """Compute forces using FMM method"""
    if not HAS_FMM:
        return None, None
    
    N = len(x)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    try:
        fmm_kernel.fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G, ax, ay)
        return ax, ay
    except Exception as e:
        print(f"FMM force computation failed: {e}")
        return None, None

def benchmark_performance():
    """Comprehensive performance benchmark"""
    print("\nPerformance Benchmark")
    print("=" * 50)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("No computation modules available")
        return
    
    test_sizes = [100, 200, 500, 1000, 2000]
    results = []
    
    print(f"{'N':>6} {'Direct':>12} {'FMM':>12} {'Speedup':>10} {'FMM/Direct':>12}")
    print("-" * 65)
    
    for N in test_sizes:
        print(f"{N:>6}", end=" ")
        
        # Create test system
        system = ParticleSystem(N)
        x, y, m = system.get_arrays()
        
        # Test direct method
        t_direct = None
        if HAS_DIRECT and N <= 2000:  # Limit direct method for large N
            try:
                # Warmup
                compute_direct_forces(x, y, m)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(3):
                    ax, ay = compute_direct_forces(x, y, m)
                    if ax is None:
                        raise Exception("Direct computation failed")
                t_direct = (time.perf_counter() - start_time) / 3
                
                print(f"{t_direct:>12.6f}", end=" ")
                
            except Exception as e:
                print(f"{'Failed':>12}", end=" ")
                t_direct = None
        else:
            print(f"{'Skipped':>12}", end=" ")
            t_direct = None
        
        # Test FMM method
        t_fmm = None
        if HAS_FMM:
            try:
                # Warmup
                compute_fmm_forces(x, y, m)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(3):
                    ax, ay = compute_fmm_forces(x, y, m)
                    if ax is None:
                        raise Exception("FMM computation failed")
                t_fmm = (time.perf_counter() - start_time) / 3
                
                print(f"{t_fmm:>12.6f}", end=" ")
                
            except Exception as e:
                print(f"{'Failed':>12}", end=" ")
                t_fmm = None
        else:
            print(f"{'N/A':>12}", end=" ")
        
        # Calculate speedup
        if t_direct and t_fmm:
            speedup = t_direct / t_fmm
            ratio = t_fmm / t_direct
            print(f"{speedup:>10.2f} {ratio:>12.4f}")
        elif t_fmm:
            print(f"{'N/A':>10} {'N/A':>12}")
        else:
            print(f"{'N/A':>10} {'N/A':>12}")
        
        results.append((N, t_direct, t_fmm))
    
    # Create performance plot
    create_performance_plot(results)
    
    return results

def test_accuracy():
    """Test accuracy of FMM vs direct method"""
    print("\nAccuracy Test")
    print("=" * 50)
    
    if not HAS_DIRECT or not HAS_FMM:
        print("Both methods required for accuracy test")
        return
    
    N = 100
    system = ParticleSystem(N, seed=123)
    x, y, m = system.get_arrays()
    
    print(f"Testing with {N} particles...")
    
    # Compute reference solution (direct method)
    print("Computing reference solution (direct method)...")
    ax_ref, ay_ref = compute_direct_forces(x, y, m)
    
    if ax_ref is None:
        print("Failed to compute reference solution")
        return
    
    # Test FMM with different theta values
    theta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    errors = []
    times = []
    
    print(f"{'Theta':>8} {'Rel Error':>12} {'Time (s)':>12}")
    print("-" * 35)
    
    for theta in theta_values:
        start_time = time.perf_counter()
        ax_fmm, ay_fmm = compute_fmm_forces(x, y, m, theta=theta)
        elapsed = time.perf_counter() - start_time
        
        if ax_fmm is not None:
            # Calculate relative error
            force_ref = np.sqrt(ax_ref**2 + ay_ref**2)
            force_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
            
            rel_error = np.mean(np.abs(force_fmm - force_ref) / (force_ref + 1e-10))
            
            errors.append(rel_error)
            times.append(elapsed)
            
            print(f"{theta:>8.1f} {rel_error:>12.4e} {elapsed:>12.6f}")
        else:
            errors.append(float('nan'))
            times.append(float('nan'))
            print(f"{theta:>8.1f} {'Failed':>12} {'Failed':>12}")
    
    # Create accuracy plot
    create_accuracy_plot(theta_values, errors, times)

def test_parallel_scaling():
    """Test parallel scaling performance"""
    print("\nParallel Scaling Test")
    print("=" * 50)
    
    if not HAS_FMM:
        print("FMM method required for parallel scaling test")
        return
    
    N = 1000
    system = ParticleSystem(N)
    x, y, m = system.get_arrays()
    
    thread_counts = [1, 2, 4, 8]
    times = []
    
    print(f"Testing with {N} particles")
    print(f"{'Threads':>8} {'Time (s)':>12} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 50)
    
    for threads in thread_counts:
        # Set thread count
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.2)  # Allow environment change to take effect
        
        try:
            # Warmup
            compute_fmm_forces(x, y, m)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(5):
                ax, ay = compute_fmm_forces(x, y, m)
                if ax is None:
                    raise Exception("FMM computation failed")
            elapsed = (time.perf_counter() - start_time) / 5
            
            times.append(elapsed)
            
            if len(times) == 1:
                speedup = 1.0
                efficiency = 1.0
            else:
                speedup = times[0] / elapsed
                efficiency = speedup / threads
            
            print(f"{threads:>8} {elapsed:>12.6f} {speedup:>10.2f} {efficiency:>12.1%}")
            
        except Exception as e:
            print(f"{threads:>8} {'Failed':>12} {'N/A':>10} {'N/A':>12}")
            times.append(float('nan'))
    
    # Restore original thread count
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    # Create scaling plot
    create_scaling_plot(thread_counts, times)

def create_performance_plot(results):
    """Create performance comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    Ns = [r[0] for r in results]
    times_direct = [r[1] for r in results if r[1] is not None]
    times_fmm = [r[2] for r in results if r[2] is not None]
    
    Ns_direct = [r[0] for r in results if r[1] is not None]
    Ns_fmm = [r[0] for r in results if r[2] is not None]
    
    # Plot results
    if times_direct:
        ax.loglog(Ns_direct, times_direct, 'o-', label='Direct O(N²)', 
                 linewidth=2.5, markersize=8, color='blue')
    
    if times_fmm:
        ax.loglog(Ns_fmm, times_fmm, 's-', label='FMM O(N log N)', 
                 linewidth=2.5, markersize=8, color='orange')
    
    # Add theoretical scaling lines
    if times_direct and len(Ns_direct) > 1:
        N_theory = np.logspace(np.log10(min(Ns_direct)), np.log10(max(Ns_direct)), 100)
        t_theory_n2 = times_direct[0] * (N_theory / Ns_direct[0])**2
        ax.loglog(N_theory, t_theory_n2, '--', alpha=0.7, color='blue', label='O(N²) theory')
    
    if times_fmm and len(Ns_fmm) > 1:
        N_theory = np.logspace(np.log10(min(Ns_fmm)), np.log10(max(Ns_fmm)), 100)
        t_theory_nlogn = times_fmm[0] * (N_theory / Ns_fmm[0]) * np.log2(N_theory) / np.log2(Ns_fmm[0])
        ax.loglog(N_theory, t_theory_nlogn, ':', alpha=0.7, color='orange', label='O(N log N) theory')
    
    ax.set_xlabel('Number of Particles (N)', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Performance Comparison: Fixed Implementation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance plot saved to {plot_path}")

def create_accuracy_plot(theta_values, errors, times):
    """Create accuracy analysis plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter valid data
    valid_data = [(t, e, tm) for t, e, tm in zip(theta_values, errors, times) 
                  if not math.isnan(e) and not math.isnan(tm)]
    
    if valid_data:
        theta_valid, errors_valid, times_valid = zip(*valid_data)
        
        # Error vs theta
        ax1.semilogy(theta_valid, errors_valid, 'o-', linewidth=2.5, markersize=8)
        ax1.set_xlabel('Theta (Opening Angle)', fontsize=12)
        ax1.set_ylabel('Relative Error', fontsize=12)
        ax1.set_title('FMM Accuracy vs Theta', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Time vs theta
        ax2.plot(theta_valid, times_valid, 's-', linewidth=2.5, markersize=8, color='orange')
        ax2.set_xlabel('Theta (Opening Angle)', fontsize=12)
        ax2.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax2.set_title('FMM Performance vs Theta', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "accuracy_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Accuracy plot saved to {plot_path}")

def create_scaling_plot(thread_counts, times):
    """Create parallel scaling plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter valid data
    valid_times = [t for t in times if not math.isnan(t)]
    valid_threads = thread_counts[:len(valid_times)]
    
    if len(valid_times) > 1:
        # Calculate speedup and efficiency
        speedups = [valid_times[0] / t for t in valid_times]
        efficiency = [s / tc for s, tc in zip(speedups, valid_threads)]
        
        # Speedup plot
        ax1.plot(valid_threads, speedups, 'o-', linewidth=2.5, markersize=8, label='Actual')
        ax1.plot(valid_threads, valid_threads, '--', alpha=0.7, label='Ideal')
        ax1.set_xlabel('Number of Threads', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Parallel Speedup', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax2.plot(valid_threads, efficiency, 's-', linewidth=2.5, markersize=8, color='orange')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Good (80%)')
        ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='Poor (50%)')
        ax2.set_xlabel('Number of Threads', fontsize=12)
        ax2.set_ylabel('Parallel Efficiency', fontsize=12)
        ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "parallel_scaling.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Parallel scaling plot saved to {plot_path}")

def energy_conservation_test():
    """Test energy conservation in time integration"""
    print("\nEnergy Conservation Test")
    print("=" * 50)
    
    if not HAS_FMM:
        print("FMM method required for energy conservation test")
        return
    
    N = 100
    system = ParticleSystem(N, domain_size=25.0)
    x, y, m = system.get_arrays()
    
    # Initialize velocities (circular motion)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        r = np.sqrt(x[i]**2 + y[i]**2)
        if r > 0:
            # Circular velocity for rough equilibrium
            v_circ = np.sqrt(np.sum(m) * 0.1 / r)  # Rough estimate
            vx[i] = -v_circ * y[i] / r
            vy[i] = v_circ * x[i] / r
    
    # Time integration parameters
    dt = 0.001
    steps = 1000
    save_every = 10
    
    # Storage for energy history
    times = []
    energies = []
    
    print(f"Running {steps} steps with dt={dt}")
    
    # Calculate initial energy
    def calculate_energy():
        ke = 0.5 * np.sum(m * (vx**2 + vy**2))
        pe = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r = np.sqrt(dx**2 + dy**2 + 0.01**2)
                pe -= m[i] * m[j] / r
        return ke + pe
    
    E0 = calculate_energy()
    print(f"Initial energy: {E0:.6e}")
    
    # Time integration loop
    for step in range(steps):
        # Compute forces
        ax, ay = compute_fmm_forces(x, y, m)
        if ax is None:
            print("Force computation failed")
            break
        
        # Leapfrog integration
        vx += 0.5 * dt * ax
        vy += 0.5 * dt * ay
        
        x += dt * vx
        y += dt * vy
        
        # Recompute forces
        ax, ay = compute_fmm_forces(x, y, m)
        if ax is None:
            break
        
        vx += 0.5 * dt * ax
        vy += 0.5 * dt * ay
        
        # Save energy
        if step % save_every == 0:
            E = calculate_energy()
            times.append(step * dt)
            energies.append(E)
    
    if energies:
        # Calculate relative energy error
        rel_errors = [abs(E - E0) / abs(E0) for E in energies]
        
        print(f"Final relative energy error: {rel_errors[-1]:.4e}")
        
        # Create energy conservation plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Total energy vs time
        ax1.plot(times, energies, 'b-', linewidth=2)
        ax1.axhline(y=E0, color='red', linestyle='--', alpha=0.7, label='Initial Energy')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Total Energy', fontsize=12)
        ax1.set_title('Energy Conservation Test', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative error vs time
        ax2.semilogy(times, rel_errors, 'r-', linewidth=2)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Relative Energy Error', fontsize=12)
        ax2.set_title('Energy Conservation Error', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "energy_conservation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Energy conservation plot saved to {plot_path}")

def system_info():
    """Display system information"""
    print("\nSystem Information")
    print("=" * 50)
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    
    # OpenMP information
    omp_vars = ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE"]
    for var in omp_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Module availability
    print(f"Direct method: {'Available' if HAS_DIRECT else 'Not available'}")
    print(f"FMM method: {'Available' if HAS_FMM else 'Not available'}")
    
    # System information
    try:
        import platform
        import multiprocessing
        print(f"Operating system: {platform.system()} {platform.release()}")
        print(f"CPU cores: {multiprocessing.cpu_count()}")
        print(f"Architecture: {platform.machine()}")
    except:
        pass

def comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    print("\nComprehensive Benchmark Suite")
    print("=" * 60)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("No computation modules available")
        return
    
    print("Running all benchmark tests...")
    print("This may take several minutes...")
    
    # Test 1: Performance benchmark
    benchmark_performance()
    
    # Test 2: Accuracy test
    test_accuracy()
    
    # Test 3: Parallel scaling
    test_parallel_scaling()
    
    # Test 4: Energy conservation
    energy_conservation_test()
    
    print("\n" + "=" * 60)
    print("✓ Comprehensive benchmark completed!")
    print(f"✓ Results saved to {OUTPUT_DIR}/")
    print("=" * 60)

def main_menu():
    """Main menu interface"""
    while True:
        print("\n" + "=" * 60)
        print("N-Body Simulation Platform - Final Working Version")
        print("=" * 60)
        print("Available functions:")
        print(" 1) Performance benchmark")
        print(" 2) Accuracy test")
        print(" 3) Parallel scaling test")
        print(" 4) Energy conservation test")
        print(" 5) Comprehensive benchmark suite")
        print(" 6) System information")
        print(" q) Exit")
        print("=" * 60)
        
        choice = input("Enter your choice: ").strip().lower()
        
        if choice == '1':
            benchmark_performance()
        elif choice == '2':
            test_accuracy()
        elif choice == '3':
            test_parallel_scaling()
        elif choice == '4':
            energy_conservation_test()
        elif choice == '5':
            comprehensive_benchmark()
        elif choice == '6':
            system_info()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("N-Body Simulation Platform Starting...")
    print("Final Working Version with Proper Parallelization")
    
    # Check module availability
    if not HAS_DIRECT and not HAS_FMM:
        print("\nError: No computation modules available!")
        print("Please compile the modules first:")
        print("  python setup_final.py build_ext --inplace")
        print("Then run the test suite:")
        print("  python test_final.py")
        sys.exit(1)
    
    main_menu()
