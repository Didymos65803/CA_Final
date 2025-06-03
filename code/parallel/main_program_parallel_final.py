#!/usr/bin/env python3
# main_program_parallel_final.py
# HW6-inspired fixed version with proper parallelization

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# HW6-inspired OpenMP environment settings
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"] = "cores"
os.environ["OMP_SCHEDULE"] = "static"  # 參考 HW6 的靜態排程
os.environ["OMP_DYNAMIC"] = "false"

# Set path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import C++ modules
try:
    import force_kernel
    HAS_DIRECT = True
    print("✓ force_kernel loaded successfully")
except ImportError as e:
    HAS_DIRECT = False
    print(f"✗ force_kernel not available: {e}")

try:
    import fmm_kernel
    HAS_FMM = True
    print("✓ fmm_kernel loaded successfully")
except ImportError as e:
    HAS_FMM = False
    print(f"✗ fmm_kernel not available: {e}")

# Global settings
OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def initialize_particles(N, domain_size):
    """Initialize N particles with HW6-inspired memory layout"""
    rng = np.random.default_rng()
    
    # Generate random positions in a circle
    angles = rng.uniform(0, 2*math.pi, N)
    radii = domain_size * np.sqrt(rng.uniform(0, 1, N))
    
    # Create memory-aligned arrays (參考 HW6 的陣列分配)
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Vectorized computation
    np.multiply(radii, np.cos(angles), out=x)
    np.multiply(radii, np.sin(angles), out=y)
    
    # Ensure contiguous memory layout
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = np.ascontiguousarray(m, dtype=np.float64)
    
    return x, y, m

def safe_direct_force(x, y, m, eps2):
    """HW6-inspired direct force calculation"""
    N = len(x)
    
    # Pre-allocate output arrays
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_DIRECT:
        try:
            # Ensure proper memory layout
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            m_arr = np.ascontiguousarray(m, dtype=np.float64)
            
            force_kernel.direct_force(x_arr, y_arr, m_arr, eps2, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"Direct force calculation failed: {e}")
            return None, None
    else:
        return None, None

def safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G):
    """HW6-inspired FMM force calculation"""
    
    # Pre-allocate output arrays
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_FMM:
        try:
            # Ensure proper memory layout
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            m_arr = np.ascontiguousarray(m, dtype=np.float64)
            
            fmm_kernel.fmm_force(x_arr, y_arr, m_arr, N, domain_size, 
                               theta, maxLeaf, eps, G, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"FMM force calculation failed: {e}")
            return None, None
    else:
        return None, None

def hw6_inspired_benchmark():
    """HW6-inspired benchmark with proper parallelization analysis"""
    print("\nHW6-Inspired N-body Benchmark")
    print("=" * 60)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("No computation modules available")
        return
    
    # 參考 HW6 的測試規模
    test_sizes = [64, 128, 256, 512, 1024, 2048]
    thread_counts = [1, 2, 4, 8]
    
    results = {}
    
    for N in test_sizes:
        print(f"\nTesting with N = {N} particles")
        
        domain_size = 100.0
        theta = 0.5
        maxLeaf = 32
        eps = 0.01
        G = 1.0
        
        x, y, m = initialize_particles(N, domain_size)
        
        times_direct = []
        times_fmm = []
        
        for threads in thread_counts:
            # 參考 HW6 的環境設定
            os.environ["OMP_NUM_THREADS"] = str(threads)
            time.sleep(0.5)
            
            # Test direct method (參考 HW6 的測量方法)
            if HAS_DIRECT:
                # Warm up
                safe_direct_force(x, y, m, eps*eps)
                
                # 參考 HW6：多次測量取平均
                run_times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times:
                    avg_time = sum(run_times) / len(run_times)
                    times_direct.append(avg_time)
                    print(f"  Direct ({threads} threads): {avg_time:.6f} seconds")
                else:
                    times_direct.append(float('nan'))
            else:
                times_direct.append(float('nan'))
            
            # Test FMM method
            if HAS_FMM:
                # Warm up
                safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                run_times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times:
                    avg_time = sum(run_times) / len(run_times)
                    times_fmm.append(avg_time)
                    print(f"  FMM ({threads} threads): {avg_time:.6f} seconds")
                else:
                    times_fmm.append(float('nan'))
            else:
                times_fmm.append(float('nan'))
        
        # 計算加速比 (參考 HW6 的計算方式)
        speedup_direct = []
        speedup_fmm = []
        efficiency_direct = []
        efficiency_fmm = []
        
        if times_direct and not math.isnan(times_direct[0]) and times_direct[0] > 0:
            speedup_direct = [times_direct[0] / t for t in times_direct if not math.isnan(t) and t > 0]
            efficiency_direct = [s / tc for s, tc in zip(speedup_direct, thread_counts[:len(speedup_direct)])]
        
        if times_fmm and not math.isnan(times_fmm[0]) and times_fmm[0] > 0:
            speedup_fmm = [times_fmm[0] / t for t in times_fmm if not math.isnan(t) and t > 0]
            efficiency_fmm = [s / tc for s, tc in zip(speedup_fmm, thread_counts[:len(speedup_fmm)])]
        
        results[N] = {
            'threads': thread_counts[:len(speedup_direct)] if speedup_direct else thread_counts[:len(speedup_fmm)],
            'times_direct': times_direct,
            'times_fmm': times_fmm,
            'speedup_direct': speedup_direct,
            'speedup_fmm': speedup_fmm,
            'efficiency_direct': efficiency_direct,
            'efficiency_fmm': efficiency_fmm
        }
        
        if speedup_direct:
            max_speedup = max(speedup_direct)
            best_threads = thread_counts[speedup_direct.index(max_speedup)]
            print(f"  Direct best speedup: {max_speedup:.2f}x at {best_threads} threads")
        
        if speedup_fmm:
            max_speedup = max(speedup_fmm)
            best_threads = thread_counts[speedup_fmm.index(max_speedup)]
            print(f"  FMM best speedup: {max_speedup:.2f}x at {best_threads} threads")
    
    # 創建類似 HW6 的圖表
    create_hw6_style_plots(results, test_sizes, thread_counts)
    
    return results

def create_hw6_style_plots(results, test_sizes, thread_counts):
    """創建類似 HW6 plot_scaling.py 的圖表"""
    
    # 參考 HW6 的圖表風格
    plt.figure(figsize=(12, 8))
    
    # 使用 HW6 相同的顏色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_sizes)))
    
    # Direct Method Speedup
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_direct']:
            threads = results[N]['threads']
            speedup = results[N]['speedup_direct']
            plt.plot(threads, speedup, 'o-', color=colors[i], 
                    label=f"Direct N={N}", linewidth=2, markersize=6)
    
    # FMM Method Speedup (虛線)
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_fmm']:
            threads = results[N]['threads']
            speedup = results[N]['speedup_fmm']
            plt.plot(threads, speedup, 's--', color=colors[i], 
                    label=f"FMM N={N}", linewidth=2, markersize=6, alpha=0.7)
    
    # 理想加速比線 (參考 HW6)
    plt.plot(thread_counts, thread_counts, 'k--', label="Ideal", linewidth=2)
    
    plt.xlabel("Number of Threads")
    plt.ylabel("Speed-up $T_1/T_p$")
    plt.title("N-body Speed-up vs. Number of Threads (HW6-Inspired)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    
    # 保存圖表
    speedup_path = os.path.join(OUTPUT_DIR, "nbody_speedup_hw6_inspired.png")
    plt.savefig(speedup_path, dpi=300)
    plt.close()
    print(f"\n✓ HW6-inspired speedup plot saved to {speedup_path}")
    
    # 創建效率圖表
    plt.figure(figsize=(12, 8))
    
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_direct']:
            threads = results[N]['threads']
            efficiency = results[N]['efficiency_direct']
            plt.plot(threads, efficiency, 'o-', color=colors[i], 
                    label=f"Direct N={N}", linewidth=2, markersize=6)
    
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_fmm']:
            threads = results[N]['threads']
            efficiency = results[N]['efficiency_fmm']
            plt.plot(threads, efficiency, 's--', color=colors[i], 
                    label=f"FMM N={N}", linewidth=2, markersize=6, alpha=0.7)
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="Good (80%)")
    plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="Fair (50%)")
    
    plt.xlabel("Number of Threads")
    plt.ylabel("Parallel Efficiency")
    plt.title("N-body Parallel Efficiency (HW6-Inspired)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    
    efficiency_path = os.path.join(OUTPUT_DIR, "nbody_efficiency_hw6_inspired.png")
    plt.savefig(efficiency_path, dpi=300)
    plt.close()
    print(f"✓ HW6-inspired efficiency plot saved to {efficiency_path}")

def quick_benchmark():
    """Quick benchmark scaling"""
    print("\nQuick Benchmark Scaling")
    print("=" * 50)
    
    Ns = [50, 100, 200, 500]
    steps = 3
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    
    print(f"Using 8 threads (OpenMP)")
    print(f"Testing particle counts: {Ns}")
    
    results = []
    
    for N in Ns:
        print(f"\nTesting N = {N}")
        
        x, y, m = initialize_particles(N, domain_size)
        
        # Test direct method
        t_direct = None
        if HAS_DIRECT:
            try:
                print("  Testing direct method...")
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is None:
                        raise Exception("Direct force failed")
                t_direct = (time.perf_counter() - t0) / steps
                print(f"  ✓ Direct method: {t_direct:.6f} seconds")
            except Exception as e:
                print(f"  ✗ Direct method: Failed ({e})")
                t_direct = float('nan')
        else:
            print("  ✗ Direct method: Not available")
            t_direct = float('nan')
        
        # Test Barnes-Hut
        t_bh = None
        if HAS_FMM:
            try:
                print("  Testing Barnes-Hut method...")
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                    if ax is None:
                        raise Exception("BH force failed")
                t_bh = (time.perf_counter() - t0) / steps
                print(f"  ✓ Barnes-Hut: {t_bh:.6f} seconds")
            except Exception as e:
                print(f"  ✗ Barnes-Hut: Failed ({e})")
                t_bh = float('nan')
        else:
            print("  ✗ Barnes-Hut: Not available")
            t_bh = float('nan')
        
        # Test FMM
        t_fmm = None
        if HAS_FMM:
            try:
                print("  Testing FMM method...")
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                    if ax is None:
                        raise Exception("FMM force failed")
                t_fmm = (time.perf_counter() - t0) / steps
                print(f"  ✓ FMM: {t_fmm:.6f} seconds")
            except Exception as e:
                print(f"  ✗ FMM: Failed ({e})")
                t_fmm = float('nan')
        else:
            print("  ✗ FMM: Not available")
            t_fmm = float('nan')
        
        results.append((N, t_direct, t_bh, t_fmm))
    
    save_scaling_results(results, "scaling_quick")
    print("\n✓ Quick benchmark completed!")

def save_trajectory_and_energy():
    """Save trajectory + energy plot"""
    print("\nSave Trajectory + Energy Plot")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 200): "))
        method = input("Choose method (direct/bh/fmm): ").strip().lower()
        steps = int(input("Enter number of steps (e.g., 100): "))
    except ValueError:
        print("Invalid input. Using default values.")
        N = 200
        method = "fmm"
        steps = 100
    
    if method not in ["direct", "bh", "fmm"]:
        print("Invalid method. Using FMM.")
        method = "fmm"
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    # Store trajectory and energy
    trajectory = []
    energies = []
    
    print(f"Running simulation with {method} method...")
    
    for step in range(steps):
        if step % 20 == 0:
            print(f"  Step {step}/{steps}")
        
        # Calculate forces
        if method == "direct" and HAS_DIRECT:
            ax, ay = safe_direct_force(x, y, m, eps*eps)
        elif method == "bh" and HAS_FMM:
            ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
        elif method == "fmm" and HAS_FMM:
            ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
        else:
            print(f"Method {method} not available")
            return
        
        if ax is None:
            print("Force calculation failed")
            return
        
        # Leapfrog integration
        vx += 0.5 * dt * ax
        vy += 0.5 * dt * ay
        x += dt * vx
        y += dt * vy
        
        # Calculate energy (kinetic + potential)
        ke = 0.5 * np.sum(m * (vx**2 + vy**2))
        pe = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r = math.sqrt(dx*dx + dy*dy + eps*eps)
                pe -= G * m[i] * m[j] / r
        
        trajectory.append((x.copy(), y.copy()))
        energies.append(ke + pe)
    
    # Create and save plots
    create_trajectory_animation(trajectory, method, N)
    create_energy_plot(energies, dt, method, N)
    print("✓ Trajectory and energy plots saved!")

def live_simulation_animation():
    """Live simulation animation"""
    print("\nLive Simulation Animation")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 100): "))
        method = input("Choose method (direct/bh/fmm): ").strip().lower()
    except ValueError:
        print("Invalid input. Using default values.")
        N = 100
        method = "fmm"
    
    if method not in ["direct", "bh", "fmm"]:
        print("Invalid method. Using FMM.")
        method = "fmm"
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    frames = 200
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    print(f"Creating animation with {method} method...")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-domain_size*1.2, domain_size*1.2)
    ax.set_ylim(-domain_size*1.2, domain_size*1.2)
    scat = ax.scatter(x, y, s=10, c='blue')
    ax.set_title(f"Live Simulation ({method.upper()}, N={N})")
    
    def update(frame):
        nonlocal x, y, vx, vy
        
        # Calculate forces
        if method == "direct" and HAS_DIRECT:
            ax_arr, ay_arr = safe_direct_force(x, y, m, eps*eps)
        elif method == "bh" and HAS_FMM:
            ax_arr, ay_arr = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
        elif method == "fmm" and HAS_FMM:
            ax_arr, ay_arr = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
        else:
            return scat,
        
        if ax_arr is None:
            return scat,
        
        # Update positions
        vx += dt * ax_arr
        vy += dt * ay_arr
        x += dt * vx
        y += dt * vy
        
        scat.set_offsets(np.column_stack((x, y)))
        return scat,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Save animation
    gif_path = os.path.join(OUTPUT_DIR, f"live_simulation_{method}_{N}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close()
    print(f"✓ Live simulation saved to {gif_path}")

def large_n_scaling():
    """Large-N scaling test"""
    print("\nLarge-N Scaling Test")
    print("=" * 50)
    
    Ns = [500, 1000, 2000, 4000, 8000]
    steps = 3
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 32
    eps = 0.01
    G = 1.0
    
    print(f"Testing large N values: {Ns}")
    
    results = []
    
    for N in Ns:
        print(f"\nTesting N = {N}")
        
        x, y, m = initialize_particles(N, domain_size)
        
        # Test direct method (skip for very large N)
        t_direct = float('nan') if N > 2000 else None
        if t_direct is None and HAS_DIRECT:
            try:
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                t_direct = (time.perf_counter() - t0) / steps
                print(f"  Direct method: {t_direct:.6f} seconds")
            except:
                t_direct = float('nan')
        
        # Test other methods
        t_bh = float('nan')
        t_fmm = float('nan')
        
        if HAS_FMM:
            try:
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                t_bh = (time.perf_counter() - t0) / steps
                print(f"  Barnes-Hut: {t_bh:.6f} seconds")
            except:
                pass
            
            try:
                t0 = time.perf_counter()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                t_fmm = (time.perf_counter() - t0) / steps
                print(f"  FMM: {t_fmm:.6f} seconds")
            except:
                pass
        
        results.append((N, t_direct, t_bh, t_fmm))
    
    save_scaling_results(results, "scaling_large")
    print("\n✓ Large-N scaling test completed!")

def energy_conservation_test():
    """Energy conservation test"""
    print("\nEnergy Conservation Test")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 200): "))
    except ValueError:
        N = 200
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    steps = 500
    
    methods = []
    if HAS_DIRECT:
        methods.append(("Direct", "direct"))
    if HAS_FMM:
        methods.append(("Barnes-Hut", "bh"))
        methods.append(("FMM", "fmm"))
    
    plt.figure(figsize=(10, 6))
    
    for method_name, method in methods:
        print(f"\nTesting {method_name}...")
        
        # Initialize particles
        x, y, m = initialize_particles(N, domain_size)
        vx = np.zeros(N, dtype=np.float64)
        vy = np.zeros(N, dtype=np.float64)
        
        # Calculate initial energy
        E0 = calculate_total_energy(x, y, vx, vy, m, G, eps)
        
        times = []
        rel_errors = []
        
        for step in range(0, steps, 5):  # Record every 5 steps
            # Calculate forces
            if method == "direct":
                ax, ay = safe_direct_force(x, y, m, eps*eps)
            elif method == "bh":
                ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
            elif method == "fmm":
                ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
            
            if ax is None:
                break
            
            # Integrate for 5 steps
            for _ in range(5):
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                x += dt * vx
                y += dt * vy
                
                # Recalculate forces
                if method == "direct":
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                elif method == "bh":
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                elif method == "fmm":
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
            
            # Calculate current energy and relative error
            E = calculate_total_energy(x, y, vx, vy, m, G, eps)
            rel_error = abs(E - E0) / abs(E0)
            
            times.append(step * dt)
            rel_errors.append(rel_error)
        
        plt.semilogy(times, rel_errors, label=method_name)
    
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy Conservation Test (N={N})")
    plt.legend()
    plt.grid(True)
    
    energy_path = os.path.join(OUTPUT_DIR, "energy_conservation.png")
    plt.savefig(energy_path, dpi=300)
    plt.close()
    print(f"✓ Energy conservation plot saved to {energy_path}")

def parameter_optimization():
    """Parameter optimization"""
    print("\nParameter Optimization")
    print("=" * 50)
    
    N = 100
    domain_size = 50.0
    eps = 0.01
    G = 1.0
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    
    # Get reference solution
    if not HAS_DIRECT:
        print("Direct method not available for reference")
        return
    
    ax_ref, ay_ref = safe_direct_force(x, y, m, eps*eps)
    if ax_ref is None:
        print("Failed to compute reference solution")
        return
    
    thetas = [0.1, 0.3, 0.5, 0.7, 1.0]
    bh_errors = []
    fmm_errors = []
    
    for theta in thetas:
        # Test Barnes-Hut
        if HAS_FMM:
            ax_bh, ay_bh = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
            if ax_bh is not None:
                bh_error = np.sqrt(np.sum((ax_bh - ax_ref)**2 + (ay_bh - ay_ref)**2)) / np.sqrt(np.sum(ax_ref**2 + ay_ref**2))
                bh_errors.append(bh_error)
            else:
                bh_errors.append(float('nan'))
            
            # Test FMM
            ax_fmm, ay_fmm = safe_fmm_force(x, y, m, N, domain_size, theta, 8, eps, G)
            if ax_fmm is not None:
                fmm_error = np.sqrt(np.sum((ax_fmm - ax_ref)**2 + (ay_fmm - ay_ref)**2)) / np.sqrt(np.sum(ax_ref**2 + ay_ref**2))
                fmm_errors.append(fmm_error)
            else:
                fmm_errors.append(float('nan'))
        else:
            bh_errors.append(float('nan'))
            fmm_errors.append(float('nan'))
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.semilogy(thetas, bh_errors, 's-', label="Barnes-Hut Error")
    plt.semilogy(thetas, fmm_errors, '^-', label="FMM Error")
    plt.xlabel("Theta (opening angle)")
    plt.ylabel("Relative Force Error")
    plt.title(f"Parameter Optimization (N={N})")
    plt.legend()
    plt.grid(True)
    
    param_path = os.path.join(OUTPUT_DIR, "parameter_optimization.png")
    plt.savefig(param_path, dpi=300)
    plt.close()
    print(f"✓ Parameter optimization plot saved to {param_path}")

def system_information():
    """System information"""
    print("\nSystem Information")
    print("=" * 50)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print(f"OpenMP proc bind: {os.environ.get('OMP_PROC_BIND', 'Not set')}")
    print(f"OpenMP places: {os.environ.get('OMP_PLACES', 'Not set')}")
    print(f"OpenMP schedule: {os.environ.get('OMP_SCHEDULE', 'Not set')}")
    print(f"Direct method module: {'Available' if HAS_DIRECT else 'Not available'}")
    print(f"FMM module: {'Available' if HAS_FMM else 'Not available'}")
    
    try:
        import platform
        print(f"Operating system: {platform.platform()}")
    except:
        pass
    
    try:
        import multiprocessing
        print(f"CPU cores: {multiprocessing.cpu_count()}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except:
        pass

# Helper functions
def calculate_total_energy(x, y, vx, vy, m, G, eps):
    """Calculate total energy (kinetic + potential)"""
    N = len(x)
    
    # Kinetic energy
    ke = 0.5 * np.sum(m * (vx**2 + vy**2))
    
    # Potential energy
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = math.sqrt(dx*dx + dy*dy + eps*eps)
            pe -= G * m[i] * m[j] / r
    
    return ke + pe

def save_scaling_results(results, filename):
    """Save scaling results to CSV and create plot"""
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    with open(csv_path, "w") as f:
        f.write("N,Direct,BH,FMM\n")
        for N, t_direct, t_bh, t_fmm in results:
            f.write(f"{N},{t_direct},{t_bh},{t_fmm}\n")
    
    # Create plot
    Ns = [r[0] for r in results]
    times_direct = [r[1] for r in results if not math.isnan(r[1])]
    times_bh = [r[2] for r in results if not math.isnan(r[2])]
    times_fmm = [r[3] for r in results if not math.isnan(r[3])]
    
    plt.figure(figsize=(8, 6))
    
    if times_direct:
        Ns_direct = [r[0] for r in results if not math.isnan(r[1])]
        plt.loglog(Ns_direct, times_direct, 'o-', label="Direct O(N²)", linewidth=2)
    
    if times_bh:
        Ns_bh = [r[0] for r in results if not math.isnan(r[2])]
        plt.loglog(Ns_bh, times_bh, 's-', label="Barnes-Hut O(N log N)", linewidth=2)
    
    if times_fmm:
        Ns_fmm = [r[0] for r in results if not math.isnan(r[3])]
        plt.loglog(Ns_fmm, times_fmm, '^-', label="FMM O(N)", linewidth=2)
    
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time per Step (seconds)")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(True)
    
    png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    print(f"✓ Results saved to {csv_path} and {png_path}")

def create_trajectory_animation(trajectory, method, N):
    """Create trajectory animation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set limits based on trajectory
    all_x = np.concatenate([pos[0] for pos in trajectory])
    all_y = np.concatenate([pos[1] for pos in trajectory])
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    ax.set_xlim(all_x.min() - margin*x_range, all_x.max() + margin*x_range)
    ax.set_ylim(all_y.min() - margin*y_range, all_y.max() + margin*y_range)
    ax.set_title(f"Trajectory ({method.upper()}, N={N})")
    
    scat = ax.scatter(trajectory[0][0], trajectory[0][1], s=10, c='blue')
    
    def animate(frame):
        x, y = trajectory[frame]
        scat.set_offsets(np.column_stack((x, y)))
        return scat,
    
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=True)
    
    gif_path = os.path.join(OUTPUT_DIR, f"trajectory_{method}_{N}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close()
    print(f"✓ Trajectory animation saved to {gif_path}")

def create_energy_plot(energies, dt, method, N):
    """Create energy vs time plot"""
    times = np.arange(len(energies)) * dt
    
    plt.figure(figsize=(8, 6))
    plt.plot(times, energies, '-')
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"Energy vs Time ({method.upper()}, N={N})")
    plt.grid(True)
    
    energy_path = os.path.join(OUTPUT_DIR, f"energy_{method}_{N}.png")
    plt.savefig(energy_path, dpi=300)
    plt.close()
    print(f"✓ Energy plot saved to {energy_path}")

def main_menu():
    """Main menu with HW6-inspired benchmark"""
    while True:
        print("\n" + "=" * 60)
        print("2D N-Body Problem Simulation Platform")
        print("(HW6-Inspired Parallel Analysis)")
        print("=" * 60)
        print("Select function:")
        print(" 1) Quick benchmark scaling")
        print(" 2) Save trajectory + energy plot")
        print(" 3) Live simulation animation")
        print(" 4) Large-N scaling test")
        print(" 5) Energy conservation test")
        print(" 6) Parameter optimization")
        print(" 7) HW6-inspired benchmark")  # 新的主要測試
        print(" 8) System information")
        print(" q) Exit program")
        print("=" * 60)
        
        choice = input("Please enter your choice: ").strip().lower()
        
        if choice == '1':
            quick_benchmark()
        elif choice == '2':
            save_trajectory_and_energy()
        elif choice == '3':
            live_simulation_animation()
        elif choice == '4':
            large_n_scaling()
        elif choice == '5':
            energy_conservation_test()
        elif choice == '6':
            parameter_optimization()
        elif choice == '7':
            hw6_inspired_benchmark()  # 主要的並行化測試
        elif choice == '8':
            system_information()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    print("2D N-Body Problem Simulation Platform starting...")
    print("HW6-inspired parallel performance analysis version")
    print(f"OpenMP configuration:")
    print(f"  - Threads: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  - Proc bind: {os.environ.get('OMP_PROC_BIND')}")
    print(f"  - Places: {os.environ.get('OMP_PLACES')}")
    print(f"  - Schedule: {os.environ.get('OMP_SCHEDULE')}")
    
    # Check module availability
    if not HAS_DIRECT and not HAS_FMM:
        print("Error: No available computation modules!")
        print("Please ensure force_kernel and fmm_kernel modules are properly compiled.")
        print("Run: python setup.py build_ext --inplace")
        sys.exit(1)
    
    main_menu()

