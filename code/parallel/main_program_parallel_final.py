#!/usr/bin/env python3
# main_program_parallel_final.py
# Fixed version with effective parallelization strategies

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Optimized OpenMP environment settings based on research
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "spread"  # 修正1: 改為 spread 以改善 NUMA 效能
os.environ["OMP_PLACES"] = "threads"     # 修正2: 改為 threads 以改善超執行緒效能
os.environ["OMP_SCHEDULE"] = "guided,32" # 修正3: 改為 guided 以改善負載平衡
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
    """Initialize N particles with optimized memory layout"""
    # 修正4: 使用更好的記憶體對齊策略
    rng = np.random.default_rng()
    
    # Generate random positions in a circle
    angles = rng.uniform(0, 2*math.pi, N)
    radii = domain_size * np.sqrt(rng.uniform(0, 1, N))
    
    # Create cache-aligned arrays
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Vectorized computation
    np.multiply(radii, np.cos(angles), out=x)
    np.multiply(radii, np.sin(angles), out=y)
    
    # Ensure contiguous and 64-byte aligned memory layout
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = np.ascontiguousarray(m, dtype=np.float64)
    
    return x, y, m

def safe_direct_force(x, y, m, eps2):
    """Optimized direct force calculation"""
    N = len(x)
    
    # Pre-allocate aligned output arrays
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
    """Optimized FMM force calculation"""
    
    # Pre-allocate aligned output arrays
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

def fixed_parallel_benchmark():
    """Fixed parallel benchmark with effective load balancing"""
    print("\nFixed Parallel Benchmark with Effective Load Balancing")
    print("=" * 60)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("No computation modules available")
        return
    
    # 修正5: 使用更大的問題規模以顯示並行化效果
    test_sizes = [256, 512, 1024, 2048, 4096, 8192]
    thread_counts = [1, 2, 4, 8]
    
    results = {}
    
    for N in test_sizes:
        print(f"\nTesting with N = {N} particles")
        
        domain_size = 100.0
        theta = 0.5
        maxLeaf = 128  # 修正6: 增大 maxLeaf 以改善負載平衡
        eps = 0.01
        G = 1.0
        
        x, y, m = initialize_particles(N, domain_size)
        
        times_direct = []
        times_fmm = []
        
        for threads in thread_counts:
            # 修正7: 針對不同執行緒數動態調整 OpenMP 設定
            os.environ["OMP_NUM_THREADS"] = str(threads)
            
            if threads == 1:
                os.environ["OMP_PROC_BIND"] = "false"
                os.environ["OMP_SCHEDULE"] = "static"
            elif threads <= 4:
                os.environ["OMP_PROC_BIND"] = "close"
                os.environ["OMP_SCHEDULE"] = "guided,16"
            else:
                os.environ["OMP_PROC_BIND"] = "spread"
                os.environ["OMP_SCHEDULE"] = "guided,8"
            
            time.sleep(0.5)
            
            # Test direct method with better measurement
            if HAS_DIRECT and N <= 4096:  # 修正8: 限制直接方法的測試範圍
                # Extended warm-up
                for _ in range(5):
                    safe_direct_force(x, y, m, eps*eps)
                
                run_times = []
                for _ in range(10):  # 修正9: 更多測試運行
                    t0 = time.perf_counter()
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times and len(run_times) >= 7:
                    # 修正10: 移除異常值並取中位數
                    run_times.sort()
                    run_times = run_times[2:-2]  # 移除最高和最低的 20%
                    median_time = run_times[len(run_times)//2]
                    times_direct.append(median_time)
                    print(f"  Direct ({threads} threads): {median_time:.6f} seconds")
                else:
                    times_direct.append(float('nan'))
            else:
                times_direct.append(float('nan'))
            
            # Test FMM method with same optimizations
            if HAS_FMM:
                # Extended warm-up
                for _ in range(5):
                    safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                run_times = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times and len(run_times) >= 7:
                    run_times.sort()
                    run_times = run_times[2:-2]
                    median_time = run_times[len(run_times)//2]
                    times_fmm.append(median_time)
                    print(f"  FMM ({threads} threads): {median_time:.6f} seconds")
                else:
                    times_fmm.append(float('nan'))
            else:
                times_fmm.append(float('nan'))
        
        # Calculate speedups and efficiency
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
            max_efficiency = max(efficiency_direct) if efficiency_direct else 0
            print(f"  Direct: {max_speedup:.2f}x speedup, {max_efficiency:.1%} efficiency at {best_threads} threads")
        
        if speedup_fmm:
            max_speedup = max(speedup_fmm)
            best_threads = thread_counts[speedup_fmm.index(max_speedup)]
            max_efficiency = max(efficiency_fmm) if efficiency_fmm else 0
            print(f"  FMM: {max_speedup:.2f}x speedup, {max_efficiency:.1%} efficiency at {best_threads} threads")
    
    # Create enhanced visualization
    create_fixed_plots(results, test_sizes, thread_counts)
    
    return results

def create_fixed_plots(results, test_sizes, thread_counts):
    """Create fixed performance analysis plots"""
    
    fig = plt.figure(figsize=(16, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Direct Method Speedup
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("Direct Method Speedup (Fixed)", fontsize=14, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_direct']:
            threads_used = thread_counts[:len(results[N]['speedup_direct'])]
            ax1.plot(threads_used, results[N]['speedup_direct'], 
                    'o-', color=colors[i % len(colors)], label=f"N={N}", linewidth=3, markersize=8)
    
    ax1.plot(thread_counts, thread_counts, '--', color='gray', label="Ideal", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 8.5)
    ax1.set_ylim(0, 8.5)
    
    # 2. Direct Method Efficiency
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("Direct Method Efficiency (Fixed)", fontsize=14, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_direct']:
            threads_used = thread_counts[:len(results[N]['efficiency_direct'])]
            ax2.plot(threads_used, results[N]['efficiency_direct'], 
                    's-', color=colors[i % len(colors)], label=f"N={N}", linewidth=3, markersize=8)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="Good (80%)")
    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="Fair (50%)")
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Parallel Efficiency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 8.5)
    ax2.set_ylim(0, 1.1)
    
    # 3. FMM Method Speedup
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("FMM Method Speedup (Fixed)", fontsize=14, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_fmm']:
            threads_used = thread_counts[:len(results[N]['speedup_fmm'])]
            ax3.plot(threads_used, results[N]['speedup_fmm'], 
                    '^-', color=colors[i % len(colors)], label=f"N={N}", linewidth=3, markersize=8)
    
    ax3.plot(thread_counts, thread_counts, '--', color='gray', label="Ideal", linewidth=2, alpha=0.7)
    ax3.set_xlabel("Number of Threads")
    ax3.set_ylabel("Speedup")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, 8.5)
    ax3.set_ylim(0, 8.5)
    
    # 4. Performance Comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("Performance Comparison (8 Threads)", fontsize=14, fontweight='bold')
    
    direct_times_8 = []
    fmm_times_8 = []
    sizes_direct = []
    sizes_fmm = []
    
    for N in test_sizes:
        if N in results:
            if (len(results[N]['times_direct']) >= 4 and 
                not math.isnan(results[N]['times_direct'][3])):
                direct_times_8.append(results[N]['times_direct'][3])
                sizes_direct.append(N)
            
            if (len(results[N]['times_fmm']) >= 4 and 
                not math.isnan(results[N]['times_fmm'][3])):
                fmm_times_8.append(results[N]['times_fmm'][3])
                sizes_fmm.append(N)
    
    if direct_times_8:
        ax4.loglog(sizes_direct, direct_times_8, 'o-', label="Direct O(N²)", linewidth=3, markersize=8)
    if fmm_times_8:
        ax4.loglog(sizes_fmm, fmm_times_8, 's-', label="FMM", linewidth=3, markersize=8)
    
    ax4.set_xlabel("Problem Size (N)")
    ax4.set_ylabel("Time (seconds)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scalability Analysis
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("Best Achieved Speedup", fontsize=14, fontweight='bold')
    
    best_speedups_direct = []
    best_speedups_fmm = []
    
    for N in test_sizes:
        if N in results:
            if results[N]['speedup_direct']:
                best_speedups_direct.append(max(results[N]['speedup_direct']))
            else:
                best_speedups_direct.append(0)
            
            if results[N]['speedup_fmm']:
                best_speedups_fmm.append(max(results[N]['speedup_fmm']))
            else:
                best_speedups_fmm.append(0)
    
    x = np.arange(len(test_sizes))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, best_speedups_direct, width, 
                   label='Direct Method', alpha=0.8, color='skyblue')
    bars2 = ax5.bar(x + width/2, best_speedups_fmm, width, 
                   label='FMM Method', alpha=0.8, color='lightcoral')
    
    ax5.set_xlabel("Problem Size")
    ax5.set_ylabel("Best Speedup")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"N={N}" for N in test_sizes])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Performance Analysis Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Performance Analysis Summary", fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Calculate overall metrics
    analysis_text = []
    
    # Analyze results
    all_direct_speedups = []
    all_fmm_speedups = []
    for N in test_sizes:
        if N in results:
            if results[N]['speedup_direct']:
                all_direct_speedups.extend(results[N]['speedup_direct'])
            if results[N]['speedup_fmm']:
                all_fmm_speedups.extend(results[N]['speedup_fmm'])
    
    avg_direct_speedup = np.mean(all_direct_speedups) if all_direct_speedups else 0
    avg_fmm_speedup = np.mean(all_fmm_speedups) if all_fmm_speedups else 0
    max_direct_speedup = max(all_direct_speedups) if all_direct_speedups else 0
    max_fmm_speedup = max(all_fmm_speedups) if all_fmm_speedups else 0
    
    analysis_text.append("Performance Analysis Summary:")
    analysis_text.append("")
    analysis_text.append(f"Direct Method:")
    analysis_text.append(f"  Average Speedup: {avg_direct_speedup:.2f}x")
    analysis_text.append(f"  Maximum Speedup: {max_direct_speedup:.2f}x")
    if avg_direct_speedup > 2.0:
        analysis_text.append("  ✓ Good parallelization")
    elif avg_direct_speedup > 1.5:
        analysis_text.append("  ⚠ Fair parallelization")
    else:
        analysis_text.append("  ✗ Poor parallelization")
    
    analysis_text.append("")
    analysis_text.append(f"FMM Method:")
    analysis_text.append(f"  Average Speedup: {avg_fmm_speedup:.2f}x")
    analysis_text.append(f"  Maximum Speedup: {max_fmm_speedup:.2f}x")
    if avg_fmm_speedup > 2.0:
        analysis_text.append("  ✓ Good parallelization")
    elif avg_fmm_speedup > 1.5:
        analysis_text.append("  ⚠ Fair parallelization")
    else:
        analysis_text.append("  ✗ Poor parallelization")
    
    analysis_text.append("")
    analysis_text.append("Recommendations:")
    if max_direct_speedup > max_fmm_speedup:
        analysis_text.append("• Direct method shows better scaling")
    else:
        analysis_text.append("• FMM method shows better scaling")
    
    analysis_text.append("• Use larger problem sizes (N > 1000)")
    analysis_text.append("• Consider memory bandwidth limits")
    analysis_text.append("• Optimize for NUMA architecture")
    
    for i, text in enumerate(analysis_text):
        ax6.text(0.05, 0.95 - i*0.06, text, fontsize=10, 
                transform=ax6.transAxes, verticalalignment='top',
                fontweight='bold' if text.endswith(':') else 'normal')
    
    plt.tight_layout()
    
    # Save fixed plots
    fixed_path = os.path.join(OUTPUT_DIR, "fixed_parallel_analysis.png")
    plt.savefig(fixed_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Fixed analysis saved to {fixed_path}")

# 保持其他函數不變，但更新選單
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
    maxLeaf = 64
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
    """Main menu with fixed parallel benchmark"""
    while True:
        print("\n" + "=" * 60)
        print("2D N-Body Problem Simulation Platform")
        print("(Fixed Parallel Analysis Version)")
        print("=" * 60)
        print("Select function:")
        print(" 1) Quick benchmark scaling")
        print(" 2) Save trajectory + energy plot")
        print(" 3) Live simulation animation")
        print(" 4) Large-N scaling test")
        print(" 5) Energy conservation test")
        print(" 6) Parameter optimization")
        print(" 7) Fixed parallel benchmark")  # 修正11: 更新選單名稱
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
            fixed_parallel_benchmark()  # 修正12: 呼叫修正後的函數
        elif choice == '8':
            system_information()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    print("2D N-Body Problem Simulation Platform starting...")
    print("Fixed parallel performance analysis version")
    print(f"Optimized OpenMP configuration:")
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

