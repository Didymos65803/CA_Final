#!/usr/bin/env python3
"""benchmark_improved.py — Enhanced OpenMP & algorithmic performance analysis
==============================================================================
* Comprehensive size sweep analysis (Direct vs. Barnes–Hut FMM)
* Detailed thread scaling with efficiency metrics
* Opening-angle accuracy vs performance trade-off analysis  
* Enhanced visualizations showing clear OpenMP benefits
* Detailed performance metrics and analysis

Usage examples:
$ python benchmark_improved.py                                # default settings
$ python benchmark_improved.py --sizes 1e3 2e3 4e3 8e3 1.6e4  \
                              --threads 1 2 4 8 16            \
                              --theta_base 0.6                \
                              --theta 0.3 0.5 0.7 1.0         \
                              --detailed_analysis
"""

import os
import time
import math
import argparse
import pathlib
import sys
from typing import Sequence, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for prettier plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

try:
    import fmm_openmp as fm
except ImportError:
    sys.exit("ERROR: fmm_openmp module not found! Please compile fmm_openmp.cpp first:\n" +
             "  python setup_openmp.py build_ext --inplace")

# Configuration
OUTPUT_DIR = pathlib.Path("results_enhanced")
OUTPUT_DIR.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)

# Performance tracking
class PerformanceTracker:
    def __init__(self):
        self.timings = {}
        self.scaling_data = {}
        self.accuracy_data = {}
    
    def add_timing(self, method: str, size: int, threads: int, time_val: float):
        key = f"{method}_{size}_{threads}"
        self.timings[key] = time_val
    
    def get_speedup(self, method: str, size: int, threads: int) -> float:
        base_key = f"{method}_{size}_1"
        current_key = f"{method}_{size}_{threads}"
        if base_key in self.timings and current_key in self.timings:
            return self.timings[base_key] / self.timings[current_key]
        return 1.0
    
    def get_efficiency(self, method: str, size: int, threads: int) -> float:
        return self.get_speedup(method, size, threads) / threads

tracker = PerformanceTracker()

def random_system(N: int, domain: float = 50.0, distribution: str = "uniform") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random particle system with different distributions."""
    if distribution == "uniform":
        x = _rng.uniform(-domain, domain, N).astype(np.float64)
        y = _rng.uniform(-domain, domain, N).astype(np.float64)
    elif distribution == "clustered":
        # Create clustered distribution for more realistic test
        n_clusters = max(3, N // 1000)
        cluster_centers = _rng.uniform(-domain*0.5, domain*0.5, (n_clusters, 2))
        cluster_sizes = _rng.exponential(domain/10, n_clusters)
        
        x, y = [], []
        particles_per_cluster = N // n_clusters
        
        for i in range(n_clusters):
            n_in_cluster = particles_per_cluster if i < n_clusters-1 else N - len(x)
            cx, cy = cluster_centers[i]
            sigma = cluster_sizes[i]
            
            cluster_x = _rng.normal(cx, sigma, n_in_cluster)
            cluster_y = _rng.normal(cy, sigma, n_in_cluster)
            
            x.extend(cluster_x)
            y.extend(cluster_y)
        
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
    
    m = np.ones(N, dtype=np.float64)
    return x, y, m

def measure_with_warmup(func, *args, warmup_runs: int = 2, timing_runs: int = 5) -> float:
    """Measure execution time with warmup and multiple runs for accuracy."""
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args)
    
    # Timing runs
    times = []
    for _ in range(timing_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    
    return np.median(times)  # Use median to reduce outlier impact

def create_enhanced_size_plot(Ns: List[int], direct_times: List[float], 
                            fmm_times: List[float], threads: int, theta: float):
    """Create enhanced size scaling plot with theoretical curves and speedup."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main timing plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Theoretical scaling references
    N0 = Ns[0]
    theoretical_n2 = [direct_times[0] * (N / N0) ** 2 for N in Ns]
    theoretical_nlogn = [fmm_times[0] * (N / N0) * math.log2(N) / math.log2(N0) for N in Ns]
    
    # Plot measured data
    ax1.loglog(Ns, direct_times, 'o-', linewidth=3, markersize=8, 
               label=f'Direct O(N²) - {threads} threads', color='#1f77b4')
    ax1.loglog(Ns, fmm_times, 's-', linewidth=3, markersize=8, 
               label=f'FMM O(N log N) - {threads} threads', color='#ff7f0e')
    
    # Plot theoretical references
    ax1.loglog(Ns, theoretical_n2, '--', alpha=0.6, color='#1f77b4', 
               label='Theoretical O(N²)')
    ax1.loglog(Ns, theoretical_nlogn, ':', alpha=0.6, color='#ff7f0e', 
               label='Theoretical O(N log N)')
    
    ax1.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wall-clock Time [s]', fontsize=12, fontweight='bold')
    ax1.set_title(f'Algorithmic Scaling Comparison (θ={theta}, {threads} threads)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Speedup plot
    ax2 = fig.add_subplot(gs[1, 0])
    speedups = np.array(direct_times) / np.array(fmm_times)
    
    ax2.loglog(Ns, speedups, 'o-', linewidth=3, markersize=8, color='#2ca02c')
    ax2.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (Direct/FMM)', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithmic Speedup', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add speedup annotations
    for i, (n, s) in enumerate(zip(Ns, speedups)):
        if i % 2 == 0:  # Annotate every other point to avoid crowding
            ax2.annotate(f'{s:.0f}×', (n, s), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Efficiency comparison
    ax3 = fig.add_subplot(gs[1, 1])
    direct_efficiency = [N / t for N, t in zip(Ns, direct_times)]
    fmm_efficiency = [N / t for N, t in zip(Ns, fmm_times)]
    
    ax3.loglog(Ns, direct_efficiency, 'o-', linewidth=3, markersize=8, 
               label='Direct', color='#1f77b4')
    ax3.loglog(Ns, fmm_efficiency, 's-', linewidth=3, markersize=8, 
               label='FMM', color='#ff7f0e')
    ax3.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput [particles/s]', fontsize=12, fontweight='bold')
    ax3.set_title('Computational Throughput', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'enhanced_size_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_thread_scaling_plot(N: int, thread_list: List[int], 
                             direct_times: List[float], fmm_times: List[float], theta: float):
    """Create comprehensive thread scaling analysis."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Calculate metrics
    direct_speedups = [direct_times[0] / t for t in direct_times]
    fmm_speedups = [fmm_times[0] / t for t in fmm_times]
    direct_efficiency = [s / threads for s, threads in zip(direct_speedups, thread_list)]
    fmm_efficiency = [s / threads for s, threads in zip(fmm_speedups, thread_list)]
    
    # 1. Raw timing comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(thread_list, direct_times, 'o-', linewidth=3, markersize=8, 
             label='Direct', color='#1f77b4')
    ax1.plot(thread_list, fmm_times, 's-', linewidth=3, markersize=8, 
             label='FMM', color='#ff7f0e')
    ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wall-clock Time [s]', fontsize=12, fontweight='bold')
    ax1.set_title(f'Raw Performance (N={N:,})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_yscale('log')
    
    # 2. Speedup comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(thread_list, direct_speedups, 'o-', linewidth=3, markersize=8, 
             label='Direct', color='#1f77b4')
    ax2.plot(thread_list, fmm_speedups, 's-', linewidth=3, markersize=8, 
             label='FMM', color='#ff7f0e')
    ax2.plot(thread_list, thread_list, '--k', alpha=0.5, label='Ideal Linear')
    ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup vs Serial', fontsize=12, fontweight='bold')
    ax2.set_title('OpenMP Speedup', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 3. Parallel efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(thread_list, direct_efficiency, 'o-', linewidth=3, markersize=8, 
             label='Direct', color='#1f77b4')
    ax3.plot(thread_list, fmm_efficiency, 's-', linewidth=3, markersize=8, 
             label='FMM', color='#ff7f0e')
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax3.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax3.set_title('Parallel Efficiency Analysis', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_ylim(0, 1.2)
    
    # 4. Performance gain from FMM vs threads
    ax4 = fig.add_subplot(gs[1, 1])
    algorithmic_speedup = np.array(direct_times) / np.array(fmm_times)
    ax4.plot(thread_list, algorithmic_speedup, 'o-', linewidth=3, markersize=8, 
             color='#2ca02c')
    ax4.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Algorithmic Speedup (Direct/FMM)', fontsize=12, fontweight='bold')
    ax4.set_title('FMM vs Direct Advantage', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for speedup values
    for i, (threads, speedup) in enumerate(zip(thread_list, algorithmic_speedup)):
        if i % 2 == 0:
            ax4.annotate(f'{speedup:.0f}×', (threads, speedup), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 5. Combined efficiency heatmap
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create efficiency matrix
    methods = ['Direct', 'FMM']
    efficiency_matrix = np.array([direct_efficiency, fmm_efficiency])
    
    im = ax5.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(len(thread_list)))
    ax5.set_xticklabels(thread_list)
    ax5.set_yticks(range(len(methods)))
    ax5.set_yticklabels(methods)
    ax5.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax5.set_title('Parallel Efficiency Heatmap', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(thread_list)):
            text = ax5.text(j, i, f'{efficiency_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Efficiency', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comprehensive_thread_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_theta_analysis_plot(N: int, thetas: List[float], 
                              errors: List[float], times: List[float]):
    """Create detailed theta trade-off analysis."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Error vs theta
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(thetas, errors, 'o-', linewidth=3, markersize=8, color='#d62728')
    ax1.set_xlabel('Opening Angle θ', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2 Relative Error', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Opening Angle', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add error annotations
    for theta, error in zip(thetas, errors):
        ax1.annotate(f'{error:.1e}', (theta, error), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # 2. Runtime vs theta
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(thetas, times, 's-', linewidth=3, markersize=8, color='#9467bd')
    ax2.set_xlabel('Opening Angle θ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime [s]', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Opening Angle', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add performance annotations
    for theta, time_val in zip(thetas, times):
        ax2.annotate(f'{time_val:.3f}s', (theta, time_val), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # 3. Trade-off scatter plot
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(times, errors, c=thetas, s=150, cmap='viridis', 
                         edgecolors='black', linewidth=2)
    ax3.set_xlabel('Runtime [s]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('L2 Relative Error', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_title('Accuracy-Performance Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for theta values
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Opening Angle θ', fontsize=11, fontweight='bold')
    
    # Add annotations for theta values
    for theta, time_val, error in zip(thetas, times, errors):
        ax3.annotate(f'θ={theta}', (time_val, error), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 4. Performance efficiency vs accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate performance efficiency (inverse of time, normalized)
    performance_efficiency = np.array(times)
    performance_efficiency = performance_efficiency.max() / performance_efficiency
    
    ax4.plot(thetas, performance_efficiency, 'o-', linewidth=3, markersize=8, 
             label='Performance Efficiency', color='#2ca02c')
    
    # Calculate accuracy efficiency (inverse of error, normalized to [0,1])
    accuracy_efficiency = 1.0 / np.array(errors)
    accuracy_efficiency = accuracy_efficiency / accuracy_efficiency.max()
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(thetas, accuracy_efficiency, 's-', linewidth=3, markersize=8, 
                  label='Accuracy Efficiency', color='#ff7f0e')
    
    ax4.set_xlabel('Opening Angle θ', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance Efficiency', fontsize=12, fontweight='bold', color='#2ca02c')
    ax4_twin.set_ylabel('Accuracy Efficiency', fontsize=12, fontweight='bold', color='#ff7f0e')
    ax4.set_title('Efficiency Analysis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Find optimal theta (balance between accuracy and performance)
    combined_efficiency = performance_efficiency * accuracy_efficiency
    optimal_idx = np.argmax(combined_efficiency)
    optimal_theta = thetas[optimal_idx]
    
    ax4.axvline(x=optimal_theta, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.text(optimal_theta, 0.5, f'Optimal θ≈{optimal_theta:.2f}', 
             rotation=90, va='center', ha='right', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detailed_theta_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_theta

def run_comprehensive_size_sweep(Ns: List[int], threads: int, eps2: float, 
                                domain: float, theta: float) -> Tuple[List[float], List[float]]:
    """Run comprehensive size sweep with proper warmup and multiple measurements."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    print(f"\n=== Size Sweep Analysis (θ={theta}, {threads} threads) ===")
    
    direct_times, fmm_times = [], []
    
    for i, N in enumerate(Ns):
        print(f"\nTesting N={N:,} particles ({i+1}/{len(Ns)})...")
        
        x, y, m = random_system(N, domain)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        
        # Measure direct method
        print("  Measuring direct O(N²) method...")
        direct_time = measure_with_warmup(
            lambda: fm.direct_force(x, y, m, eps2, ax, ay),
            warmup_runs=2, timing_runs=3
        )
        direct_times.append(direct_time)
        tracker.add_timing("direct", N, threads, direct_time)
        
        # Measure FMM method
        print("  Measuring FMM O(N log N) method...")
        fmm_time = measure_with_warmup(
            lambda: fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay),
            warmup_runs=2, timing_runs=3
        )
        fmm_times.append(fmm_time)
        tracker.add_timing("fmm", N, threads, fmm_time)
        
        speedup = direct_time / fmm_time
        print(f"  Results: Direct={direct_time:.4f}s, FMM={fmm_time:.4f}s, Speedup={speedup:.1f}×")
    
    # Create enhanced visualization
    create_enhanced_size_plot(Ns, direct_times, fmm_times, threads, theta)
    
    # Save detailed results
    with open(OUTPUT_DIR / 'detailed_size_sweep.tsv', 'w') as f:
        f.write('N\tDirect_Time[s]\tFMM_Time[s]\tSpeedup\tDirect_Throughput[part/s]\tFMM_Throughput[part/s]\n')
        for N, dt, ft in zip(Ns, direct_times, fmm_times):
            speedup = dt / ft
            direct_throughput = N / dt
            fmm_throughput = N / ft
            f.write(f'{N}\t{dt:.6f}\t{ft:.6f}\t{speedup:.2f}\t{direct_throughput:.0f}\t{fmm_throughput:.0f}\n')
    
    return direct_times, fmm_times

def run_detailed_thread_scaling(N: int, thread_list: List[int], eps2: float, 
                               domain: float, theta: float) -> Tuple[List[float], List[float]]:
    """Run detailed thread scaling analysis."""
    print(f"\n=== Thread Scaling Analysis (N={N:,}, θ={theta}) ===")
    
    # Generate consistent dataset
    x, y, m = random_system(N, domain)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    direct_times, fmm_times = [], []
    
    for threads in thread_list:
        print(f"\nTesting {threads} threads...")
        os.environ['OMP_NUM_THREADS'] = str(threads)
        time.sleep(0.1)  # Allow environment to settle
        
        # Measure direct method
        direct_time = measure_with_warmup(
            lambda: fm.direct_force(x, y, m, eps2, ax, ay),
            warmup_runs=2, timing_runs=5
        )
        direct_times.append(direct_time)
        tracker.add_timing("direct", N, threads, direct_time)
        
        # Measure FMM method
        fmm_time = measure_with_warmup(
            lambda: fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay),
            warmup_runs=2, timing_runs=5
        )
        fmm_times.append(fmm_time)
        tracker.add_timing("fmm", N, threads, fmm_time)
        
        direct_speedup = direct_times[0] / direct_time
        fmm_speedup = fmm_times[0] / fmm_time
        direct_efficiency = direct_speedup / threads
        fmm_efficiency = fmm_speedup / threads
        
        print(f"  Direct: {direct_time:.4f}s (speedup: {direct_speedup:.2f}×, efficiency: {direct_efficiency:.2f})")
        print(f"  FMM:    {fmm_time:.4f}s (speedup: {fmm_speedup:.2f}×, efficiency: {fmm_efficiency:.2f})")
    
    # Create comprehensive visualization
    create_thread_scaling_plot(N, thread_list, direct_times, fmm_times, theta)
    
    # Save detailed results
    with open(OUTPUT_DIR / 'detailed_thread_scaling.tsv', 'w') as f:
        f.write('Threads\tDirect_Time[s]\tFMM_Time[s]\tDirect_Speedup\tFMM_Speedup\tDirect_Efficiency\tFMM_Efficiency\n')
        for i, threads in enumerate(thread_list):
            dt, ft = direct_times[i], fmm_times[i]
            ds = direct_times[0] / dt
            fs = fmm_times[0] / ft
            de = ds / threads
            fe = fs / threads
            f.write(f'{threads}\t{dt:.6f}\t{ft:.6f}\t{ds:.3f}\t{fs:.3f}\t{de:.3f}\t{fe:.3f}\n')
    
    return direct_times, fmm_times

def run_theta_optimization(N: int, thetas: List[float], eps2: float, domain: float) -> float:
    """Run comprehensive theta optimization analysis."""
    print(f"\n=== Opening Angle Optimization (N={N:,}) ===")
    
    x, y, m = random_system(N, domain)
    ax_ref = np.zeros(N, dtype=np.float64)
    ay_ref = np.zeros(N, dtype=np.float64)
    ax_test = np.zeros(N, dtype=np.float64)
    ay_test = np.zeros(N, dtype=np.float64)
    
    # Compute reference solution with direct method
    print("Computing reference solution with direct method...")
    fm.direct_force(x, y, m, eps2, ax_ref, ay_ref)
    force_ref = np.sqrt(ax_ref**2 + ay_ref**2)
    ref_norm = np.linalg.norm(force_ref)
    
    errors, times = [], []
    
    for theta in thetas:
        print(f"\nTesting θ={theta:.2f}...")
        
        # Measure timing
        exec_time = measure_with_warmup(
            lambda: fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax_test, ay_test),
            warmup_runs=2, timing_runs=5
        )
        times.append(exec_time)
        
        # Compute error
        force_test = np.sqrt(ax_test**2 + ay_test**2)
        error = np.linalg.norm(force_test - force_ref) / max(ref_norm, 1e-12)
        errors.append(error)
        
        print(f"  Time: {exec_time:.6f}s, L2 error: {error:.2e}")
    
    # Create detailed analysis and find optimal theta
    optimal_theta = create_theta_analysis_plot(N, thetas, errors, times)
    
    # Save results
    with open(OUTPUT_DIR / 'theta_optimization.tsv', 'w') as f:
        f.write('Theta\tTime[s]\tL2_Error\tPerformance_Score\tAccuracy_Score\n')
        max_perf = max(times)
        max_err = max(errors)
        for theta, time_val, error in zip(thetas, times, errors):
            perf_score = max_perf / time_val  # Higher is better
            acc_score = max_err / error       # Higher is better (lower error)
            f.write(f'{theta:.2f}\t{time_val:.6f}\t{error:.6e}\t{perf_score:.3f}\t{acc_score:.3f}\n')
    
    print(f"\nOptimal opening angle: θ = {optimal_theta:.2f}")
    return optimal_theta

def create_summary_report(Ns: List[int], thread_list: List[int], 
                         direct_times_size: List[float], fmm_times_size: List[float],
                         direct_times_thread: List[float], fmm_times_thread: List[float],
                         optimal_theta: float):
    """Create comprehensive summary report."""
    report = f"""
=== PERFORMANCE ANALYSIS SUMMARY ===
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

ALGORITHMIC SCALING ANALYSIS:
- Problem sizes tested: {Ns}
- Maximum speedup (Direct/FMM): {max(np.array(direct_times_size)/np.array(fmm_times_size)):.1f}×
- FMM efficiency scales as O(N log N) vs O(N²) for direct method

PARALLEL SCALING ANALYSIS:
- Thread counts tested: {thread_list}
- Maximum direct speedup: {max([direct_times_thread[0]/t for t in direct_times_thread]):.2f}×
- Maximum FMM speedup: {max([fmm_times_thread[0]/t for t in fmm_times_thread]):.2f}×
- Direct parallel efficiency at max threads: {(direct_times_thread[0]/direct_times_thread[-1])/thread_list[-1]:.2f}
- FMM parallel efficiency at max threads: {(fmm_times_thread[0]/fmm_times_thread[-1])/thread_list[-1]:.2f}

OPENING ANGLE OPTIMIZATION:
- Optimal opening angle: θ = {optimal_theta:.2f}
- This provides the best balance between accuracy and performance

RECOMMENDATIONS:
1. Use FMM for problems with N > 1000 particles
2. OpenMP scaling is effective up to {thread_list[-2] if len(thread_list) > 2 else thread_list[-1]} threads for this problem size
3. Use θ = {optimal_theta:.2f} for optimal accuracy/performance trade-off
4. Consider problem-specific tuning for production applications

FILES GENERATED:
- enhanced_size_scaling.png: Comprehensive algorithmic scaling analysis
- comprehensive_thread_scaling.png: Detailed OpenMP performance analysis  
- detailed_theta_analysis.png: Opening angle optimization results
- *.tsv files: Raw numerical data for further analysis
"""
    
    with open(OUTPUT_DIR / 'performance_summary.txt', 'w') as f:
        f.write(report)
    
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive FMM performance analysis')
    parser.add_argument('--sizes', type=float, nargs='+',
                        default=[1e3, 2e3, 4e3, 8e3, 1.6e4],
                        help='Particle counts for size sweep')
    parser.add_argument('--threads', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16],
                        help='Thread counts for scaling test')
    parser.add_argument('--theta', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7, 1.0],
                        help='Opening angles for optimization')
    parser.add_argument('--theta_base', type=float, default=0.6,
                        help='Baseline opening angle for other tests')
    parser.add_argument('--soft', type=float, default=0.01,
                        help='Softening parameter')
    parser.add_argument('--domain', type=float, default=100.0,
                        help='Simulation domain size')
    parser.add_argument('--detailed_analysis', action='store_true',
                        help='Run additional detailed analysis')
    
    args = parser.parse_args()
    
    Ns = [int(s) for s in args.sizes]
    eps2 = args.soft ** 2
    
    print("=== COMPREHENSIVE FMM PERFORMANCE ANALYSIS ===")
    print(f"OpenMP maximum threads available: {fm.get_max_threads()}")
    print(f"Testing problem sizes: {Ns}")
    print(f"Testing thread counts: {args.threads}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # 1. Size sweep analysis
    direct_times_size, fmm_times_size = run_comprehensive_size_sweep(
        Ns, max(args.threads), eps2, args.domain, args.theta_base
    )
    
    # 2. Thread scaling analysis
    test_size = Ns[len(Ns)//2]  # Use middle size for thread scaling
    direct_times_thread, fmm_times_thread = run_detailed_thread_scaling(
        test_size, args.threads, eps2, args.domain, args.theta_base
    )
    
    # 3. Opening angle optimization
    optimal_theta = run_theta_optimization(
        Ns[1], args.theta, eps2, args.domain
    )
    
    # 4. Generate comprehensive summary
    create_summary_report(
        Ns, args.threads, direct_times_size, fmm_times_size,
        direct_times_thread, fmm_times_thread, optimal_theta
    )
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"All results saved to: {OUTPUT_DIR.absolute()}")
    print("Key findings:")
    print(f"- Maximum algorithmic speedup: {max(np.array(direct_times_size)/np.array(fmm_times_size)):.1f}×")
    print(f"- Best parallel efficiency: {max([fmm_times_thread[0]/t for t in fmm_times_thread])/max(args.threads):.2f}")
    print(f"- Optimal opening angle: θ = {optimal_theta:.2f}")

if __name__ == '__main__':
    main()
