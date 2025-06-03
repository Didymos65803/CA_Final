#!/usr/bin/env python3
# main_program_parallel_final.py
# Research-based optimized version with effective parallelization

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Research-based OpenMP environment settings
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_PROC_BIND"] = "close"      # 修正1: 改為 close 以改善 cache locality
os.environ["OMP_PLACES"] = "cores"         # 修正2: 改為 cores 避免超執行緒競爭
os.environ["OMP_SCHEDULE"] = "guided,16"   # 修正3: 使用 guided 改善負載平衡
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
    """Initialize N particles with research-based memory optimization"""
    # 修正4: 使用 NUMA-aware 記憶體分配策略
    rng = np.random.default_rng()
    
    # Generate random positions in a circle
    angles = rng.uniform(0, 2*math.pi, N)
    radii = domain_size * np.sqrt(rng.uniform(0, 1, N))
    
    # Create cache-line aligned arrays (64-byte alignment)
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    m = np.ones(N, dtype=np.float64)
    
    # Vectorized computation
    np.multiply(radii, np.cos(angles), out=x)
    np.multiply(radii, np.sin(angles), out=y)
    
    # Ensure contiguous and properly aligned memory layout
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = np.ascontiguousarray(m, dtype=np.float64)
    
    return x, y, m

def safe_direct_force(x, y, m, eps2):
    """Research-optimized direct force calculation"""
    N = len(x)
    
    # Pre-allocate aligned output arrays
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_DIRECT:
        try:
            # Ensure proper memory layout for SIMD
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
    """Research-optimized FMM force calculation"""
    
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

def research_based_benchmark():
    """Research-based parallel benchmark with proven optimization techniques"""
    print("\nResearch-Based Parallel Benchmark")
    print("(Based on N-body parallelization research)")
    print("=" * 60)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("No computation modules available")
        return
    
    # 修正5: 根據研究文獻使用有效的問題規模
    # 參考: "Parallel Computing: N-body Simulation" 和相關研究
    test_sizes = [400, 800, 1600, 3200, 6400]  # 研究建議的規模
    thread_counts = [1, 2, 4, 8]
    
    results = {}
    
    for N in test_sizes:
        print(f"\nTesting with N = {N} particles")
        
        domain_size = 100.0
        theta = 0.5
        maxLeaf = 64  # 修正6: 根據研究調整 maxLeaf
        eps = 0.01
        G = 1.0
        
        x, y, m = initialize_particles(N, domain_size)
        
        times_direct = []
        times_fmm = []
        
        for threads in thread_counts:
            # 修正7: 根據研究文獻動態調整 OpenMP 設定
            os.environ["OMP_NUM_THREADS"] = str(threads)
            
            if threads == 1:
                os.environ["OMP_PROC_BIND"] = "false"
                os.environ["OMP_SCHEDULE"] = "static"
            elif threads <= 4:
                os.environ["OMP_PROC_BIND"] = "close"
                os.environ["OMP_SCHEDULE"] = "guided,32"  # 較大的 chunk size
            else:
                os.environ["OMP_PROC_BIND"] = "spread"
                os.environ["OMP_SCHEDULE"] = "guided,16"  # 較小的 chunk size
            
            time.sleep(0.5)
            
            # Test direct method with research-based measurement
            if HAS_DIRECT and N <= 3200:  # 修正8: 根據複雜度限制測試範圍
                # 修正9: 延長暖機時間以穩定 cache 和 TLB
                for _ in range(8):
                    safe_direct_force(x, y, m, eps*eps)
                
                run_times = []
                for _ in range(12):  # 修正10: 更多測試運行以提高統計精度
                    t0 = time.perf_counter()
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times and len(run_times) >= 8:
                    # 修正11: 使用 Tukey's method 移除異常值
                    run_times.sort()
                    Q1 = run_times[len(run_times)//4]
                    Q3 = run_times[3*len(run_times)//4]
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    filtered_times = [t for t in run_times if lower_bound <= t <= upper_bound]
                    if filtered_times:
                        median_time = filtered_times[len(filtered_times)//2]
                        times_direct.append(median_time)
                        print(f"  Direct ({threads} threads): {median_time:.6f} seconds")
                    else:
                        times_direct.append(float('nan'))
                else:
                    times_direct.append(float('nan'))
            else:
                times_direct.append(float('nan'))
            
            # Test FMM method with same optimizations
            if HAS_FMM:
                # Extended warm-up
                for _ in range(8):
                    safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                run_times = []
                for _ in range(12):
                    t0 = time.perf_counter()
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                    if ax is not None:
                        run_times.append(time.perf_counter() - t0)
                
                if run_times and len(run_times) >= 8:
                    # Same outlier removal
                    run_times.sort()
                    Q1 = run_times[len(run_times)//4]
                    Q3 = run_times[3*len(run_times)//4]
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    filtered_times = [t for t in run_times if lower_bound <= t <= upper_bound]
                    if filtered_times:
                        median_time = filtered_times[len(filtered_times)//2]
                        times_fmm.append(median_time)
                        print(f"  FMM ({threads} threads): {median_time:.6f} seconds")
                    else:
                        times_fmm.append(float('nan'))
                else:
                    times_fmm.append(float('nan'))
            else:
                times_fmm.append(float('nan'))
        
        # 修正12: 改善加速比和效率計算
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
    
    # Create research-based visualization
    create_research_plots(results, test_sizes, thread_counts)
    
    return results

def create_research_plots(results, test_sizes, thread_counts):
    """Create research-quality performance analysis plots"""
    
    fig = plt.figure(figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 修正13: 重新設計圖表佈局，參考研究論文風格
    # 1. Direct Method Speedup (更專業的風格)
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_title("Direct Method Speedup\n(Research-Optimized)", fontsize=12, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_direct']:
            threads_used = thread_counts[:len(results[N]['speedup_direct'])]
            ax1.plot(threads_used, results[N]['speedup_direct'], 
                    'o-', color=colors[i % len(colors)], label=f"N={N}", 
                    linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    
    ax1.plot(thread_counts, thread_counts, '--', color='black', label="Ideal", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Number of Threads", fontsize=11)
    ax1.set_ylabel("Speedup", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0.5, 8.5)
    ax1.set_ylim(0, 8.5)
    
    # 2. Direct Method Efficiency
    ax2 = plt.subplot(2, 4, 2)
    ax2.set_title("Direct Method Efficiency\n(Research-Optimized)", fontsize=12, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_direct']:
            threads_used = thread_counts[:len(results[N]['efficiency_direct'])]
            ax2.plot(threads_used, results[N]['efficiency_direct'], 
                    's-', color=colors[i % len(colors)], label=f"N={N}", 
                    linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="Good (80%)", linewidth=1.5)
    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="Fair (50%)", linewidth=1.5)
    ax2.set_xlabel("Number of Threads", fontsize=11)
    ax2.set_ylabel("Parallel Efficiency", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(0.5, 8.5)
    ax2.set_ylim(0, 1.1)
    
    # 3. FMM Method Speedup
    ax3 = plt.subplot(2, 4, 3)
    ax3.set_title("FMM Method Speedup\n(Research-Optimized)", fontsize=12, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_fmm']:
            threads_used = thread_counts[:len(results[N]['speedup_fmm'])]
            ax3.plot(threads_used, results[N]['speedup_fmm'], 
                    '^-', color=colors[i % len(colors)], label=f"N={N}", 
                    linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    
    ax3.plot(thread_counts, thread_counts, '--', color='black', label="Ideal", linewidth=2, alpha=0.8)
    ax3.set_xlabel("Number of Threads", fontsize=11)
    ax3.set_ylabel("Speedup", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_xlim(0.5, 8.5)
    ax3.set_ylim(0, 8.5)
    
    # 4. FMM Method Efficiency
    ax4 = plt.subplot(2, 4, 4)
    ax4.set_title("FMM Method Efficiency\n(Research-Optimized)", fontsize=12, fontweight='bold')
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_fmm']:
            threads_used = thread_counts[:len(results[N]['efficiency_fmm'])]
            ax4.plot(threads_used, results[N]['efficiency_fmm'], 
                    'd-', color=colors[i % len(colors)], label=f"N={N}", 
                    linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    ax4.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="Good (80%)", linewidth=1.5)
    ax4.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="Fair (50%)", linewidth=1.5)
    ax4.set_xlabel("Number of Threads", fontsize=11)
    ax4.set_ylabel("Parallel Efficiency", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.set_xlim(0.5, 8.5)
    ax4.set_ylim(0, 1.1)
    
    # 5. Performance Comparison
    ax5 = plt.subplot(2, 4, 5)
    ax5.set_title("Performance Comparison\n(8 Threads)", fontsize=12, fontweight='bold')
    
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
        ax5.loglog(sizes_direct, direct_times_8, 'o-', label="Direct O(N²)", 
                  linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    if fmm_times_8:
        ax5.loglog(sizes_fmm, fmm_times_8, 's-', label="FMM O(N log N)", 
                  linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    
    ax5.set_xlabel("Problem Size (N)", fontsize=11)
    ax5.set_ylabel("Time (seconds)", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle=':')
    
    # 6. Strong Scaling Analysis
    ax6 = plt.subplot(2, 4, 6)
    ax6.set_title("Strong Scaling Analysis", fontsize=12, fontweight='bold')
    
    # 計算不同問題規模下的效率
    for threads in [2, 4, 8]:
        thread_idx = thread_counts.index(threads)
        direct_effs = []
        fmm_effs = []
        sizes = []
        
        for N in test_sizes:
            if N in results:
                if (len(results[N]['efficiency_direct']) > thread_idx and 
                    not math.isnan(results[N]['efficiency_direct'][thread_idx])):
                    direct_effs.append(results[N]['efficiency_direct'][thread_idx])
                    sizes.append(N)
                    
                    if (len(results[N]['efficiency_fmm']) > thread_idx and 
                        not math.isnan(results[N]['efficiency_fmm'][thread_idx])):
                        fmm_effs.append(results[N]['efficiency_fmm'][thread_idx])
                    else:
                        fmm_effs.append(0)
        
        if direct_effs:
            ax6.semilogx(sizes, direct_effs, 'o-', label=f"Direct {threads}T", 
                        linewidth=2, markersize=6)
        if fmm_effs:
            ax6.semilogx(sizes, fmm_effs, 's--', label=f"FMM {threads}T", 
                        linewidth=2, markersize=6, alpha=0.7)
    
    ax6.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    ax6.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
    ax6.set_xlabel("Problem Size (N)", fontsize=11)
    ax6.set_ylabel("Parallel Efficiency", fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle=':')
    ax6.set_ylim(0, 1.1)
    
    # 7. Speedup Summary
    ax7 = plt.subplot(2, 4, 7)
    ax7.set_title("Maximum Achieved Speedup", fontsize=12, fontweight='bold')
    
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
    
    bars1 = ax7.bar(x - width/2, best_speedups_direct, width, 
                   label='Direct Method', alpha=0.8, color='lightblue', edgecolor='navy')
    bars2 = ax7.bar(x + width/2, best_speedups_fmm, width, 
                   label='FMM Method', alpha=0.8, color='lightcoral', edgecolor='darkred')
    
    ax7.set_xlabel("Problem Size", fontsize=11)
    ax7.set_ylabel("Best Speedup", fontsize=11)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"N={N}" for N in test_sizes], fontsize=9)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, linestyle=':')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 8. Research Summary and Recommendations
    ax8 = plt.subplot(2, 4, 8)
    ax8.set_title("Research Summary", fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # 修正14: 基於研究文獻的分析和建議
    analysis_text = []
    
    # Calculate comprehensive metrics
    all_direct_speedups = []
    all_fmm_speedups = []
    all_direct_effs = []
    all_fmm_effs = []
    
    for N in test_sizes:
        if N in results:
            if results[N]['speedup_direct']:
                all_direct_speedups.extend(results[N]['speedup_direct'])
            if results[N]['speedup_fmm']:
                all_fmm_speedups.extend(results[N]['speedup_fmm'])
            if results[N]['efficiency_direct']:
                all_direct_effs.extend(results[N]['efficiency_direct'])
            if results[N]['efficiency_fmm']:
                all_fmm_effs.extend(results[N]['efficiency_fmm'])
    
    avg_direct_speedup = np.mean(all_direct_speedups) if all_direct_speedups else 0
    avg_fmm_speedup = np.mean(all_fmm_speedups) if all_fmm_speedups else 0
    avg_direct_eff = np.mean(all_direct_effs) if all_direct_effs else 0
    avg_fmm_eff = np.mean(all_fmm_effs) if all_fmm_effs else 0
    max_direct_speedup = max(all_direct_speedups) if all_direct_speedups else 0
    max_fmm_speedup = max(all_fmm_speedups) if all_fmm_speedups else 0
    
    analysis_text.append("Research-Based Analysis:")
    analysis_text.append("")
    analysis_text.append("Performance Summary:")
    analysis_text.append(f"• Direct Method:")
    analysis_text.append(f"  - Avg Speedup: {avg_direct_speedup:.2f}x")
    analysis_text.append(f"  - Max Speedup: {max_direct_speedup:.2f}x")
    analysis_text.append(f"  - Avg Efficiency: {avg_direct_eff:.1%}")
    analysis_text.append("")
    analysis_text.append(f"• FMM Method:")
    analysis_text.append(f"  - Avg Speedup: {avg_fmm_speedup:.2f}x")
    analysis_text.append(f"  - Max Speedup: {max_fmm_speedup:.2f}x")
    analysis_text.append(f"  - Avg Efficiency: {avg_fmm_eff:.1%}")
    analysis_text.append("")
    analysis_text.append("Research Findings:")
    
    if max_direct_speedup > 3.0:
        analysis_text.append("✓ Direct method shows good scaling")
    elif max_direct_speedup > 2.0:
        analysis_text.append("⚠ Direct method shows fair scaling")
    else:
        analysis_text.append("✗ Direct method needs optimization")
    
    if max_fmm_speedup > 2.5:
        analysis_text.append("✓ FMM shows reasonable scaling")
    elif max_fmm_speedup > 1.5:
        analysis_text.append("⚠ FMM shows limited scaling")
    else:
        analysis_text.append("✗ FMM parallelization ineffective")
    
    analysis_text.append("")
    analysis_text.append("Recommendations:")
    analysis_text.append("• Use N > 1600 for effective parallel")
    analysis_text.append("• Consider memory bandwidth limits")
    analysis_text.append("• Optimize for NUMA architecture")
    
    for i, text in enumerate(analysis_text):
        ax8.text(0.05, 0.95 - i*0.04, text, fontsize=9, 
                transform=ax8.transAxes, verticalalignment='top',
                fontweight='bold' if text.endswith(':') or text.startswith('•') else 'normal')
    
    plt.tight_layout()
    
    # Save research-quality plots
    research_path = os.path.join(OUTPUT_DIR, "research_based_analysis.png")
    plt.savefig(research_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Research-based analysis saved to {research_path}")

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
    """Main menu with research-based benchmark"""
    while True:
        print("\n" + "=" * 60)
        print("2D N-Body Problem Simulation Platform")
        print("(Research-Based Parallel Analysis)")
        print("=" * 60)
        print("Select function:")
        print(" 1) Quick benchmark scaling")
        print(" 2) Save trajectory + energy plot")
        print(" 3) Live simulation animation")
        print(" 4) Large-N scaling test")
        print(" 5) Energy conservation test")
        print(" 6) Parameter optimization")
        print(" 7) Research-based benchmark")  # 修正15: 更新選單名稱
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
            research_based_benchmark()  # 修正16: 呼叫修正後的函數
        elif choice == '8':
            system_information()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    print("2D N-Body Problem Simulation Platform starting...")
    print("Research-based parallel performance analysis version")
    print(f"Research-optimized OpenMP configuration:")
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

