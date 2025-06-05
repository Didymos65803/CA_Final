#!/usr/bin/env python3
"""
benchmark_true_on.py

測試真正的 O(N) FMM 實現並與 O(N log N) Barnes-Hut 比較
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 導入模組
try:
    import fmm_true_on  # 真正的 O(N) FMM
    import fmm_omp      # O(N log N) Barnes-Hut
except ImportError as e:
    print(f"請先編譯模組: {e}")
    print("運行: python3 setup_true_on.py build_ext --inplace")
    print("和: python3 setup_openmp.py build_ext --inplace")
    exit(1)

def benchmark_comparison(sizes, threads_list, eps2=1e-6, theta=0.6):
    """
    比較真正的 O(N) FMM 與 O(N log N) Barnes-Hut 的性能
    """
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    
    results = {
        'sizes': sizes,
        'threads': threads_list,
        'fmm_on_times': {},    # {threads: [times for each size]}
        'fmm_nlogn_times': {}  # {threads: [times for each size]}
    }
    
    for P in threads_list:
        os.environ["OMP_NUM_THREADS"] = str(P)
        print(f"\n=== Testing with {P} threads ===")
        
        fmm_on_times = []
        fmm_nlogn_times = []
        
        for N in sizes:
            print(f"N = {N:>6d}", end=" ")
            
            # 生成隨機資料
            np.random.seed(42)  # 固定種子確保一致性
            x = np.random.rand(N).astype(np.float64)
            y = np.random.rand(N).astype(np.float64)
            m = np.ones(N, dtype=np.float64)
            
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            # 測試 O(N) FMM
            t0 = time.time()
            fmm_true_on.fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)
            t_on = time.time() - t0
            fmm_on_times.append(t_on)
            
            # 測試 O(N log N) Barnes-Hut
            ax.fill(0.0)
            ay.fill(0.0)
            t0 = time.time()
            fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
            t_nlogn = time.time() - t0
            fmm_nlogn_times.append(t_nlogn)
            
            speedup = t_nlogn / t_on if t_on > 0 else 0
            print(f"O(N): {t_on:.4f}s, O(N log N): {t_nlogn:.4f}s, Speed-up: {speedup:.2f}x")
        
        results['fmm_on_times'][P] = fmm_on_times
        results['fmm_nlogn_times'][P] = fmm_nlogn_times
    
    return results

def plot_scaling_comparison(results):
    """
    繪製縮放性比較圖表
    """
    sizes = results['sizes']
    threads_list = results['threads']
    
    # 1. 演算法複雜度比較 (single thread)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time vs N (1 thread)
    P = 1
    if P in results['fmm_on_times']:
        times_on = results['fmm_on_times'][P]
        times_nlogn = results['fmm_nlogn_times'][P]
        
        axes[0,0].loglog(sizes, times_on, 'o-', label='True O(N) FMM', color='green', linewidth=2)
        axes[0,0].loglog(sizes, times_nlogn, 's-', label='O(N log N) Barnes-Hut', color='blue', linewidth=2)
        
        # 理論參考線
        scale_on = times_on[0] / sizes[0]
        scale_nlogn = times_nlogn[0] / (sizes[0] * np.log(sizes[0]))
        
        ref_on = [scale_on * n for n in sizes]
        ref_nlogn = [scale_nlogn * n * np.log(n) for n in sizes]
        
        axes[0,0].loglog(sizes, ref_on, '--', color='green', alpha=0.5, label='O(N) reference')
        axes[0,0].loglog(sizes, ref_nlogn, '--', color='blue', alpha=0.5, label='O(N log N) reference')
        
        axes[0,0].set_xlabel('N')
        axes[0,0].set_ylabel('Time [s]')
        axes[0,0].set_title('Algorithmic Complexity (1 thread)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 計算實際縮放指數
        log_n = np.log(sizes)
        slope_on, _, r2_on, _, _ = linregress(log_n, np.log(times_on))
        slope_nlogn, _, r2_nlogn, _, _ = linregress(log_n, np.log(times_nlogn))
        
        print(f"\n實際縮放：")
        print(f"True O(N) FMM: N^{slope_on:.2f} (R²={r2_on:.3f})")
        print(f"Barnes-Hut: N^{slope_nlogn:.2f} (R²={r2_nlogn:.3f})")
    
    # Plot 2: Speed-up vs N (1 thread)
    if P in results['fmm_on_times']:
        speedups = [results['fmm_nlogn_times'][P][i] / results['fmm_on_times'][P][i] 
                   for i in range(len(sizes))]
        axes[0,1].loglog(sizes, speedups, 'o-', color='red', linewidth=2)
        axes[0,1].set_xlabel('N')
        axes[0,1].set_ylabel('Speed-up (Barnes-Hut / True FMM)')
        axes[0,1].set_title('Algorithmic Speed-up')
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Parallel scaling (largest N)
    largest_n_idx = -1
    largest_n = sizes[largest_n_idx]
    
    threads = []
    times_on_par = []
    times_nlogn_par = []
    
    for P in threads_list:
        if P in results['fmm_on_times']:
            threads.append(P)
            times_on_par.append(results['fmm_on_times'][P][largest_n_idx])
            times_nlogn_par.append(results['fmm_nlogn_times'][P][largest_n_idx])
    
    if threads:
        axes[1,0].plot(threads, times_on_par, 'o-', label='True O(N) FMM', color='green', linewidth=2)
        axes[1,0].plot(threads, times_nlogn_par, 's-', label='O(N log N) Barnes-Hut', color='blue', linewidth=2)
        axes[1,0].set_xlabel('Threads')
        axes[1,0].set_ylabel('Time [s]')
        axes[1,0].set_title(f'Parallel Scaling (N={largest_n})')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Parallel efficiency
    if threads:
        base_time_on = times_on_par[0]
        base_time_nlogn = times_nlogn_par[0]
        
        efficiency_on = [base_time_on / (P * times_on_par[i]) for i, P in enumerate(threads)]
        efficiency_nlogn = [base_time_nlogn / (P * times_nlogn_par[i]) for i, P in enumerate(threads)]
        
        axes[1,1].plot(threads, efficiency_on, 'o-', label='True O(N) FMM', color='green', linewidth=2)
        axes[1,1].plot(threads, efficiency_nlogn, 's-', label='O(N log N) Barnes-Hut', color='blue', linewidth=2)
        axes[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect efficiency')
        axes[1,1].set_xlabel('Threads')
        axes[1,1].set_ylabel('Parallel Efficiency')
        axes[1,1].set_title(f'Parallel Efficiency (N={largest_n})')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.show()

def main():
    # 測試參數
    sizes = [10000, 50000, 100000, 200000, 400000]
    threads_list = [1, 2, 4, 8, 16]
    
    print("比較真正的 O(N) FMM 與 O(N log N) Barnes-Hut")
    print("=" * 50)
    
    # 執行測試
    results = benchmark_comparison(sizes, threads_list)
    
    # 繪製結果
    plot_scaling_comparison(results)

if __name__ == "__main__":
    main()
