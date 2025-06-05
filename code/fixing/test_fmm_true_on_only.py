#!/usr/bin/env python3
"""
test_fmm_true_on_only.py

測試並比較不同版本的FMM性能
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 導入真正的 O(N) FMM
try:
    import fmm_true_on
    print("Successfully imported fmm_true_on")
except ImportError as e:
    print(f"Error importing fmm_true_on: {e}")
    print("Please compile with: python3 setup_true_on.py build_ext --inplace")
    exit(1)

def test_true_fmm_scaling(sizes, threads_list, eps2=1e-6, theta=0.6):
    """
    測試真正的 O(N) FMM 的複雜度和並行性能
    """
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    
    print("Testing True O(N) FMM scaling...")
    print("=" * 50)
    
    # 存儲結果
    results = {
        'sizes': sizes,
        'times_by_threads': {},  # {threads: [times for each size]}
        'speedups': {}           # {threads: [speedups relative to 1 thread]}
    }
    
    # 對每個線程數進行測試
    for P in threads_list:
        os.environ["OMP_NUM_THREADS"] = str(P)
        print(f"\n--- Testing with {P} threads ---")
        
        times = []
        
        for N in sizes:
            print(f"N = {N:>6d}", end=" ")
            
            # 生成隨機資料（固定種子確保一致性）
            np.random.seed(42)
            x = np.random.rand(N).astype(np.float64)
            y = np.random.rand(N).astype(np.float64)
            m = np.ones(N, dtype=np.float64)
            
            # 準備輸出陣列
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            # Warm-up run
            fmm_true_on.fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)
            
            # 執行並計時（多次測量取平均）
            num_runs = 3 if N < 100000 else 1
            total_time = 0.0
            
            for run in range(num_runs):
                ax.fill(0.0)
                ay.fill(0.0)
                t0 = time.time()
                fmm_true_on.fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)
                total_time += time.time() - t0
            
            elapsed = total_time / num_runs
            times.append(elapsed)
            
            print(f"time: {elapsed:.4f}s")
        
        results['times_by_threads'][P] = times
    
    # 計算相對於1線程的加速比
    if 1 in results['times_by_threads']:
        base_times = results['times_by_threads'][1]
        for P in threads_list:
            if P in results['times_by_threads']:
                speedups = [base_times[i] / results['times_by_threads'][P][i] 
                           for i in range(len(sizes))]
                results['speedups'][P] = speedups
    
    return results

def plot_fmm_results(results):
    """
    繪製 FMM 測試結果，包含與之前結果的比較
    """
    sizes = results['sizes']
    threads_list = list(results['times_by_threads'].keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: 時間 vs N（不同線程數）
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(threads_list)))
    
    for i, P in enumerate(sorted(threads_list)):
        times = results['times_by_threads'][P]
        ax1.loglog(sizes, times, 'o-', color=colors[i], 
                  label=f'{P} threads', linewidth=2, markersize=6)
    
    # 添加 O(N) 和 O(N log N) 參考線
    if 1 in results['times_by_threads']:
        base_time = results['times_by_threads'][1][0]
        base_n = sizes[0]
        
        # O(N) 參考線
        scale_n = base_time / base_n
        ref_n = [scale_n * n for n in sizes]
        ax1.loglog(sizes, ref_n, '--', color='green', 
                  alpha=0.7, label='O(N) reference', linewidth=2)
        
        # O(N log N) 參考線
        scale_nlogn = base_time / (base_n * np.log(base_n))
        ref_nlogn = [scale_nlogn * n * np.log(n) for n in sizes]
        ax1.loglog(sizes, ref_nlogn, '--', color='blue', 
                  alpha=0.7, label='O(N log N) reference', linewidth=2)
        
        # O(N²) 參考線
        scale_n2 = base_time / (base_n * base_n)
        ref_n2 = [scale_n2 * n * n for n in sizes]
        ax1.loglog(sizes, ref_n2, '--', color='red', 
                  alpha=0.7, label='O(N²) reference', linewidth=2)
    
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Time [s]')
    ax1.set_title('True O(N) FMM: Time vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 並行加速比（最大 N）
    ax2 = axes[0, 1]
    if results['speedups']:
        largest_n_idx = -1
        threads = []
        speedups_largest = []
        efficiencies = []
        
        for P in sorted(threads_list):
            if P in results['speedups']:
                threads.append(P)
                speedup = results['speedups'][P][largest_n_idx]
                speedups_largest.append(speedup)
                efficiencies.append(speedup / P)
        
        ax2.plot(threads, speedups_largest, 'o-', linewidth=2, 
                label='Speed-up', color='blue', markersize=8)
        ax2.plot(threads, threads, '--', color='gray', alpha=0.7, 
                label='Perfect scaling', linewidth=2)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(threads, efficiencies, 's-', linewidth=2, 
                     label='Efficiency', color='red', markersize=6)
        ax2_twin.set_ylabel('Parallel Efficiency', color='red')
        ax2_twin.set_ylim(0, 1.2)
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Speed-up')
        ax2.set_title(f'Parallel Scaling (N={sizes[-1]})')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: 複雜度分析
    ax3 = axes[0, 2]
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        
        # 計算實際縮放指數
        log_n = np.log(sizes)
        log_times = np.log(times_1thread)
        slope, intercept, r_value, p_value, std_err = linregress(log_n, log_times)
        
        ax3.loglog(sizes, times_1thread, 'o-', linewidth=2, 
                  label=f'Measured: N^{slope:.2f} (R²={r_value**2:.3f})', 
                  color='blue', markersize=8)
        
        # 理論線
        scale = times_1thread[0] / sizes[0]
        theory_n = [scale * n for n in sizes]
        ax3.loglog(sizes, theory_n, '--', color='green', 
                  alpha=0.7, label='Theoretical O(N)', linewidth=2)
        
        scale_nlogn = times_1thread[0] / (sizes[0] * np.log(sizes[0]))
        theory_nlogn = [scale_nlogn * n * np.log(n) for n in sizes]
        ax3.loglog(sizes, theory_nlogn, '--', color='orange', 
                  alpha=0.7, label='Theoretical O(N log N)', linewidth=2)
        
        ax3.set_xlabel('Number of Particles')
        ax3.set_ylabel('Time [s]')
        ax3.set_title('Complexity Analysis (1 thread)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        print(f"\n複雜度分析：")
        print(f"實際測量：N^{slope:.2f} (R²={r_value**2:.3f})")
        if slope < 1.3:
            print("✓ 接近理論 O(N) 複雜度")
        elif slope < 1.8:
            print("⚠ 接近 O(N log N) 複雜度")
        else:
            print("✗ 偏離預期複雜度")
    
    # Plot 4: 不同 N 下的加速比
    ax4 = axes[1, 0]
    if results['speedups'] and len(sizes) > 1:
        for i, N in enumerate(sizes):
            threads = []
            speedups_n = []
            for P in sorted(threads_list):
                if P in results['speedups'] and i < len(results['speedups'][P]):
                    threads.append(P)
                    speedups_n.append(results['speedups'][P][i])
            
            if threads:
                ax4.plot(threads, speedups_n, 'o-', 
                        label=f'N={N}', linewidth=1.5, markersize=5)
        
        # 完美縮放參考線
        max_threads = max(threads_list) if threads_list else 16
        perfect_threads = range(1, max_threads + 1)
        ax4.plot(perfect_threads, perfect_threads, '--', 
                color='gray', alpha=0.7, label='Perfect scaling', linewidth=2)
        
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Speed-up')
        ax4.set_title('Speed-up vs Threads (different N)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: 與之前結果的比較
    ax5 = axes[1, 1]
    
    # 添加之前的糟糕結果作為比較
    previous_bad_times = {
        1: [0.0404, 0.1972, 0.3970, 0.8358, 3.0229],
        8: [0.0648, 0.6657, 0.6365, 0.9536, 2.7640],
        16: [0.1046, 0.4448, 0.7715, 1.9355, 3.6095]
    }
    
    # 繪製當前結果
    if 1 in results['times_by_threads']:
        ax5.loglog(sizes, results['times_by_threads'][1], 'o-', 
                  label='Current O(N) FMM', color='green', linewidth=2, markersize=6)
    
    # 繪製之前的結果（如果大小匹配）
    if len(sizes) == len(previous_bad_times[1]):
        ax5.loglog(sizes, previous_bad_times[1], 's-', 
                  label='Previous inefficient version', color='red', linewidth=2, markersize=6)
    
    ax5.set_xlabel('Number of Particles')
    ax5.set_ylabel('Time [s]')
    ax5.set_title('Comparison with Previous Version')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: 效率摘要表
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 創建摘要表格
    summary_data = []
    if results['speedups']:
        largest_n = sizes[-1]
        for P in sorted(threads_list):
            if P in results['speedups']:
                speedup = results['speedups'][P][-1]
                efficiency = speedup / P
                time_1thread = results['times_by_threads'][1][-1] if 1 in results['times_by_threads'] else 0
                time_p = results['times_by_threads'][P][-1]
                
                summary_data.append([
                    f'{P}',
                    f'{time_p:.3f}s',
                    f'{speedup:.2f}x',
                    f'{efficiency:.1%}'
                ])
    
    if summary_data:
        table = ax6.table(
            cellText=summary_data,
            colLabels=['Threads', f'Time (N={sizes[-1]})', 'Speed-up', 'Efficiency'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title(f'Performance Summary (N={sizes[-1]})', pad=20)
    
    plt.tight_layout()
    plt.show()

def print_performance_summary(results):
    """
    印出性能摘要
    """
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    sizes = results['sizes']
    threads_list = sorted(results['times_by_threads'].keys())
    
    # 印出時間表格
    print(f"\n{'Threads':>8} ", end="")
    for N in sizes:
        print(f"{'N='+str(N):>10}", end="")
    print()
    print("-" * (8 + 10 * len(sizes)))
    
    for P in threads_list:
        print(f"{P:8d} ", end="")
        times = results['times_by_threads'][P]
        for t in times:
            print(f"{t:10.4f}", end="")
        print()
    
    # 印出加速比
    if results['speedups']:
        print(f"\nSpeed-up vs 1 thread:")
        print(f"{'Threads':>8} ", end="")
        for N in sizes:
            print(f"{'N='+str(N):>10}", end="")
        print()
        print("-" * (8 + 10 * len(sizes)))
        
        for P in threads_list:
            if P in results['speedups']:
                print(f"{P:8d} ", end="")
                speedups = results['speedups'][P]
                for s in speedups:
                    print(f"{s:10.2f}", end="")
                print()
    
    # 複雜度分析
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        log_n = np.log(sizes)
        log_times = np.log(times_1thread)
        slope, _, r_value, _, _ = linregress(log_n, log_times)
        
        print(f"\n複雜度分析 (1 thread):")
        print(f"  實際縮放: N^{slope:.2f} (R²={r_value**2:.3f})")
        print(f"  理論期望: O(N) = N^1.0")
        
        if slope < 1.2:
            print("  ✓ 符合 O(N) 複雜度")
        elif slope < 1.8:
            print("  ⚠ 接近 O(N log N) 複雜度")
        else:
            print("  ✗ 偏離預期複雜度")
    
    # 並行效率分析
    if results['speedups'] and len(threads_list) > 1:
        largest_n_idx = -1
        print(f"\n並行效率分析 (N={sizes[largest_n_idx]}):")
        print(f"{'Threads':>8} {'Speed-up':>10} {'Efficiency':>12} {'Rating':>10}")
        print("-" * 42)
        
        for P in threads_list:
            if P in results['speedups']:
                speedup = results['speedups'][P][largest_n_idx]
                efficiency = speedup / P
                
                if efficiency > 0.8:
                    rating = "Excellent"
                elif efficiency > 0.6:
                    rating = "Good"
                elif efficiency > 0.4:
                    rating = "Fair"
                else:
                    rating = "Poor"
                
                print(f"{P:8d} {speedup:10.2f} {efficiency:11.1%} {rating:>10}")

def compare_with_barnes_hut():
    """
    與理論 Barnes-Hut 性能比較
    """
    print(f"\n與 Barnes-Hut O(N log N) 的理論比較:")
    print("=" * 40)
    
    sizes = [10000, 50000, 100000, 200000, 400000]
    
    print(f"{'N':>8} {'O(N)':>10} {'O(N log N)':>12} {'Ratio':>8}")
    print("-" * 40)
    
    for N in sizes:
        on_relative = N / sizes[0]
        onlogn_relative = (N * np.log(N)) / (sizes[0] * np.log(sizes[0]))
        ratio = onlogn_relative / on_relative
        
        print(f"{N:8d} {on_relative:10.1f} {onlogn_relative:12.1f} {ratio:8.2f}")
    
    print("\n說明：當 N 增大時，O(N) 相對於 O(N log N) 的優勢會越來越明顯")

def main():
    # 測試參數
    sizes = [10000, 50000, 100000, 200000, 400000]
    threads_list = [1, 2, 4, 8, 16]
    
    print("Testing True O(N) Fast Multipole Method (Optimized Version)")
    print("=" * 60)
    
    # 執行測試
    results = test_true_fmm_scaling(sizes, threads_list)
    
    # 印出結果摘要
    print_performance_summary(results)
    
    # 理論比較
    compare_with_barnes_hut()
    
    # 繪製圖表
    plot_fmm_results(results)

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
test_fmm_true_on_only.py

只測試真正的 O(N) FMM，不依賴其他模組
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 導入真正的 O(N) FMM
try:
    import fmm_true_on
    print("Successfully imported fmm_true_on")
except ImportError as e:
    print(f"Error importing fmm_true_on: {e}")
    print("Please compile with: python3 setup_true_on.py build_ext --inplace")
    exit(1)

def test_true_fmm_scaling(sizes, threads_list, eps2=1e-6, theta=0.6):
    """
    測試真正的 O(N) FMM 的複雜度和並行性能
    """
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    
    print("Testing True O(N) FMM scaling...")
    print("=" * 50)
    
    # 存儲結果
    results = {
        'sizes': sizes,
        'times_by_threads': {},  # {threads: [times for each size]}
        'speedups': {}           # {threads: [speedups relative to 1 thread]}
    }
    
    # 對每個線程數進行測試
    for P in threads_list:
        os.environ["OMP_NUM_THREADS"] = str(P)
        print(f"\n--- Testing with {P} threads ---")
        
        times = []
        
        for N in sizes:
            print(f"N = {N:>6d}", end=" ")
            
            # 生成隨機資料（固定種子確保一致性）
            np.random.seed(42)
            x = np.random.rand(N).astype(np.float64)
            y = np.random.rand(N).astype(np.float64)
            m = np.ones(N, dtype=np.float64)
            
            # 準備輸出陣列
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            # 執行並計時
            t0 = time.time()
            fmm_true_on.fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)
            elapsed = time.time() - t0
            times.append(elapsed)
            
            print(f"time: {elapsed:.4f}s")
        
        results['times_by_threads'][P] = times
    
    # 計算相對於1線程的加速比
    if 1 in results['times_by_threads']:
        base_times = results['times_by_threads'][1]
        for P in threads_list:
            if P in results['times_by_threads']:
                speedups = [base_times[i] / results['times_by_threads'][P][i] 
                           for i in range(len(sizes))]
                results['speedups'][P] = speedups
    
    return results

def plot_fmm_results(results):
    """
    繪製 FMM 測試結果
    """
    sizes = results['sizes']
    threads_list = list(results['times_by_threads'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: 時間 vs N（不同線程數）
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(threads_list)))
    
    for i, P in enumerate(sorted(threads_list)):
        times = results['times_by_threads'][P]
        ax1.loglog(sizes, times, 'o-', color=colors[i], 
                  label=f'{P} threads', linewidth=2)
    
    # 添加 O(N) 參考線
    if 1 in results['times_by_threads']:
        base_time = results['times_by_threads'][1][0]
        base_n = sizes[0]
        scale = base_time / base_n
        ref_line = [scale * n for n in sizes]
        ax1.loglog(sizes, ref_line, '--', color='gray', 
                  alpha=0.7, label='O(N) reference')
    
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Time [s]')
    ax1.set_title('True O(N) FMM: Time vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 並行效率（最大 N）
    ax2 = axes[0, 1]
    if results['speedups']:
        largest_n_idx = -1
        threads = []
        speedups_largest = []
        efficiencies = []
        
        for P in sorted(threads_list):
            if P in results['speedups']:
                threads.append(P)
                speedup = results['speedups'][P][largest_n_idx]
                speedups_largest.append(speedup)
                efficiencies.append(speedup / P)
        
        ax2.plot(threads, speedups_largest, 'o-', linewidth=2, 
                label='Speed-up', color='blue')
        ax2.plot(threads, threads, '--', color='gray', alpha=0.7, 
                label='Perfect scaling')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(threads, efficiencies, 's-', linewidth=2, 
                     label='Efficiency', color='red')
        ax2_twin.set_ylabel('Parallel Efficiency', color='red')
        ax2_twin.set_ylim(0, 1.2)
        
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Speed-up')
        ax2.set_title(f'Parallel Scaling (N={sizes[-1]})')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: 複雜度分析
    ax3 = axes[1, 0]
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        
        # 計算實際縮放指數
        log_n = np.log(sizes)
        log_times = np.log(times_1thread)
        slope, intercept, r_value, p_value, std_err = linregress(log_n, log_times)
        
        ax3.loglog(sizes, times_1thread, 'o-', linewidth=2, 
                  label=f'Measured: N^{slope:.2f} (R²={r_value**2:.3f})')
        
        # 理論 O(N) 線
        scale = times_1thread[0] / sizes[0]
        theory_line = [scale * n for n in sizes]
        ax3.loglog(sizes, theory_line, '--', color='green', 
                  alpha=0.7, label='Theoretical O(N)')
        
        ax3.set_xlabel('Number of Particles')
        ax3.set_ylabel('Time [s]')
        ax3.set_title('Complexity Analysis (1 thread)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        print(f"\n複雜度分析：")
        print(f"實際測量：N^{slope:.2f} (R²={r_value**2:.3f})")
        if slope < 1.2:
            print("✓ 接近理論 O(N) 複雜度")
        else:
            print("⚠ 偏離理論 O(N) 複雜度")
    
    # Plot 4: 不同 N 下的加速比
    ax4 = axes[1, 1]
    if results['speedups'] and len(sizes) > 1:
        for i, N in enumerate(sizes):
            threads = []
            speedups_n = []
            for P in sorted(threads_list):
                if P in results['speedups'] and i < len(results['speedups'][P]):
                    threads.append(P)
                    speedups_n.append(results['speedups'][P][i])
            
            if threads:
                ax4.plot(threads, speedups_n, 'o-', 
                        label=f'N={N}', linewidth=1.5)
        
        # 完美縮放參考線
        max_threads = max(threads_list) if threads_list else 16
        perfect_threads = range(1, max_threads + 1)
        ax4.plot(perfect_threads, perfect_threads, '--', 
                color='gray', alpha=0.7, label='Perfect scaling')
        
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Speed-up')
        ax4.set_title('Speed-up vs Threads (different N)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_performance_summary(results):
    """
    印出性能摘要
    """
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    sizes = results['sizes']
    threads_list = sorted(results['times_by_threads'].keys())
    
    # 印出時間表格
    print(f"\n{'Threads':>8} ", end="")
    for N in sizes:
        print(f"{'N='+str(N):>10}", end="")
    print()
    print("-" * (8 + 10 * len(sizes)))
    
    for P in threads_list:
        print(f"{P:8d} ", end="")
        times = results['times_by_threads'][P]
        for t in times:
            print(f"{t:10.4f}", end="")
        print()
    
    # 印出加速比
    if results['speedups']:
        print(f"\nSpeed-up vs 1 thread:")
        print(f"{'Threads':>8} ", end="")
        for N in sizes:
            print(f"{'N='+str(N):>10}", end="")
        print()
        print("-" * (8 + 10 * len(sizes)))
        
        for P in threads_list:
            if P in results['speedups']:
                print(f"{P:8d} ", end="")
                speedups = results['speedups'][P]
                for s in speedups:
                    print(f"{s:10.2f}", end="")
                print()

def main():
    # 測試參數
    sizes = [10000, 50000, 100000, 200000, 400000]
    threads_list = [1, 2, 4, 8, 16]
    
    print("Testing True O(N) Fast Multipole Method")
    print("=" * 50)
    
    # 執行測試
    results = test_true_fmm_scaling(sizes, threads_list)
    
    # 印出結果摘要
    print_performance_summary(results)
    
    # 繪製圖表
    plot_fmm_results(results)

if __name__ == "__main__":
    main()
