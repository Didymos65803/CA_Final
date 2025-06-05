#!/usr/bin/env python3
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

def analyze_parallel_efficiency(results):
    """
    詳細分析並行效率
    """
    sizes = results['sizes']
    threads_list = sorted(results['times_by_threads'].keys())
    
    print(f"\n{'='*60}")
    print("PARALLEL EFFICIENCY ANALYSIS")
    print(f"{'='*60}")
    
    for i, N in enumerate(sizes):
        if N < 1000:
            continue  # 只分析較大的問題
            
        print(f"\nN = {N:,}")
        print(f"{'Threads':>8} {'Time(s)':>10} {'Speed-up':>10} {'Efficiency':>12} {'Strategy':>15}")
        print("-" * 65)
        
        base_time = results['times_by_threads'][1][i] if 1 in results['times_by_threads'] else 0
        
        for P in threads_list:
            if P in results['times_by_threads'] and i < len(results['times_by_threads'][P]):
                time_p = results['times_by_threads'][P][i]
                speedup = base_time / time_p if time_p > 0 else 0
                efficiency = speedup / P if P > 0 else 0
                
                # 判斷使用的策略
                if N < 500:
                    strategy = "Ultra-Parallel"
                elif N < 2000:
                    strategy = "NUMA-Parallel"
                elif N < 50000:
                    strategy = "Barnes-Hut"
                else:
                    strategy = "True O(N) FMM"
                
                print(f"{P:8d} {time_p:10.4f} {speedup:10.2f} {efficiency:11.1%} {strategy:>15}")

def compare_with_theoretical(results):
    """
    與理論性能比較
    """
    sizes = results['sizes']
    
    print(f"\n{'='*60}")
    print("THEORETICAL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        
        print(f"\n{'N':>8} {'Measured(s)':>12} {'O(N)':>10} {'O(NlogN)':>12} {'O(N²)':>10}")
        print("-" * 54)
        
        base_n = sizes[0]
        base_time = times_1thread[0]
        
        for i, N in enumerate(sizes):
            if i < len(times_1thread):
                measured = times_1thread[i]
                
                # 理論預測
                on_pred = base_time * (N / base_n)
                onlogn_pred = base_time * (N * np.log(N)) / (base_n * np.log(base_n))
                on2_pred = base_time * (N / base_n) ** 2
                
                print(f"{N:8,} {measured:12.4f} {on_pred:10.4f} {onlogn_pred:12.4f} {on2_pred:10.4f}")

def plot_enhanced_results(results):
    """
    增強的結果繪圖
    """
    sizes = results['sizes']
    threads_list = list(results['times_by_threads'].keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: 時間 vs N（不同線程數） - 包含更多參考線
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(threads_list)))
    
    for i, P in enumerate(sorted(threads_list)):
        times = results['times_by_threads'][P]
        ax1.loglog(sizes[:len(times)], times, 'o-', color=colors[i], 
                  label=f'{P} threads', linewidth=2, markersize=4)
    
    # 多條參考線
    if 1 in results['times_by_threads']:
        base_time = results['times_by_threads'][1][0]
        base_n = sizes[0]
        
        # O(N) 參考線
        scale_n = base_time / base_n
        ref_n = [scale_n * n for n in sizes]
        ax1.loglog(sizes, ref_n, '--', color='green', alpha=0.7, label='O(N)', linewidth=2)
        
        # O(N log N) 參考線
        scale_nlogn = base_time / (base_n * np.log(base_n))
        ref_nlogn = [scale_nlogn * n * np.log(n) for n in sizes]
        ax1.loglog(sizes, ref_nlogn, '--', color='blue', alpha=0.7, label='O(N log N)', linewidth=2)
        
        # O(N²) 參考線（僅顯示小N部分）
        small_sizes = [s for s in sizes if s <= 10000]
        scale_n2 = base_time / (base_n * base_n)
        ref_n2 = [scale_n2 * n * n for n in small_sizes]
        ax1.loglog(small_sizes, ref_n2, '--', color='red', alpha=0.7, label='O(N²)', linewidth=2)
    
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Time [s]')
    ax1.set_title('Algorithmic Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 並行效率熱圖
    ax2 = axes[0, 1]
    if results['speedups']:
        speedup_matrix = []
        efficiency_matrix = []
        
        for P in sorted(threads_list):
            if P in results['speedups']:
                speedups = results['speedups'][P]
                efficiencies = [s/P for s in speedups]
                speedup_matrix.append(speedups[:len(sizes)])
                efficiency_matrix.append(efficiencies[:len(sizes)])
        
        # 繪製效率熱圖
        im = ax2.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in sizes])
        ax2.set_yticks(range(len(threads_list)))
        ax2.set_yticklabels(sorted(threads_list))
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Threads')
        ax2.set_title('Parallel Efficiency')
        plt.colorbar(im, ax=ax2, label='Efficiency')
    
    # Plot 3: 不同N下的加速比曲線
    ax3 = axes[0, 2]
    if results['speedups']:
        # 選擇幾個代表性的N值
        representative_ns = [100, 1000, 10000, 100000, 400000]
        available_ns = [n for n in representative_ns if n in sizes]
        
        for N in available_ns:
            idx = sizes.index(N)
            threads = []
            speedups = []
            
            for P in sorted(threads_list):
                if P in results['speedups'] and idx < len(results['speedups'][P]):
                    threads.append(P)
                    speedups.append(results['speedups'][P][idx])
            
            if threads:
                ax3.plot(threads, speedups, 'o-', label=f'N={N:,}', linewidth=2, markersize=6)
        
        # 理想加速線
        max_threads = max(threads_list)
        ideal_threads = range(1, max_threads + 1)
        ax3.plot(ideal_threads, ideal_threads, '--', color='gray', alpha=0.7, 
                label='Ideal scaling', linewidth=2)
        
        ax3.set_xlabel('Threads')
        ax3.set_ylabel('Speed-up')
        ax3.set_title('Speed-up vs Threads')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: 複雜度分析（更詳細）
    ax4 = axes[1, 0]
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        
        # 分段分析複雜度
        small_mask = np.array(sizes) <= 2000
        medium_mask = (np.array(sizes) > 2000) & (np.array(sizes) <= 50000)
        large_mask = np.array(sizes) > 50000
        
        small_sizes = np.array(sizes)[small_mask]
        small_times = np.array(times_1thread)[small_mask]
        
        medium_sizes = np.array(sizes)[medium_mask]
        medium_times = np.array(times_1thread)[medium_mask]
        
        large_sizes = np.array(sizes)[large_mask]
        large_times = np.array(times_1thread)[large_mask]
        
        if len(small_sizes) > 1:
            slope_small, _, r2_small, _, _ = linregress(np.log(small_sizes), np.log(small_times))
            ax4.loglog(small_sizes, small_times, 'o-', color='blue', 
                      label=f'Small N: N^{slope_small:.2f} (R²={r2_small:.3f})', linewidth=2)
        
        if len(medium_sizes) > 1:
            slope_medium, _, r2_medium, _, _ = linregress(np.log(medium_sizes), np.log(medium_times))
            ax4.loglog(medium_sizes, medium_times, 's-', color='orange', 
                      label=f'Medium N: N^{slope_medium:.2f} (R²={r2_medium:.3f})', linewidth=2)
        
        if len(large_sizes) > 1:
            slope_large, _, r2_large, _, _ = linregress(np.log(large_sizes), np.log(large_times))
            ax4.loglog(large_sizes, large_times, '^-', color='red', 
                      label=f'Large N: N^{slope_large:.2f} (R²={r2_large:.3f})', linewidth=2)
    
    ax4.set_xlabel('Problem Size')
    ax4.set_ylabel('Time [s]')
    ax4.set_title('Complexity Analysis by Size Range')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 記憶體效率分析
    ax5 = axes[1, 1]
    if 1 in results['times_by_threads']:
        times_1thread = results['times_by_threads'][1]
        
        # 計算每秒處理的粒子對數（反映記憶體效率）
        particles_per_sec = [n / t for n, t in zip(sizes[:len(times_1thread)], times_1thread)]
        
        ax5.semilogx(sizes[:len(times_1thread)], particles_per_sec, 'o-', 
                    linewidth=2, markersize=6)
        ax5.set_xlabel('Problem Size')
        ax5.set_ylabel('Particles/sec')
        ax5.set_title('Memory/Computational Efficiency')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: 策略效果比較
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 創建策略效果摘要表
    strategy_data = []
    if results['speedups'] and 16 in results['speedups']:
        for i, N in enumerate(sizes):
            if i < len(results['speedups'][16]):
                speedup_16 = results['speedups'][16][i]
                efficiency_16 = speedup_16 / 16
                
                if N < 500:
                    strategy = "Ultra-Parallel O(N²)"
                elif N < 2000:
                    strategy = "NUMA-Parallel O(N²)"
                elif N < 50000:
                    strategy = "Barnes-Hut O(N log N)"
                else:
                    strategy = "True O(N) FMM"
                
                strategy_data.append([
                    f'{N:,}',
                    strategy,
                    f'{speedup_16:.2f}x',
                    f'{efficiency_16:.1%}'
                ])
    
    if strategy_data:
        table = ax6.table(
            cellText=strategy_data[:8],  # 只顯示前8行
            colLabels=['N', 'Strategy', '16-thread\nSpeed-up', 'Efficiency'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Strategy Effectiveness Summary', pad=20)
    
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
    # 更合理的測試參數範圍
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    threads_list = [1, 2, 4, 8, 16]
    
    print("Testing Optimized FMM Implementation (Based on Successful C++ Version)")
    print("=" * 70)
    
    # 執行測試
    results = test_true_fmm_scaling(sizes, threads_list)
    
    # 詳細分析
    analyze_parallel_efficiency(results)
    compare_with_theoretical(results)
    
    # 印出結果摘要
    print_performance_summary(results)
    
    # 理論比較
    compare_with_barnes_hut()
    
    # 增強的繪圖
    plot_enhanced_results(results)

if __name__ == "__main__":
    main()
