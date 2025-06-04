#!/usr/bin/env python3
# main_program_optimized.py
# 完全優化版本，解決所有並行化問題

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Optional, List

# 根據研究文獻設定最優的OpenMP環境變數
def setup_openmp_environment():
    """設定最優的OpenMP環境變數"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # 動態調整線程數量
    if cpu_count <= 4:
        num_threads = cpu_count
        proc_bind = "close"
        schedule = "static"
    elif cpu_count <= 8:
        num_threads = cpu_count
        proc_bind = "spread"
        schedule = "guided,32"
    else:
        num_threads = min(cpu_count, 16)  # 避免過度並行化
        proc_bind = "spread"
        schedule = "guided,64"
    
    env_settings = {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": proc_bind,
        "OMP_PLACES": "cores",
        "OMP_SCHEDULE": schedule,
        "OMP_DYNAMIC": "false",
        "OMP_NESTED": "false",
        "OMP_WAIT_POLICY": "passive",  # 減少CPU佔用
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",  # Intel編譯器優化
        "KMP_BLOCKTIME": "0",  # 減少等待時間
    }
    
    for key, value in env_settings.items():
        os.environ[key] = value
    
    print(f"OpenMP環境設定完成：{num_threads}線程，{proc_bind}綁定")
    return num_threads

# 在導入C++模組前設定環境
num_threads = setup_openmp_environment()

# 設定路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 嘗試導入優化的C++模組
try:
    import force_kernel
    HAS_DIRECT = True
    print("✓ 優化版force_kernel載入成功")
except ImportError as e:
    HAS_DIRECT = False
    print(f"✗ force_kernel不可用: {e}")

try:
    import fmm_kernel
    HAS_FMM = True
    print("✓ 優化版fmm_kernel載入成功")
except ImportError as e:
    HAS_FMM = False
    print(f"✗ fmm_kernel不可用: {e}")

# 全域設定
OUTPUT_DIR = "output_optimized"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class OptimizedParticleSystem:
    """優化的粒子系統類"""
    
    def __init__(self, N: int, domain_size: float = 50.0):
        self.N = N
        self.domain_size = domain_size
        
        # 使用連續記憶體佈局
        self.x = np.empty(N, dtype=np.float64)
        self.y = np.empty(N, dtype=np.float64)
        self.m = np.ones(N, dtype=np.float64)
        
        # 確保記憶體對齊
        self.x = np.ascontiguousarray(self.x)
        self.y = np.ascontiguousarray(self.y)
        self.m = np.ascontiguousarray(self.m)
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """初始化粒子位置"""
        rng = np.random.default_rng(42)  # 固定種子確保重現性
        
        # 使用極坐標生成均勻分佈
        angles = rng.uniform(0, 2*math.pi, self.N)
        radii = self.domain_size * np.sqrt(rng.uniform(0, 1, self.N))
        
        # 向量化計算
        np.multiply(radii, np.cos(angles), out=self.x)
        np.multiply(radii, np.sin(angles), out=self.y)
        
        # 確保質量的變化
        self.m[:] = rng.uniform(0.8, 1.2, self.N)

def optimized_direct_force(x: np.ndarray, y: np.ndarray, m: np.ndarray, 
                          eps: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """優化的直接力計算"""
    N = len(x)
    
    # 預分配對齊的輸出陣列
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_DIRECT:
        try:
            # 確保記憶體佈局正確
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            m_arr = np.ascontiguousarray(m, dtype=np.float64)
            
            force_kernel.direct_force(x_arr, y_arr, m_arr, eps*eps, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"直接力計算失敗: {e}")
            return None, None
    else:
        return None, None

def optimized_fmm_force(x: np.ndarray, y: np.ndarray, m: np.ndarray, N: int,
                       domain_size: float, theta: float, maxLeaf: int, 
                       eps: float, G: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """優化的FMM力計算"""
    
    # 預分配對齊的輸出陣列
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_FMM:
        try:
            # 確保記憶體佈局正確
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            m_arr = np.ascontiguousarray(m, dtype=np.float64)
            
            fmm_kernel.fmm_force(x_arr, y_arr, m_arr, N, domain_size, 
                               theta, maxLeaf, eps, G, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"FMM力計算失敗: {e}")
            return None, None
    else:
        return None, None

def advanced_performance_benchmark():
    """進階性能基準測試"""
    print("\n進階性能基準測試")
    print("=" * 60)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("無可用的計算模組")
        return
    
    # 測試參數
    test_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
    thread_counts = [1, 2, 4, 8] if num_threads >= 8 else [1, 2, 4]
    
    # 基準參數
    domain_size = 100.0
    theta = 0.5
    maxLeaf = 128
    eps = 0.01
    G = 1.0
    
    results = {}
    
    for N in test_sizes:
        print(f"\n測試N = {N}個粒子")
        
        # 創建測試粒子系統
        system = OptimizedParticleSystem(N, domain_size)
        
        times_direct = []
        times_fmm = []
        
        for threads in thread_counts:
            # 動態調整OpenMP設定
            os.environ["OMP_NUM_THREADS"] = str(threads)
            time.sleep(0.1)  # 等待設定生效
            
            # 測試直接方法
            if HAS_DIRECT and N <= 20000:  # 擴大測試範圍
                warmup_runs = 3
                test_runs = 10
                
                # 熱身運行
                for _ in range(warmup_runs):
                    optimized_direct_force(system.x, system.y, system.m, eps)
                
                # 測試運行
                run_times = []
                for _ in range(test_runs):
                    start_time = time.perf_counter()
                    ax, ay = optimized_direct_force(system.x, system.y, system.m, eps)
                    if ax is not None:
                        run_times.append(time.perf_counter() - start_time)
                
                if run_times:
                    # 使用中位數減少異常值影響
                    median_time = np.median(run_times)
                    times_direct.append(median_time)
                    print(f"  直接方法 ({threads} 線程): {median_time:.6f} 秒")
                else:
                    times_direct.append(float('nan'))
            else:
                times_direct.append(float('nan'))
            
            # 測試FMM方法
            if HAS_FMM:
                warmup_runs = 3
                test_runs = 10
                
                # 熱身運行
                for _ in range(warmup_runs):
                    optimized_fmm_force(system.x, system.y, system.m, N, 
                                      domain_size, theta, maxLeaf, eps, G)
                
                # 測試運行
                run_times = []
                for _ in range(test_runs):
                    start_time = time.perf_counter()
                    ax, ay = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, maxLeaf, eps, G)
                    if ax is not None:
                        run_times.append(time.perf_counter() - start_time)
                
                if run_times:
                    median_time = np.median(run_times)
                    times_fmm.append(median_time)
                    print(f"  FMM方法 ({threads} 線程): {median_time:.6f} 秒")
                else:
                    times_fmm.append(float('nan'))
            else:
                times_fmm.append(float('nan'))
        
        # 計算加速比和效率
        speedup_direct = []
        speedup_fmm = []
        efficiency_direct = []
        efficiency_fmm = []
        
        if times_direct and not math.isnan(times_direct[0]) and times_direct[0] > 0:
            speedup_direct = [times_direct[0] / t for t in times_direct 
                            if not math.isnan(t) and t > 0]
            efficiency_direct = [s / tc for s, tc in zip(speedup_direct, 
                               thread_counts[:len(speedup_direct)])]
        
        if times_fmm and not math.isnan(times_fmm[0]) and times_fmm[0] > 0:
            speedup_fmm = [times_fmm[0] / t for t in times_fmm 
                         if not math.isnan(t) and t > 0]
            efficiency_fmm = [s / tc for s, tc in zip(speedup_fmm, 
                            thread_counts[:len(speedup_fmm)])]
        
        results[N] = {
            'threads': thread_counts,
            'times_direct': times_direct,
            'times_fmm': times_fmm,
            'speedup_direct': speedup_direct,
            'speedup_fmm': speedup_fmm,
            'efficiency_direct': efficiency_direct,
            'efficiency_fmm': efficiency_fmm
        }
        
        # 輸出性能摘要
        if speedup_direct:
            max_speedup = max(speedup_direct)
            best_threads = thread_counts[speedup_direct.index(max_speedup)]
            max_efficiency = max(efficiency_direct) if efficiency_direct else 0
            print(f"  直接方法最佳: {max_speedup:.2f}x 加速, "
                  f"{max_efficiency:.1%} 效率 @ {best_threads} 線程")
        
        if speedup_fmm:
            max_speedup = max(speedup_fmm)
            best_threads = thread_counts[speedup_fmm.index(max_speedup)]
            max_efficiency = max(efficiency_fmm) if efficiency_fmm else 0
            print(f"  FMM方法最佳: {max_speedup:.2f}x 加速, "
                  f"{max_efficiency:.1%} 效率 @ {best_threads} 線程")
    
    # 創建進階性能圖表
    create_advanced_performance_plots(results, test_sizes, thread_counts)
    
    return results

def create_advanced_performance_plots(results, test_sizes, thread_counts):
    """創建進階性能分析圖表"""
    
    fig = plt.figure(figsize=(20, 16))
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_sizes)))
    
    # 1. 直接方法加速比
    ax1 = plt.subplot(3, 4, 1)
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_direct']:
            used_threads = thread_counts[:len(results[N]['speedup_direct'])]
            ax1.plot(used_threads, results[N]['speedup_direct'], 
                    'o-', color=colors[i], label=f"N={N}", 
                    linewidth=2.5, markersize=8)
    
    ax1.plot(thread_counts, thread_counts, '--k', alpha=0.7, label="理想加速比")
    ax1.set_xlabel("線程數量", fontsize=12)
    ax1.set_ylabel("加速比", fontsize=12)
    ax1.set_title("直接方法加速比", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, max(thread_counts) + 0.5)
    
    # 2. 直接方法效率
    ax2 = plt.subplot(3, 4, 2)
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_direct']:
            used_threads = thread_counts[:len(results[N]['efficiency_direct'])]
            ax2.plot(used_threads, results[N]['efficiency_direct'], 
                    's-', color=colors[i], label=f"N={N}", 
                    linewidth=2.5, markersize=8)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="良好 (80%)")
    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="尚可 (50%)")
    ax2.set_xlabel("線程數量", fontsize=12)
    ax2.set_ylabel("並行效率", fontsize=12)
    ax2.set_title("直接方法效率", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, max(thread_counts) + 0.5)
    ax2.set_ylim(0, 1.1)
    
    # 3. FMM方法加速比
    ax3 = plt.subplot(3, 4, 3)
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['speedup_fmm']:
            used_threads = thread_counts[:len(results[N]['speedup_fmm'])]
            ax3.plot(used_threads, results[N]['speedup_fmm'], 
                    '^-', color=colors[i], label=f"N={N}", 
                    linewidth=2.5, markersize=8)
    
    ax3.plot(thread_counts, thread_counts, '--k', alpha=0.7, label="理想加速比")
    ax3.set_xlabel("線程數量", fontsize=12)
    ax3.set_ylabel("加速比", fontsize=12)
    ax3.set_title("FMM方法加速比", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, max(thread_counts) + 0.5)
    
    # 4. FMM方法效率
    ax4 = plt.subplot(3, 4, 4)
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_fmm']:
            used_threads = thread_counts[:len(results[N]['efficiency_fmm'])]
            ax4.plot(used_threads, results[N]['efficiency_fmm'], 
                    'd-', color=colors[i], label=f"N={N}", 
                    linewidth=2.5, markersize=8)
    
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    ax4.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="良好 (80%)")
    ax4.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label="尚可 (50%)")
    ax4.set_xlabel("線程數量", fontsize=12)
    ax4.set_ylabel("並行效率", fontsize=12)
    ax4.set_title("FMM方法效率", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, max(thread_counts) + 0.5)
    ax4.set_ylim(0, 1.1)
    
    # 5. 絕對性能比較（單線程）
    ax5 = plt.subplot(3, 4, 5)
    valid_sizes = []
    times_direct_1t = []
    times_fmm_1t = []
    
    for N in test_sizes:
        if N in results:
            if results[N]['times_direct'] and not math.isnan(results[N]['times_direct'][0]):
                valid_sizes.append(N)
                times_direct_1t.append(results[N]['times_direct'][0])
                times_fmm_1t.append(results[N]['times_fmm'][0] if results[N]['times_fmm'] and not math.isnan(results[N]['times_fmm'][0]) else np.nan)
    
    if valid_sizes:
        ax5.loglog(valid_sizes, times_direct_1t, 'o-', label="直接方法 O(N²)", linewidth=2.5, markersize=8)
        valid_fmm = [(N, t) for N, t in zip(valid_sizes, times_fmm_1t) if not math.isnan(t)]
        if valid_fmm:
            N_fmm, t_fmm = zip(*valid_fmm)
            ax5.loglog(N_fmm, t_fmm, 's-', label="FMM方法 O(N log N)", linewidth=2.5, markersize=8)
    
    ax5.set_xlabel("粒子數量 (N)", fontsize=12)
    ax5.set_ylabel("計算時間 (秒)", fontsize=12)
    ax5.set_title("單線程性能比較", fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. 最佳線程數分析
    ax6 = plt.subplot(3, 4, 6)
    optimal_threads_direct = []
    optimal_threads_fmm = []
    problem_sizes = []
    
    for N in test_sizes:
        if N in results:
            problem_sizes.append(N)
            
            # 找直接方法的最佳線程數
            if results[N]['speedup_direct']:
                best_idx = np.argmax(results[N]['speedup_direct'])
                optimal_threads_direct.append(thread_counts[best_idx])
            else:
                optimal_threads_direct.append(np.nan)
            
            # 找FMM方法的最佳線程數
            if results[N]['speedup_fmm']:
                best_idx = np.argmax(results[N]['speedup_fmm'])
                optimal_threads_fmm.append(thread_counts[best_idx])
            else:
                optimal_threads_fmm.append(np.nan)
    
    if problem_sizes:
        ax6.semilogx(problem_sizes, optimal_threads_direct, 'o-', label="直接方法", linewidth=2.5, markersize=8)
        ax6.semilogx(problem_sizes, optimal_threads_fmm, 's-', label="FMM方法", linewidth=2.5, markersize=8)
    
    ax6.set_xlabel("粒子數量 (N)", fontsize=12)
    ax6.set_ylabel("最佳線程數", fontsize=12)
    ax6.set_title("最佳線程數分析", fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0.5, max(thread_counts) + 0.5)
    
    # 7. 效率vs問題大小熱圖（FMM）
    ax7 = plt.subplot(3, 4, 7)
    efficiency_matrix = np.full((len(thread_counts), len(test_sizes)), np.nan)
    
    for i, N in enumerate(test_sizes):
        if N in results and results[N]['efficiency_fmm']:
            for j, eff in enumerate(results[N]['efficiency_fmm']):
                if j < len(thread_counts):
                    efficiency_matrix[j, i] = eff
    
    im = ax7.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=1, interpolation='nearest')
    ax7.set_xticks(range(len(test_sizes)))
    ax7.set_xticklabels([f"{N//1000}K" for N in test_sizes], rotation=45)
    ax7.set_yticks(range(len(thread_counts)))
    ax7.set_yticklabels(thread_counts)
    ax7.set_xlabel("粒子數量", fontsize=12)
    ax7.set_ylabel("線程數", fontsize=12)
    ax7.set_title("FMM效率熱圖", fontsize=14, fontweight='bold')
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('並行效率', fontsize=10)
    
    # 8. 加速比隨問題大小變化
    ax8 = plt.subplot(3, 4, 8)
    max_speedup_direct = []
    max_speedup_fmm = []
    
    for N in test_sizes:
        if N in results:
            if results[N]['speedup_direct']:
                max_speedup_direct.append(max(results[N]['speedup_direct']))
            else:
                max_speedup_direct.append(np.nan)
            
            if results[N]['speedup_fmm']:
                max_speedup_fmm.append(max(results[N]['speedup_fmm']))
            else:
                max_speedup_fmm.append(np.nan)
    
    ax8.semilogx(test_sizes, max_speedup_direct, 'o-', label="直接方法", linewidth=2.5, markersize=8)
    ax8.semilogx(test_sizes, max_speedup_fmm, 's-', label="FMM方法", linewidth=2.5, markersize=8)
    ax8.axhline(y=max(thread_counts), color='black', linestyle='--', alpha=0.7, label="理論最大值")
    
    ax8.set_xlabel("粒子數量 (N)", fontsize=12)
    ax8.set_ylabel("最大加速比", fontsize=12)
    ax8.set_title("最大加速比vs問題大小", fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # 9-12. 詳細性能分析
    # 9. 內存帶寬分析
    ax9 = plt.subplot(3, 4, 9)
    memory_bandwidth_direct = []
    memory_bandwidth_fmm = []
    
    for N in test_sizes:
        if N in results:
            # 估算記憶體頻寬需求 (bytes/second)
            # 直接方法: N^2 * 8 bytes (double) * 操作數
            if results[N]['times_direct'] and not math.isnan(results[N]['times_direct'][0]):
                ops_per_pair = 20  # 大約的操作數
                memory_ops = N * N * ops_per_pair * 8  # bytes
                bandwidth = memory_ops / results[N]['times_direct'][0] / 1e9  # GB/s
                memory_bandwidth_direct.append(bandwidth)
            else:
                memory_bandwidth_direct.append(np.nan)
            
            # FMM方法的頻寬需求較低
            if results[N]['times_fmm'] and not math.isnan(results[N]['times_fmm'][0]):
                ops_fmm = N * math.log2(N) * 50 * 8  # 估算
                bandwidth = ops_fmm / results[N]['times_fmm'][0] / 1e9  # GB/s
                memory_bandwidth_fmm.append(bandwidth)
            else:
                memory_bandwidth_fmm.append(np.nan)
    
    valid_direct = [(N, bw) for N, bw in zip(test_sizes, memory_bandwidth_direct) if not math.isnan(bw)]
    valid_fmm = [(N, bw) for N, bw in zip(test_sizes, memory_bandwidth_fmm) if not math.isnan(bw)]
    
    if valid_direct:
        N_d, bw_d = zip(*valid_direct)
        ax9.semilogx(N_d, bw_d, 'o-', label="直接方法", linewidth=2.5, markersize=8)
    if valid_fmm:
        N_f, bw_f = zip(*valid_fmm)
        ax9.semilogx(N_f, bw_f, 's-', label="FMM方法", linewidth=2.5, markersize=8)
    
    ax9.set_xlabel("粒子數量 (N)", fontsize=12)
    ax9.set_ylabel("記憶體頻寬 (GB/s)", fontsize=12)
    ax9.set_title("記憶體頻寬需求", fontsize=14, fontweight='bold')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)
    
    # 10. 強擴展性分析
    ax10 = plt.subplot(3, 4, 10)
    
    # 選擇一個中等大小的問題進行強擴展性分析
    if 10000 in results:
        N_fixed = 10000
        result = results[N_fixed]
        
        if result['times_direct']:
            ax10.plot(thread_counts[:len(result['times_direct'])], 
                     result['times_direct'], 'o-', 
                     label=f"直接方法 (N={N_fixed})", linewidth=2.5, markersize=8)
        
        if result['times_fmm']:
            ax10.plot(thread_counts[:len(result['times_fmm'])], 
                     result['times_fmm'], 's-', 
                     label=f"FMM方法 (N={N_fixed})", linewidth=2.5, markersize=8)
    
    ax10.set_xlabel("線程數量", fontsize=12)
    ax10.set_ylabel("計算時間 (秒)", fontsize=12)
    ax10.set_title("強擴展性分析", fontsize=14, fontweight='bold')
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3)
    ax10.set_yscale('log')
    
    # 11. 弱擴展性分析 (每線程固定工作量)
    ax11 = plt.subplot(3, 4, 11)
    
    # 計算每線程固定粒子數的性能
    particles_per_thread = 1000
    weak_scaling_times_direct = []
    weak_scaling_times_fmm = []
    weak_scaling_threads = []
    
    for threads in thread_counts:
        target_N = particles_per_thread * threads
        # 找最接近的測試大小
        closest_N = min(test_sizes, key=lambda x: abs(x - target_N))
        
        if abs(closest_N - target_N) / target_N < 0.2 and closest_N in results:  # 20%誤差內
            result = results[closest_N]
            thread_idx = thread_counts.index(threads) if threads in thread_counts else -1
            
            if thread_idx >= 0:
                if (result['times_direct'] and thread_idx < len(result['times_direct']) 
                    and not math.isnan(result['times_direct'][thread_idx])):
                    weak_scaling_times_direct.append(result['times_direct'][thread_idx])
                    weak_scaling_threads.append(threads)
                
                if (result['times_fmm'] and thread_idx < len(result['times_fmm']) 
                    and not math.isnan(result['times_fmm'][thread_idx])):
                    weak_scaling_times_fmm.append(result['times_fmm'][thread_idx])
    
    if weak_scaling_times_direct:
        ax11.plot(weak_scaling_threads[:len(weak_scaling_times_direct)], 
                 weak_scaling_times_direct, 'o-', 
                 label="直接方法", linewidth=2.5, markersize=8)
    
    if weak_scaling_times_fmm:
        ax11.plot(weak_scaling_threads[:len(weak_scaling_times_fmm)], 
                 weak_scaling_times_fmm, 's-', 
                 label="FMM方法", linewidth=2.5, markersize=8)
    
    ax11.set_xlabel("線程數量", fontsize=12)
    ax11.set_ylabel("計算時間 (秒)", fontsize=12)
    ax11.set_title(f"弱擴展性分析\n(每線程{particles_per_thread}粒子)", fontsize=14, fontweight='bold')
    ax11.legend(fontsize=10)
    ax11.grid(True, alpha=0.3)
    
    # 12. 性能摘要表
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # 創建性能摘要文本
    summary_text = "性能分析摘要\n" + "="*30 + "\n\n"
    
    # 計算平均效率
    avg_eff_direct = []
    avg_eff_fmm = []
    
    for N in test_sizes:
        if N in results:
            if results[N]['efficiency_direct']:
                avg_eff_direct.extend(results[N]['efficiency_direct'])
            if results[N]['efficiency_fmm']:
                avg_eff_fmm.extend(results[N]['efficiency_fmm'])
    
    if avg_eff_direct:
        summary_text += f"直接方法平均效率: {np.mean(avg_eff_direct):.1%}\n"
        summary_text += f"直接方法最高效率: {np.max(avg_eff_direct):.1%}\n\n"
    
    if avg_eff_fmm:
        summary_text += f"FMM方法平均效率: {np.mean(avg_eff_fmm):.1%}\n"
        summary_text += f"FMM方法最高效率: {np.max(avg_eff_fmm):.1%}\n\n"
    
    # 找出最佳配置
    best_configs = []
    for N in test_sizes:
        if N in results:
            if results[N]['efficiency_fmm']:
                best_eff = max(results[N]['efficiency_fmm'])
                best_threads = thread_counts[results[N]['efficiency_fmm'].index(best_eff)]
                if best_eff > 0.5:  # 只顯示效率>50%的配置
                    best_configs.append(f"N={N}: {best_threads}線程, {best_eff:.1%}")
    
    if best_configs:
        summary_text += "推薦配置 (FMM):\n"
        for config in best_configs[:5]:  # 只顯示前5個
            summary_text += f"  {config}\n"
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 保存高質量圖片
    advanced_path = os.path.join(OUTPUT_DIR, "advanced_performance_analysis.png")
    plt.savefig(advanced_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ 進階性能分析圖表已保存至 {advanced_path}")

def quick_scaling_test():
    """快速擴展性測試"""
    print("\n快速擴展性測試")
    print("=" * 50)
    
    test_sizes = [100, 500, 1000, 2000, 5000]
    steps = 3
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 64
    eps = 0.01
    G = 1.0
    
    print(f"使用 {num_threads} 線程進行測試")
    print(f"測試粒子數量: {test_sizes}")
    
    results = []
    
    for N in test_sizes:
        print(f"\n測試 N = {N}")
        
        system = OptimizedParticleSystem(N, domain_size)
        
        # 測試直接方法
        t_direct = None
        if HAS_DIRECT:
            try:
                print("  測試直接方法...")
                start_time = time.perf_counter()
                for _ in range(steps):
                    ax, ay = optimized_direct_force(system.x, system.y, system.m, eps)
                    if ax is None:
                        raise Exception("直接力計算失敗")
                t_direct = (time.perf_counter() - start_time) / steps
                print(f"  ✓ 直接方法: {t_direct:.6f} 秒")
            except Exception as e:
                print(f"  ✗ 直接方法: 失敗 ({e})")
                t_direct = float('nan')
        else:
            print("  ✗ 直接方法: 不可用")
            t_direct = float('nan')
        
        # 測試FMM方法
        t_fmm = None
        if HAS_FMM:
            try:
                print("  測試FMM方法...")
                start_time = time.perf_counter()
                for _ in range(steps):
                    ax, ay = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, maxLeaf, eps, G)
                    if ax is None:
                        raise Exception("FMM力計算失敗")
                t_fmm = (time.perf_counter() - start_time) / steps
                print(f"  ✓ FMM方法: {t_fmm:.6f} 秒")
            except Exception as e:
                print(f"  ✗ FMM方法: 失敗 ({e})")
                t_fmm = float('nan')
        else:
            print("  ✗ FMM方法: 不可用")
            t_fmm = float('nan')
        
        results.append((N, t_direct, t_fmm))
    
    # 保存結果並創建圖表
    save_scaling_results(results, "quick_scaling_optimized")
    print("\n✓ 快速擴展性測試完成！")

def accuracy_verification_test():
    """精度驗證測試"""
    print("\n精度驗證測試")
    print("=" * 50)
    
    N = 100
    domain_size = 50.0
    theta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    eps = 0.01
    G = 1.0
    
    system = OptimizedParticleSystem(N, domain_size)
    
    # 獲取參考解（直接方法）
    if not HAS_DIRECT:
        print("直接方法不可用，無法進行精度驗證")
        return
    
    print("計算參考解（直接方法）...")
    ax_ref, ay_ref = optimized_direct_force(system.x, system.y, system.m, eps)
    if ax_ref is None:
        print("參考解計算失敗")
        return
    
    print("測試FMM方法在不同theta值下的精度...")
    
    theta_errors = []
    theta_times = []
    
    for theta in theta_values:
        if HAS_FMM:
            print(f"  測試 theta = {theta}")
            
            # 計算FMM解
            start_time = time.perf_counter()
            ax_fmm, ay_fmm = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, 64, eps, G)
            compute_time = time.perf_counter() - start_time
            
            if ax_fmm is not None:
                # 計算相對誤差
                force_magnitude_ref = np.sqrt(ax_ref**2 + ay_ref**2)
                force_magnitude_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
                
                relative_error = np.mean(np.abs(force_magnitude_fmm - force_magnitude_ref) / 
                                       (force_magnitude_ref + 1e-10))
                
                theta_errors.append(relative_error)
                theta_times.append(compute_time)
                
                print(f"    相對誤差: {relative_error:.2e}")
                print(f"    計算時間: {compute_time:.6f} 秒")
            else:
                theta_errors.append(float('nan'))
                theta_times.append(float('nan'))
                print(f"    計算失敗")
    
    # 創建精度-性能權衡圖
    if theta_errors and not all(math.isnan(e) for e in theta_errors):
        create_accuracy_plot(theta_values, theta_errors, theta_times)

def create_accuracy_plot(theta_values, errors, times):
    """創建精度分析圖表"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 誤差vs theta
    valid_data = [(t, e) for t, e in zip(theta_values, errors) if not math.isnan(e)]
    if valid_data:
        theta_valid, errors_valid = zip(*valid_data)
        ax1.semilogy(theta_valid, errors_valid, 'o-', linewidth=2.5, markersize=8)
        ax1.set_xlabel("Theta (開放角)", fontsize=12)
        ax1.set_ylabel("相對誤差", fontsize=12)
        ax1.set_title("FMM精度分析", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # 2. 時間vs theta
    valid_time_data = [(t, time) for t, time in zip(theta_values, times) if not math.isnan(time)]
    if valid_time_data:
        theta_valid, times_valid = zip(*valid_time_data)
        ax2.plot(theta_valid, times_valid, 's-', linewidth=2.5, markersize=8, color='orange')
        ax2.set_xlabel("Theta (開放角)", fontsize=12)
        ax2.set_ylabel("計算時間 (秒)", fontsize=12)
        ax2.set_title("FMM性能分析", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. 精度-性能權衡
    if valid_data and valid_time_data:
        ax3.loglog(errors_valid, times_valid, '^-', linewidth=2.5, markersize=8, color='green')
        
        # 標記每個點的theta值
        for i, (error, time, theta) in enumerate(zip(errors_valid, times_valid, theta_valid)):
            ax3.annotate(f'θ={theta}', (error, time), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel("相對誤差", fontsize=12)
        ax3.set_ylabel("計算時間 (秒)", fontsize=12)
        ax3.set_title("精度-性能權衡", fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    accuracy_path = os.path.join(OUTPUT_DIR, "accuracy_analysis.png")
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 精度分析圖表已保存至 {accuracy_path}")

def system_info():
    """系統信息"""
    print("\n系統信息")
    print("=" * 50)
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"OpenMP線程數: {os.environ.get('OMP_NUM_THREADS', '未設定')}")
    print(f"OpenMP處理器綁定: {os.environ.get('OMP_PROC_BIND', '未設定')}")
    print(f"OpenMP放置策略: {os.environ.get('OMP_PLACES', '未設定')}")
    print(f"OpenMP調度策略: {os.environ.get('OMP_SCHEDULE', '未設定')}")
    print(f"直接方法模組: {'可用' if HAS_DIRECT else '不可用'}")
    print(f"FMM模組: {'可用' if HAS_FMM else '不可用'}")
    
    try:
        import platform
        print(f"作業系統: {platform.platform()}")
        print(f"處理器: {platform.processor()}")
    except:
        pass
    
    try:
        import multiprocessing
        print(f"CPU核心數: {multiprocessing.cpu_count()}")
    except:
        pass
    
    try:
        print(f"NumPy版本: {np.__version__}")
        print(f"NumPy配置:")
        np.show_config()
    except:
        pass

# 輔助函數
def save_scaling_results(results, filename):
    """保存擴展性結果"""
    # 保存CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    with open(csv_path, "w") as f:
        f.write("N,Direct,FMM\n")
        for N, t_direct, t_fmm in results:
            f.write(f"{N},{t_direct},{t_fmm}\n")
    
    # 創建圖表
    valid_results = [(N, t_d, t_f) for N, t_d, t_f in results 
                    if not (math.isnan(t_d) and math.isnan(t_f))]
    
    if valid_results:
        Ns, times_direct, times_fmm = zip(*valid_results)
        
        plt.figure(figsize=(10, 7))
        
        # 直接方法
        valid_direct = [(N, t) for N, t in zip(Ns, times_direct) if not math.isnan(t)]
        if valid_direct:
            N_direct, t_direct = zip(*valid_direct)
            plt.loglog(N_direct, t_direct, 'o-', label="直接方法 O(N²)", 
                      linewidth=2.5, markersize=8)
        
        # FMM方法
        valid_fmm = [(N, t) for N, t in zip(Ns, times_fmm) if not math.isnan(t)]
        if valid_fmm:
            N_fmm, t_fmm = zip(*valid_fmm)
            plt.loglog(N_fmm, t_fmm, 's-', label="FMM方法 O(N log N)", 
                      linewidth=2.5, markersize=8)
        
        # 理論曲線
        if valid_direct and len(N_direct) > 1:
            N_theory = np.logspace(np.log10(min(N_direct)), np.log10(max(N_direct)), 100)
            # O(N^2) 理論曲線
            t_theory_n2 = t_direct[0] * (N_theory / N_direct[0])**2
            plt.loglog(N_theory, t_theory_n2, '--', alpha=0.7, label="O(N²) 理論")
        
        if valid_fmm and len(N_fmm) > 1:
            N_theory_fmm = np.logspace(np.log10(min(N_fmm)), np.log10(max(N_fmm)), 100)
            # O(N log N) 理論曲線
            t_theory_nlogn = t_fmm[0] * (N_theory_fmm / N_fmm[0]) * np.log2(N_theory_fmm) / np.log2(N_fmm[0])
            plt.loglog(N_theory_fmm, t_theory_nlogn, ':', alpha=0.7, label="O(N log N) 理論")
        
        plt.xlabel("粒子數量 (N)", fontsize=12)
        plt.ylabel("每步計算時間 (秒)", fontsize=12)
        plt.title("優化版性能比較", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 結果已保存至 {csv_path} 和 {png_path}")

def energy_conservation_test():
    """能量守恆測試"""
    print("\n能量守恆測試")
    print("=" * 50)
    
    try:
        N = int(input("輸入粒子數量 (例: 200): "))
    except ValueError:
        N = 200
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 64
    eps = 0.01
    G = 1.0
    dt = 0.001
    steps = 1000
    
    methods = []
    if HAS_DIRECT and N <= 1000:
        methods.append(("直接方法", "direct"))
    if HAS_FMM:
        methods.append(("FMM方法", "fmm"))
    
    if not methods:
        print("無可用的計算方法")
        return
    
    plt.figure(figsize=(12, 8))
    
    for method_name, method in methods:
        print(f"\n測試 {method_name}...")
        
        # 初始化粒子
        system = OptimizedParticleSystem(N, domain_size)
        vx = np.zeros(N, dtype=np.float64)
        vy = np.zeros(N, dtype=np.float64)
        
        # 計算初始能量
        E0 = calculate_total_energy(system.x, system.y, vx, vy, system.m, G, eps)
        
        times = []
        rel_errors = []
        
        print(f"  初始能量: {E0:.6e}")
        
        for step in range(0, steps, 10):  # 每10步記錄一次
            # 計算力
            if method == "direct":
                ax, ay = optimized_direct_force(system.x, system.y, system.m, eps)
            elif method == "fmm":
                ax, ay = optimized_fmm_force(system.x, system.y, system.m, N,
                                           domain_size, theta, maxLeaf, eps, G)
            
            if ax is None:
                print(f"  {method_name} 計算失敗")
                break
            
            # Leapfrog積分10步
            for _ in range(10):
                # 半步速度更新
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                
                # 位置更新
                system.x += dt * vx
                system.y += dt * vy
                
                # 重新計算力
                if method == "direct":
                    ax, ay = optimized_direct_force(system.x, system.y, system.m, eps)
                elif method == "fmm":
                    ax, ay = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, maxLeaf, eps, G)
                
                if ax is None:
                    break
                
                # 半步速度更新
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
            
            if ax is None:
                break
            
            # 計算當前能量和相對誤差
            E = calculate_total_energy(system.x, system.y, vx, vy, system.m, G, eps)
            rel_error = abs(E - E0) / abs(E0)
            
            times.append(step * dt)
            rel_errors.append(rel_error)
            
            if step % 100 == 0:
                print(f"  步數 {step}: 相對能量誤差 = {rel_error:.2e}")
        
        if times and rel_errors:
            plt.semilogy(times, rel_errors, label=method_name, linewidth=2)
    
    plt.xlabel("時間", fontsize=12)
    plt.ylabel("相對能量誤差", fontsize=12)
    plt.title(f"能量守恆測試 (N={N})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    energy_path = os.path.join(OUTPUT_DIR, "energy_conservation_optimized.png")
    plt.savefig(energy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 能量守恆圖表已保存至 {energy_path}")

def calculate_total_energy(x, y, vx, vy, m, G, eps):
    """計算總能量（動能+位能）"""
    N = len(x)
    
    # 動能
    ke = 0.5 * np.sum(m * (vx**2 + vy**2))
    
    # 位能
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = math.sqrt(dx*dx + dy*dy + eps*eps)
            pe -= G * m[i] * m[j] / r
    
    return ke + pe

def interactive_visualization():
    """互動式視覺化"""
    print("\n互動式視覺化")
    print("=" * 50)
    
    try:
        N = int(input("輸入粒子數量 (例: 100): "))
        method = input("選擇方法 (direct/fmm): ").strip().lower()
        frames = int(input("動畫幀數 (例: 200): "))
    except ValueError:
        print("輸入無效，使用預設值")
        N = 100
        method = "fmm"
        frames = 200
    
    if method not in ["direct", "fmm"]:
        print("方法無效，使用FMM")
        method = "fmm"
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 64
    eps = 0.01
    G = 1.0
    dt = 0.001
    
    # 初始化粒子
    system = OptimizedParticleSystem(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    print(f"創建動畫，使用 {method} 方法...")
    
    # 創建動畫
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左側：粒子分佈
    ax1.set_xlim(-domain_size*1.2, domain_size*1.2)
    ax1.set_ylim(-domain_size*1.2, domain_size*1.2)
    ax1.set_aspect('equal')
    ax1.set_title(f"粒子分佈 ({method.upper()}, N={N})")
    scat = ax1.scatter(system.x, system.y, s=20, c='blue', alpha=0.6)
    
    # 右側：能量變化
    ax2.set_xlabel("時間")
    ax2.set_ylabel("總能量")
    ax2.set_title("能量變化")
    energy_line, = ax2.plot([], [], 'r-', linewidth=2)
    energy_times = []
    energy_values = []
    
    def update(frame):
        nonlocal system, vx, vy, energy_times, energy_values
        
        # 計算力
        if method == "direct" and HAS_DIRECT:
            ax_arr, ay_arr = optimized_direct_force(system.x, system.y, system.m, eps)
        elif method == "fmm" and HAS_FMM:
            ax_arr, ay_arr = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, maxLeaf, eps, G)
        else:
            return scat, energy_line
        
        if ax_arr is None:
            return scat, energy_line
        
        # 更新位置和速度（Leapfrog）
        vx += 0.5 * dt * ax_arr
        vy += 0.5 * dt * ay_arr
        system.x += dt * vx
        system.y += dt * vy
        
        # 重新計算力用於第二次速度更新
        if method == "direct" and HAS_DIRECT:
            ax_arr, ay_arr = optimized_direct_force(system.x, system.y, system.m, eps)
        elif method == "fmm" and HAS_FMM:
            ax_arr, ay_arr = optimized_fmm_force(system.x, system.y, system.m, N,
                                               domain_size, theta, maxLeaf, eps, G)
        
        if ax_arr is not None:
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr
        
        # 更新視覺化
        scat.set_offsets(np.column_stack((system.x, system.y)))
        
        # 更新能量圖
        if frame % 5 == 0:  # 每5幀更新一次能量
            E = calculate_total_energy(system.x, system.y, vx, vy, system.m, G, eps)
            energy_times.append(frame * dt)
            energy_values.append(E)
            
            if len(energy_times) > 100:  # 只保留最近100個點
                energy_times = energy_times[-100:]
                energy_values = energy_values[-100:]
            
            energy_line.set_data(energy_times, energy_values)
            if energy_values:
                ax2.set_xlim(energy_times[0], energy_times[-1])
                ax2.set_ylim(min(energy_values)*1.1, max(energy_values)*1.1)
        
        return scat, energy_line
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, 
                                 blit=True, repeat=True)
    
    # 保存動畫
    gif_path = os.path.join(OUTPUT_DIR, f"interactive_simulation_{method}_{N}.gif")
    try:
        ani.save(gif_path, writer='pillow', fps=20)
        print(f"✓ 互動式模擬已保存至 {gif_path}")
    except Exception as e:
        print(f"保存動畫失敗: {e}")
    
    plt.show()

def main_menu():
    """主選單"""
    while True:
        print("\n" + "=" * 70)
        print("2D N-Body 問題模擬平台 - 完全優化版")
        print("(解決並行化、記憶體存取、false sharing等問題)")
        print("=" * 70)
        print("選擇功能:")
        print(" 1) 快速擴展性測試")
        print(" 2) 進階性能基準測試")
        print(" 3) 精度驗證測試")
        print(" 4) 能量守恆測試")
        print(" 5) 互動式視覺化")
        print(" 6) 系統信息")
        print(" q) 退出程式")
        print("=" * 70)
        
        choice = input("請輸入選擇: ").strip().lower()
        
        if choice == '1':
            quick_scaling_test()
        elif choice == '2':
            advanced_performance_benchmark()
        elif choice == '3':
            accuracy_verification_test()
        elif choice == '4':
            energy_conservation_test()
        elif choice == '5':
            interactive_visualization()
        elif choice == '6':
            system_info()
        elif choice == 'q':
            print("再見！")
            break
        else:
            print("無效選擇，請重新輸入。")

if __name__ == "__main__":
    print("2D N-Body 問題模擬平台啟動中...")
    print("完全優化版：解決並行化問題")
    print(f"優化的OpenMP設定：")
    print(f"  - 線程數: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  - 處理器綁定: {os.environ.get('OMP_PROC_BIND')}")
    print(f"  - 放置策略: {os.environ.get('OMP_PLACES')}")
    print(f"  - 調度策略: {os.environ.get('OMP_SCHEDULE')}")
    
    # 檢查模組可用性
    if not HAS_DIRECT and not HAS_FMM:
        print("錯誤：無可用的計算模組！")
        print("請確保 force_kernel 和 fmm_kernel 模組已正確編譯。")
        print("執行: python setup_optimized.py build_ext --inplace")
        sys.exit(1)
    
    main_menu()
