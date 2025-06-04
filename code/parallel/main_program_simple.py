#!/usr/bin/env python3
# main_program_simple.py
# 簡化版主程序，確保基本功能正常運行

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# 設定OpenMP環境變數
def setup_openmp():
    """設定OpenMP環境變數"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # 保守的線程設定
    num_threads = min(cpu_count, 8)
    
    env_settings = {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_SCHEDULE": "guided",
        "OMP_DYNAMIC": "false"
    }
    
    for key, value in env_settings.items():
        os.environ[key] = value
    
    print(f"OpenMP設定：{num_threads}線程")
    return num_threads

# 在導入前設定環境
num_threads = setup_openmp()

# 嘗試導入模組
try:
    import force_kernel
    HAS_DIRECT = True
    print("✓ force_kernel載入成功")
except ImportError as e:
    HAS_DIRECT = False
    print(f"✗ force_kernel載入失敗: {e}")

try:
    import fmm_kernel
    HAS_FMM = True
    print("✓ fmm_kernel載入成功")
except ImportError as e:
    HAS_FMM = False
    print(f"✗ fmm_kernel載入失敗: {e}")

# 輸出目錄
OUTPUT_DIR = "output_simple"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def initialize_particles(N, domain_size=50.0):
    """初始化粒子"""
    rng = np.random.default_rng(42)
    
    # 極坐標分佈
    angles = rng.uniform(0, 2*math.pi, N)
    radii = domain_size * np.sqrt(rng.uniform(0, 1, N))
    
    # 確保記憶體連續
    x = np.ascontiguousarray(radii * np.cos(angles), dtype=np.float64)
    y = np.ascontiguousarray(radii * np.sin(angles), dtype=np.float64)
    m = np.ascontiguousarray(np.ones(N), dtype=np.float64)
    
    return x, y, m

def safe_direct_force(x, y, m, eps2):
    """安全的直接力計算"""
    N = len(x)
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_DIRECT:
        try:
            force_kernel.direct_force(x, y, m, eps2, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"直接力計算失敗: {e}")
    
    return None, None

def safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G):
    """安全的FMM力計算"""
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_FMM:
        try:
            fmm_kernel.fmm_force(x, y, m, N, domain_size, 
                               theta, maxLeaf, eps, G, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"FMM力計算失敗: {e}")
    
    return None, None

def simple_benchmark():
    """簡單基準測試"""
    print("\n簡單基準測試")
    print("=" * 40)
    
    if not HAS_DIRECT and not HAS_FMM:
        print("無可用的計算模組")
        return
    
    # 測試參數
    test_sizes = [100, 200, 500, 1000]
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 16
    eps = 0.01
    G = 1.0
    
    print(f"使用{num_threads}線程測試")
    
    results = []
    
    for N in test_sizes:
        print(f"\n測試N = {N}個粒子")
        
        # 初始化粒子
        x, y, m = initialize_particles(N, domain_size)
        
        # 測試直接方法
        t_direct = None
        if HAS_DIRECT and N <= 1000:  # 限制直接方法的測試範圍
            try:
                print("  測試直接方法...")
                
                # 熱身
                safe_direct_force(x, y, m, eps*eps)
                
                # 測試
                start_time = time.perf_counter()
                for _ in range(3):
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is None:
                        raise Exception("計算失敗")
                t_direct = (time.perf_counter() - start_time) / 3
                
                print(f"  ✓ 直接方法: {t_direct:.6f} 秒")
                
            except Exception as e:
                print(f"  ✗ 直接方法失敗: {e}")
                t_direct = float('nan')
        else:
            t_direct = float('nan')
        
        # 測試FMM方法
        t_fmm = None
        if HAS_FMM:
            try:
                print("  測試FMM方法...")
                
                # 熱身
                safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                # 測試
                start_time = time.perf_counter()
                for _ in range(3):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, 
                                          theta, maxLeaf, eps, G)
                    if ax is None:
                        raise Exception("計算失敗")
                t_fmm = (time.perf_counter() - start_time) / 3
                
                print(f"  ✓ FMM方法: {t_fmm:.6f} 秒")
                
            except Exception as e:
                print(f"  ✗ FMM方法失敗: {e}")
                t_fmm = float('nan')
        else:
            t_fmm = float('nan')
        
        results.append((N, t_direct, t_fmm))
    
    # 創建性能圖表
    create_performance_plot(results)
    
    # 顯示結果摘要
    print("\n性能摘要:")
    print("-" * 40)
    for N, t_direct, t_fmm in results:
        print(f"N={N:4d}: ", end="")
        if not math.isnan(t_direct):
            print(f"Direct={t_direct:.6f}s ", end="")
        if not math.isnan(t_fmm):
            print(f"FMM={t_fmm:.6f}s ", end="")
            if not math.isnan(t_direct) and t_direct > 0:
                speedup = t_direct / t_fmm
                print(f"Speedup={speedup:.2f}x", end="")
        print()

def create_performance_plot(results):
    """創建性能圖表"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 分離數據
    Ns = [r[0] for r in results]
    times_direct = [r[1] for r in results if not math.isnan(r[1])]
    times_fmm = [r[2] for r in results if not math.isnan(r[2])]
    
    Ns_direct = [r[0] for r in results if not math.isnan(r[1])]
    Ns_fmm = [r[0] for r in results if not math.isnan(r[2])]
    
    # 繪製圖表
    if times_direct:
        ax.loglog(Ns_direct, times_direct, 'o-', 
                 label="直接方法 O(N²)", linewidth=2, markersize=8)
    
    if times_fmm:
        ax.loglog(Ns_fmm, times_fmm, 's-', 
                 label="FMM方法 O(N log N)", linewidth=2, markersize=8)
    
    ax.set_xlabel("粒子數量 (N)", fontsize=12)
    ax.set_ylabel("計算時間 (秒)", fontsize=12)
    ax.set_title("性能比較", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 保存圖表
    plot_path = os.path.join(OUTPUT_DIR, "simple_benchmark.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 性能圖表已保存至: {plot_path}")

def accuracy_test():
    """精度測試"""
    print("\n精度測試")
    print("=" * 40)
    
    if not HAS_DIRECT or not HAS_FMM:
        print("需要兩種方法才能進行精度比較")
        return
    
    N = 100
    domain_size = 50.0
    eps = 0.01
    G = 1.0
    
    print(f"使用{N}個粒子進行精度測試")
    
    # 初始化粒子
    x, y, m = initialize_particles(N, domain_size)
    
    # 計算參考解（直接方法）
    print("計算參考解（直接方法）...")
    ax_ref, ay_ref = safe_direct_force(x, y, m, eps*eps)
    
    if ax_ref is None:
        print("參考解計算失敗")
        return
    
    # 測試不同theta值的FMM精度
    theta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    errors = []
    
    print("測試FMM精度...")
    for theta in theta_values:
        ax_fmm, ay_fmm = safe_fmm_force(x, y, m, N, domain_size, 
                                       theta, 16, eps, G)
        
        if ax_fmm is not None:
            # 計算相對誤差
            force_ref = np.sqrt(ax_ref**2 + ay_ref**2)
            force_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
            
            rel_error = np.mean(np.abs(force_fmm - force_ref) / 
                               (force_ref + 1e-10))
            errors.append(rel_error)
            
            print(f"  theta={theta}: 相對誤差={rel_error:.4e}")
        else:
            errors.append(float('nan'))
    
    # 創建精度圖表
    if errors and not all(math.isnan(e) for e in errors):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        valid_data = [(t, e) for t, e in zip(theta_values, errors) 
                     if not math.isnan(e)]
        if valid_data:
            theta_valid, errors_valid = zip(*valid_data)
            ax.semilogy(theta_valid, errors_valid, 'o-', 
                       linewidth=2, markersize=8)
            ax.set_xlabel("Theta (開放角)", fontsize=12)
            ax.set_ylabel("相對誤差", fontsize=12)
            ax.set_title("FMM精度分析", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            accuracy_path = os.path.join(OUTPUT_DIR, "accuracy_test.png")
            plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 精度圖表已保存至: {accuracy_path}")

def parallel_scaling_test():
    """並行擴展性測試"""
    print("\n並行擴展性測試")
    print("=" * 40)
    
    if not HAS_FMM:
        print("需要FMM方法進行並行測試")
        return
    
    N = 1000
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 16
    eps = 0.01
    G = 1.0
    
    # 測試不同線程數
    thread_counts = [1, 2, 4, 8]
    times = []
    
    print(f"使用{N}個粒子測試並行擴展性")
    
    # 初始化粒子
    x, y, m = initialize_particles(N, domain_size)
    
    for threads in thread_counts:
        print(f"\n測試{threads}線程:")
        
        # 設定線程數
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)  # 等待設定生效
        
        try:
            # 熱身
            safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
            
            # 測試
            start_time = time.perf_counter()
            for _ in range(5):
                ax, ay = safe_fmm_force(x, y, m, N, domain_size, 
                                      theta, maxLeaf, eps, G)
                if ax is None:
                    raise Exception("計算失敗")
            elapsed = (time.perf_counter() - start_time) / 5
            
            times.append(elapsed)
            print(f"  時間: {elapsed:.6f} 秒")
            
        except Exception as e:
            print(f"  失敗: {e}")
            times.append(float('nan'))
    
    # 計算加速比
    if times and not math.isnan(times[0]):
        speedups = [times[0] / t for t in times if not math.isnan(t)]
        efficiency = [s / tc for s, tc in zip(speedups, thread_counts[:len(speedups)])]
        
        print(f"\n並行性能:")
        for i, (threads, speedup, eff) in enumerate(zip(thread_counts[:len(speedups)], 
                                                        speedups, efficiency)):
            print(f"  {threads}線程: {speedup:.2f}x 加速, {eff:.1%} 效率")
        
        # 創建擴展性圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 加速比
        ax1.plot(thread_counts[:len(speedups)], speedups, 'o-', 
                linewidth=2, markersize=8, label="實際")
        ax1.plot(thread_counts[:len(speedups)], thread_counts[:len(speedups)], 
                '--', alpha=0.7, label="理想")
        ax1.set_xlabel("線程數", fontsize=12)
        ax1.set_ylabel("加速比", fontsize=12)
        ax1.set_title("加速比", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 效率
        ax2.plot(thread_counts[:len(efficiency)], efficiency, 's-', 
                linewidth=2, markersize=8, color='orange')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label="80%")
        ax2.set_xlabel("線程數", fontsize=12)
        ax2.set_ylabel("並行效率", fontsize=12)
        ax2.set_title("並行效率", fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        scaling_path = os.path.join(OUTPUT_DIR, "parallel_scaling.png")
        plt.savefig(scaling_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 擴展性圖表已保存至: {scaling_path}")

def system_info():
    """系統信息"""
    print("\n系統信息")
    print("=" * 40)
    
    print(f"Python版本: {sys.version.split()[0]}")
    
    # OpenMP設定
    omp_vars = ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE"]
    for var in omp_vars:
        print(f"{var}: {os.environ.get(var, '未設定')}")
    
    # 模組狀態
    print(f"direct方法: {'可用' if HAS_DIRECT else '不可用'}")
    print(f"FMM方法: {'可用' if HAS_FMM else '不可用'}")
    
    # 系統信息
    try:
        import platform
        import multiprocessing
        print(f"作業系統: {platform.system()} {platform.release()}")
        print(f"CPU核心數: {multiprocessing.cpu_count()}")
        print(f"NumPy版本: {np.__version__}")
    except:
        pass

def main_menu():
    """主選單"""
    while True:
        print("\n" + "=" * 50)
        print("2D N-Body 問題模擬平台 - 簡化版")
        print("=" * 50)
        print("選擇功能:")
        print(" 1) 簡單基準測試")
        print(" 2) 精度測試")
        print(" 3) 並行擴展性測試")
        print(" 4) 系統信息")
        print(" q) 退出")
        print("=" * 50)
        
        choice = input("請輸入選擇: ").strip().lower()
        
        if choice == '1':
            simple_benchmark()
        elif choice == '2':
            accuracy_test()
        elif choice == '3':
            parallel_scaling_test()
        elif choice == '4':
            system_info()
        elif choice == 'q':
            print("再見！")
            break
        else:
            print("無效選擇，請重新輸入。")

if __name__ == "__main__":
    print("2D N-Body 問題模擬平台啟動中...")
    print("簡化版 - 確保基本功能正常")
    
    # 檢查模組可用性
    if not HAS_DIRECT and not HAS_FMM:
        print("\n錯誤：無可用的計算模組！")
        print("請先編譯模組:")
        print("  python setup_optimized.py build_ext --inplace")
        print("然後運行測試:")
        print("  python test_compilation.py")
        sys.exit(1)
    
    main_menu()
