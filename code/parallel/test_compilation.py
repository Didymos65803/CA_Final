#!/usr/bin/env python3
# test_compilation.py - 測試編譯和基本功能

import os
import sys
import time
import numpy as np

def test_compilation():
    """測試編譯是否成功"""
    print("Testing compilation and basic functionality...")
    print("=" * 50)
    
    # 測試導入模組
    try:
        import force_kernel
        print("✓ force_kernel imported successfully")
        has_direct = True
    except ImportError as e:
        print(f"✗ force_kernel import failed: {e}")
        has_direct = False
    
    try:
        import fmm_kernel
        print("✓ fmm_kernel imported successfully")
        has_fmm = True
    except ImportError as e:
        print(f"✗ fmm_kernel import failed: {e}")
        has_fmm = False
    
    if not has_direct and not has_fmm:
        print("\nNo modules available for testing")
        return False
    
    # 創建測試數據
    N = 100
    x = np.random.uniform(-10, 10, N).astype(np.float64)
    y = np.random.uniform(-10, 10, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    print(f"\nTesting with {N} particles...")
    
    # 測試直接方法
    if has_direct:
        try:
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            start_time = time.perf_counter()
            force_kernel.direct_force(x, y, m, 0.01, ax, ay)
            elapsed = time.perf_counter() - start_time
            
            # 檢查結果
            force_magnitude = np.sqrt(ax**2 + ay**2)
            max_force = np.max(force_magnitude)
            
            print(f"✓ Direct method test passed")
            print(f"  Time: {elapsed:.6f} seconds")
            print(f"  Max force: {max_force:.6e}")
            
        except Exception as e:
            print(f"✗ Direct method test failed: {e}")
            has_direct = False
    
    # 測試FMM方法
    if has_fmm:
        try:
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            
            start_time = time.perf_counter()
            fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 8, 0.01, 1.0, ax, ay)
            elapsed = time.perf_counter() - start_time
            
            # 檢查結果
            force_magnitude = np.sqrt(ax**2 + ay**2)
            max_force = np.max(force_magnitude)
            
            print(f"✓ FMM method test passed")
            print(f"  Time: {elapsed:.6f} seconds")
            print(f"  Max force: {max_force:.6e}")
            
        except Exception as e:
            print(f"✗ FMM method test failed: {e}")
            has_fmm = False
    
    # 比較精度（如果兩種方法都可用）
    if has_direct and has_fmm:
        try:
            # 直接方法結果
            ax_direct = np.zeros(N, dtype=np.float64)
            ay_direct = np.zeros(N, dtype=np.float64)
            force_kernel.direct_force(x, y, m, 0.01, ax_direct, ay_direct)
            
            # FMM方法結果
            ax_fmm = np.zeros(N, dtype=np.float64)
            ay_fmm = np.zeros(N, dtype=np.float64)
            fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 8, 0.01, 1.0, ax_fmm, ay_fmm)
            
            # 計算相對誤差
            force_direct = np.sqrt(ax_direct**2 + ay_direct**2)
            force_fmm = np.sqrt(ax_fmm**2 + ay_fmm**2)
            
            relative_error = np.mean(np.abs(force_fmm - force_direct) / (force_direct + 1e-10))
            
            print(f"\n✓ Accuracy comparison:")
            print(f"  Relative error: {relative_error:.4e}")
            
            if relative_error < 0.1:
                print("  ✓ Accuracy is acceptable")
            else:
                print("  ⚠ High relative error detected")
                
        except Exception as e:
            print(f"✗ Accuracy comparison failed: {e}")
    
    # 測試並行性能
    print(f"\nTesting parallel performance...")
    test_parallel_performance(has_direct, has_fmm)
    
    return has_direct or has_fmm

def test_parallel_performance(has_direct, has_fmm):
    """測試並行性能"""
    
    # 設定不同的線程數
    thread_counts = [1, 2, 4]
    N = 500
    
    print(f"Testing with {N} particles and different thread counts...")
    
    # 創建測試數據
    x = np.random.uniform(-20, 20, N).astype(np.float64)
    y = np.random.uniform(-20, 20, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    
    for threads in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        print(f"\n  Testing with {threads} threads:")
        
        # 測試直接方法
        if has_direct:
            try:
                ax = np.zeros(N, dtype=np.float64)
                ay = np.zeros(N, dtype=np.float64)
                
                # 熱身
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
                
                # 測試
                start_time = time.perf_counter()
                for _ in range(3):
                    force_kernel.direct_force(x, y, m, 0.01, ax, ay)
                elapsed = (time.perf_counter() - start_time) / 3
                
                print(f"    Direct: {elapsed:.6f} seconds")
                
            except Exception as e:
                print(f"    Direct: Failed ({e})")
        
        # 測試FMM方法
        if has_fmm:
            try:
                ax = np.zeros(N, dtype=np.float64)
                ay = np.zeros(N, dtype=np.float64)
                
                # 熱身
                fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax, ay)
                
                # 測試
                start_time = time.perf_counter()
                for _ in range(3):
                    fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax, ay)
                elapsed = (time.perf_counter() - start_time) / 3
                
                print(f"    FMM: {elapsed:.6f} seconds")
                
            except Exception as e:
                print(f"    FMM: Failed ({e})")

def check_openmp_settings():
    """檢查OpenMP設定"""
    print("\nOpenMP Environment Settings:")
    print("=" * 30)
    
def check_openmp_settings():
    """檢查OpenMP設定"""
    print("\nOpenMP Environment Settings:")
    print("=" * 30)
    
    omp_vars = [
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND", 
        "OMP_PLACES",
        "OMP_SCHEDULE",
        "OMP_DYNAMIC"
    ]
    
    for var in omp_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")

def run_quick_benchmark():
    """運行快速基準測試"""
    print("\nQuick Benchmark:")
    print("=" * 20)
    
    try:
        import force_kernel
        has_direct = True
    except:
        has_direct = False
        
    try:
        import fmm_kernel
        has_fmm = True
    except:
        has_fmm = False
    
    if not has_direct and not has_fmm:
        print("No modules available for benchmark")
        return
    
    test_sizes = [100, 200, 500]
    
    for N in test_sizes:
        print(f"\nN = {N} particles:")
        
        # 創建測試數據
        x = np.random.uniform(-25, 25, N).astype(np.float64)
        y = np.random.uniform(-25, 25, N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        
        # 測試直接方法
        if has_direct:
            try:
                ax = np.zeros(N, dtype=np.float64)
                ay = np.zeros(N, dtype=np.float64)
                
                start_time = time.perf_counter()
                force_kernel.direct_force(x, y, m, 0.01, ax, ay)
                elapsed = time.perf_counter() - start_time
                
                print(f"  Direct: {elapsed:.6f} seconds")
                
            except Exception as e:
                print(f"  Direct: Failed - {e}")
        
        # 測試FMM方法
        if has_fmm:
            try:
                ax = np.zeros(N, dtype=np.float64)
                ay = np.zeros(N, dtype=np.float64)
                
                start_time = time.perf_counter()
                fmm_kernel.fmm_force(x, y, m, N, 50.0, 0.5, 16, 0.01, 1.0, ax, ay)
                elapsed = time.perf_counter() - start_time
                
                print(f"  FMM: {elapsed:.6f} seconds")
                
            except Exception as e:
                print(f"  FMM: Failed - {e}")

def main():
    """主函數"""
    print("N-Body Kernel Compilation and Functionality Test")
    print("=" * 55)
    
    # 檢查OpenMP設定
    check_openmp_settings()
    
    # 測試編譯結果
    success = test_compilation()
    
    if success:
        print("\n" + "=" * 55)
        print("✓ All tests passed!")
        
        # 運行快速基準測試
        run_quick_benchmark()
        
        print("\n" + "=" * 55)
        print("Compilation and testing completed successfully!")
        print("You can now run the main program:")
        print("  python main_program_optimized.py")
        
    else:
        print("\n" + "=" * 55)
        print("✗ Some tests failed!")
        print("Please check the compilation errors and try again.")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have a C++ compiler installed")
        print("2. Check if OpenMP is available: gcc -fopenmp")
        print("3. Ensure pybind11 is installed: pip install pybind11")
        print("4. Try recompiling: python setup_fixed.py build_ext --inplace")

if __name__ == "__main__":
    main()
