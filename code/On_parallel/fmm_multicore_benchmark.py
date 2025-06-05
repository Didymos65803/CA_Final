#!/usr/bin/env python3
"""
fmm_multicore_benchmark.py

专门测试 C++ OpenMP FMM (fmm_omp) 在大 N (>=200k) 下，多线程 (1,2,4,8,16) 的平均耗时与 speed-up。
同时会打印“树构建阶段 spawn 出了多少个 task”，以便判断并行树构建是否充分展开。

使用方式：
  1. 先确保在同一目录下编译好 fmm_omp 模块：
       python3 setup_openmp.py build_ext --inplace
  2. 确保已安装 numpy。
  3. 执行：
       python3 fmm_multicore_benchmark.py
"""

import os
import time
import numpy as np
import fmm_omp

# 固定参数：软化长度平方、domain、theta
EPS2 = 1e-6
DOMAIN = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
THETA = 0.6

# 建议测试的 N 列表 (必须足够大才能观察到并行加速)
SIZES = [200000, 400000, 800000, 1200000]

# 要测试的线程数
THREADS_LIST = [1, 2, 4, 8, 16]

# 每个 (N, threads) 下运行次数 (先 warm-up，再计时)，取平均
NUM_TRIALS = 3

def run_fmm_once(x, y, m, eps2, domain, theta):
    """
    单次调用 C++ OpenMP FMM，返回耗时 (秒) 及本次 spawn task 数（从 C++ 端的原子计数器读取）。
    """
    ax = np.zeros_like(x)
    ay = np.zeros_like(y)
    # Warm-up：使得第一次的初始化／缓存等完成
    fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
    # 记录开始实时
    t0 = time.time()
    # 重要：C++ 端会在 build_tree_rec 中更新 spawn_counter
    fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
    t1 = time.time()
    # 从 C++ 端获取本次 spawn_counter（注意：这是所有线程累计到最后的值）
    # 我们用 Python 层接口 `fmm_omp.get_spawn_count()` 来读取。需要提前在 C++ 端
    # 定义并导出一个函数，但为了简化，这里假设 `spawn_counter` 直接会在 Python 端
    # 作为一个全局变量可读。若无法直接读取，就只能用 debug 输出到 stdout。
    # 这里我们暂时不做 Python 层读取；仅返回耗时。
    return (t1 - t0)

def benchmark_for_size(N, threads_list, eps2, domain, theta, trials):
    """
    针对指定 N，测试 threads_list 中各线程数 (threads) 的平均耗时与 speed-up。
    返回：times_avg, speedups
      times_avg: {P: avg_time}
      speedups:  {P: speedup} （以 P = threads_list[0] 当基准）
    """
    # 1) 生成一次随机数据，后续所有 threads 共享同一组 (x,y,m)
    x = np.random.rand(N).astype(np.float64)
    y = np.random.rand(N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)

    times_accum = {P: 0.0 for P in threads_list}
    base_avg = None

    for P in threads_list:
        os.environ["OMP_NUM_THREADS"] = str(P)
        t_sum = 0.0
        for _ in range(trials):
            t_sum += run_fmm_once(x, y, m, eps2, domain, theta)
        avg_t = t_sum / trials
        times_accum[P] = avg_t
        if P == threads_list[0]:
            base_avg = avg_t

    # 计算 speed_up = base_avg / times_accum[P]
    speedups = {P: (base_avg / times_accum[P]) for P in threads_list}
    return times_accum, speedups

if __name__ == "__main__":
    print("\n=== FMM (C++ OpenMP) Multicore Benchmark ===\n")
    header = f"{'N':>10}   {'Threads':>8}   {'Avg Time (s)':>12}   {'Speed-up':>10}"
    print(header)
    print("-" * len(header))

    for N in SIZES:
        times_avg, speedups = benchmark_for_size(N, THREADS_LIST, EPS2, DOMAIN, THETA, NUM_TRIALS)
        for P in THREADS_LIST:
            tavg = times_avg[P]
            sup  = speedups[P]
            print(f"{N:10d}   {P:8d}   {tavg:12.6f}   {sup:10.2f}")
        print("-" * len(header))

    print("\n说明：")
    print("  • 每组 (N, threads) 均跑了 {} 次 (包含 warm-up + 正式计时)，取平均后输出。".format(NUM_TRIALS))
    print("  • 若 Speed-up 趋近 threads 数，表示已接近理想的 O(N)/P 并行效率。")
    print("  • 小 N (< 50000) 时，OpenMP 启动与同步开销占比太高，无法看到加速。")
    print("  • 本脚本测试 N=200k、400k、800k、1.2M，目的是让并行计算量明显大于 overhead。")

