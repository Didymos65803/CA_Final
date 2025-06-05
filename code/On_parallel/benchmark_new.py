#!/usr/bin/env python3
"""
benchmark_new.py

修正版：先為 P=1 建立 base_times 字典，之後各 P 下才使用對應 N 的基準時間計算 Speed-up。
"""

import os
import numpy as np
import time
import fmm_omp

def single_run(N, eps2, domain, theta):
    """
    對 N 顆粒子執行一次 FMM 計算，回傳 wall-clock 耗時 (秒)。
    """
    # 隨機產生 N 顆粒子的 x, y 座標、unit mass
    x = np.random.rand(N).astype(np.float64)
    y = np.random.rand(N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)

    # 輸出陣列
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)

    # Warm-up (避免第一次呼叫時可能的初始化 overhead)
    fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)

    # 正式計時
    t0 = time.time()
    fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
    t1 = time.time()

    return t1 - t0

if __name__ == "__main__":
    # 要測試的 N 值（可以自行增減、放大）
    sizes = [50000, 100000, 200000, 400000]

    # Domain bounding box: [xmin, xmax, ymin, ymax]
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)

    # Softening length squared, opening angle
    eps2  = 1e-6
    theta = 0.6

    # 印出表頭
    print(f"{'Threads':>8}   {'N':>10}   {'Time (s)':>10}   {'Speed-up':>10}")
    print("-" * 46)

    # 先初始化 base_times 字典，專門儲存 P=1 時各 N 下的時間
    base_times = {}

    for P in [1, 2, 4, 8, 16]:
        # 設定 OpenMP 執行緒數
        os.environ["OMP_NUM_THREADS"] = str(P)

        for N in sizes:
            # 執行一次 FMM 計算，取得耗時
            t_elapsed = single_run(N, eps2, domain, theta)

            if P == 1:
                # P=1 時，把該 N 的耗時記錄到 base_times
                base_times[N] = t_elapsed
                speedup = 1.0
            else:
                # 其它 P 值，去對應的 base_times[N] 算出 speedup
                speedup = base_times[N] / t_elapsed

            print(f"{P:8d}   {N:10d}   {t_elapsed:10.4f}   {speedup:10.2f}")

