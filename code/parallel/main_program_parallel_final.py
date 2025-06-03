#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_program_parallel_final.py
==============================
Interactive 2-D N-body playground with optimized high-precision kernels

支持菜单（共 8 个选项）：
 1) Quick benchmark scaling              → test_scaling() 输出 performance_comparison.png + scaling_smallN.csv
 2) Save trajectory + energy plot        → 生成 trajectory.gif + energy_vs_time.png
 3) Live animation (real-time)           → 生成 live.gif
 4) Large-N scaling test                 → test_largeN_scaling() 输出 scaling_largeN.png + scaling_largeN.csv
 5) Energy conservation test             → test_energy_conservation() 输出 energy_conservation.png
 6) Parameter optimization               → optimize_parameters() 输出 parameter_optimization.png
 7) OpenMP thread benchmark              → thread_benchmark() 输出 openmp_thread_benchmark.png + openmp_thread_benchmark.csv
 8) System information                   → show_system_info() 打印 OpenMP/CPU/系统信息
  q) Quit
"""

import os
import sys
import math
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 导入 C++ 模块
# ----------------------------
try:
    import fmm_kernel
    HAS_FMM = True
except ImportError:
    HAS_FMM = False

try:
    import force_kernel
    direct_omp = force_kernel.direct_omp
    bh_omp     = force_kernel.bh_omp
    HAS_DIRECT_BH = True
except ImportError:
    HAS_DIRECT_BH = False
    direct_omp = None
    bh_omp     = None

# ----------------------------
# 导入本地测试模块 comprehensive_test.py
# ----------------------------
try:
    import comprehensive_test as ctest
except ImportError:
    ctest = None

# ----------------------------
# 全局物理常数与默认参数
# ----------------------------
G = 1.0
SOFT = 0.005               # 近场 softening
DOMAIN = 100.0             # 模拟域：[-50,50]^2
DT = 0.0005                # Leapfrog 时步
STAR_M = 100.0             # 固定中心恒星质量

OPTIMIZED_PARAMS = {
    'bh_theta': 0.3,       # BH 开放角 (accuracy vs speed)
    'fmm_theta': 0.2,      # FMM 开放角
    'bh_domain': DOMAIN,
    'fmm_domain': DOMAIN,
    'distribution_size': 50.0,
}

# 默认使用哪种求力方法 (DIRECT / BH / FMM)
USE_SOLVER = "DIRECT"


# ----------------------------
# Leapfrog 积分步骤
# ----------------------------
def leapfrog_step(bodies, ax, ay, include_central=False):
    """
    bodies: numpy ndarray (N,5) → [x, y, vx, vy, m]
    ax, ay: list 或 numpy 数组 (N,) → 当前加速度
    include_central: True 时第 0 号当作固定恒星
    """
    N = bodies.shape[0]

    # (1) 半步速度更新 (kick)
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 2] += 0.5 * DT * ax[i]
        bodies[i, 3] += 0.5 * DT * ay[i]

    # (2) 全步位置更新 (drift)
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 0] += DT * bodies[i, 2]
        bodies[i, 1] += DT * bodies[i, 3]

    # (3) 重新计算加速度
    x_list = bodies[:, 0].tolist()
    y_list = bodies[:, 1].tolist()
    m_list = bodies[:, 4].tolist()

    if USE_SOLVER.upper() == 'DIRECT':
        if not HAS_DIRECT_BH:
            raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
        ax_new, ay_new = direct_omp(x_list, y_list, m_list, G=G, soft=SOFT)

    elif USE_SOLVER.upper() == 'BH':
        if not HAS_DIRECT_BH:
            raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
        theta = OPTIMIZED_PARAMS['bh_theta']
        ax_new, ay_new = bh_omp(
            x_list, y_list, m_list,
            OPTIMIZED_PARAMS['bh_domain'],
            theta, G, SOFT
        )

    elif USE_SOLVER.upper() == 'FMM':
        if not HAS_FMM:
            raise RuntimeError("FMM 模块 fmm_kernel 未加载！")
        theta = OPTIMIZED_PARAMS['fmm_theta']
        ax_new, ay_new = fmm_kernel.fmm_omp(
            x_list, y_list, m_list,
            OPTIMIZED_PARAMS['fmm_domain'],
            theta, G, SOFT
        )

    else:
        raise ValueError(f"Unknown solver: {USE_SOLVER}")

    # (4) 半步速度更新 (kick)
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 2] += 0.5 * DT * ax_new[i]
        bodies[i, 3] += 0.5 * DT * ay_new[i]

    return ax_new, ay_new


# ----------------------------
# Total Energy 计算 (动能 + 位能)
# include_central=True 时第 0 号当作固定恒星
# ----------------------------
def total_energy(bodies, include_central=False):
    N = bodies.shape[0]
    KE = 0.0
    for i in range(N):
        if include_central and i == 0:
            continue
        KE += 0.5 * bodies[i, 4] * (bodies[i, 2]**2 + bodies[i, 3]**2)

    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if include_central and (i == 0 or j == 0):
                if i == 0:
                    dx = bodies[j, 0] - bodies[i, 0]
                    dy = bodies[j, 1] - bodies[i, 1]
                    dist = math.hypot(dx, dy) + SOFT
                    PE -= G * bodies[i, 4] * bodies[j, 4] / dist
                else:  # j == 0
                    dx = bodies[i, 0] - bodies[j, 0]
                    dy = bodies[i, 1] - bodies[j, 1]
                    dist = math.hypot(dx, dy) + SOFT
                    PE -= G * bodies[i, 4] * bodies[j, 4] / dist
                continue
            dx = bodies[i, 0] - bodies[j, 0]
            dy = bodies[i, 1] - bodies[j, 1]
            dist = math.hypot(dx, dy) + SOFT
            PE -= G * bodies[i, 4] * bodies[j, 4] / dist

    return KE + PE


def generate_disk(n, radius=OPTIMIZED_PARAMS['distribution_size'], include_central=False):
    """
    随机生成 n 颗粒子（圆盘分布），质心在 (0,0)，
    如果 include_central=True，则第 0 号为质量 STAR_M 的固定恒星。
    返回 bodies: ndarray (N_total, 5) → [x, y, vx, vy, m]
    """
    if include_central:
        N_total = n + 1
    else:
        N_total = n

    bodies = np.zeros((N_total, 5), dtype=np.float64)
    if include_central:
        bodies[0] = [0.0, 0.0, 0.0, 0.0, STAR_M]

    for i in range(1 if include_central else 0, N_total):
        r = random.random()**0.5 * radius
        theta = random.random() * 2.0 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        bodies[i, 0] = x
        bodies[i, 1] = y
        bodies[i, 4] = 1.0
        bodies[i, 2] = 0.0
        bodies[i, 3] = 0.0

    return bodies


# ----------------------------
# 交互式菜单主程序
# ----------------------------
def main_menu():
    global USE_SOLVER

    menu_text = """
=== 2D N-body Playground (Parallel, High-Precision) ===
Select option:
 1) Quick benchmark scaling
 2) Save trajectory + energy plot
 3) Live animation (real-time)
 4) Large-N scaling test
 5) Energy conservation test
 6) Parameter optimization
 7) OpenMP thread benchmark
 8) System information
  q) Quit
==============================================
"""
    print(menu_text)

    while True:
        try:
            choice = input("Enter choice: ").strip().lower()
            if choice in ["1", "benchmark"]:
                # Option 1: Quick benchmark scaling
                if ctest is not None:
                    ctest.test_scaling()
                else:
                    print("Error: comprehensive_test.py 未找到，无法执行 Quick benchmark。\n")

            elif choice in ["2", "save", "save trajectory"]:
                # Option 2: Save trajectory + energy plot
                print("\nSave trajectory + energy plot")
                parser = argparse.ArgumentParser()
                parser.add_argument("--solver", type=str, default="direct",
                                    choices=["direct", "bh", "fmm"],
                                    help="Solver to use: direct / bh / fmm")
                parser.add_argument("--n", type=int, default=200,
                                    help="Number of particles (excluding central star)")
                parser.add_argument("--steps", type=int, default=2000,
                                    help="Number of integration steps")
                parser.add_argument("--threads", type=int, default=4,
                                    help="Number of OpenMP threads")
                parser.add_argument("--fixed-star", action="store_true",
                                    help="Include a fixed central star at index 0")
                args = parser.parse_args()

                USE_SOLVER = args.solver
                N = args.n
                STEPS = args.steps
                THREADS = args.threads
                fixed_star = args.fixed_star

                os.environ["OMP_NUM_THREADS"] = str(THREADS)
                bodies = generate_disk(N, OPTIMIZED_PARAMS['distribution_size'], include_central=fixed_star)
                total_N = bodies.shape[0]
                E0 = total_energy(bodies, include_central=fixed_star)

                # 初始加速度
                x0 = bodies[:, 0].tolist()
                y0 = bodies[:, 1].tolist()
                m0 = bodies[:, 4].tolist()
                if USE_SOLVER.upper() == 'DIRECT':
                    if not HAS_DIRECT_BH:
                        raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
                    ax0, ay0 = direct_omp(x0, y0, m0, G=G, soft=SOFT)
                elif USE_SOLVER.upper() == 'BH':
                    if not HAS_DIRECT_BH:
                        raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
                    theta = OPTIMIZED_PARAMS['bh_theta']
                    ax0, ay0 = bh_omp(x0, y0, m0, OPTIMIZED_PARAMS['bh_domain'], theta, G, SOFT)
                elif USE_SOLVER.upper() == 'FMM':
                    if not HAS_FMM:
                        raise RuntimeError("FMM 模块 fmm_kernel 未加载！")
                    theta = OPTIMIZED_PARAMS['fmm_theta']
                    ax0, ay0 = fmm_kernel.fmm_omp(x0, y0, m0, OPTIMIZED_PARAMS['fmm_domain'], theta, G, SOFT)
                else:
                    raise ValueError(f"Unknown solver: {USE_SOLVER}")

                ax = ax0
                ay = ay0

                xs = []
                ys = []
                E_list = []

                try:
                    for s in range(STEPS):
                        if s % 10 == 0:
                            xs.append(bodies[:, 0].copy())
                            ys.append(bodies[:, 1].copy())

                        ax, ay = leapfrog_step(bodies, ax, ay, include_central=fixed_star)

                        if s % 10 == 0:
                            E = total_energy(bodies, include_central=fixed_star)
                            rel_error = abs(E - E0) / (abs(E0) + 1e-16)
                            E_list.append((s * DT, E, rel_error))
                except KeyboardInterrupt:
                    print("\nIntegration interrupted by user.")
                except Exception as e:
                    print(f"\nError during integration: {e}\n")
                    return

                # (a) 保存轨迹动画
                if xs and ys:
                    print("Creating animation GIF...")
                    fig, axg = plt.subplots(figsize=(8, 8))
                    points = []
                    if fixed_star:
                        colors = ["red"] + ["blue"] * (total_N - 1)
                        sizes = [10] + [3] * (total_N - 1)
                    else:
                        colors = ["blue"] * total_N
                        sizes = [3] * total_N

                    for i in range(total_N):
                        pt, = axg.plot([], [], 'o', color=colors[i], markersize=sizes[i], alpha=0.8)
                        points.append(pt)

                    axg.set_xlim(-DOMAIN, DOMAIN)
                    axg.set_ylim(-DOMAIN, DOMAIN)
                    axg.set_title(f"Trajectory ({USE_SOLVER.upper()}, N={N}, threads={THREADS})")

                    def update(frame):
                        for i, pt in enumerate(points):
                            pt.set_data([xs[frame][i]], [ys[frame][i]])
                        return points

                    import matplotlib.animation as animation
                    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=50, blit=True)
                    gif_name = f"trajectory_{USE_SOLVER}_{N}_{THREADS}_{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"
                    ani.save(gif_name, fps=20, dpi=80)
                    plt.close(fig)
                    print(f"✓ Saved trajectory GIF: {gif_name}")

                # (b) 保存能量-时间曲线
                if E_list:
                    print("Creating energy vs time plot...")
                    times = [item[0] for item in E_list]
                    energies = [item[1] for item in E_list]

                    fig, axp = plt.subplots(figsize=(8, 4))
                    axp.plot(times, energies, label="Total Energy")
                    axp.axhline(E0, color="gray", linestyle="--", label="Initial Energy")
                    axp.set_xlabel("Time")
                    axp.set_ylabel("Energy")
                    axp.set_title(f"Energy vs Time ({USE_SOLVER.upper()}, N={N}, threads={THREADS})")
                    axp.legend()

                    energy_plot = f"energy_{USE_SOLVER}_{N}_{THREADS}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                    fig.tight_layout()
                    fig.savefig(energy_plot, dpi=150)
                    plt.close(fig)
                    print(f"✓ Saved energy plot: {energy_plot}\n")

            elif choice in ["3", "live", "live animation"]:
                # Option 3: Live animation (real-time)
                print("\nLive simulation animation")
                parser = argparse.ArgumentParser()
                parser.add_argument("--solver", type=str, default="fmm",
                                    choices=["direct", "bh", "fmm"],
                                    help="Solver to use: direct / bh / fmm")
                parser.add_argument("--n", type=int, default=100,
                                    help="Number of particles (excluding central star)")
                parser.add_argument("--steps", type=int, default=500,
                                    help="Number of integration steps")
                parser.add_argument("--threads", type=int, default=8,
                                    help="Number of OpenMP threads")
                parser.add_argument("--fixed-star", action="store_true",
                                    help="Include a fixed central star at index 0")
                args = parser.parse_args()

                USE_SOLVER = args.solver
                N = args.n
                STEPS = args.steps
                THREADS = args.threads
                fixed_star = args.fixed_star

                os.environ["OMP_NUM_THREADS"] = str(THREADS)
                bodies = generate_disk(N, OPTIMIZED_PARAMS['distribution_size'], include_central=fixed_star)
                total_N = bodies.shape[0]

                x0 = bodies[:, 0].tolist()
                y0 = bodies[:, 1].tolist()
                m0 = bodies[:, 4].tolist()
                if USE_SOLVER.upper() == 'DIRECT':
                    if not HAS_DIRECT_BH:
                        raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
                    ax0, ay0 = direct_omp(x0, y0, m0, G=G, soft=SOFT)
                elif USE_SOLVER.upper() == 'BH':
                    if not HAS_DIRECT_BH:
                        raise RuntimeError("Direct/BH 模块 force_kernel 未加载！")
                    theta = OPTIMIZED_PARAMS['bh_theta']
                    ax0, ay0 = bh_omp(x0, y0, m0, OPTIMIZED_PARAMS['bh_domain'], theta, G, SOFT)
                elif USE_SOLVER.upper() == 'FMM':
                    if not HAS_FMM:
                        raise RuntimeError("FMM 模块 fmm_kernel 未加载！")
                    theta = OPTIMIZED_PARAMS['fmm_theta']
                    ax0, ay0 = fmm_kernel.fmm_omp(x0, y0, m0, OPTIMIZED_PARAMS['fmm_domain'], theta, G, SOFT)
                else:
                    raise ValueError(f"Unknown solver: {USE_SOLVER}")

                ax = ax0
                ay = ay0

                print("Creating live simulation GIF...")
                fig, ax_live = plt.subplots(figsize=(8, 8))
                points = []
                if fixed_star:
                    colors = ["red"] + ["blue"] * (total_N - 1)
                    sizes = [10] + [3] * (total_N - 1)
                else:
                    colors = ["blue"] * total_N
                    sizes = [3] * total_N

                for i in range(total_N):
                    pt, = ax_live.plot([], [], 'o', color=colors[i], markersize=sizes[i], alpha=0.8)
                    points.append(pt)

                ax_live.set_xlim(-DOMAIN, DOMAIN)
                ax_live.set_ylim(-DOMAIN, DOMAIN)
                ax_live.set_title(f"Live Simulation ({USE_SOLVER.upper()}, N={N}, threads={THREADS})")

                def update(frame):
                    nonlocal ax, ay
                    ax, ay = leapfrog_step(bodies, ax, ay, include_central=fixed_star)
                    for idx, pt in enumerate(points):
                        pt.set_data([bodies[idx, 0]], [bodies[idx, 1]])
                    return points

                import matplotlib.animation as animation
                ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=50, blit=True)
                gif_name = f"live_{USE_SOLVER}_{N}_{THREADS}_{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"
                ani.save(gif_name, fps=20, dpi=80)
                plt.close(fig)
                print(f"✓ Saved live simulation GIF: {gif_name}\n")

            elif choice in ["4", "large-n", "large-n scaling"]:
                # Option 4: Large-N scaling test
                print("\nLarge-N Scaling Test")
                if ctest is not None:
                    ctest.test_largeN_scaling()
                else:
                    print("Error: comprehensive_test.py 未找到，无法执行 Large-N scaling。\n")

            elif choice in ["5", "energy", "energy conservation"]:
                # Option 5: Energy conservation test
                print("\nEnergy conservation test")
                if ctest is not None:
                    ctest.test_energy_conservation()
                else:
                    print("Error: comprehensive_test.py 未找到，无法执行 Energy conservation test。\n")

            elif choice in ["6", "optimize", "parameter optimization"]:
                # Option 6: Parameter optimization
                print("\nParameter optimization")
                if ctest is not None:
                    ctest.optimize_parameters()
                else:
                    print("Error: comprehensive_test.py 未找到，无法执行 Parameter optimization。\n")

            elif choice in ["7", "thread", "openmp thread benchmark"]:
                # Option 7: OpenMP thread benchmark
                print("\nOpenMP thread benchmark")
                if ctest is not None:
                    ctest.thread_benchmark()
                else:
                    print("Error: comprehensive_test.py 未找到，无法执行 Thread benchmark。\n")

            elif choice in ["8", "system", "system information"]:
                # Option 8: System information
                print("\nSystem information")
                if ctest is not None:
                    ctest.show_system_info()
                else:
                    print("Error: comprehensive_test.py 未找到，无法显示 System information。\n")

            elif choice in ["q", "quit", "exit"]:
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.\n")

        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user. Returning to menu...\n")
        except Exception as e:
            print(f"\nError: {e}\nPlease try again or choose a different option.\n")


if __name__ == "__main__":
    main_menu()

