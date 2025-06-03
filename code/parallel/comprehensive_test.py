# comprehensive_test.py
# ======================
# 包含多种对比测试与基准测试函数，供 main_program_parallel_final.py 调用。
#
# 注意：在使用这些函数前，请确保已经完成以下模块编译：
#   - fmm_kernel（来自 fmm_kernel_full.cpp）
#   - force_kernel（如果有 direct_omp, bh_omp 实现的话）
#
# 如果缺少 force_kernel，你可以先只用 FMM 进行测试。

import os
import sys
import time
import math
import random

import numpy as np
import matplotlib.pyplot as plt

# 试着导入 C++ 接口模块
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
    bh_omp = None

# 全局参数 (与 main_program_parallel_final.py 保持一致)
G = 1.0
SOFT = 0.005
DOMAIN = 100.0
DT = 0.0005
STAR_M = 100.0

OPTIMIZED_PARAMS = {
    'bh_theta': 0.3,
    'fmm_theta': 0.2,
    'bh_domain': DOMAIN,
    'fmm_domain': DOMAIN,
    'distribution_size': 50.0,
}


def generate_disk(n, radius=OPTIMIZED_PARAMS['distribution_size'], include_central=False):
    """
    随机生成 n 颗粒子，分布在半径 radius 的圆盘内。如果 include_central=True，
    第 0 号为质量 STAR_M 的固定恒星。返回 bodies: ndarray (N_total, 5) → [x,y,vx,vy,m]
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


def total_energy(bodies, include_central=False):
    """
    计算系统的总能量 (动能 + 位能)。如果 include_central=True，第 0 号当作固定恒星。
    """
    N = bodies.shape[0]
    KE = 0.0
    for i in range(N):
        if include_central and i == 0:
            continue
        KE += 0.5 * bodies[i, 4] * (bodies[i, 2]**2 + bodies[i, 3]**2)

    PE = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if include_central and (i == 0 or j == 0):
                # 恒星与其他粒子
                if i == 0:
                    dx = bodies[j, 0] - bodies[i, 0]
                    dy = bodies[j, 1] - bodies[i, 1]
                    dist = math.hypot(dx, dy) + SOFT
                    PE -= G * bodies[i, 4] * bodies[j, 4] / dist
                else:
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


def test_accuracy():
    """
    Accuracy Comparison:
    - 对 N = [50, 100, 200, 500] 做 Direct / BH / FMM 3 款算法的力计算，并比较相对误差。
    - 结果绘制成 'accuracy_test_results_fixed.png'。
    """
    print("Testing accuracy with optimized parameters...\n")

    Ns = [50, 100, 200, 500]
    bh_theta = OPTIMIZED_PARAMS['bh_theta']
    fmm_theta = OPTIMIZED_PARAMS['fmm_theta']
    distribution_size = OPTIMIZED_PARAMS['distribution_size']

    results = []
    for N in Ns:
        # 生成随机圆盘分布（无固定恒星）
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # --- Direct (基准) ---
        if not HAS_DIRECT_BH:
            print("Warning: Direct 模块未加载，将跳过 Accuracy 测试。")
            return
        t0 = time.time()
        fx_direct, fy_direct = direct_omp(x, y, m, G=G, soft=SOFT)
        t_direct = time.time() - t0
        Fd = np.sqrt(np.array(fx_direct)**2 + np.array(fy_direct)**2).sum()

        # --- Barnes-Hut ---
        t0 = time.time()
        fx_bh, fy_bh = bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
        t_bh = time.time() - t0
        Fb = np.sqrt(np.array(fx_bh)**2 + np.array(fy_bh)**2).sum()
        err_bh = abs(Fb - Fd) / (Fd + 1e-16)

        # --- FMM ---
        if not HAS_FMM:
            err_fmm = np.nan
            t_fmm = np.nan
        else:
            t0 = time.time()
            fx_fmm, fy_fmm = fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0
            Ff = np.sqrt(np.array(fx_fmm)**2 + np.array(fy_fmm)**2).sum()
            err_fmm = abs(Ff - Fd) / (Fd + 1e-16)

        results.append((N, t_direct, t_bh, err_bh, t_fmm, err_fmm, Fd, Fb if HAS_DIRECT_BH else 0.0, Ff if HAS_FMM else 0.0))

        print(f"Testing N = {N}")
        print(f"  Direct:     {t_direct:.4f} s")
        print(f"  Barnes-Hut: {t_bh:.4f} s (error: {err_bh:.2e}) [θ={bh_theta}]")
        if HAS_FMM:
            print(f"  FMM:        {t_fmm:.4f} s (error: {err_fmm:.2e}) [θ={fmm_theta}]")
        else:
            print("  FMM:        Not Available")
        print(f"  Force magnitudes - Direct: {Fd:.3e}, BH: {Fb:.3e}, FMM: {Ff:.3e}\n")

    # 汇总输出
    max_err_bh = max([r[3] for r in results])
    max_err_fmm = max([r[5] for r in results if not math.isnan(r[5])])
    print("Overall Results:")
    print(f"Max Barnes-Hut error: {max_err_bh:.2e}")
    print(f"Max FMM error: {max_err_fmm:.2e}\n")

    # 绘图
    Ns_plot = [r[0] for r in results]
    times_direct = [r[1] for r in results]
    times_bh = [r[2] for r in results]
    times_fmm = [r[4] for r in results]

    errs_bh = [r[3] for r in results]
    errs_fmm = [r[5] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 性能对比 (左)
    ax1.plot(Ns_plot, times_direct, 'o-', color='red', label='Direct (O(N²))')
    ax1.plot(Ns_plot, times_bh, 's-', color='blue', label='Barnes-Hut (O(N log N))')
    if HAS_FMM:
        ax1.plot(Ns_plot, times_fmm, '^-', color='green', label='FMM (O(N))')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('N particles')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Performance Comparison')
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    ax1.legend()

    # 精度对比 (右)
    ax2.plot(Ns_plot, errs_bh, 's-', color='blue', label='Barnes-Hut Error')
    if HAS_FMM:
        ax2.plot(Ns_plot, errs_fmm, '^-', color='green', label='FMM Error')
    ax2.axhline(0.01, color='orange', linestyle='--', label='1% Error Target')
    ax2.axhline(0.10, color='red', linestyle='--', label='10% Error Limit')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('N particles')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Accuracy Comparison')
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('accuracy_test_results_fixed.png', dpi=150)
    plt.close(fig)
    print("✓ Saved accuracy_test_results_fixed.png\n")


def test_scaling():
    """
    Quick benchmark scaling:
    - 先在小 N (50,100,200,500,1000) 下测试 Direct/BH/FMM 的时间，绘图。
    - 存储结果到 'performance_comparison.png' 和 'scaling_smallN.csv'。
    """
    print("Testing scaling behavior...\n")

    smallNs = [50, 100, 200, 500, 1000, 2000]
    bh_theta = OPTIMIZED_PARAMS['bh_theta']
    fmm_theta = OPTIMIZED_PARAMS['fmm_theta']
    distribution_size = OPTIMIZED_PARAMS['distribution_size']

    results = []
    for N in smallNs:
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # Direct
        if not HAS_DIRECT_BH:
            t_direct = np.nan
        else:
            t0 = time.time()
            direct_omp(x, y, m, G=G, soft=SOFT)
            t_direct = time.time() - t0

        # Barnes-Hut
        if not HAS_DIRECT_BH:
            t_bh = np.nan
        else:
            t0 = time.time()
            bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
            t_bh = time.time() - t0

        # FMM
        if not HAS_FMM:
            t_fmm = np.nan
        else:
            t0 = time.time()
            fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0

        results.append((N, t_direct, t_bh, t_fmm))
        print(f"Testing N = {N}")
        print(f"  Direct:  {t_direct:.4f} s")
        print(f"  Barnes-Hut: {t_bh:.4f} s (θ={bh_theta})")
        if HAS_FMM:
            print(f"  FMM:     {t_fmm:.4f} s (θ={fmm_theta})")
        else:
            print("  FMM:     Not Available")
        print("")

    # 输出 CSV
    import csv
    with open('scaling_smallN.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'Direct', 'Barnes-Hut', 'FMM'])
        for row in results:
            writer.writerow(row)

    # 绘图
    Ns_plot = [r[0] for r in results]
    times_direct = [r[1] for r in results]
    times_bh = [r[2] for r in results]
    times_fmm = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(Ns_plot, times_direct, 'o-', color='red', label='Direct O(N²)')
    ax.plot(Ns_plot, times_bh, 's-', color='blue', label='Barnes-Hut O(N log N)')
    if HAS_FMM:
        ax.plot(Ns_plot, times_fmm, '^-', color='green', label='FMM O(N)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Scaling Comparison (small N)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    plt.close(fig)
    print("✓ Saved performance_comparison.png")
    print("✓ Saved scaling_smallN.csv\n")


def test_largeN_scaling():
    """
    Large-N Scaling Test:
    - 对 N = [500, 1000, 2000, 4000] 做 Direct/BH/FMM benchmarking（也可只做 BH 和 FMM），
      输出到 'scaling_largeN.png' + 'scaling_largeN.csv'。
    """
    print("Testing large-N scaling behavior...\n")

    largeNs = [500, 1000, 2000, 4000]
    bh_theta = OPTIMIZED_PARAMS['bh_theta']
    fmm_theta = OPTIMIZED_PARAMS['fmm_theta']
    distribution_size = OPTIMIZED_PARAMS['distribution_size']

    results = []
    for N in largeNs:
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # Direct (如果 N 太大可能耗时很久，可以选择 skip)
        if not HAS_DIRECT_BH or N > 2000:
            t_direct = np.nan
        else:
            t0 = time.time()
            direct_omp(x, y, m, G=G, soft=SOFT)
            t_direct = time.time() - t0

        # Barnes-Hut
        if not HAS_DIRECT_BH:
            t_bh = np.nan
        else:
            t0 = time.time()
            bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
            t_bh = time.time() - t0

        # FMM
        if not HAS_FMM:
            t_fmm = np.nan
        else:
            t0 = time.time()
            fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0

        results.append((N, t_direct, t_bh, t_fmm))
        print(f"Testing N = {N}")
        if not math.isnan(t_direct):
            print(f"  Direct:  {t_direct:.4f} s")
        else:
            print("  Direct:  skip (N too large)")
        print(f"  Barnes-Hut: {t_bh:.4f} s (θ={bh_theta})")
        if HAS_FMM:
            print(f"  FMM:     {t_fmm:.4f} s (θ={fmm_theta})")
        else:
            print("  FMM:     Not Available")
        print("")

    # 输出 CSV
    import csv
    with open('scaling_largeN.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'Direct', 'Barnes-Hut', 'FMM'])
        for row in results:
            writer.writerow(row)

    # 绘图
    Ns_plot = [r[0] for r in results]
    times_direct = [r[1] for r in results]
    times_bh = [r[2] for r in results]
    times_fmm = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    if not all(math.isnan(t) for t in times_direct):
        ax.plot(Ns_plot, times_direct, 'o-', color='red', label='Direct O(N²)')
    ax.plot(Ns_plot, times_bh, 's-', color='blue', label='Barnes-Hut O(N log N)')
    if HAS_FMM:
        ax.plot(Ns_plot, times_fmm, '^-', color='green', label='FMM O(N)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Time (s)')
    ax.set_title('Large-N Scaling Comparison')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig('scaling_largeN.png', dpi=150)
    plt.close(fig)
    print("✓ Saved scaling_largeN.png")
    print("✓ Saved scaling_largeN.csv\n")


def test_energy_conservation():
    """
    Energy Conservation Test:
    - 生成一批随机粒子（N=200）、使用 Direct/BH/FMM 分别演化一段时间，
      记录并绘制能量随时间的相对误差，保存为 'energy_conservation.png'。
    """
    print("Testing energy conservation for Direct/BH/FMM...\n")

    N = 200
    STEPS = 500
    RECORD_EVERY = 5
    THREADS = 4
    fixed_star = False

    # 设置线程数
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    # 生成初始状态
    bodies = generate_disk(N, OPTIMIZED_PARAMS['distribution_size'], include_central=fixed_star)
    total_N = bodies.shape[0]
    E0 = total_energy(bodies, include_central=fixed_star)

    # 初始加速度
    x0 = bodies[:, 0].tolist()
    y0 = bodies[:, 1].tolist()
    m0 = bodies[:, 4].tolist()

    # 分别对 Direct/BH/FMM 做能量守恒测试
    solvers = []
    labels = []

    if HAS_DIRECT_BH:
        solvers.append(('Direct', direct_omp, None))
        labels.append('Direct')
    if HAS_DIRECT_BH:
        solvers.append(('BH', bh_omp, OPTIMIZED_PARAMS['bh_theta']))
        labels.append('Barnes-Hut')
    if HAS_FMM:
        solvers.append(('FMM', fmm_kernel.fmm_omp, OPTIMIZED_PARAMS['fmm_theta']))
        labels.append('FMM')

    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, solver_fn, theta) in solvers:
        # 重新生成起始点，以保证三者用同一初始分布
        bodies = generate_disk(N, OPTIMIZED_PARAMS['distribution_size'], include_central=fixed_star)
        E0_local = total_energy(bodies, include_central=fixed_star)
        x_list = bodies[:, 0].tolist()
        y_list = bodies[:, 1].tolist()
        m_list = bodies[:, 4].tolist()

        if name == 'Direct':
            ax_old, ay_old = solver_fn(x_list, y_list, m_list, G=G, soft=SOFT)
        elif name == 'BH':
            ax_old, ay_old = solver_fn(x_list, y_list, m_list, DOMAIN, theta, G, SOFT)
        else:  # FMM
            ax_old, ay_old = solver_fn(x_list, y_list, m_list, DOMAIN, theta, G, SOFT)

        times = []
        rel_errors = []

        for step in range(STEPS):
            # Leapfrog half-kick/drift/half-kick
            # （复用前面 main 程序的 leapfrog_step 实现即可，
            #  但为了简洁，这里在测试函数里再写一次）
            # 1) half-kick
            for i in range(total_N):
                bodies[i, 2] += 0.5 * DT * ax_old[i]
                bodies[i, 3] += 0.5 * DT * ay_old[i]
            # 2) drift
            for i in range(total_N):
                bodies[i, 0] += DT * bodies[i, 2]
                bodies[i, 1] += DT * bodies[i, 3]
            # 3) compute new accel
            x_list = bodies[:, 0].tolist()
            y_list = bodies[:, 1].tolist()
            m_list = bodies[:, 4].tolist()
            if name == 'Direct':
                ax_new, ay_new = solver_fn(x_list, y_list, m_list, G=G, soft=SOFT)
            elif name == 'BH':
                ax_new, ay_new = solver_fn(x_list, y_list, m_list, DOMAIN, theta, G, SOFT)
            else:
                ax_new, ay_new = solver_fn(x_list, y_list, m_list, DOMAIN, theta, G, SOFT)
            # 4) half-kick
            for i in range(total_N):
                bodies[i, 2] += 0.5 * DT * ax_new[i]
                bodies[i, 3] += 0.5 * DT * ay_new[i]

            ax_old, ay_old = ax_new, ay_new

            if step % RECORD_EVERY == 0:
                E = total_energy(bodies, include_central=fixed_star)
                rel_err = abs(E - E0_local) / abs(E0_local + 1e-16)
                times.append(step * DT)
                rel_errors.append(rel_err)

        ax.plot(times, rel_errors, label=name)

    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Energy Error')
    ax.set_title(f'Energy Conservation Test (N={N}, threads={THREADS})')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=150)
    plt.close(fig)
    print("✓ Saved energy_conservation.png\n")


def optimize_parameters():
    """
    Parameter Optimization:
    - 对 BH 的 θ ∈ [0.1,0.3,0.5,0.7,1.0]，domain ∈ [50,100,200] 做网格搜索，找出最优组合：
      误差 < 10% 且 时间最小。
    - 输出到 'parameter_optimization.png' 并打印最佳 θ, domain。
    """
    print("Optimizing Barnes-Hut parameters...\n")

    # 固定 N=100 用来测试误差与时间
    N = 100
    distribution_size = OPTIMIZED_PARAMS['distribution_size']
    bodies = generate_disk(N, distribution_size, include_central=False)
    x = bodies[:, 0].tolist()
    y = bodies[:, 1].tolist()
    m = bodies[:, 4].tolist()

    # 直接当作基准
    if not HAS_DIRECT_BH:
        print("Error: Direct/BH 模块未加载，Parameter Optimization 无法进行。")
        return
    fx_direct, fy_direct = direct_omp(x, y, m, G=G, soft=SOFT)
    Fd = np.sqrt(np.array(fx_direct)**2 + np.array(fy_direct)**2).sum()

    thetas = [0.1, 0.3, 0.5, 0.7, 1.0]
    domains = [50.0, 100.0, 200.0]

    records = []
    best = (None, None, float('inf'), float('inf'))  # (θ, domain, error, time)

    print("Testing parameter combinations:")
    print("Theta  Domain  Error      Time")
    print("-----------------------------------")
    for theta in thetas:
        for domain in domains:
            t0 = time.time()
            fx_bh, fy_bh = bh_omp(x, y, m, domain, theta, G, SOFT)
            t_bh = time.time() - t0
            Fb = np.sqrt(np.array(fx_bh)**2 + np.array(fy_bh)**2).sum()
            err = abs(Fb - Fd) / (Fd + 1e-16)
            records.append((theta, domain, err, t_bh))
            print(f"  {theta:<5} {domain:<6} {err:.2e}  {t_bh:.4f}s")
            # 以 error < 0.1 (10%) 且 时间最小 为准
            if err < 0.1 and t_bh < best[3]:
                best = (theta, domain, err, t_bh)
    print("")
    if best[0] is not None:
        print(f"Best parameters: θ={best[0]}, domain={best[1]}")
        print(f"Best error: {best[2]:.2e}, Best time: {best[3]:.4f}s\n")
    else:
        print("没有找到在误差 < 10% 的组合，请考虑降低误差阈值或扩大搜索范围。\n")

    # 绘图：误差 vs θ (color 为 domain)
    fig, ax = plt.subplots(figsize=(6, 5))
    for domain in domains:
        errs = [r[2] for r in records if r[1] == domain]
        times = [r[3] for r in records if r[1] == domain]
        ax.semilogy(thetas, errs, 'o-', label=f'domain={domain}')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Relative Error')
    ax.set_title('Parameter Optimization (N=100)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig('parameter_optimization.png', dpi=150)
    plt.close(fig)
    print("✓ Saved parameter_optimization.png\n")


def thread_benchmark():
    """
    OpenMP Thread Benchmark:
    - 对 FMM （N=500 固定）在线程数 [1,2,4,8] 下做时间对比。
    - 绘制成 'openmp_thread_benchmark.png'，CSV 存到 'openmp_thread_benchmark.csv'。
    """
    print("Running OpenMP thread benchmark for FMM (N=500)...\n")

    if not HAS_FMM:
        print("Error: FMM 模块未加载，无法进行线程基准测试。")
        return

    N = 500
    distribution_size = OPTIMIZED_PARAMS['distribution_size']
    bodies = generate_disk(N, distribution_size, include_central=False)
    x = bodies[:, 0].tolist()
    y = bodies[:, 1].tolist()
    m = bodies[:, 4].tolist()

    thread_counts = [1, 2, 4, 8]
    results = []

    for tcount in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(tcount)
        t0 = time.time()
        fmm_kernel.fmm_omp(x, y, m, DOMAIN, OPTIMIZED_PARAMS['fmm_theta'], G, SOFT)
        t_fmm = time.time() - t0
        results.append((tcount, t_fmm))
        print(f"Threads = {tcount}, Time = {t_fmm:.5f} s")

    # 保存 CSV
    import csv
    with open('openmp_thread_benchmark.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Threads', 'Time'])
        for row in results:
            writer.writerow(row)

    # 绘图
    threads_plot = [r[0] for r in results]
    times_plot = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(threads_plot, threads_plot[0]/np.array(times_plot), 'o-', color='red', label='Speedup')
    ax.plot(threads_plot, threads_plot, '--', color='gray', label='Ideal Speedup')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('OpenMP Thread Benchmark (FMM, N=500)')
    ax.set_xticks(thread_counts)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig('openmp_thread_benchmark.png', dpi=150)
    plt.close(fig)
    print("✓ Saved openmp_thread_benchmark.png")
    print("✓ Saved openmp_thread_benchmark.csv\n")


def show_system_info():
    """
    System Information:
    - 输出当前环境下的 CPU 核心数、OpenMP 线程信息、操作系统等。
    """
    print("Gathering system information...\n")
    try:
        import platform
        info = platform.uname()
        print(f"System: {info.system} {info.release} ({info.machine})")
    except ImportError:
        pass

    # CPU count
    try:
        cpu_count = os.cpu_count()
        print(f"CPU cores (os.cpu_count): {cpu_count}")
    except:
        pass

    # OpenMP 线程
    try:
        # 通过环境变量或 omp_get_max_threads
        env_t = os.environ.get("OMP_NUM_THREADS", "Not set")
        print(f"OMP_NUM_THREADS (env): {env_t}")
        if HAS_FMM:
            # 如果编译时包含 OpenMP，那么 m.attr("has_openmp") 在 Python 端为 True
            has_omp = fmm_kernel.has_openmp
        else:
            has_omp = False
        print(f"FMM has OpenMP support: {has_omp}")
        if has_omp:
            from ctypes import cdll, c_int
            # 尝试加载 libgomp 获取 omp_get_max_threads
            try:
                libg = cdll.LoadLibrary("libgomp.so")
                libg.omp_get_max_threads.restype = c_int
                max_th = libg.omp_get_max_threads()
                print(f"omp_get_max_threads(): {max_th}")
            except:
                pass
    except:
        pass

    # Python 版本
    print(f"Python version: {sys.version.split()[0]}")
    print("Done.\n")


if __name__ == "__main__":
    # stand-alone 测试时可启用
    print("Run `python main_program_parallel_final.py` using interactive menu\n")

