# benchmark_fmm.py — OpenMP & algorithmic performance analysis (REV-3)
# ====================================================================
# Produces four figures in ./results_bench_rev3/
#   size_vs_time.png, size_vs_speedup.png, thread_scaling.png,
#   theta_tradeoff.png
# Usage:
#   python benchmark_fmm.py
#   python benchmark_fmm.py --sizes 2e3 4e3 8e3 1.6e4
#                           --threads 1 2 4 8 16
#                           --theta 0.3 0.5 0.7 1.0
# --------------------------------------------------------------------
from __future__ import annotations
import os, time, math, argparse, pathlib, sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import fmm_openmp as fm               # compiled C++ kernels
except ImportError:
    sys.exit("fmm_openmp module not found – compile fmm_openmp.cpp first!")

OUT = pathlib.Path("results_bench_rev6")
OUT.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)


def random_system(N: int, domain: float = 50.0):
    x = _rng.uniform(-domain, domain, N).astype(np.float64)
    y = _rng.uniform(-domain, domain, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    return x, y, m


# --------------------------------------------------------------------
# 1) size sweep – Direct vs FMM
# --------------------------------------------------------------------
def run_size_sweep(Ns, threads: int, eps2: float, domain: float):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    direct_t, fmm_t = [], []

    for N in Ns:
        x, y, m = random_system(N)
        ax = np.zeros(N); ay = np.zeros(N)

        t0 = time.perf_counter(); fm.direct_force(x, y, m, eps2, ax, ay)
        direct_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter(); fm.fmm_force(x, y, m, eps2, domain, ax, ay)
        fmm_t.append(time.perf_counter() - t0)

        print(f"N={N:6d}  direct={direct_t[-1]:.4g}s  fmm={fmm_t[-1]:.4g}s  "
              f"speed-up={direct_t[-1]/fmm_t[-1]:.2f}")

    N0 = Ns[0]
    th_n2 = [direct_t[0] * (N / N0) ** 2 for N in Ns]
    th_nl = [fmm_t[0]    * (N / N0) * math.log2(N) / math.log2(N0) for N in Ns]

    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, direct_t, 'o-', label='Direct  O(N²)')
    plt.loglog(Ns, fmm_t,    's-', label='FMM     O(N log N)')
    plt.loglog(Ns, th_n2, '--', color='C0', alpha=0.4)
    plt.loglog(Ns, th_nl, ':',  color='C1', alpha=0.4)
    plt.title(f'Timing @ {threads} threads')
    plt.xlabel('N'); plt.ylabel('time [s]'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / 'size_vs_time.png', dpi=300); plt.close()

    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, np.array(direct_t) / np.array(fmm_t), 'o-')
    plt.xlabel('N'); plt.ylabel('Direct / FMM'); plt.title('Algorithmic speed-up')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / 'size_vs_speedup.png', dpi=300); plt.close()


# --------------------------------------------------------------------
# 2) thread scaling – both solvers
# --------------------------------------------------------------------
def run_scaling(N: int, thread_list, eps2: float, domain: float):
    x, y, m = random_system(N)
    ax = np.zeros(N); ay = np.zeros(N)

    d_times, f_times = [], []
    for t in thread_list:
        os.environ['OMP_NUM_THREADS'] = str(t); time.sleep(0.05)

        t0 = time.perf_counter(); fm.direct_force(x, y, m, eps2, ax, ay)
        d_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter(); fm.fmm_force(x, y, m, eps2, domain, ax, ay)
        f_times.append(time.perf_counter() - t0)

        print(f" {t:3d} thr  direct={d_times[-1]:.6f}s  fmm={f_times[-1]:.6f}s")

    speed_d = d_times[0] / np.array(d_times)
    speed_f = f_times[0] / np.array(f_times)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(thread_list, d_times, 'o-', label='Direct time')
    ax1.plot(thread_list, f_times, 's-', label='FMM time')
    ax1.set_xlabel('#threads'); ax1.set_ylabel('wall-time [s]'); ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thread_list, speed_d, 'o--', color='C0', label='Direct speed-up')
    ax2.plot(thread_list, speed_f, 's--', color='C1', label='FMM speed-up')
    ax2.set_ylabel('speed-up')

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)

    plt.title(f'Thread scaling  N={N}')
    plt.tight_layout(); plt.savefig(OUT / 'thread_scaling.png', dpi=300); plt.close()


# --------------------------------------------------------------------
# 3) θ trade-off – L² error (θ still hard-wired in C++)
# --------------------------------------------------------------------
def run_theta(N: int, thetas, eps2: float, domain: float):
    x, y, m = random_system(N)
    ax = np.zeros(N); ay = np.zeros(N)

    fm.direct_force(x, y, m, eps2, ax, ay)
    a_ref = np.vstack((ax, ay)).T

    errs, times = [], []
    for th in thetas:
        t0 = time.perf_counter()
        fm.fmm_force(x, y, m, eps2, domain, ax, ay)   # θ ignored inside C++
        times.append(time.perf_counter() - t0)

        a_fmm = np.vstack((ax, ay)).T
        err = np.linalg.norm(a_fmm - a_ref) / max(np.linalg.norm(a_ref), 1e-12)
        errs.append(err)
        print(f" θ={th:.2f}  t={times[-1]:.3e}s  L2-err={err:.2e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.semilogy(thetas, errs, 'o-'); a1.set_xlabel('θ'); a1.set_ylabel('L2 relative error')
    a2.plot(thetas, times, 's-');    a2.set_xlabel('θ'); a2.set_ylabel('time [s]')
    fig.suptitle(f'Accuracy vs runtime  N={N}')
    plt.tight_layout(); plt.savefig(OUT / 'theta_tradeoff.png', dpi=300); plt.close()


# --------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Direct vs FMM benchmark')
    parser.add_argument('--sizes',   type=float, nargs='+',
                        default=[2e3, 4e3, 8e3, 1.6e4])
    parser.add_argument('--threads', type=int,   nargs='+',
                        default=[1, 2, 4, 8, 16])
    parser.add_argument('--theta',   type=float, nargs='+',
                        default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument('--soft',    type=float, default=0.01)
    parser.add_argument('--domain',  type=float, default=100.0)
    args = parser.parse_args()

    Ns     = [int(s) for s in args.sizes]
    eps2   = args.soft ** 2
    domain = args.domain

    print('\\n=== Size sweep ===');     run_size_sweep(Ns, max(args.threads), eps2, domain)
    print('\\n=== Thread scaling ==='); run_scaling(Ns[len(Ns)//2], args.threads, eps2, domain)
    print('\\n=== θ trade-off ===');    run_theta(Ns[1], args.theta, eps2, domain)

    print('\\nPlots saved to', OUT)

