#!/usr/bin/env python3
"""
benchmark_fmm.py — OpenMP & algorithmic performance analysis (English)

1) Size sweep   (Direct vs. Barnes–Hut FMM)   — O(N^2) vs O(N log N)
2) Thread scaling (varying OMP_NUM_THREADS at fixed N)
3) Theta trade‐off (accuracy vs runtime for different opening angles)

Usage examples:
  python3 benchmark_fmm.py
  python3 benchmark_fmm.py --sizes 20000 50000 100000 --threads 1 2 4 8 16 --theta_base 0.6 --theta 0.3 0.5 0.7 1.0
"""

from __future__ import annotations
import os
import sys
import time
import math
import argparse
import pathlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) Ensure current directory is in sys.path
# ---------------------------------------------------------------------------
here = os.path.abspath(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

# ---------------------------------------------------------------------------
# 2) Import the direct solver (force_openmp) and FMM solver (fmm_openmp).
#    If "import fmm_openmp" fails, try loading "fmm_openmp.so" explicitly.
# ---------------------------------------------------------------------------
try:
    import fmm_openmp as fm
except ImportError:
    so_path = os.path.join(here, "fmm_openmp.so")
    if os.path.isfile(so_path):
        spec = importlib.util.spec_from_file_location("fmm_openmp", so_path)
        if spec and spec.loader:
            fmm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fmm_module)
            fm = fmm_module
        else:
            sys.exit("Failed to create spec for fmm_openmp.so")
    else:
        sys.exit("fmm_openmp module not found — ensure fmm_openmp.so is in the same folder")

try:
    import force_openmp as direct_module
except ImportError:
    so_path2 = os.path.join(here, "force_openmp.so")
    if os.path.isfile(so_path2):
        spec2 = importlib.util.spec_from_file_location("force_openmp", so_path2)
        if spec2 and spec2.loader:
            dir_mod = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(dir_mod)
            direct_module = dir_mod
        else:
            sys.exit("Failed to create spec for force_openmp.so")
    else:
        sys.exit("force_openmp module not found — ensure force_openmp.so is in the same folder")

# ---------------------------------------------------------------------------
OUT = pathlib.Path("results_bench_rev6")
OUT.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1) Generate random N‐body system: uniform in [-100,100], unit masses
# ---------------------------------------------------------------------------
def random_system(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _rng.uniform(-100, 100, size=N)
    y = _rng.uniform(-100, 100, size=N)
    m = np.ones(N, dtype=float)
    return x, y, m

# ---------------------------------------------------------------------------
# 2) Size sweep: for each N, run direct (O(N^2)) vs. FMM (O(N log N))
# ---------------------------------------------------------------------------
def run_size_sweep(
    Ns: list[int],
    threads: int,
    eps2: float,
    domain: float,
    theta: float
):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    direct_times, fmm_times = [], []

    for N in Ns:
        # 2.1) Print a header so we know it has reached this iteration
        print(f"\n--- Starting N = {N} ---", flush=True)

        x, y, m = random_system(N)
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)

        # --- Direct O(N^2)
        print(f"→ Running direct solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        direct_module.direct_symm(x, y, m, eps2, ax, ay)
        dt_direct = time.perf_counter() - t0
        direct_times.append(dt_direct)
        print(f"→ Direct solver for N={N} finished in {dt_direct:.4g}s", flush=True)

        # --- FMM O(N log N)
        print(f"→ Running FMM solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        dt_fmm = time.perf_counter() - t0
        fmm_times.append(dt_fmm)
        print(f"→ FMM solver for N={N} finished in {dt_fmm:.4g}s", flush=True)

        speedup = dt_direct / dt_fmm if dt_fmm > 0 else float("inf")
        print(f"   N={N:6d}  direct={dt_direct:.4g}s  fmm={dt_fmm:.4g}s  speed-up={speedup:.2f}", flush=True)

    # Write TSV: N vs. speed‐up
    tsv_path = OUT / "speedup_vs_size.tsv"
    with open(tsv_path, "w") as f:
        f.write("N\tspeedup\n")
        for i, N in enumerate(Ns):
            f.write(f"{N}\t{direct_times[i]/fmm_times[i]:.6g}\n")

    # Plot times (log‐log) with theoretical O(N^2) and O(N log N) references
    N0 = Ns[0]
    ref_n2 = [direct_times[0] * (N/N0) ** 2 for N in Ns]
    ref_nl = [fmm_times[0] * (N/N0) * math.log2(N)/math.log2(N0) for N in Ns]

    plt.figure(figsize=(6,4))
    plt.loglog(Ns, direct_times, 'o-', label='Direct O(N²)')
    plt.loglog(Ns, fmm_times,    's-', label='FMM O(N log N)')
    plt.loglog(Ns, ref_n2, '--', color='C0', alpha=0.35)
    plt.loglog(Ns, ref_nl, ':',  color='C1', alpha=0.35)
    plt.xlabel('N')
    plt.ylabel('time [s]')
    plt.title(f'Algorithmic timing (threads={threads}, θ={theta})')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_time.png", dpi=300)
    plt.close()

    # Plot speed‐up vs. N
    plt.figure(figsize=(6,4))
    plt.loglog(Ns, [direct_times[i]/fmm_times[i] for i in range(len(Ns))], 'o-')
    plt.xlabel('N')
    plt.ylabel('Direct / FMM')
    plt.title('Algorithmic speed‐up')
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_speedup.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 3) Thread scaling: fix N, vary OMP_NUM_THREADS, compare times & speed‐ups
# ---------------------------------------------------------------------------
def run_scaling(
    N: int,
    thread_list: list[int],
    eps2: float,
    domain: float,
    theta: float
):
    print(f"\n--- Thread scaling for N = {N} ---", flush=True)
    x, y, m = random_system(N)
    base_ax = np.zeros(N, dtype=float)
    base_ay = np.zeros(N, dtype=float)

    # Warm‐up (load any “cold” overhead)
    direct_module.direct_symm(x, y, m, eps2, base_ax, base_ay)
    fm.fmm_force_theta(x, y, m, eps2, domain, theta, base_ax, base_ay)

    direct_tlist, fmm_tlist = [], []
    for thr in thread_list:
        print(f"→ Running with threads={thr}", flush=True)
        os.environ["OMP_NUM_THREADS"] = str(thr)

        # Direct
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)
        t0 = time.perf_counter()
        direct_module.direct_symm(x, y, m, eps2, ax, ay)
        direct_tlist.append(time.perf_counter() - t0)

        # FMM
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        fmm_tlist.append(time.perf_counter() - t0)

        print(f"   threads={thr:2d}  direct={direct_tlist[-1]:.4g}s  fmm={fmm_tlist[-1]:.4g}s", flush=True)

    base_direct = direct_tlist[0]
    base_fmm    = fmm_tlist[0]
    speed_d = [base_direct / t for t in direct_tlist]
    speed_f = [base_fmm    / t for t in fmm_tlist]

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()

    ax1.plot(thread_list, direct_tlist, 'o-', color='C0', label='Direct time')
    ax1.plot(thread_list, fmm_tlist,    's--', color='C1', label='FMM time')
    ax2.plot(thread_list, speed_d, 'o:', color='C0', label='Direct speed‐up')
    ax2.plot(thread_list, speed_f, 's--', color='C1', label='FMM speed‐up')

    ax1.set_xlabel('#threads')
    ax1.set_ylabel('wall-time [s]')
    ax2.set_ylabel('speed‐up')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    plt.title(f'Thread scaling   N={N}   (θ={theta})')
    plt.tight_layout()
    plt.savefig(OUT / "thread_scaling.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 4) Theta trade‐off: fix N, measure L2 error vs runtime for different θ
# ---------------------------------------------------------------------------
def run_theta(
    N: int,
    thetas: list[float],
    eps2: float,
    domain: float
):
    print(f"\n--- Theta trade‐off for N = {N} ---", flush=True)
    x, y, m = random_system(N)
    ax = np.zeros(N, dtype=float)
    ay = np.zeros(N, dtype=float)

    # Reference: Direct
    direct_module.direct_symm(x, y, m, eps2, ax, ay)
    a_ref = np.vstack((ax, ay)).T

    errs, times = [], []
    for th in thetas:
        print(f"→ Running FMM with θ = {th}", flush=True)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, th, ax, ay)
        times.append(time.perf_counter() - t0)
        a_fmm = np.vstack((ax, ay)).T
        err = np.linalg.norm(a_fmm - a_ref) / max(np.linalg.norm(a_ref), 1e-12)
        errs.append(err)
        print(f"   θ={th:.2f}  t={times[-1]:.3e}s  L2-err={err:.2e}", flush=True)

    # Plot L2 error vs θ (log‐y) and runtime vs θ (linear)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

    ax1.semilogy(thetas, errs, 'o-', color='C0')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('L2 relative error')
    ax1.set_title('Accuracy vs θ')

    ax2.plot(thetas, times, 's-', color='C1')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('time [s]')
    ax2.set_title(f'Runtime vs θ   N={N}')

    plt.tight_layout()
    plt.savefig(OUT / "theta_tradeoff.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 5) main: parse arguments & run experiments
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark_fmm.py: compare Direct vs. FMM + OpenMP"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=float,
        default=[2e3, 4e3, 8e3, 1.6e4],
        help="List of N values for size sweep (e.g. 2e3 4e3 8e3 1.6e4)"
    )
    parser.add_argument(
        "--threads", nargs="+", type=int,
        default=[1, 2, 4, 8, 16],
        help="List of OMP_NUM_THREADS to test (e.g. 1 2 4 8 16)"
    )
    parser.add_argument(
        "--theta_base", type=float, default=0.6,
        help="Base θ value for thread scaling"
    )
    parser.add_argument(
        "--theta", nargs="+", type=float,
        default=[0.3, 0.5, 0.7, 1.0],
        help="List of θ values for the theta trade-off"
    )
    parser.add_argument(
        "--soft", type=float, default=1.0,
        help="Softening length ε (computes eps2 = ε^2)"
    )
    parser.add_argument(
        "--domain", type=float, default=100.0,
        help="Domain half‐width (e.g. 100.0 means [-100, +100])"
    )
    args = parser.parse_args()

    Ns     = [int(s) for s in args.sizes]
    eps2   = args.soft ** 2
    domain = args.domain

    print("\n=== Size sweep ===", flush=True)
    run_size_sweep(Ns, max(args.threads), eps2, domain, args.theta_base)

    print("\n=== Thread scaling ===", flush=True)
    run_scaling(Ns[len(Ns)//2], args.threads, eps2, domain, args.theta_base)

    print("\n=== θ trade‐off ===", flush=True)
    run_theta(Ns[1], args.theta, eps2, domain)

    print("\nFigures + tables saved to", OUT.resolve(), flush=True)

