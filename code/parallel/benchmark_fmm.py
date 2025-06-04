#!/usr/bin/env python3
"""
benchmark_fmm.py  —  Compare Direct (O(N²)) vs FMM (O(N log N)) + OpenMP

 1) Size sweep:  for each N in --sizes, run Direct (1 thread) and FMM (1 thread).
 2) Thread scaling: pick one N (second‐smallest if ≥2), vary OMP_NUM_THREADS, measure Direct & FMM.
 3) Θ trade‐off: pick one N (second‐smallest if ≥2), vary θ, measure L² error vs runtime.

Usage example:
  python3 benchmark_fmm.py \
    --sizes 2000 20000 200000 \
    --threads 1 2 4 8 16 \
    --theta_base 0.6 \
    --theta 0.3 0.5 0.7 1.0
"""

import os
import sys
import time
import math
import argparse
import pathlib
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────
# 1) Ensure current directory is on sys.path so we can import “*.so” modules
# ───────────────────────────────────────────────────────────────────────────
here = pathlib.Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

# ───────────────────────────────────────────────────────────────────────────
# 2) Load `force_openmp` (Direct O(N²)) and `fmm_openmp` (our FMM) modules
# ───────────────────────────────────────────────────────────────────────────
def load_module(name: str, so: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        so_path = here / f"{so}.so"
        if not so_path.exists():
            sys.exit(f"Error: Cannot find `{name}` or `{so_path}`.")
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod  # type: ignore

fm     = load_module("fmm_openmp",   "fmm_openmp")
direct = load_module("force_openmp", "force_openmp")

# ───────────────────────────────────────────────────────────────────────────
# 3) Create output directory if it does not exist
# ───────────────────────────────────────────────────────────────────────────
OUT = here / "results_bench_rev6"
OUT.mkdir(exist_ok=True)

_rng = np.random.default_rng(42)

# ───────────────────────────────────────────────────────────────────────────
# 4) Random N‐body system: uniform in [−domain, +domain], all masses = 1
# ───────────────────────────────────────────────────────────────────────────
def random_system(N: int, domain: float):
    x = _rng.uniform(-domain, domain, size=N).astype(np.float64)
    y = _rng.uniform(-domain, domain, size=N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    return x, y, m

# ───────────────────────────────────────────────────────────────────────────
# 5) Size sweep: for each N, measure Direct (1 thr) & FMM (1 thr)
# ───────────────────────────────────────────────────────────────────────────
def run_size_sweep(Ns, eps2, domain, theta):
    direct_times = []
    fmm_times    = []

    for N in Ns:
        print(f"\n--- Starting N = {N} ---", flush=True)
        x, y, m = random_system(N, domain)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)

        # Direct O(N²) solver (serial)
        print(f"→ Running direct solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        direct.direct_symm(x, y, m, eps2, ax, ay)
        dt = time.perf_counter() - t0
        direct_times.append(dt)
        print(f"→ Direct solver for N={N} finished in {dt:.6g}s", flush=True)

        # FMM solver (1 thread)
        os.environ["OMP_NUM_THREADS"] = "1"
        print(f"→ Running FMM solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        tf = time.perf_counter() - t0
        fmm_times.append(tf)
        print(f"→ FMM solver for N={N} finished in {tf:.6g}s", flush=True)

        speedup = dt / tf if tf > 0 else float("inf")
        print(f"   N={N:7d}  direct={dt:.6g}s  fmm={tf:.6g}s  speed-up={speedup:.2f}", flush=True)

    # Save raw timing data
    with open(OUT / "size_vs_times.tsv", "w") as fout:
        fout.write("N\tdirect_time\tfmm_time\n")
        for i, N in enumerate(Ns):
            fout.write(f"{N}\t{direct_times[i]:.6g}\t{fmm_times[i]:.6g}\n")

    # Plot wall‐time vs N (log‐log)
    N0 = Ns[0]
    ref_n2 = [direct_times[0] * (N/N0)**2 for N in Ns]
    ref_nl = [fmm_times[0] * (N/N0)*math.log2(N)/math.log2(N0) for N in Ns]

    plt.figure(figsize=(6,4))
    plt.loglog(Ns, direct_times, 'o-', label='Direct O(N²)')
    plt.loglog(Ns, fmm_times,    's-', label='FMM O(N log N)')
    plt.loglog(Ns, ref_n2, '--', color='C0', alpha=0.3)
    plt.loglog(Ns, ref_nl, ':',  color='C1', alpha=0.3)
    plt.xlabel('N')
    plt.ylabel('wall-time [s]')
    plt.title(f'Algorithmic timings (θ={theta})')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_time.png", dpi=300)
    plt.close()

    # Plot speed‐up vs N
    plt.figure(figsize=(6,4))
    plt.loglog(Ns, [direct_times[i]/fmm_times[i] for i in range(len(Ns))], 'o-')
    plt.xlabel('N')
    plt.ylabel('Direct / FMM speed-up')
    plt.title('Algorithmic speed-up')
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_speedup.png", dpi=300)
    plt.close()

# ───────────────────────────────────────────────────────────────────────────
# 6) Thread scaling: fix N, vary threads, measure Direct (serial) & FMM (parallel)
# ───────────────────────────────────────────────────────────────────────────
def run_thread_scaling(N, threads, eps2, domain, theta):
    print(f"\n--- Thread scaling for N = {N} ---", flush=True)
    x, y, m = random_system(N, domain)

    direct_times = []
    fmm_times    = []

    for thr in threads:
        print(f"→ Running with threads = {thr}", flush=True)
        os.environ["OMP_NUM_THREADS"] = str(thr)

        # Direct (serial)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        t0 = time.perf_counter()
        direct.direct_symm(x, y, m, eps2, ax, ay)
        dt = time.perf_counter() - t0
        direct_times.append(dt)

        # FMM (parallel traversal only)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        tf = time.perf_counter() - t0
        fmm_times.append(tf)

        print(f"   threads={thr:2d}  direct={dt:.6g}s  fmm={tf:.6g}s", flush=True)

    # Save raw data
    with open(OUT / "thread_scaling.tsv", "w") as fout:
        fout.write("threads\tdirect_time\tfmm_time\n")
        for i, thr in enumerate(threads):
            fout.write(f"{thr}\t{direct_times[i]:.6g}\t{fmm_times[i]:.6g}\n")

    # Plot FMM wall-time vs threads
    plt.figure(figsize=(6,4))
    plt.plot(threads, fmm_times, 'o-', color='C1', label='FMM time')
    plt.xlabel('# threads')
    plt.ylabel('FMM wall-time [s]')
    plt.title(f'Thread scaling (N={N}, θ={theta})')
    plt.tight_layout()
    plt.savefig(OUT / "threads_plot.png", dpi=300)
    plt.close()

# ───────────────────────────────────────────────────────────────────────────
# 7) Θ trade-off: fix N, vary θ, measure L² error vs runtime
# ───────────────────────────────────────────────────────────────────────────
def run_theta_tradeoff(N, thetas, eps2, domain):
    print(f"\n--- Θ trade-off for N = {N} ---", flush=True)
    x, y, m = random_system(N, domain)

    # Reference via Direct
    ax_ref = np.zeros(N, dtype=np.float64)
    ay_ref = np.zeros(N, dtype=np.float64)
    direct.direct_symm(x, y, m, eps2, ax_ref, ay_ref)
    a_ref = np.vstack((ax_ref, ay_ref)).T

    errs, times = [], []
    for th in thetas:
        print(f"→ Running FMM with θ = {th}", flush=True)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, th, ax, ay)
        tf = time.perf_counter() - t0
        times.append(tf)

        a_fmm = np.vstack((ax, ay), axis=1)
        err = np.linalg.norm(a_fmm - a_ref) / max(np.linalg.norm(a_ref), 1e-12)
        errs.append(err)

        print(f"   θ={th:.2f}  t={tf:.3e}s  L2-err={err:.3e}", flush=True)

    # Plot error vs θ and time vs θ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    ax1.semilogy(thetas, errs, 'o-', color='C0')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('L2 relative error')
    ax1.set_title(f'Accuracy vs θ (N={N})')

    ax2.plot(thetas, times, 's-', color='C1')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('time [s]')
    ax2.set_title(f'Runtime vs θ (N={N})')

    plt.tight_layout()
    plt.savefig(OUT / "theta_tradeoff.png", dpi=300)
    plt.close()

# ───────────────────────────────────────────────────────────────────────────
# 8) main: parse arguments and run everything
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark_fmm.py: Direct vs FMM + OpenMP"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=float,
        default=[2000, 20000, 200000],
        help="List of N values for size sweep"
    )
    parser.add_argument(
        "--threads", nargs="+", type=int,
        default=[1, 2, 4, 8, 16],
        help="List of OMP_NUM_THREADS for thread scaling"
    )
    parser.add_argument(
        "--theta_base", type=float, default=0.6,
        help="θ for size sweep & thread scaling"
    )
    parser.add_argument(
        "--theta", nargs="+", type=float,
        default=[0.3, 0.5, 0.7, 1.0],
        help="List of θ values for Θ trade-off"
    )
    parser.add_argument(
        "--soft", type=float, default=1.0,
        help="Softening length ε (eps2 = ε²)"
    )
    parser.add_argument(
        "--domain", type=float, default=100.0,
        help="Half-width of domain (coordinates ∈ [–domain, +domain])"
    )
    args = parser.parse_args()

    Ns     = [int(s) for s in args.sizes]
    eps2   = args.soft * args.soft
    domain = args.domain

    print("\n=== Size sweep ===", flush=True)
    run_size_sweep(Ns, eps2, domain, args.theta_base)

    print("\n=== Thread scaling ===", flush=True)
    if len(Ns) >= 2:
        idx = 1
    else:
        idx = 0
    run_thread_scaling(Ns[idx], args.threads, eps2, domain, args.theta_base)

    print("\n=== Θ trade-off ===", flush=True)
    if len(Ns) >= 2:
        idx_theta = 1
    else:
        idx_theta = 0
    run_theta_tradeoff(Ns[idx_theta], args.theta, eps2, domain)

    print("\nPlots & data saved to", OUT.resolve(), flush=True)

