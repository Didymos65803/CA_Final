#!/usr/bin/env python3
# benchmark_fmm.py  —  Direct (O(N²)) vs. FMM (O(N log N)) + OpenMP

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

# ───────────────────────────────────────────────────────────────────────────
#  1) Ensure current directory is on sys.path
# ───────────────────────────────────────────────────────────────────────────
here = pathlib.Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

# ───────────────────────────────────────────────────────────────────────────
#  2) Load the two extension modules:
#       - force_openmp  (direct O(N²) solver)
#       - fmm_openmp    (our FMM solver)
#    If a normal import fails, try loading so‐file explicitly.
# ───────────────────────────────────────────────────────────────────────────
def load_module(name: str, short: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        so_path = here / f"{short}.so"
        if not so_path.exists():
            sys.exit(f"{name} not found and {so_path} does not exist")
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod  # type: ignore

fm     = load_module("fmm_openmp",   "fmm_openmp")
direct = load_module("force_openmp", "force_openmp")

# ───────────────────────────────────────────────────────────────────────────
#  3) Output folder
# ───────────────────────────────────────────────────────────────────────────
OUT = here / "results_bench_rev6"
OUT.mkdir(exist_ok=True)

_rng = np.random.default_rng(42)

# ───────────────────────────────────────────────────────────────────────────
#  4) Build a random N-body system: uniform in [-domain, +domain], unit mass
# ───────────────────────────────────────────────────────────────────────────
def random_system(N: int, domain: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _rng.uniform(-domain, domain, size=N)
    y = _rng.uniform(-domain, domain, size=N)
    m = np.ones(N, dtype=float)
    return x, y, m

# ───────────────────────────────────────────────────────────────────────────
#  5) Size sweep: for each N, run Direct vs. FMM (1 thread for both).
# ───────────────────────────────────────────────────────────────────────────
def run_size_sweep(Ns: list[int], eps2: float, domain: float, theta: float):
    direct_times, fmm_times = [], []

    for N in Ns:
        print(f"\n--- Starting N = {N} ---", flush=True)

        x, y, m = random_system(N, domain)
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)

        # --- Direct O(N²)
        print(f"→ Running direct solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        direct.direct_symm(x, y, m, eps2, ax, ay)
        dt = time.perf_counter() - t0
        direct_times.append(dt)
        print(f"→ Direct solver for N={N} finished in {dt:.6g}s", flush=True)

        # --- FMM O(N log N) with 1 thread
        os.environ["OMP_NUM_THREADS"] = "1"
        print(f"→ Running FMM solver for N={N}...", flush=True)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        tf = time.perf_counter() - t0
        fmm_times.append(tf)
        print(f"→ FMM solver for N={N} finished in {tf:.6g}s", flush=True)

        speedup = dt / tf if tf > 0 else float("inf")
        print(f"   N={N:7d}  direct={dt:.6g}s  fmm={tf:.6g}s  speed-up={speedup:.2f}", flush=True)

    # Save a table of N vs. speed-up
    tsv_path = OUT / "size_vs_speedup.tsv"
    with open(tsv_path, "w") as f:
        f.write("N\tspeedup\n")
        for i, N in enumerate(Ns):
            f.write(f"{N}\t{direct_times[i]/fmm_times[i]:.6g}\n")

    # Plot times (log-log)
    N0 = Ns[0]
    ref_n2 = [direct_times[0] * (N/N0)**2 for N in Ns]
    ref_nl = [fmm_times[0] * (N/N0) * math.log2(N)/math.log2(N0) for N in Ns]

    plt.figure(figsize=(6,4))
    plt.loglog(Ns, direct_times, 'o-', label='Direct O(N²)')
    plt.loglog(Ns, fmm_times,    's-', label='FMM O(N log N)')
    plt.loglog(Ns, ref_n2, '--', color='C0', alpha=0.3)
    plt.loglog(Ns, ref_nl, ':',  color='C1', alpha=0.3)
    plt.xlabel('N')
    plt.ylabel('wall-time [s]')
    plt.title(f'Algorithmic timing (θ={theta})')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_time.png", dpi=300)
    plt.close()

    # Plot speed-up vs. N
    plt.figure(figsize=(6,4))
    plt.loglog(Ns, [direct_times[i]/fmm_times[i] for i in range(len(Ns))], 'o-')
    plt.xlabel('N')
    plt.ylabel('Direct / FMM')
    plt.title('Algorithmic speed-up')
    plt.tight_layout()
    plt.savefig(OUT / "size_vs_speedup.png", dpi=300)
    plt.close()

# ───────────────────────────────────────────────────────────────────────────
#  6) Thread scaling: fix N, vary thread count, record FMM time (and direct)
# ───────────────────────────────────────────────────────────────────────────
def run_thread_scaling(
    N: int,
    threads: list[int],
    eps2: float,
    domain: float,
    theta: float
):
    print(f"\n--- Thread scaling for N = {N} ---", flush=True)
    x, y, m = random_system(N, domain)

    fmm_times = []
    direct_times = []
    for thr in threads:
        print(f"→ Running FMM with threads = {thr}", flush=True)
        os.environ["OMP_NUM_THREADS"] = str(thr)

        # Direct (does not actually use threads in our code, but we measure anyway)
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)
        t0 = time.perf_counter()
        direct.direct_symm(x, y, m, eps2, ax, ay)
        td = time.perf_counter() - t0
        direct_times.append(td)

        # FMM
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        tf = time.perf_counter() - t0
        fmm_times.append(tf)

        print(f"   threads={thr:2d}  direct={td:.6g}s  fmm={tf:.6g}s", flush=True)

    # Save raw times
    tsv_path = OUT / "thread_scaling.tsv"
    with open(tsv_path, "w") as f:
        f.write("threads\tdirect_time\tfmm_time\n")
        for i, thr in enumerate(threads):
            f.write(f"{thr}\t{direct_times[i]:.6g}\t{fmm_times[i]:.6g}\n")

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
#  7) Theta trade-off: fix N, vary θ, measure L² error vs runtime
# ───────────────────────────────────────────────────────────────────────────
def run_theta_tradeoff(
    N: int,
    thetas: list[float],
    eps2: float,
    domain: float
):
    print(f"\n--- Θ trade-off for N = {N} ---", flush=True)
    x, y, m = random_system(N, domain)
    ax = np.zeros(N, dtype=float)
    ay = np.zeros(N, dtype=float)

    # Compute a reference acceleration via Direct O(N²)
    direct.direct_symm(x, y, m, eps2, ax, ay)
    a_ref = np.vstack((ax, ay)).T

    errs, times = [], []
    for th in thetas:
        print(f"→ Running FMM with θ = {th}", flush=True)
        ax = np.zeros(N, dtype=float)
        ay = np.zeros(N, dtype=float)
        t0 = time.perf_counter()
        fm.fmm_force_theta(x, y, m, eps2, domain, th, ax, ay)
        tf = time.perf_counter() - t0
        times.append(tf)

        a_fmm = np.vstack((ax, ay)).T
        err = np.linalg.norm(a_fmm - a_ref) / max(np.linalg.norm(a_ref), 1e-12)
        errs.append(err)

        print(f"   θ={th:.2f}  t={tf:.3e}s  L2-err={err:.3e}", flush=True)

    # Plot L² error vs θ and runtime vs θ
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
#  8) main: parse arguments & run all three experiments
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark_fmm.py: compare Direct vs FMM + OpenMP"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=float,
        default=[2000, 20000, 200000],
        help="List of N values for size sweep, e.g., 2000 20000 200000"
    )
    parser.add_argument(
        "--threads", nargs="+", type=int,
        default=[1, 2, 4, 8, 16],
        help="List of OMP_NUM_THREADS to test"
    )
    parser.add_argument(
        "--theta_base", type=float, default=0.6,
        help="Base θ value for size sweep & thread scaling"
    )
    parser.add_argument(
        "--theta", nargs="+", type=float,
        default=[0.3, 0.5, 0.7, 1.0],
        help="List of θ values for the θ trade-off"
    )
    parser.add_argument(
        "--soft", type=float, default=1.0,
        help="Softening length ε (eps2 = ε^2)"
    )
    parser.add_argument(
        "--domain", type=float, default=100.0,
        help="Domain half-width (e.g. 100 means coordinates ∈ [−100, +100])"
    )
    args = parser.parse_args()

    Ns     = [int(s) for s in args.sizes]
    eps2   = args.soft ** 2
    domain = args.domain

    print("\n=== Size sweep ===", flush=True)
    run_size_sweep(Ns, eps2, domain, args.theta_base)

    print("\n=== Thread scaling ===", flush=True)
    # Pick the “middle” size for thread scaling if ≥ 2, otherwise use the only size
    if len(Ns) >= 2:
        idx = len(Ns) // 2
    else:
        idx = 0
    run_thread_scaling(Ns[idx], args.threads, eps2, domain, args.theta_base)

    print("\n=== Θ trade-off ===", flush=True)
    # If only one size was provided, use that; else use the second‐smallest
    if len(Ns) >= 2:
        idx_theta = 1
    else:
        idx_theta = 0
    run_theta_tradeoff(Ns[idx_theta], args.theta, eps2, domain)

    print("\nPlots & data saved to", OUT.resolve(), flush=True)

