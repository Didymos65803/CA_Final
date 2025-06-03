# comprehensive_test.py
# =====================
# This file contains all of the test / benchmark / plotting routines for the
# 2D N-body playground. It is called by main_program_parallel_final.py.
#
# Fixes included:
#   • test_scaling() now runs the Direct kernel 5× and averages to remove “first-call noise.”
#   • thread_benchmark() now computes Speedup = time@1-thread / time@N-threads (instead of 1/time).
#   • All output files go into an "output" folder (adjust OUTPUT_DIR to change or remove this behavior).

import os
import sys
import time
import math
import random
import csv

import numpy as np
import matplotlib.pyplot as plt

# If you want to save everything under a subfolder called "output",
# uncomment these lines and use OUTPUT_PATH everywhere:
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to import the C++ modules. If they are not compiled, HAS_FMM or HAS_DIRECT_BH will be False.
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

# Global physics constants and default parameters (used in both files):
G = 1.0
SOFT = 0.005               # softening length for near‐field
DOMAIN = 100.0             # Domain size (half‐width 50)
DT = 0.0005                # Time step (Leapfrog)
STAR_M = 100.0             # Mass of central fixed star (if used)

# Optimized Barnes-Hut / FMM parameters (used in accuracy & scaling tests)
OPTIMIZED_PARAMS = {
    "bh_theta": 0.3,
    "fmm_theta": 0.2,
    "bh_domain": DOMAIN,
    "fmm_domain": DOMAIN,
    "distribution_size": 50.0,
}


def generate_disk(n, radius=OPTIMIZED_PARAMS["distribution_size"], include_central=False):
    """
    Generate `n` particles uniformly in a disk of given `radius` in the XY-plane.
    If include_central=True, the 0-th particle is a fixed star of mass STAR_M at the origin.
    Returns a numpy array of shape (N_total, 5): columns = [x, y, vx, vy, m].
    """
    if include_central:
        N_total = n + 1
    else:
        N_total = n

    bodies = np.zeros((N_total, 5), dtype=np.float64)
    if include_central:
        # Place a fixed star at index 0
        bodies[0, :] = [0.0, 0.0, 0.0, 0.0, STAR_M]

    for i in range(1 if include_central else 0, N_total):
        r = math.sqrt(random.random()) * radius
        theta = random.random() * 2.0 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        bodies[i, 0] = x
        bodies[i, 1] = y
        bodies[i, 2] = 0.0  # initial vx
        bodies[i, 3] = 0.0  # initial vy
        bodies[i, 4] = 1.0  # mass = 1
    return bodies


def total_energy(bodies, include_central=False):
    """
    Compute total energy (kinetic + potential) of the system.
    If include_central=True, skip i=0 when summing KE and include i=0 in PE.
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
                # central star interactions
                dx = bodies[i, 0] - bodies[j, 0]
                dy = bodies[i, 1] - bodies[j, 1]
                dist = math.hypot(dx, dy) + SOFT
                PE -= G * bodies[i, 4] * bodies[j, 4] / dist
            else:
                dx = bodies[i, 0] - bodies[j, 0]
                dy = bodies[i, 1] - bodies[j, 1]
                dist = math.hypot(dx, dy) + SOFT
                PE -= G * bodies[i, 4] * bodies[j, 4] / dist
    return KE + PE


def test_accuracy():
    """
    Accuracy Comparison:
    - For N in [50, 100, 200, 500], compute forces with Direct, BH, FMM.
    - Compare the SUM of force magnitudes against Direct to find relative error.
    - Plot performance (time) vs error in a 1×2 subplot, save as accuracy_test_results_fixed.png.
    """
    print("Testing accuracy with optimized parameters...\n")

    Ns = [50, 100, 200, 500]
    bh_theta = OPTIMIZED_PARAMS["bh_theta"]
    fmm_theta = OPTIMIZED_PARAMS["fmm_theta"]
    distribution_size = OPTIMIZED_PARAMS["distribution_size"]

    results = []
    for N in Ns:
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # --- Direct (Baseline) ---
        if not HAS_DIRECT_BH:
            print("Warning: force_kernel (Direct/BH) not available; skipping accuracy test.")
            return
        t0 = time.time()
        fx_direct, fy_direct = direct_omp(x, y, m, G=G, soft=SOFT)
        t_direct = time.time() - t0
        Fd = np.linalg.norm(np.vstack((fx_direct, fy_direct)), axis=0).sum()

        # --- Barnes-Hut ---
        t0 = time.time()
        fx_bh, fy_bh = bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
        t_bh = time.time() - t0
        Fb = np.linalg.norm(np.vstack((fx_bh, fy_bh)), axis=0).sum()
        err_bh = abs(Fb - Fd) / (Fd + 1e-16)

        # --- FMM ---
        if not HAS_FMM:
            err_fmm = np.nan
            t_fmm = np.nan
            Ff = 0.0
        else:
            t0 = time.time()
            fx_fmm, fy_fmm = fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0
            Ff = np.linalg.norm(np.vstack((fx_fmm, fy_fmm)), axis=0).sum()
            err_fmm = abs(Ff - Fd) / (Fd + 1e-16)

        results.append((N, t_direct, t_bh, err_bh, t_fmm, err_fmm, Fd, Fb, Ff))

        print(f"Testing N = {N}")
        print(f"  Direct:     {t_direct:.6f} s")
        print(f"  Barnes-Hut: {t_bh:.6f} s (error = {err_bh:.2e}, θ={bh_theta})")
        if HAS_FMM:
            print(f"  FMM:        {t_fmm:.6f} s (error = {err_fmm:.2e}, θ={fmm_theta})")
        else:
            print("  FMM not available.")
        print(f"  Force magnitudes → Direct: {Fd:.3e}, BH: {Fb:.3e}, FMM: {Ff:.3e}\n")

    # Summary
    max_err_bh = max(r[3] for r in results)
    max_err_fmm = max(r[5] for r in results if not math.isnan(r[5]))
    print("Overall Results:")
    print(f"  Max Barnes-Hut relative error: {max_err_bh:.2e}")
    print(f"  Max FMM relative error: {max_err_fmm:.2e}\n")

    # Prepare plot
    Ns_plot = [r[0] for r in results]
    times_direct = [r[1] for r in results]
    times_bh = [r[2] for r in results]
    times_fmm = [r[4] for r in results]

    errs_bh = [r[3] for r in results]
    errs_fmm = [r[5] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Performance (time)
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

    # Right: Accuracy (relative error)
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
    outpath = os.path.join(OUTPUT_DIR, 'accuracy_test_results_fixed.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved accuracy plot to {outpath}\n")


def test_scaling():
    """
    Quick benchmark scaling (small N):
    - Test for N = [50, 100, 200, 500, 1000, 2000].
    - Measure Direct (averaged over 5 runs), BH (1 run), FMM (1 run).
    - Output CSV = scaling_smallN.csv and plot performance_comparison.png.
    """
    print("Testing small-N scaling behavior...\n")

    smallNs = [50, 100, 200, 500, 1000, 2000]
    bh_theta = OPTIMIZED_PARAMS["bh_theta"]
    fmm_theta = OPTIMIZED_PARAMS["fmm_theta"]
    distribution_size = OPTIMIZED_PARAMS["distribution_size"]

    results = []
    for N in smallNs:
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # --- Direct (averaged over 5 runs) ---
        if not HAS_DIRECT_BH:
            t_direct = np.nan
        else:
            direct_times = []
            for _ in range(5):
                t0 = time.time()
                direct_omp(x, y, m, G=G, soft=SOFT)
                direct_times.append(time.time() - t0)
            t_direct = sum(direct_times) / len(direct_times)

        # --- Barnes-Hut (1 run) ---
        if not HAS_DIRECT_BH:
            t_bh = np.nan
        else:
            t0 = time.time()
            bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
            t_bh = time.time() - t0

        # --- FMM (1 run) ---
        if not HAS_FMM:
            t_fmm = np.nan
        else:
            t0 = time.time()
            fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0

        results.append((N, t_direct, t_bh, t_fmm))
        print(f"N={N:<5}  Direct={t_direct:.6f}s  BH={t_bh:.6f}s  FMM={t_fmm:.6f}s")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'scaling_smallN.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'Direct', 'Barnes-Hut', 'FMM'])
        for row in results:
            writer.writerow(row)
    print(f"✓ Saved CSV to {csv_path}")

    # Plot
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
    ax.set_title('Scaling Comparison (Small N)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved performance plot to {outpath}\n")


def test_largeN_scaling():
    """
    Large-N Scaling Test:
    - Test for N = [500, 1000, 2000, 4000].
    - Direct is skipped for N > 2000 (NaN).
    - Output CSV = scaling_largeN.csv and plot scaling_largeN.png.
    """
    print("Testing large-N scaling behavior...\n")

    largeNs = [500, 1000, 2000, 4000]
    bh_theta = OPTIMIZED_PARAMS["bh_theta"]
    fmm_theta = OPTIMIZED_PARAMS["fmm_theta"]
    distribution_size = OPTIMIZED_PARAMS["distribution_size"]

    results = []
    for N in largeNs:
        bodies = generate_disk(N, distribution_size, include_central=False)
        x = bodies[:, 0].tolist()
        y = bodies[:, 1].tolist()
        m = bodies[:, 4].tolist()

        # --- Direct (skip if N > 2000) ---
        if not HAS_DIRECT_BH or N > 2000:
            t_direct = np.nan
        else:
            t0 = time.time()
            direct_omp(x, y, m, G=G, soft=SOFT)
            t_direct = time.time() - t0

        # --- Barnes-Hut ---
        if not HAS_DIRECT_BH:
            t_bh = np.nan
        else:
            t0 = time.time()
            bh_omp(x, y, m, DOMAIN, bh_theta, G, SOFT)
            t_bh = time.time() - t0

        # --- FMM ---
        if not HAS_FMM:
            t_fmm = np.nan
        else:
            t0 = time.time()
            fmm_kernel.fmm_omp(x, y, m, DOMAIN, fmm_theta, G, SOFT)
            t_fmm = time.time() - t0

        results.append((N, t_direct, t_bh, t_fmm))
        print(f"N={N:<5}  Direct={t_direct}  BH={t_bh:.6f}s  FMM={t_fmm:.6f}s")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'scaling_largeN.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'Direct', 'Barnes-Hut', 'FMM'])
        for row in results:
            writer.writerow(row)
    print(f"✓ Saved CSV to {csv_path}")

    # Plot
    Ns_plot = [r[0] for r in results]
    times_direct = [r[1] for r in results]
    times_bh = [r[2] for r in results]
    times_fmm = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    # Only plot Direct if it is not all NaN
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
    outpath = os.path.join(OUTPUT_DIR, 'scaling_largeN.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved plot to {outpath}\n")


def test_energy_conservation():
    """
    Energy Conservation Test:
    - For N=200, run Direct/BH/FMM for STEPS=500, record relative energy error every 5 steps.
    - Plot relative energy error (log‐scale) vs time, save as energy_conservation.png.
    """
    print("Testing energy conservation for Direct/BH/FMM...\n")

    N = 200
    STEPS = 500
    RECORD_EVERY = 5
    THREADS = 4
    include_central = False

    # Set OpenMP threads
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    # Generate initial state
    bodies = generate_disk(N, OPTIMIZED_PARAMS["distribution_size"], include_central=include_central)
    total_N = bodies.shape[0]
    E0 = total_energy(bodies, include_central=include_central)

    # Prepare solver list
    solvers = []
    if HAS_DIRECT_BH:
        solvers.append(("Direct", direct_omp, None))
        solvers.append(("Barnes-Hut", bh_omp, OPTIMIZED_PARAMS["bh_theta"]))
    if HAS_FMM:
        solvers.append(("FMM", fmm_kernel.fmm_omp, OPTIMIZED_PARAMS["fmm_theta"]))

    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, solver_fn, theta) in solvers:
        bodies_copy = generate_disk(N, OPTIMIZED_PARAMS["distribution_size"], include_central=include_central)
        E0_local = total_energy(bodies_copy, include_central=include_central)
        x = bodies_copy[:, 0].tolist()
        y = bodies_copy[:, 1].tolist()
        m = bodies_copy[:, 4].tolist()

        # Initial acceleration
        if label == "Direct":
            ax_old, ay_old = solver_fn(x, y, m, G=G, soft=SOFT)
        elif label == "Barnes-Hut":
            ax_old, ay_old = solver_fn(x, y, m, DOMAIN, theta, G, SOFT)
        else:  # FMM
            ax_old, ay_old = solver_fn(x, y, m, DOMAIN, theta, G, SOFT)

        times = []
        rel_errors = []

        for step in range(STEPS):
            # Leapfrog half‐kick, drift, compute new accel, half‐kick
            # 1) half-kick
            for i in range(total_N):
                if include_central and i == 0:
                    continue
                bodies_copy[i, 2] += 0.5 * DT * ax_old[i]
                bodies_copy[i, 3] += 0.5 * DT * ay_old[i]
            # 2) drift
            for i in range(total_N):
                if include_central and i == 0:
                    continue
                bodies_copy[i, 0] += DT * bodies_copy[i, 2]
                bodies_copy[i, 1] += DT * bodies_copy[i, 3]
            # 3) new accel
            x = bodies_copy[:, 0].tolist()
            y = bodies_copy[:, 1].tolist()
            m = bodies_copy[:, 4].tolist()
            if label == "Direct":
                ax_new, ay_new = solver_fn(x, y, m, G=G, soft=SOFT)
            elif label == "Barnes-Hut":
                ax_new, ay_new = solver_fn(x, y, m, DOMAIN, theta, G, SOFT)
            else:  # FMM
                ax_new, ay_new = solver_fn(x, y, m, DOMAIN, theta, G, SOFT)
            # 4) half-kick
            for i in range(total_N):
                if include_central and i == 0:
                    continue
                bodies_copy[i, 2] += 0.5 * DT * ax_new[i]
                bodies_copy[i, 3] += 0.5 * DT * ay_new[i]

            ax_old, ay_old = ax_new, ay_new

            if step % RECORD_EVERY == 0:
                E = total_energy(bodies_copy, include_central=include_central)
                rel_err = abs(E - E0_local) / (abs(E0_local) + 1e-16)
                times.append(step * DT)
                rel_errors.append(rel_err)

        ax.plot(times, rel_errors, label=label)

    ax.set_yscale('log')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Energy Error')
    ax.set_title(f'Energy Conservation Test (N={N}, threads={THREADS})')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'energy_conservation.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved energy conservation plot to {outpath}\n")


def optimize_parameters():
    """
    Parameter Optimization:
    - For N=100, test BH with theta ∈ [0.1, 0.3, 0.5, 0.7, 1.0] and domain ∈ [50, 100, 200].
    - Find the combination that yields relative error < 10% with minimal time.
    - Plot relative error vs theta (for each domain) and save as parameter_optimization.png.
    """
    print("Optimizing Barnes-Hut parameters (N=100)...\n")

    N = 100
    distribution_size = OPTIMIZED_PARAMS["distribution_size"]
    bodies = generate_disk(N, distribution_size, include_central=False)
    x = bodies[:, 0].tolist()
    y = bodies[:, 1].tolist()
    m = bodies[:, 4].tolist()

    if not HAS_DIRECT_BH:
        print("Error: Direct/BH module not available. Cannot optimize parameters.\n")
        return

    # Baseline direct force
    fx_direct, fy_direct = direct_omp(x, y, m, G=G, soft=SOFT)
    Fd = np.linalg.norm(np.vstack((fx_direct, fy_direct)), axis=0).sum()

    thetas = [0.1, 0.3, 0.5, 0.7, 1.0]
    domains = [50.0, 100.0, 200.0]

    records = []
    best_combo = (None, None, float('inf'), float('inf'))  # (theta, domain, error, time)

    print(" Theta  Domain    Error    Time(s)")
    print("------------------------------------")
    for theta in thetas:
        for domain in domains:
            t0 = time.time()
            fx_bh, fy_bh = bh_omp(x, y, m, domain, theta, G, SOFT)
            t_bh = time.time() - t0
            Fb = np.linalg.norm(np.vstack((fx_bh, fy_bh)), axis=0).sum()
            err = abs(Fb - Fd) / (Fd + 1e-16)
            records.append((theta, domain, err, t_bh))
            print(f" {theta:<5}  {domain:<6}  {err:.2e}  {t_bh:.4f}")
            # Choose best: error < 0.1 and minimal time
            if err < 0.1 and t_bh < best_combo[3]:
                best_combo = (theta, domain, err, t_bh)
    print("")

    if best_combo[0] is not None:
        print(f"Best combination → θ={best_combo[0]}, domain={best_combo[1]} (err={best_combo[2]:.2e}, time={best_combo[3]:.4f}s)\n")
    else:
        print("No combination found with error < 10%. Consider expanding search.\n")

    # Plot: error vs theta (one curve per domain)
    fig, ax = plt.subplots(figsize=(6, 5))
    for domain in domains:
        errs_for_domain = [r[2] for r in records if r[1] == domain]
        ax.plot(thetas, errs_for_domain, 'o-', label=f"domain={domain}")
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Relative Error')
    ax.set_title('Parameter Optimization (N=100)')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'parameter_optimization.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved parameter optimization plot to {outpath}\n")


def thread_benchmark():
    """
    OpenMP Thread Benchmark:
    - For FMM (N=500), measure time under threads = [1, 2, 4, 8].
    - Compute Speedup = t(1-thread) / t(N-threads). Plot and save as openmp_thread_benchmark.png.
    """
    print("Running OpenMP thread benchmark for FMM (N=500)...\n")

    if not HAS_FMM:
        print("Error: FMM module not available. Cannot run thread benchmark.\n")
        return

    N = 500
    distribution_size = OPTIMIZED_PARAMS["distribution_size"]
    bodies = generate_disk(N, distribution_size, include_central=False)
    x = bodies[:, 0].tolist()
    y = bodies[:, 1].tolist()
    m = bodies[:, 4].tolist()

    thread_counts = [1, 2, 4, 8]
    results = []

    for tcount in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(tcount)
        t0 = time.time()
        fmm_kernel.fmm_omp(x, y, m, DOMAIN, OPTIMIZED_PARAMS["fmm_theta"], G, SOFT)
        t_run = time.time() - t0
        results.append((tcount, t_run))
        print(f"Threads={tcount:<2}  Time={t_run:.6f}s")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'openmp_thread_benchmark.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Threads', 'Time'])
        for row in results:
            writer.writerow(row)
    print(f"✓ Saved CSV to {csv_path}")

    # Plot Speedup
    threads_plot = [r[0] for r in results]
    times_plot = [r[1] for r in results]
    baseline_time = times_plot[0]
    speedups = baseline_time / np.array(times_plot)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(threads_plot, speedups, 'o-', color='red', label='Speedup')
    ax.plot(threads_plot, threads_plot, '--', color='gray', label='Ideal Speedup')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('OpenMP Thread Benchmark (FMM, N=500)')
    ax.set_xticks(thread_counts)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'openmp_thread_benchmark.png')
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved thread benchmark plot to {outpath}\n")


def show_system_info():
    """
    System Information:
    - Print OS, CPU cores, OMP_NUM_THREADS, whether FMM has OpenMP, and Python version.
    """
    print("Gathering system information...\n")
    try:
        import platform
        info = platform.uname()
        print(f"System: {info.system} {info.release} ({info.machine})")
    except:
        pass

    try:
        cpu_count = os.cpu_count()
        print(f"CPU cores (os.cpu_count): {cpu_count}")
    except:
        pass

    env_t = os.environ.get("OMP_NUM_THREADS", "Not set")
    print(f"OMP_NUM_THREADS (env): {env_t}")

    if HAS_FMM:
        has_omp = getattr(fmm_kernel, "has_openmp", False)
    else:
        has_omp = False
    print(f"FMM has OpenMP support: {has_omp}")

    if has_omp:
        try:
            from ctypes import cdll, c_int
            libg = cdll.LoadLibrary("libgomp.so")
            libg.omp_get_max_threads.restype = c_int
            max_th = libg.omp_get_max_threads()
            print(f"omp_get_max_threads(): {max_th}")
        except:
            pass

    print(f"Python version: {sys.version.split()[0]}")
    print("Done.\n")


if __name__ == "__main__":
    # If you run this file standalone, nothing special happens.
    print("This file provides testing functions. Run \"python main_program_parallel_final.py\" to use the menu.\n")

