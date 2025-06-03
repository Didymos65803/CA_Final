#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_program_parallel_final.py
==============================
This is a “2D N-body Playground” that lets you:
  1) Quick small-N scaling (Direct vs BH vs FMM)
  2) Save trajectory + energy plot (direct, BH, or FMM)
  3) Live simulation animation (real-time) for any method
  4) Large-N scaling test (Direct vs BH vs FMM)
  5) Energy conservation test (Relative energy error vs time)
  6) Parameter optimization (choose best theta vs domain for BH)
  7) OpenMP thread benchmark (speedup vs number of threads)
  8) System information (print OpenMP details)
  q) Quit

Usage:
  - Must have already built `force_kernel` and `fmm_kernel` via setup.py.
  - Run: python main_program_parallel_final.py
  - Enter the menu choice (e.g. “1”, “2”, …, “q”).
  - Follow the on-screen prompts to choose method (direct/bh/fmm), N, #steps, #threads, etc.
  - All output plots and CSVs are saved into `output/` subfolder.

Author: Your Name (2025-06-XX)
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import kernels (built in the same directory via setup.py)
try:
    import force_kernel   # exposes direct_omp(...) and bh_omp(...)
except ImportError:
    print("Error: force_kernel module not found. Run `python setup.py build_ext --inplace` first.")
    sys.exit(1)

try:
    import fmm_kernel     # exposes fmm_omp(...)
except ImportError:
    print("Error: fmm_kernel module not found. Run `python setup.py build_ext --inplace` first.")
    sys.exit(1)

# Create output folder if needed
OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ============================
# === UTILITY & SETUP CODE ===
# ============================

def get_openmp_info():
    """
    Print available OpenMP info to screen.
    """
    print("\n=== OpenMP / NumPy / Python Info ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version : {np.__version__}")
    print(f"force_kernel.has_openmp = {force_kernel.has_openmp}")
    print(f"fmm_kernel.has_openmp   = {fmm_kernel.has_openmp}")
    print(f"Max OpenMP threads on this machine: {os.getenv('OMP_NUM_THREADS', 'Not set')}")
    print("=====================================\n")


def generate_initial_conditions(N, domain):
    """
    Generate N particles uniformly on a circle of radius domain/2,
    random initial velocities (small), and unit masses.

    Returns:
      x0, y0 : 1D arrays of length N (initial positions)
      vx0, vy0 : 1D arrays (initial velocities, set to zero here)
      m0     : 1D array (all masses = 1.0)
    """
    angles = np.random.uniform(0, 2 * math.pi, size=N)
    r = domain * 0.5 * np.sqrt(np.random.uniform(0.0, 1.0, size=N))
    x0 = r * np.cos(angles)
    y0 = r * np.sin(angles)
    vx0 = np.zeros(N)
    vy0 = np.zeros(N)
    m0  = np.ones(N)
    return x0, y0, vx0, vy0, m0


def compute_accelerations(method, x, y, m, domain, theta, G, soft, threads):
    """
    Wrapper to call the appropriate kernel based on method:
      - 'direct' : uses force_kernel.direct_omp()
      - 'bh'     : uses force_kernel.bh_omp()
      - 'fmm'    : uses fmm_kernel.fmm_omp()

    We set environment variable OMP_NUM_THREADS=threads before calling.
    Returns: (ax, ay) NumPy arrays
    """
    os.environ["OMP_NUM_THREADS"] = str(threads)

    if method == "direct":
        ax, ay = force_kernel.direct_omp(x, y, m, G, soft)
    elif method == "bh":
        ax, ay = force_kernel.bh_omp(x, y, m, domain, theta, G, soft)
    elif method == "fmm":
        ax, ay = fmm_kernel.fmm_omp(x, y, m, domain, theta, G, soft)
    else:
        raise ValueError(f"Unknown method '{method}'")

    return ax, ay


def kinetic_energy(vx, vy, m):
    """
    Kinetic energy = 0.5 * sum(m[i] * (vx[i]^2 + vy[i]^2))
    """
    return 0.5 * np.sum(m * (vx**2 + vy**2))


def potential_energy_direct(x, y, m, G, soft):
    """
    Compute potential energy directly (O(N^2)/2).
    U = - G * sum_{i<j} m_i m_j / sqrt(r2 + soft^2)
    """
    N = x.size
    U = 0.0
    soft2 = soft * soft
    for i in range(N):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        r2 = dx*dx + dy*dy + soft2
        inv_r = 1.0 / np.sqrt(r2)
        U -= G * m[i] * np.sum(m[i+1:] * inv_r)
    return U


def compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads):
    """
    Compute total energy = K + U.  Use direct O(N^2) for potential energy,
    always, to measure “true” energy.  K via velocities, U via direct formula.
    """
    K = kinetic_energy(vx, vy, m)
    U = potential_energy_direct(x, y, m, G, soft)
    return K + U


# ============================
# === MENU OPTION #1: QUICK SMALL‐N SCALING ===
# ============================

def benchmark_scaling_smallN(methods, Ns, domain, theta, G, soft, threads, max_steps=1):
    """
    For each N in Ns (e.g. [50,100,200,500,1000,2000]), generate random initial pos/mass,
    then measure the time to compute accelerations once (max_steps=1).  Do this for
    each method in methods=['direct','bh','fmm'], and return a dict of times.
    """
    results = {}  # results[N][method] = elapsed_seconds

    for N in Ns:
        x, y, vx, vy, m = generate_initial_conditions(N, domain)
        results[N] = {}

        for method in methods:
            # Warm‐up call once
            compute_accelerations(method, x, y, m, domain, theta, G, soft, threads)

            # Timed call
            start = time.time()
            for _ in range(max_steps):
                ax, ay = compute_accelerations(method, x, y, m, domain, theta, G, soft, threads)
            elapsed = (time.time() - start) / max_steps
            results[N][method] = elapsed

        # Print intermediate results
        print(f"N={N:<5d}", end=" ")
        for method in methods:
            print(f"{method:<6s}={results[N][method]:.4e}s", end="  ")
        print()

    # Save to CSV
    csv_file = os.path.join(OUTPUT_DIR, "scaling_smallN.csv")
    with open(csv_file, "w") as f:
        f.write("N,direct,bh,fmm\n")
        for N in Ns:
            f.write(f"{N},{results[N]['direct']},{results[N]['bh']},{results[N]['fmm']}\n")
    print(f"✓ Saved CSV to {csv_file}")

    # Plot results
    plt.figure(figsize=(6,5))
    for method, color, marker in zip(methods, ['r','b','g'], ['o','s','^']):
        times = [results[N][method] for N in Ns]
        plt.loglog(Ns, times, marker=marker, color=color, label=f"{method.upper()}")
    plt.xlabel("Number of Particles N")
    plt.ylabel("Time (s)")
    plt.title("Scaling Comparison (small N)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "scaling_smallN.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved plot to {plot_file}")


# ================================
# === MENU OPTION #4: LARGE‐N SCALING ===
# ================================

def benchmark_scaling_largeN(methods, Ns, domain, theta, G, soft, threads):
    """
    Similar to small‐N, but use larger N vector (e.g. [500, 1000, 2000, 3000, 4000]).
    Save times to CSV “scaling_largeN.csv” and plot “scaling_largeN.png”.
    """
    results = {}
    for N in Ns:
        x, y, vx, vy, m = generate_initial_conditions(N, domain)
        results[N] = {}

        for method in methods:
            # Warm‐up call once
            compute_accelerations(method, x, y, m, domain, theta, G, soft, threads)

            start = time.time()
            ax, ay = compute_accelerations(method, x, y, m, domain, theta, G, soft, threads)
            elapsed = time.time() - start
            results[N][method] = elapsed

        print(f"N={N:<5d}", end=" ")
        for method in methods:
            print(f"{method:<6s}={results[N][method]:.4e}s", end="  ")
        print()

    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, "scaling_largeN.csv")
    with open(csv_file, "w") as f:
        f.write("N,direct,bh,fmm\n")
        for N in Ns:
            direct_t = results[N].get('direct', np.nan)
            bh_t     = results[N].get('bh', np.nan)
            fmm_t    = results[N].get('fmm', np.nan)
            f.write(f"{N},{direct_t},{bh_t},{fmm_t}\n")
    print(f"✓ Saved CSV to {csv_file}")

    # Plot
    plt.figure(figsize=(6,5))
    for method, color, marker in zip(methods, ['r','b','g'], ['o','s','^']):
        times = [results[N][method] for N in Ns]
        plt.loglog(Ns, times, marker=marker, color=color, label=f"{method.upper()}")
    plt.xlabel("Number of Particles N")
    plt.ylabel("Time (s)")
    plt.title("Large-N Scaling Comparison")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "scaling_largeN.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved plot to {plot_file}")


# ============================================
# === MENU OPTION #2: SAVE TRAJECTORY+ENERGY ===
# ============================================

def simulate_trajectory(
    method,
    N,
    dt,
    steps,
    domain,
    theta,
    G,
    soft,
    threads
):
    """
    Simulate N-body for “steps” frames using leapfrog:
      x(t+dt/2) = x(t) + (dt/2)*vx(t)
      Compute a(t+dt/2) from positions x(t+dt/2)
      vx(t+dt) = vx(t) + dt * a(t+dt/2)
      x(t+dt) = x(t+dt/2) + (dt/2) * vx(t+dt)
    Record trajectory frames for animation and energy(t) for plotting.
    Save:
      - trajectory_{method}_{N}_{threads}.gif
      - energy_{method}_{N}_{threads}.png
    """
    x, y, vx, vy, m = generate_initial_conditions(N, domain)
    positions = []   # store (x,y) for each frame
    energies  = []   # store (t, E_rel) for each frame

    # Initial total energy E0
    E0 = compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads)

    for k in range(steps):
        t = k * dt
        # Half‐kick: advance positions by dt/2
        x_mid = x + 0.5 * dt * vx
        y_mid = y + 0.5 * dt * vy

        # Acceleration at mid‐step
        ax, ay = compute_accelerations(method, x_mid, y_mid, m, domain, theta, G, soft, threads)

        # Full kick: update velocities
        vx_new = vx + dt * ax
        vy_new = vy + dt * ay

        # Drift: update positions to full step
        x_new = x_mid + 0.5 * dt * vx_new
        y_new = y_mid + 0.5 * dt * vy_new

        # Update for next iteration
        x, y, vx, vy = x_new, y_new, vx_new, vy_new

        # Record positions for this frame
        positions.append((x.copy(), y.copy()))

        # Compute energy relative error
        E = compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads)
        rel_err = abs((E - E0) / E0)
        energies.append((t + dt, rel_err))

    # 1) Save trajectory as GIF
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    scat = ax.scatter([], [], s=10, c='blue')
    ax.set_title(f"Trajectory ({method.upper()}, N={N}, threads={threads})")

    def update_frame(i):
        arrx, arry = positions[i]
        scat.set_offsets(np.column_stack([arrx, arry]))
        return scat,

    ani = animation.FuncAnimation(fig, update_frame, frames=len(positions), interval=50, blit=True)
    gif_file = os.path.join(OUTPUT_DIR, f"trajectory_{method}_{N}_{threads}.gif")
    ani.save(gif_file, writer='pillow', fps=15)
    plt.close()
    print(f"✓ Saved trajectory GIF: {gif_file}")

    # 2) Save energy vs time plot (log‐scale y)
    times = [e[0] for e in energies]
    errs = [e[1] for e in energies]
    plt.figure(figsize=(6,4))
    plt.semilogy(times, errs, lw=2, label=method.upper())
    plt.axhline(1e-6, color='k', ls='--', lw=1, label='1e-6')
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy vs Time ({method.upper()}, N={N}, threads={threads})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, f"energy_{method}_{N}_{threads}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved energy plot: {plot_file}")


# ============================================
# === MENU OPTION #3: LIVE SIMULATION ANIMATION ===
# ============================================

def live_simulation(
    method,
    N,
    dt,
    steps,
    domain,
    theta,
    G,
    soft,
    threads
):
    """
    Similar to simulate_trajectory, but display on‐screen and save GIF.
    """
    x, y, vx, vy, m = generate_initial_conditions(N, domain)
    E0 = compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    scat = ax.scatter(x, y, s=10, c='blue')
    ax.set_title(f"Live Simulation ({method.upper()}, N={N}, threads={threads})")

    def update(i):
        nonlocal x, y, vx, vy, m
        # Leapfrog step
        x_mid = x + 0.5 * dt * vx
        y_mid = y + 0.5 * dt * vy
        ax_, ay_ = compute_accelerations(method, x_mid, y_mid, m, domain, theta, G, soft, threads)
        vx = vx + dt * ax_
        vy = vy + dt * ay_
        x = x_mid + 0.5 * dt * vx
        y = y_mid + 0.5 * dt * vy

        scat.set_offsets(np.column_stack([x, y]))
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    gif_file = os.path.join(OUTPUT_DIR, f"live_{method}_{N}_{threads}.gif")
    ani.save(gif_file, writer='pillow', fps=15)
    plt.close()
    print(f"✓ Saved live simulation GIF: {gif_file}")


# ================================================
# === MENU OPTION #5: ENERGY CONSERVATION TEST ===
# ================================================

def energy_conservation_test(
    methods,
    N,
    dt,
    steps,
    domain,
    theta,
    G,
    soft,
    threads
):
    """
    For each method in methods, run a short simulation (N, dt, steps, threads).
    At each time step, record the relative energy error.  After all steps,
    plot all three error‐vs‐time curves on a single log‐y plot, save “energy_conservation.png”.
    """
    plt.figure(figsize=(6,5))

    for method, color in zip(methods, ['r','b','g']):
        x, y, vx, vy, m = generate_initial_conditions(N, domain)
        E0 = compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads)

        times = []
        errs  = []
        for k in range(steps):
            t = k * dt
            # Leapfrog
            x_mid = x + 0.5 * dt * vx
            y_mid = y + 0.5 * dt * vy
            ax_, ay_ = compute_accelerations(method, x_mid, y_mid, m, domain, theta, G, soft, threads)
            vx = vx + dt * ax_
            vy = vy + dt * ay_
            x = x_mid + 0.5 * dt * vx
            y = y_mid + 0.5 * dt * vy

            E = compute_total_energy(method, x, y, vx, vy, m, domain, theta, G, soft, threads)
            rel_err = abs((E - E0) / E0)
            times.append(t + dt)
            errs.append(rel_err)

        plt.semilogy(times, errs, color=color, lw=2, label=method.upper())

    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy Conservation Test (N={N}, threads={threads})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, f"energy_conservation.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved energy‐conservation plot: {plot_file}")


# ======================================================
# === MENU OPTION #6: PARAMETER OPTIMIZATION (BH only) ===
# ======================================================

def parameter_optimization(
    domain_values,
    theta_values,
    N,
    G,
    soft,
    threads
):
    """
    For a fixed N and G,soft,threads, vary domain in domain_values and theta in theta_values,
    compute Barnes‐Hut error (||f_BH - f_direct|| / ||f_direct||) on a single random snapshot.
    We choose the combination with error < 0.05 that yields minimal BH time.
    Save a plot “parameter_optimization.png” showing relative error vs theta for each domain.
    """
    # Generate one random snapshot
    x, y, vx, vy, m = generate_initial_conditions(N, max(domain_values))
    # Compute “true” (direct) force
    ax_true, ay_true = force_kernel.direct_omp(x, y, m, G, soft)
    mag_true = np.sqrt(ax_true**2 + ay_true**2)

    results = {}  # results[(domain,theta)] = (error, time)
    best = None   # best = (domain,theta,error,time)

    print("\nTesting BH parameter combinations:")
    print("Domain  Theta   Error       Time(s)")
    print("-----------------------------------")
    for domain in domain_values:
        for theta in theta_values:
            # Measure BH time
            start = time.time()
            ax_bh, ay_bh = force_kernel.bh_omp(x, y, m, domain, theta, G, soft)
            elapsed = time.time() - start

            mag_bh = np.sqrt(ax_bh**2 + ay_bh**2)
            # compute relative L2‐norm error
            diff = mag_bh - mag_true
            err = np.linalg.norm(diff) / np.linalg.norm(mag_true)

            results[(domain,theta)] = (err, elapsed)

            print(f"{domain:<6.1f} {theta:<6.1f} {err:<10.3e} {elapsed:<8.4f}")

            if err < 0.05:
                if best is None or elapsed < best[3]:
                    best = (domain, theta, err, elapsed)

    if best is None:
        print("No combination found with error < 0.05")
    else:
        print("\nBest BH parameters:")
        print(f"  domain={best[0]:.1f}, theta={best[1]:.2f}, error={best[2]:.3e}, time={best[3]:.4f}s")

    # Plot relative error vs theta for each domain
    plt.figure(figsize=(6,5))
    for domain in domain_values:
        errs = [ results[(domain,theta)][0] for theta in theta_values ]
        plt.loglog(theta_values, errs, marker='o', label=f"domain={domain}")
    plt.axhline(0.05, color='k', ls='--', lw=1, label="0.05 error")
    plt.xlabel("Theta")
    plt.ylabel("Relative Force Error")
    plt.title(f"Parameter Optimization (N={N})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "parameter_optimization.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved plot to {plot_file}")


# =======================================================
# === MENU OPTION #7: OPENMP THREAD BENCHMARK (FMM only) ===
# =======================================================

def openmp_thread_benchmark_fmm(N, domain, theta, G, soft):
    """
    For N fixed, run FMM with threads = [1,2,4,8], measure time of fmm_omp once.
    Compute speedup relative to 1 thread. Save CSV and plot “openmp_thread_benchmark.csv/png”.
    """
    times = {}
    for thr in [1, 2, 4, 8]:
        x, y, vx, vy, m = generate_initial_conditions(N, domain)
        # Warm up
        compute_accelerations("fmm", x, y, m, domain, theta, G, soft, thr)

        start = time.time()
        ax, ay = compute_accelerations("fmm", x, y, m, domain, theta, G, soft, thr)
        elapsed = time.time() - start
        times[thr] = elapsed
        print(f"Threads={thr:<2d}  time={elapsed:.4e}s")

    base = times[1]
    speedups = { thr: base / times[thr] for thr in times }

    # Save CSV
    csv_file = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.csv")
    with open(csv_file, "w") as f:
        f.write("threads,time,speedup\n")
        for thr in sorted(times):
            f.write(f"{thr},{times[thr]},{speedups[thr]}\n")
    print(f"✓ Saved CSV to {csv_file}")

    # Plot
    plt.figure(figsize=(6,5))
    thr_list = sorted(speedups.keys())
    sp_list  = [ speedups[t] for t in thr_list ]
    plt.plot(thr_list, sp_list, 'ro-', label="Actual Speedup")
    plt.plot(thr_list, thr_list, 'k--', label="Ideal Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"OpenMP Thread Benchmark (FMM, N={N})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"✓ Saved plot to {plot_file}")


# =============================
# === MAIN MENU & ARG PARSER ===
# =============================

def main():
    print("\n=== 2D N-body Playground (Parallel, High-Precision) ===")
    print("Select option:")
    print("  1) Quick benchmark scaling")
    print("  2) Save trajectory + energy plot")
    print("  3) Live animation (real-time)")
    print("  4) Large-N scaling test")
    print("  5) Energy conservation test")
    print("  6) Parameter optimization")
    print("  7) OpenMP thread benchmark")
    print("  8) System information")
    print("  q) Quit")
    print("==============================================")

    while True:
        choice = input("\nEnter choice: ").strip()
        if choice == 'q':
            print("Goodbye!")
            break

        if choice == '1':
            # Quick small‐N scaling
            Ns      = [50, 100, 200, 500, 1000, 2000]
            methods = ["direct", "bh", "fmm"]
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter BH/FMM theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            threads = int(input("Enter #OpenMP threads (e.g. 4): ") or "4")
            benchmark_scaling_smallN(methods, Ns, domain, theta, G, soft, threads)

        elif choice == '2':
            # Save trajectory + energy plot
            method  = input("Select method (direct / bh / fmm): ").strip().lower()
            if method not in ["direct","bh","fmm"]:
                print("Invalid method.")
                continue
            N       = int(input("Enter N particles (e.g. 200): ") or "200")
            dt      = float(input("Enter dt (e.g. 0.001): ") or "0.001")
            steps   = int(input("Enter # steps/frames (e.g. 200): ") or "200")
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            threads = int(input("Enter #OpenMP threads: ") or "4")
            simulate_trajectory(method, N, dt, steps, domain, theta, G, soft, threads)

        elif choice == '3':
            # Live simulation animation
            method  = input("Select method (direct / bh / fmm): ").strip().lower()
            if method not in ["direct","bh","fmm"]:
                print("Invalid method.")
                continue
            N       = int(input("Enter N particles (e.g. 100): ") or "100")
            dt      = float(input("Enter dt (e.g. 0.001): ") or "0.001")
            steps   = int(input("Enter # steps/frames (e.g. 200): ") or "200")
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            threads = int(input("Enter #OpenMP threads: ") or "4")
            live_simulation(method, N, dt, steps, domain, theta, G, soft, threads)

        elif choice == '4':
            # Large‐N scaling test
            method_list = ["direct", "bh", "fmm"]
            Ns      = [500, 1000, 2000, 3000, 4000]
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            threads = int(input("Enter #OpenMP threads: ") or "4")
            benchmark_scaling_largeN(method_list, Ns, domain, theta, G, soft, threads)

        elif choice == '5':
            # Energy conservation test
            methods = ["direct", "bh", "fmm"]
            N       = int(input("Enter N particles (e.g. 200): ") or "200")
            dt      = float(input("Enter dt (e.g. 0.001): ") or "0.001")
            steps   = int(input("Enter # steps (e.g. 200): ") or "200")
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            threads = int(input("Enter #OpenMP threads: ") or "4")
            energy_conservation_test(methods, N, dt, steps, domain, theta, G, soft, threads)

        elif choice == '6':
            # Parameter optimization (BH only)
            domain_values = list(map(float, input("Enter domain values (comma‐sep, e.g. 50,100,200): ").split(',')))
            theta_values  = list(map(float, input("Enter theta values (comma‐sep, e.g. 0.1,0.3,0.5,0.7,1.0): ").split(',')))
            N             = int(input("Enter N particles (e.g. 100): ") or "100")
            G             = float(input("Enter G (default=1.0): ") or "1.0")
            soft          = float(input("Enter soft (default=0.05): ") or "0.05")
            threads       = int(input("Enter #OpenMP threads: ") or "4")
            parameter_optimization(domain_values, theta_values, N, G, soft, threads)

        elif choice == '7':
            # OpenMP thread benchmark (FMM only)
            N       = int(input("Enter N particles (e.g. 500): ") or "500")
            domain  = float(input("Enter domain (e.g. 100.0): ") or "50.0")
            theta   = float(input("Enter theta (e.g. 0.5): ") or "0.5")
            G       = float(input("Enter G (default=1.0): ") or "1.0")
            soft    = float(input("Enter soft (default=0.05): ") or "0.05")
            openmp_thread_benchmark_fmm(N, domain, theta, G, soft)

        elif choice == '8':
            # System information (OpenMP, versions)
            get_openmp_info()

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

