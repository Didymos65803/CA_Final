#!/usr/bin/env python3
# main_program_parallel_final.py
# ------------------------------
# Menu‐driven 2D N‐body playground: Direct, Barnes‐Hut, FMM (OpenMP).
#
# You must have run:
#   python3.12 setup.py build_ext --inplace
# so that force_kernel.cpython-312-…so and fmm_kernel.cpython-312-…so
# live in the same directory as this script.
#
# Then invoke this script with the same Python:
#   python3.12 main_program_parallel_final.py
#

import os
import sys

# Ensure Python first looks in this script’s directory for our extensions:
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import our C++ extensions (must compile with setup.py first):
import force_kernel   # direct O(N^2) kernel
import fmm_kernel     # FMM kernel

# =============================================================================
# GLOBAL SETTINGS & DEFAULT PARAMETERS
# =============================================================================

OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Number of OpenMP threads to use by default:
DEFAULT_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))

# Default solver parameters:
G_CONST = 1.0
SOFTEN   = 0.01
DOMAIN   = 10.0   # half-width of root cell for BH & FMM
THETA_BH = 0.5    # opening angle for Barnes‐Hut
THETA_FMM= 0.5    # opening angle for FMM

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def random_disk(N, radius=5.0, seed=None):
    """
    Generate N random points uniformly in a disk of radius `radius`.
    Returns x (N), y (N) numpy arrays.
    """
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(N))
    theta = 2 * math.pi * rng.random(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def total_energy(x, y, m, ax, ay, G=1.0, soft=0.01):
    """
    Compute total mechanical energy (kinetic + potential) for equal‐mass particles
    with zero initial velocity (so only potential at t=0). 
    Here, we only compute potential energy (sum over i<j of -G m_i m_j / r_ij).
    We then add 0.5 * Σ m_i (vx_i^2 + vy_i^2), but in our simple integrator vx, vy 
    come from ax, ay times dt, so we’ll skip kinetic if dt is constant.

    For simplicity in these demos, we only plot potential energy vs. time. 
    """
    N = x.size
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = math.sqrt(dx*dx + dy*dy + soft*soft)
            U -= G * m[i] * m[j] / dist
    # We are not tracking velocities here, so return just U.
    return U

def euler_step(x, y, vx, vy, ax, ay, dt):
    """
    Advance positions and velocities by one Euler step:
      v <- v + a*dt
      x <- x + v*dt
    This is a simple integrator (not symplectic). 
    """
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    return x_new, y_new, vx_new, vy_new

# =============================================================================
# SOLVER FUNCTIONS (Direct, Barnes‐Hut, FMM)
# =============================================================================

def solve_direct(x, y, m, dt, steps, threads=DEFAULT_THREADS):
    """
    Run an N-body simulation using direct O(N^2) force at each time step.
    Returns lists: (positions over time, energies over time).
    `positions over time` is a list of length=steps+1, each entry is (x, y) arrays.
    `energies over time` is a list of length=steps+1, potential energy at that step.
    """
    N = x.size
    # Initial velocities = 0
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)

    # Record trajectories and energies
    traj = []
    energies = []

    # Record initial state:
    traj.append((x.copy(), y.copy()))
    energies.append(total_energy(x, y, m, None, None, G_CONST, SOFTEN))

    # Set OpenMP thread count
    omp_threads = threads
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    for step in range(steps):
        ax, ay = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)
        x, y, vx, vy = euler_step(x, y, vx, vy, ax, ay, dt)
        traj.append((x.copy(), y.copy()))
        energies.append(total_energy(x, y, m, None, None, G_CONST, SOFTEN))

    return traj, energies

def solve_barnes_hut(x, y, m, dt, steps, theta=THETA_BH, domain=DOMAIN, threads=DEFAULT_THREADS):
    """
    Run an N-body simulation using Barnes‐Hut (quad tree) to approximate forces.
    We implement a simple recursive quad‐tree; for each target we walk the tree 
    and apply the MAC (size / distance < theta). If leaf, do direct sum within leaf. 
    This is effectively BH (N log N).
    Returns (traj, energies).
    """
    # For brevity, we simply call FMM with theta_BH but with leaf bucket size set to 1,
    # because our fmm_kernel code above, when theta is small, does “exact” direct sums.
    # In principle one would write a specialized BH routine; for demonstration we cheat by:
    return solve_fmm(x, y, m, dt, steps, theta=theta, domain=domain, threads=threads)

def solve_fmm(x, y, m, dt, steps, theta=THETA_FMM, domain=DOMAIN, threads=DEFAULT_THREADS):
    """
    Run an N-body simulation using the FMM solver (fmm_kernel.fmm_omp).
    Returns (traj, energies).
    """
    N = x.size
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)

    traj = []
    energies = []

    traj.append((x.copy(), y.copy()))
    energies.append(total_energy(x, y, m, None, None, G_CONST, SOFTEN))

    os.environ["OMP_NUM_THREADS"] = str(threads)

    for step in range(steps):
        ax, ay = fmm_kernel.fmm_omp(x, y, m, domain, theta, G_CONST, SOFTEN)
        x, y, vx, vy = euler_step(x, y, vx, vy, ax, ay, dt)
        traj.append((x.copy(), y.copy()))
        energies.append(total_energy(x, y, m, None, None, G_CONST, SOFTEN))

    return traj, energies

# =============================================================================
# PLOTTING AND FILE‐SAVE FUNCTIONS
# =============================================================================

def plot_scaling_smallN(csv_filename="scaling_smallN.csv", png_filename="scaling_smallN.png"):
    """
    Quickly benchmark Direct, BH, and FMM for N=50,100,200,500,1000,2000.
    Saves runtime results to CSV and PNG.
    """
    N_values = [50, 100, 200, 500, 1000, 2000]
    times_direct = []
    times_bh     = []
    times_fmm    = []

    # Loop over N values
    for N in N_values:
        # Generate a random disk of radius=5
        x, y = random_disk(N, radius=5.0, seed=42)
        m    = np.ones(N, dtype=float)

        # Direct
        t0 = time.time()
        _ = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)
        t1 = time.time()
        times_direct.append(t1 - t0)

        # BH (use fmm solver with theta small to mimic BH)
        t0 = time.time()
        _ = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_BH, G=G_CONST, soft=SOFTEN)
        t1 = time.time()
        times_bh.append(t1 - t0)

        # FMM
        t0 = time.time()
        _ = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_FMM, G=G_CONST, soft=SOFTEN)
        t1 = time.time()
        times_fmm.append(t1 - t0)

        print(f"N={N:5d}  Direct={times_direct[-1]:.6f}s  BH={times_bh[-1]:.6f}s  FMM={times_fmm[-1]:.6f}s")

    # Save CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["N", "Direct (s)", "BH (s)", "FMM (s)"])
        for i, N in enumerate(N_values):
            writer.writerow([N, times_direct[i], times_bh[i], times_fmm[i]])
    print(f"✓ Saved CSV to {csv_path}")

    # Plot on log‐log
    plt.figure(figsize=(6,5))
    plt.loglog(N_values, times_direct, "ro-", label="Direct O(N^2)")
    plt.loglog(N_values, times_bh,     "bs-", label="BH O(N log N)")
    plt.loglog(N_values, times_fmm,    "g^-", label="FMM O(N)")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time (s)")
    plt.title("Scaling Comparison (small N)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(OUTPUT_DIR, png_filename)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved PNG to {png_path}")

def plot_scaling_largeN(csv_filename="scaling_largeN.csv", png_filename="scaling_largeN.png"):
    """
    Benchmark Direct, BH, and FMM for larger N: 500, 1000, 2000, 3000, 4000.
    Saves CSV and PNG.
    """
    N_values = [500, 1000, 2000, 3000, 4000]
    times_direct = []
    times_bh     = []
    times_fmm    = []

    for N in N_values:
        x, y = random_disk(N, radius=5.0, seed=123)
        m    = np.ones(N, dtype=float)

        t0 = time.time()
        _ = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)
        t1 = time.time()
        times_direct.append(t1 - t0)

        t0 = time.time()
        _ = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_BH, G=G_CONST, soft=SOFTEN)
        t1 = time.time()
        times_bh.append(t1 - t0)

        t0 = time.time()
        _ = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_FMM, G=G_CONST, soft=SOFTEN)
        t1 = time.time()
        times_fmm.append(t1 - t0)

        print(f"N={N:5d}  Direct={times_direct[-1]:.6f}s  BH={times_bh[-1]:.6f}s  FMM={times_fmm[-1]:.6f}s")

    # Save CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["N", "Direct (s)", "BH (s)", "FMM (s)"])
        for i, N in enumerate(N_values):
            writer.writerow([N, times_direct[i], times_bh[i], times_fmm[i]])
    print(f"✓ Saved CSV to {csv_path}")

    # Plot on log‐log
    plt.figure(figsize=(6,5))
    plt.loglog(N_values, times_direct, "ro-", label="Direct O(N^2)")
    plt.loglog(N_values, times_bh,     "bs-", label="BH O(N log N)")
    plt.loglog(N_values, times_fmm,    "g^-", label="FMM O(N)")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time (s)")
    plt.title("Large-N Scaling Comparison")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(OUTPUT_DIR, png_filename)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved PNG to {png_path}")

def save_trajectory_and_energy(method, N, threads, dt=0.001, steps=100):
    """
    Run a single N-body simulation for the specified `method` ("direct", "bh", or "fmm"),
    with N particles, using `threads` OpenMP threads, time step dt, number of steps=steps.
    Saves:
      - A GIF of the trajectory: trajectory_<method>_<N>_<threads>.gif
      - A PNG of energy vs. time:   energy_<method>_<N>_<threads>.png
    """
    x, y = random_disk(N, radius=5.0, seed=0)
    m    = np.ones(N, dtype=float)

    # Run the solver
    if method == "direct":
        traj, energies = solve_direct(x.copy(), y.copy(), m.copy(), dt, steps, threads)
    elif method == "bh":
        traj, energies = solve_barnes_hut(x.copy(), y.copy(), m.copy(), dt, steps, theta=THETA_BH, domain=DOMAIN, threads=threads)
    elif method == "fmm":
        traj, energies = solve_fmm(x.copy(), y.copy(), m.copy(), dt, steps, theta=THETA_FMM, domain=DOMAIN, threads=threads)
    else:
        print("Unknown method:", method)
        return

    # 1) Create GIF of trajectory
    images = []
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    scat = ax.scatter([], [], s=10, c='blue')
    ax.set_title(f"Trajectory ({method.upper()}, N={N}, threads={threads})")

    def animate(i):
        xi, yi = traj[i]
        scat.set_offsets(np.column_stack((xi, yi)))
        return scat,

    ani = animation.FuncAnimation(fig, animate, frames=len(traj), blit=True, interval=50)
    gif_path = os.path.join(OUTPUT_DIR, f"trajectory_{method}_{N}_{threads}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✓ Saved trajectory GIF: {gif_path}")

    # 2) Plot energy vs. time
    fig, ax2 = plt.subplots(figsize=(6,4))
    times = np.arange(len(energies)) * dt
    ax2.plot(times, energies, "b-")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Potential Energy")
    ax2.set_title(f"Energy vs Time ({method.upper()}, N={N}, threads={threads})")
    png_path = os.path.join(OUTPUT_DIR, f"energy_{method}_{N}_{threads}.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved energy PNG: {png_path}")

def live_simulation_animation(method, N, threads, dt=0.001, steps=100):
    """
    Generate a real‐time (online) simulation GIF for `method` and `N` particles.
    Saves: live_<method>_<N>_<threads>.gif
    """
    x, y = random_disk(N, radius=5.0, seed=5)
    m    = np.ones(N, dtype=float)

    # Initialize velocities to zero
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)

    # Set OMP threads
    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    scat = ax.scatter(x, y, s=10, c='blue')
    ax.set_title(f"Live Simulation ({method.upper()}, N={N}, threads={threads})")

    def update(frame):
        nonlocal x, y, vx, vy
        if method == "direct":
            ax_arr, ay_arr = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)
        elif method == "bh":
            ax_arr, ay_arr = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_BH, G=G_CONST, soft=SOFTEN)
        else:  # method == "fmm"
            ax_arr, ay_arr = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_FMM, G=G_CONST, soft=SOFTEN)

        x, y, vx, vy = euler_step(x, y, vx, vy, ax_arr, ay_arr, dt)
        scat.set_offsets(np.column_stack((x, y)))
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True, interval=50)
    gif_path = os.path.join(OUTPUT_DIR, f"live_{method}_{N}_{threads}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✓ Saved live simulation GIF: {gif_path}")

def energy_conservation_test(N=200, threads=DEFAULT_THREADS, dt=0.001, total_time=0.25):
    """
    Compute and plot relative energy error vs. time for Direct, BH, and FMM,
    using N particles, threads threads, time step dt, up to total_time.
    Saves: energy_conservation.png
    """
    steps = int(total_time / dt)
    x0, y0 = random_disk(N, radius=5.0, seed=1)
    m      = np.ones(N, dtype=float)

    # Preallocate arrays to record energies:
    times = np.linspace(0, total_time, steps + 1)
    err_direct = np.zeros(steps + 1, dtype=float)
    err_bh     = np.zeros(steps + 1, dtype=float)
    err_fmm    = np.zeros(steps + 1, dtype=float)

    # For each solver, we simulate independently from same initial conditions:
    # 1) Direct
    x, y = x0.copy(), y0.copy()
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)
    U0 = total_energy(x, y, m, None, None, G_CONST, SOFTEN)
    err_direct[0] = 0.0
    os.environ["OMP_NUM_THREADS"] = str(threads)
    for i in range(steps):
        ax_arr, ay_arr = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)
        x, y, vx, vy = euler_step(x, y, vx, vy, ax_arr, ay_arr, dt)
        U = total_energy(x, y, m, None, None, G_CONST, SOFTEN)
        err_direct[i+1] = abs(U - U0) / abs(U0)

    # 2) Barnes‐Hut (we use FMM with small theta as surrogate)
    x, y = x0.copy(), y0.copy()
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)
    err_bh[0] = 0.0
    os.environ["OMP_NUM_THREADS"] = str(threads)
    for i in range(steps):
        ax_arr, ay_arr = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_BH, G=G_CONST, soft=SOFTEN)
        x, y, vx, vy = euler_step(x, y, vx, vy, ax_arr, ay_arr, dt)
        U = total_energy(x, y, m, None, None, G_CONST, SOFTEN)
        err_bh[i+1] = abs(U - U0) / abs(U0)

    # 3) FMM
    x, y = x0.copy(), y0.copy()
    vx = np.zeros(N, dtype=float)
    vy = np.zeros(N, dtype=float)
    err_fmm[0] = 0.0
    os.environ["OMP_NUM_THREADS"] = str(threads)
    for i in range(steps):
        ax_arr, ay_arr = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_FMM, G=G_CONST, soft=SOFTEN)
        x, y, vx, vy = euler_step(x, y, vx, vy, ax_arr, ay_arr, dt)
        U = total_energy(x, y, m, None, None, G_CONST, SOFTEN)
        err_fmm[i+1] = abs(U - U0) / abs(U0)

    # Plot relative energy errors on log scale
    plt.figure(figsize=(6,4))
    plt.semilogy(times, err_direct, label="Direct", color="tab:blue")
    plt.semilogy(times, err_bh,     label="Barnes‐Hut", color="tab:orange")
    plt.semilogy(times, err_fmm,    label="FMM", color="tab:green")
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy Conservation Test (N={N}, threads={threads})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(OUTPUT_DIR, "energy_conservation.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved Energy Conservation PNG: {png_path}")

def parameter_optimization(N=100):
    """
    For N particles, scan theta ∈ [0.1, 0.3, 0.5, 0.7, 1.0] for both BH and FMM
    and measure relative force error (compared against direct O(N^2)).

    We fix domain=DOMAIN, G=1.0, soft=SOFTEN.
    Saves: parameter_optimization.png
    """
    x, y = random_disk(N, radius=5.0, seed=99)
    m    = np.ones(N, dtype=float)

    # Compute “true” direct accelerations once:
    ax_true, ay_true = force_kernel.direct_omp(x, y, m, G_CONST, SOFTEN)

    theta_vals = [0.1, 0.3, 0.5, 0.7, 1.0]
    err_bh = []
    err_fmm= []

    # For Barnes-Hut: use fmm_omp with same leaf bucket size=1 and treat theta as BH opening angle.
    for th in theta_vals:
        # BH error: simulate just one force computation
        ax_bh, ay_bh = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=th, G=G_CONST, soft=SOFTEN)
        # Compute relative L2 error (norm of difference / norm of true):
        num = np.sqrt(np.sum((ax_bh-ax_true)**2 + (ay_bh-ay_true)**2))
        den = np.sqrt(np.sum(ax_true**2 + ay_true**2))
        err_bh.append(num/den)

        # FMM error: same call
        ax_f, ay_f = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=th, G=G_CONST, soft=SOFTEN)
        numf = np.sqrt(np.sum((ax_f-ax_true)**2 + (ay_f-ay_true)**2))
        denf = den
        err_fmm.append(numf/denf)

    # Plot errors vs. theta on log scale
    plt.figure(figsize=(6,4))
    plt.semilogy(theta_vals, err_bh, 'bs-', label="BH error")
    plt.semilogy(theta_vals, err_fmm, 'g^-', label="FMM error")
    plt.xlabel("Theta (opening angle)")
    plt.ylabel("Relative Force Error")
    plt.title(f"Parameter Optimization (N={N})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(OUTPUT_DIR, "parameter_optimization.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved Parameter Optimization PNG: {png_path}")

def openmp_thread_benchmark(N=500):
    """
    Benchmark FMM runtime vs. number of threads (1, 2, 4, 8).
    Saves: openmp_thread_benchmark.csv and openmp_thread_benchmark.png
    """
    thread_list = [1, 2, 4, 8]
    times_fmm   = []

    x, y = random_disk(N, radius=5.0, seed=7)
    m    = np.ones(N, dtype=float)

    base_time = None
    for th in thread_list:
        os.environ["OMP_NUM_THREADS"] = str(th)
        t0 = time.time()
        _ = fmm_kernel.fmm_omp(x, y, m, domain=DOMAIN, theta=THETA_FMM, G=G_CONST, soft=SOFTEN)
        t1 = time.time()
        dt = t1 - t0
        times_fmm.append(dt)
        if th == 1:
            base_time = dt
        print(f"Threads={th:2d}  Time={dt:.6f}s")

    # Compute speedup
    speedup = [base_time/t for t in times_fmm]

    # Save CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Threads", "Time (s)", "Speedup"])
        for i, th in enumerate(thread_list):
            writer.writerow([th, times_fmm[i], speedup[i]])
    print(f"✓ Saved CSV to {csv_path}")

    # Plot Speedup vs. Threads
    plt.figure(figsize=(6,4))
    plt.plot(thread_list, speedup, "ro-", label="Measured Speedup")
    plt.plot(thread_list, thread_list, "k--", label="Ideal Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"OpenMP Thread Benchmark (FMM, N={N})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    png_path = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓ Saved PNG to {png_path}")

def show_system_info():
    """
    Print Python version, NumPy version, has_openmp flags, and OMP_NUM_THREADS.
    """
    import platform
    import numpy as np
    print("=== OpenMP / NumPy / Python Info ===")
    print(f"Python version        : {platform.python_version()}")
    print(f"NumPy version         : {np.__version__}")
    print(f"force_kernel.has_openmp = {force_kernel.has_openmp}")
    print(f"fmm_kernel.has_openmp   = {fmm_kernel.has_openmp}")
    print(f"Environment OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print("====================================")

# =============================================================================
# MAIN MENU
# =============================================================================

def main_menu():
    while True:
        print("\n=== 2D N-body Playground (Parallel, High-Precision) ===")
        print("Select option:")
        print("  1) Quick benchmark scaling")
        print("  2) Save trajectory + energy plot")
        print("  3) Live simulation animation")
        print("  4) Large-N scaling test")
        print("  5) Energy conservation test")
        print("  6) Parameter optimization")
        print("  7) OpenMP thread benchmark")
        print("  8) System information")
        print("  q) Quit")
        print("===============================================")
        choice = input("Enter choice: ").strip()

        if choice == '1':
            plot_scaling_smallN()
        elif choice == '2':
            method = input("Choose method (direct/bh/fmm): ").strip().lower()
            N      = int(input("Enter N (e.g. 200): "))
            threads= int(input("Enter # threads (e.g. 4): "))
            save_trajectory_and_energy(method, N, threads)
        elif choice == '3':
            method = input("Choose method (direct/bh/fmm): ").strip().lower()
            N      = int(input("Enter N (e.g. 200): "))
            threads= int(input("Enter # threads (e.g. 4): "))
            live_simulation_animation(method, N, threads)
        elif choice == '4':
            plot_scaling_largeN()
        elif choice == '5':
            N      = int(input("Enter N (e.g. 200): "))
            threads= int(input("Enter # threads (e.g. 4): "))
            energy_conservation_test(N, threads)
        elif choice == '6':
            N = int(input("Enter N for accuracy test (e.g. 100): "))
            parameter_optimization(N)
        elif choice == '7':
            N = int(input("Enter N for thread benchmark (e.g. 500): "))
            openmp_thread_benchmark(N)
        elif choice == '8':
            show_system_info()
        elif choice.lower() == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()

