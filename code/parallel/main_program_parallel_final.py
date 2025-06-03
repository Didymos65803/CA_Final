#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_program_parallel_final.py
==============================
Interactive 2D N-body Playground (Parallel, High-Precision)

Menu (8 options):
 1) Quick benchmark scaling
 2) Save trajectory + energy plot
 3) Live animation (real-time)
 4) Large-N scaling test
 5) Energy conservation test
 6) Parameter optimization
 7) OpenMP thread benchmark
 8) System information
  q) Quit

Key changes in this version (English):
  - Optional log‐file input at startup (type a filename to log all output; press Enter to skip).
  - Menu options 2 & 3 now “prompt line by line” for solver, N, steps, threads, fixed‐star.
  - All generated files placed in subfolder “output/” (remove OUTPUT_DIR lines if you prefer current dir).
"""

import os
import sys
import math
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Optional Logging (to file + console)
# ----------------------------
LOG_FILE = None

def log_print(*args, **kwargs):
    """
    Print to stdout and simultaneously append to LOG_FILE if it is set.
    """
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    sys.stdout.write(text)
    sys.stdout.flush()
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a") as f:
                f.write(text)
        except Exception as e:
            sys.stdout.write(f"[Warning] Cannot write to log file: {e}\n")
            sys.stdout.flush()

# ----------------------------
# Import C++ modules
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
    bh_omp = None

# ----------------------------
# Import comprehensive_test (all the functions above)
# ----------------------------
try:
    import comprehensive_test as ctest
except ImportError:
    ctest = None

# ----------------------------
# Global physics constants (same as in comprehensive_test.py)
# ----------------------------
G = 1.0
SOFT = 0.005
DOMAIN = 100.0
DT = 0.0005
STAR_M = 100.0

OPTIMIZED_PARAMS = {
    "bh_theta": 0.3,
    "fmm_theta": 0.2,
    "bh_domain": DOMAIN,
    "fmm_domain": DOMAIN,
    "distribution_size": 50.0,
}

# Default solver selection
USE_SOLVER = "DIRECT"

# If you want all outputs in a subfolder, uncomment these two lines:
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def leapfrog_step(bodies, ax, ay, include_central=False):
    """
    Perform one Leapfrog step on `bodies` (shape (N,5) = [x,y,vx,vy,m]).
    `ax`, `ay` are current accelerations. Returns new (ax, ay).
    """
    N = bodies.shape[0]

    # 1) half-kick
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 2] += 0.5 * DT * ax[i]
        bodies[i, 3] += 0.5 * DT * ay[i]

    # 2) drift
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 0] += DT * bodies[i, 2]
        bodies[i, 1] += DT * bodies[i, 3]

    # 3) compute new accel
    x_list = bodies[:, 0].tolist()
    y_list = bodies[:, 1].tolist()
    m_list = bodies[:, 4].tolist()

    if USE_SOLVER.upper() == "DIRECT":
        if not HAS_DIRECT_BH:
            raise RuntimeError("Direct/BH module not loaded!")
        ax_new, ay_new = direct_omp(x_list, y_list, m_list, G=G, soft=SOFT)

    elif USE_SOLVER.upper() == "BH":
        if not HAS_DIRECT_BH:
            raise RuntimeError("Direct/BH module not loaded!")
        theta = OPTIMIZED_PARAMS["bh_theta"]
        ax_new, ay_new = bh_omp(x_list, y_list, m_list,
                                OPTIMIZED_PARAMS["bh_domain"],
                                theta, G, SOFT)

    elif USE_SOLVER.upper() == "FMM":
        if not HAS_FMM:
            raise RuntimeError("FMM module not loaded!")
        theta = OPTIMIZED_PARAMS["fmm_theta"]
        ax_new, ay_new = fmm_kernel.fmm_omp(x_list, y_list, m_list,
                                            OPTIMIZED_PARAMS["fmm_domain"],
                                            theta, G, SOFT)
    else:
        raise ValueError(f"Unknown solver: {USE_SOLVER}")

    # 4) half-kick
    for i in range(N):
        if include_central and i == 0:
            continue
        bodies[i, 2] += 0.5 * DT * ax_new[i]
        bodies[i, 3] += 0.5 * DT * ay_new[i]

    return ax_new, ay_new


def total_energy(bodies, include_central=False):
    """
    Compute the total energy of `bodies` = [x,y,vx,vy,m].
    If include_central=True, skip index 0 for KE and include in PE.
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
            dx = bodies[i, 0] - bodies[j, 0]
            dy = bodies[i, 1] - bodies[j, 1]
            dist = math.hypot(dx, dy) + SOFT
            PE -= G * bodies[i, 4] * bodies[j, 4] / dist
    return KE + PE


def generate_disk(n, radius=OPTIMIZED_PARAMS["distribution_size"], include_central=False):
    """
    Generate n bodies in a disk of radius `radius`. If include_central=True,
    body[0] is a central star at the origin with mass STAR_M. Return a numpy array
    of shape (N_total, 5) = [x,y,vx,vy,m].
    """
    if include_central:
        N_total = n + 1
    else:
        N_total = n

    bodies = np.zeros((N_total, 5), dtype=np.float64)
    if include_central:
        bodies[0] = [0.0, 0.0, 0.0, 0.0, STAR_M]

    for i in range(1 if include_central else 0, N_total):
        r = math.sqrt(random.random()) * radius
        theta = 2.0 * math.pi * random.random()
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        bodies[i, 0] = x
        bodies[i, 1] = y
        bodies[i, 2] = 0.0
        bodies[i, 3] = 0.0
        bodies[i, 4] = 1.0
    return bodies


def main_menu():
    global USE_SOLVER, LOG_FILE

    # 1) Ask for an optional log file name
    log_print("Enter log filename (or press Enter to skip): ", end="")
    choice = input().strip()
    if choice != "":
        LOG_FILE = choice
        # Overwrite any existing file
        try:
            with open(LOG_FILE, "w") as f:
                f.write(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            LOG_FILE = None
            print(f"[Warning] Cannot create log file '{choice}': {e}\n")

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
    log_print(menu_text)

    while True:
        try:
            log_print("Enter choice: ", end="")
            choice = input().strip().lower()

            if choice in ["1", "benchmark"]:
                # Option 1: Quick benchmark scaling (small N)
                if ctest is not None:
                    ctest.test_scaling()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["2", "save", "save trajectory"]:
                # Option 2: Save trajectory + energy plot (step-by-step input)
                log_print("\n[Option 2] Save trajectory + energy plot")

                # Prompt for solver
                log_print("Enter Solver (direct / bh / fmm): ", end="")
                sol = input().strip().lower()
                if sol not in ["direct", "bh", "fmm"]:
                    log_print("Invalid solver. Choose direct, bh, or fmm.\n")
                    continue
                USE_SOLVER = sol

                # Prompt for N
                log_print("Enter number of particles N (e.g. 200): ", end="")
                try:
                    N = int(input().strip())
                except:
                    log_print("Invalid N. Must be integer.\n")
                    continue

                # Prompt for STEPS
                log_print("Enter number of integration steps STEPS (e.g. 2000): ", end="")
                try:
                    STEPS = int(input().strip())
                except:
                    log_print("Invalid STEPS. Must be integer.\n")
                    continue

                # Prompt for THREADS
                log_print("Enter number of OpenMP threads THREADS (e.g. 4): ", end="")
                try:
                    THREADS = int(input().strip())
                except:
                    log_print("Invalid THREADS. Must be integer.\n")
                    continue

                # Prompt for fixed central star
                log_print("Include fixed central star? (y / n): ", end="")
                ans = input().strip().lower()
                include_central = (ans == "y")

                # Set OMP_NUM_THREADS
                os.environ["OMP_NUM_THREADS"] = str(THREADS)

                # Generate initial configuration
                bodies = generate_disk(N, OPTIMIZED_PARAMS["distribution_size"], include_central=include_central)
                total_N = bodies.shape[0]
                E0 = total_energy(bodies, include_central=include_central)

                # Compute initial acceleration
                x0 = bodies[:, 0].tolist()
                y0 = bodies[:, 1].tolist()
                m0 = bodies[:, 4].tolist()

                try:
                    if USE_SOLVER.upper() == "DIRECT":
                        if not HAS_DIRECT_BH:
                            raise RuntimeError("Direct/BH module not loaded.")
                        ax0, ay0 = direct_omp(x0, y0, m0, G=G, soft=SOFT)

                    elif USE_SOLVER.upper() == "BH":
                        if not HAS_DIRECT_BH:
                            raise RuntimeError("Direct/BH module not loaded.")
                        theta = OPTIMIZED_PARAMS["bh_theta"]
                        ax0, ay0 = bh_omp(x0, y0, m0, OPTIMIZED_PARAMS["bh_domain"], theta, G, SOFT)

                    else:  # FMM
                        if not HAS_FMM:
                            raise RuntimeError("FMM module not loaded.")
                        theta = OPTIMIZED_PARAMS["fmm_theta"]
                        ax0, ay0 = fmm_kernel.fmm_omp(x0, y0, m0, OPTIMIZED_PARAMS["fmm_domain"], theta, G, SOFT)

                except Exception as e:
                    log_print(f"Error computing initial acceleration: {e}\n")
                    continue

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

                        ax, ay = leapfrog_step(bodies, ax, ay, include_central=include_central)

                        if s % 10 == 0:
                            E = total_energy(bodies, include_central=include_central)
                            rel_error = abs(E - E0) / (abs(E0) + 1e-16)
                            E_list.append((s * DT, E, rel_error))

                except KeyboardInterrupt:
                    log_print("\nIntegration interrupted by user.\n")
                except Exception as e:
                    log_print(f"\nError during integration: {e}\n")
                    continue

                # (a) Save trajectory GIF
                if xs and ys:
                    log_print("Creating animation GIF...")
                    fig, axg = plt.subplots(figsize=(8, 8))
                    points = []
                    if include_central:
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

                    def update_traj(frame):
                        for idx, pt in enumerate(points):
                            pt.set_data([xs[frame][idx]], [ys[frame][idx]])
                        return points

                    import matplotlib.animation as animation
                    ani = animation.FuncAnimation(fig, update_traj, frames=len(xs), interval=50, blit=True)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    gif_name = f"trajectory_{USE_SOLVER}_{N}_{THREADS}_{timestamp}.gif"
                    outpath = os.path.join(OUTPUT_DIR, gif_name)
                    ani.save(outpath, fps=20, dpi=80)
                    plt.close(fig)
                    log_print(f"✓ Saved trajectory GIF: {outpath}")

                # (b) Save energy-vs-time plot
                if E_list:
                    log_print("Creating energy vs time plot...")
                    times = [item[0] for item in E_list]
                    energies = [item[1] for item in E_list]

                    fig, axp = plt.subplots(figsize=(8, 4))
                    axp.plot(times, energies, label="Total Energy")
                    axp.axhline(E0, color="gray", linestyle="--", label="Initial Energy")
                    axp.set_xlabel("Time")
                    axp.set_ylabel("Energy")
                    axp.set_title(f"Energy vs Time ({USE_SOLVER.upper()}, N={N}, threads={THREADS})")
                    axp.legend()
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    plot_name = f"energy_{USE_SOLVER}_{N}_{THREADS}_{timestamp}.png"
                    outpath = os.path.join(OUTPUT_DIR, plot_name)
                    fig.tight_layout()
                    fig.savefig(outpath, dpi=150)
                    plt.close(fig)
                    log_print(f"✓ Saved energy plot: {outpath}\n")

            elif choice in ["3", "live", "live animation"]:
                # Option 3: Live animation (real‐time)
                log_print("\n[Option 3] Live simulation animation")

                # Prompt for solver
                log_print("Enter Solver (direct / bh / fmm): ", end="")
                sol = input().strip().lower()
                if sol not in ["direct", "bh", "fmm"]:
                    log_print("Invalid solver. Choose direct, bh, or fmm.\n")
                    continue
                USE_SOLVER = sol

                # Prompt for N
                log_print("Enter number of particles N (e.g. 100): ", end="")
                try:
                    N = int(input().strip())
                except:
                    log_print("Invalid N. Must be integer.\n")
                    continue

                # Prompt for STEPS
                log_print("Enter number of integration steps STEPS (e.g. 500): ", end="")
                try:
                    STEPS = int(input().strip())
                except:
                    log_print("Invalid STEPS. Must be integer.\n")
                    continue

                # Prompt for THREADS
                log_print("Enter number of OpenMP threads THREADS (e.g. 8): ", end="")
                try:
                    THREADS = int(input().strip())
                except:
                    log_print("Invalid THREADS. Must be integer.\n")
                    continue

                # Prompt for fixed central star
                log_print("Include fixed central star? (y / n): ", end="")
                ans = input().strip().lower()
                include_central = (ans == "y")

                # Set OMP_NUM_THREADS
                os.environ["OMP_NUM_THREADS"] = str(THREADS)

                bodies = generate_disk(N, OPTIMIZED_PARAMS["distribution_size"], include_central=include_central)
                total_N = bodies.shape[0]
                x0 = bodies[:, 0].tolist()
                y0 = bodies[:, 1].tolist()
                m0 = bodies[:, 4].tolist()

                try:
                    if USE_SOLVER.upper() == "DIRECT":
                        if not HAS_DIRECT_BH:
                            raise RuntimeError("Direct/BH module not loaded.")
                        ax0, ay0 = direct_omp(x0, y0, m0, G=G, soft=SOFT)

                    elif USE_SOLVER.upper() == "BH":
                        if not HAS_DIRECT_BH:
                            raise RuntimeError("Direct/BH module not loaded.")
                        theta = OPTIMIZED_PARAMS["bh_theta"]
                        ax0, ay0 = bh_omp(x0, y0, m0, OPTIMIZED_PARAMS["bh_domain"], theta, G, SOFT)

                    else:  # FMM
                        if not HAS_FMM:
                            raise RuntimeError("FMM module not loaded.")
                        theta = OPTIMIZED_PARAMS["fmm_theta"]
                        ax0, ay0 = fmm_kernel.fmm_omp(x0, y0, m0, OPTIMIZED_PARAMS["fmm_domain"], theta, G, SOFT)
                except Exception as e:
                    log_print(f"Error computing initial acceleration: {e}\n")
                    continue

                ax = ax0
                ay = ay0

                log_print("Creating live simulation GIF...")
                fig, ax_live = plt.subplots(figsize=(8, 8))
                points = []
                if include_central:
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

                def update_live(frame):
                    nonlocal ax, ay
                    ax, ay = leapfrog_step(bodies, ax, ay, include_central=include_central)
                    for idx, pt in enumerate(points):
                        pt.set_data([bodies[idx, 0]], [bodies[idx, 1]])
                    return points

                import matplotlib.animation as animation
                ani = animation.FuncAnimation(fig, update_live, frames=STEPS, interval=50, blit=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                gif_name = f"live_{USE_SOLVER}_{N}_{THREADS}_{timestamp}.gif"
                outpath = os.path.join(OUTPUT_DIR, gif_name)
                ani.save(outpath, fps=20, dpi=80)
                plt.close(fig)
                log_print(f"✓ Saved live simulation GIF: {outpath}\n")

            elif choice in ["4", "large-n", "large-n scaling"]:
                # Option 4: Large-N scaling test
                log_print("\n[Option 4] Large-N Scaling Test")
                if ctest is not None:
                    ctest.test_largeN_scaling()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["5", "energy", "energy conservation"]:
                # Option 5: Energy conservation test
                log_print("\n[Option 5] Energy Conservation Test")
                if ctest is not None:
                    ctest.test_energy_conservation()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["6", "optimize", "parameter optimization"]:
                # Option 6: Parameter optimization
                log_print("\n[Option 6] Parameter Optimization")
                if ctest is not None:
                    ctest.optimize_parameters()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["7", "thread", "openmp thread benchmark"]:
                # Option 7: OpenMP thread benchmark
                log_print("\n[Option 7] OpenMP Thread Benchmark")
                if ctest is not None:
                    ctest.thread_benchmark()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["8", "system", "system information"]:
                # Option 8: System information
                log_print("\n[Option 8] System Information")
                if ctest is not None:
                    ctest.show_system_info()
                else:
                    log_print("Error: comprehensive_test.py not found.\n")

            elif choice in ["q", "quit", "exit"]:
                log_print("Goodbye!")
                break

            else:
                log_print("Invalid choice. Please try again.\n")

        except KeyboardInterrupt:
            log_print("\nOperation interrupted. Returning to menu...\n")
        except Exception as e:
            log_print(f"\nError: {e}\nPlease try a different choice.\n")


if __name__ == "__main__":
    main_menu()

