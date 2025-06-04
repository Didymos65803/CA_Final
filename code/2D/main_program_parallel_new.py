#!/usr/bin/env python3
"""
main_program_parallel_fixed.py
==============================
Interactive 2-D N-body playground that calls the **parallel** C++ kernels
compiled from `force_kernel.cpp`  (direct_omp, bh_omp)
                          and `fmm_kernel.cpp`    (fmm_omp).

Fixed segmentation faults and energy calculation issues.
Algorithm now matches fmm_scaling_test.py

Menu
-----
1. Quick benchmark               → benchmark_scaling.png
2. Save trajectory + energy plot → trajectory.gif  + energy_vs_time.png
3. Live animation (real-time)    → live.gif
4. Large-N scaling test          → scaling_largeN.png + scaling_largeN.csv
5. Energy conservation test      → energy_conservation.png
q. Quit

CLI flags
---------
--threads N   cap the number of OpenMP threads (default 8)
"""

# ────────────────────────────────────────────────────────────────────────────
# 0. SAFE OPENMP SETTINGS  (prevent seg-faults on big machines)
# ────────────────────────────────────────────────────────────────────────────
import os
import argparse
import sys

_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--threads", type=int, default=8,
                  help="max OpenMP threads (default 8)")
_args, _ = _cli.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(_args.threads)   # hard cap
os.environ.setdefault("OMP_STACKSIZE", "64M")        # adequate per-thread stack

# ────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ────────────────────────────────────────────────────────────────────────────
import math
import time
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

# Try to import compiled C++ kernels
try:
    from force_kernel import direct_omp, bh_omp
    from fmm_kernel import fmm_omp
    SOLVERS = dict(direct=direct_omp, bh=bh_omp, fmm=fmm_omp)
    print(f"Successfully loaded C++ kernels with {_args.threads} OpenMP threads")
except ImportError as e:
    print(f"Error importing C++ kernels: {e}")
    print("Please compile the kernels first using setup.py or pybind11")
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────
# 2. PHYSICAL CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
G = 1.0      # gravitation in code units
SOFT = 0.01  # Plummer softening length (reduced from 0.03 to match fmm_scaling_test)
DOMAIN = 100.0    # half-box size for BH / FMM kernels
DT = 0.01     # time-step for leapfrog
STAR_M = 100.0    # central massive star (kept fixed at origin)

# ────────────────────────────────────────────────────────────────────────────
# 3. SIMPLE PARTICLE CLASS
# ────────────────────────────────────────────────────────────────────────────
class Body:
    """minimal container: position (x,y), velocity (vx,vy), mass m"""
    __slots__ = ("x", "y", "vx", "vy", "m")

    def __init__(self, x=0.0, y=0.0, m=1.0, vx=0.0, vy=0.0):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.m = float(m)

# ────────────────────────────────────────────────────────────────────────────
# 4.  INITIAL CONDITIONS
# ────────────────────────────────────────────────────────────────────────────
def init_system(N: int, with_central: bool = True, rng_seed: int = 0, distribution: str = "disc"):
    """
    Create initial conditions matching fmm_scaling_test.py
    
    distribution options:
    - "disc": thin rotating disc around central star
    - "random": random distribution in square (matches fmm_scaling_test.py)
    - "cluster": concentrated cluster
    """
    rng = np.random.default_rng(rng_seed)
    bodies = []

    if distribution == "random":
        # Match fmm_scaling_test.py exactly
        for i in range(N):
            x = (rng.random() - 0.5) * 100.0
            y = (rng.random() - 0.5) * 100.0
            mass = rng.uniform(1.0, 5.0)
            bodies.append(Body(x, y, mass, 0.0, 0.0))
    
    elif distribution == "disc":
        # Central star (index 0) – we will NOT update its motion
        if with_central:
            bodies.append(Body(0.0, 0.0, STAR_M, 0.0, 0.0))

        # Orbiters in thin disc
        for _ in range(N):
            r = rng.uniform(8.0, 30.0)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(ang), r * math.sin(ang)
            v = math.sqrt(G * STAR_M / r)        # circular speed
            vx, vy = -v * math.sin(ang), v * math.cos(ang)
            bodies.append(Body(x, y, 1.0, vx, vy))
    
    elif distribution == "cluster":
        # Concentrated cluster for testing
        for i in range(N):
            # Gaussian distribution
            r = abs(rng.normal(0, 10.0))
            ang = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(ang), r * math.sin(ang)
            mass = rng.uniform(1.0, 3.0)
            # Small random velocities
            vx = rng.normal(0, 1.0)
            vy = rng.normal(0, 1.0)
            bodies.append(Body(x, y, mass, vx, vy))

    return bodies

# ────────────────────────────────────────────────────────────────────────────
# 5. ACCELERATION COMPUTATION  (wrap C++ kernels)
# ────────────────────────────────────────────────────────────────────────────
def compute_acc(bodies, solver: str, theta: float = 0.5):
    """
    Return (ax, ay) for all bodies using the requested solver.
    Fixed memory management and error checking.
    """
    if not bodies:
        return np.array([]), np.array([])
    
    # Convert to numpy arrays with proper type checking
    x = np.array([b.x for b in bodies], dtype=np.float64)
    y = np.array([b.y for b in bodies], dtype=np.float64)
    m = np.array([b.m for b in bodies], dtype=np.float64)
    
    # Validate input
    if len(x) == 0:
        return np.array([]), np.array([])
    
    try:
        if solver == "direct":
            return direct_omp(x, y, m, G, SOFT)
        elif solver == "bh":
            return bh_omp(x, y, m, DOMAIN, theta, G, SOFT)
        elif solver == "fmm":
            return fmm_omp(x, y, m, DOMAIN, theta, G, SOFT)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    except Exception as e:
        print(f"Error in compute_acc with solver {solver}: {e}")
        # Return zero acceleration as fallback
        return np.zeros_like(x), np.zeros_like(y)

# ────────────────────────────────────────────────────────────────────────────
# 6.  LEAPFROG INTEGRATOR  (fixed to handle central star properly)
# ────────────────────────────────────────────────────────────────────────────
def leapfrog(bodies, solver: str, fixed_star: bool = True):
    """
    Leapfrog integration with proper error handling.
    If fixed_star=True, keeps index 0 (central star) fixed.
    """
    if not bodies:
        return
    
    try:
        ax, ay = compute_acc(bodies, solver)
        if len(ax) != len(bodies) or len(ay) != len(bodies):
            print(f"Warning: acceleration array size mismatch")
            return
        
        # First half-kick + drift
        start_idx = 1 if fixed_star else 0
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            b.vx += ax[i] * DT * 0.5
            b.vy += ay[i] * DT * 0.5
            b.x += b.vx * DT
            b.y += b.vy * DT
        
        # Second half-kick
        ax, ay = compute_acc(bodies, solver)
        if len(ax) != len(bodies) or len(ay) != len(bodies):
            print(f"Warning: acceleration array size mismatch in second half")
            return
            
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            b.vx += ax[i] * DT * 0.5
            b.vy += ay[i] * DT * 0.5
            
    except Exception as e:
        print(f"Error in leapfrog integration: {e}")

# ────────────────────────────────────────────────────────────────────────────
# 7.  TOTAL ENERGY  (fixed calculation, matches fmm_scaling_test.py)
# ────────────────────────────────────────────────────────────────────────────
def total_energy(bodies, include_central: bool = True):
    """
    Calculate total energy (kinetic + potential).
    Fixed to match the calculation in fmm_scaling_test.py
    """
    if not bodies:
        return 0.0
    
    try:
        ke = 0.0
        pe = 0.0
        
        start_idx = 0 if include_central else 1
        
        # Kinetic energy
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            ke += 0.5 * b.m * (b.vx**2 + b.vy**2)
        
        # Potential energy (pairwise interactions)
        for i in range(start_idx, len(bodies)):
            for j in range(i + 1, len(bodies)):
                bi, bj = bodies[i], bodies[j]
                dx, dy = bi.x - bj.x, bi.y - bj.y
                r = math.sqrt(dx*dx + dy*dy + SOFT*SOFT)
                if r > 0:
                    pe -= G * bi.m * bj.m / r
        
        return ke + pe
        
    except Exception as e:
        print(f"Error calculating total energy: {e}")
        return 0.0

# ════════════════════════════════════════════════════════════════════════════
#                      MENU 1 ― QUICK  BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
def quick_benchmark():
    """Quick benchmark comparing all three solvers"""
    print("Running quick benchmark...")
    Ns = [100, 200, 500, 1000, 2000]
    times = defaultdict(list)
    errors = defaultdict(list)

    for N in Ns:
        print(f"\nTesting N = {N}")
        bodies = init_system(N, with_central=False, distribution="random")
        
        # Store reference solution (direct method for small N)
        if N <= 1000:
            bodies_ref = [Body(b.x, b.y, b.m, b.vx, b.vy) for b in bodies]
            t0 = time.time()
            ax_ref, ay_ref = compute_acc(bodies_ref, "direct")
            direct_time = time.time() - t0
            times['direct'].append(direct_time)
            print(f"  Direct:    {direct_time:.4e} s")
        else:
            times['direct'].append(np.nan)
            ax_ref, ay_ref = None, None
        
        for solver in ["bh", "fmm"]:
            bodies_copy = [Body(b.x, b.y, b.m, b.vx, b.vy) for b in bodies]
            t0 = time.time()
            ax, ay = compute_acc(bodies_copy, solver)
            solver_time = time.time() - t0
            times[solver].append(solver_time)
            print(f"  {solver.upper():8}: {solver_time:.4e} s")
            
            # Calculate error relative to direct method
            if ax_ref is not None and len(ax) > 0:
                error = np.mean(np.sqrt((ax - ax_ref)**2 + (ay - ay_ref)**2) / 
                               (np.sqrt(ax_ref**2 + ay_ref**2) + 1e-10))
                errors[solver].append(error)
                print(f"    Error: {error:.4e}")
            else:
                errors[solver].append(np.nan)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance plot
    markers = ["o", "s", "^"]
    colors = ["red", "blue", "green"]
    for i, (solver, marker, color) in enumerate(zip(SOLVERS.keys(), markers, colors)):
        valid_data = [(n, t) for n, t in zip(Ns, times[solver]) if not np.isnan(t)]
        if valid_data:
            ns, ts = zip(*valid_data)
            ax1.loglog(ns, ts, marker + "-", label=solver.upper(), color=color)
    
    ax1.set_xlabel("N")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Solver scaling")
    ax1.grid(True, which="both")
    ax1.legend()
    
    # Error plot
    for solver, color in zip(["bh", "fmm"], ["blue", "green"]):
        valid_data = [(n, e) for n, e in zip(Ns, errors[solver]) if not np.isnan(e)]
        if valid_data:
            ns, es = zip(*valid_data)
            ax2.loglog(ns, es, "o-", label=f"{solver.upper()} Error", color=color)
    
    ax2.set_xlabel("N")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Accuracy vs Direct")
    ax2.grid(True, which="both")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_scaling.png", dpi=200)
    plt.show()
    print("Saved benchmark_scaling.png")

# ════════════════════════════════════════════════════════════════════════════
#                MENU 2 ― TRAJECTORY  + ENERGY  + GIF
# ════════════════════════════════════════════════════════════════════════════
def save_trajectory():
    """Save trajectory animation and energy plot"""
    print("\n=== Trajectory + Energy Analysis ===")
    
    # Get parameters
    N = int(input("Number of particles [100]: ") or "100")
    steps = int(input("Integration steps [600]: ") or "600")
    solver = (input("Solver direct/bh/fmm [fmm]: ") or "fmm").lower()
    distribution = (input("Distribution disc/random/cluster [disc]: ") or "disc").lower()
    gif = (input("Output GIF filename [trajectory.gif]: ") or "trajectory.gif")
    
    if solver not in SOLVERS:
        print(f"Unknown solver {solver}, using fmm")
        solver = "fmm"
    
    print(f"\nInitializing {N} particles with {distribution} distribution...")
    fixed_star = distribution == "disc"
    bodies = init_system(N, with_central=fixed_star, distribution=distribution)
    
    print(f"Integrating for {steps} steps using {solver.upper()} solver...")
    
    # Storage for animation and energy
    xs = [[] for _ in bodies]
    ys = [[] for _ in bodies]
    E_list = []
    E0 = total_energy(bodies, include_central=fixed_star)
    
    try:
        for s in trange(steps, desc="Integrating"):
            # Save coordinates for animation
            for i, b in enumerate(bodies):
                xs[i].append(b.x)
                ys[i].append(b.y)
            
            # Advance one time-step
            leapfrog(bodies, solver, fixed_star=fixed_star)
            
            # Log energy every 10 steps
            if s % 10 == 0:
                E = total_energy(bodies, include_central=fixed_star)
                E_list.append((s * DT, E, (E - E0) / abs(E0) if E0 != 0 else 0))
    
    except KeyboardInterrupt:
        print("\nIntegration interrupted by user")
    except Exception as e:
        print(f"\nError during integration: {e}")
        return

    # Energy plot
    if E_list:
        times, energies, rel_errors = zip(*E_list)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        ax1.plot(times, energies, 'b-', linewidth=1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Total Energy")
        ax1.set_title("Energy vs Time")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(times, rel_errors, 'r-', linewidth=1)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Relative Energy Error")
        ax2.set_title("Energy Conservation")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("energy_vs_time.png", dpi=180)
        plt.close()
        print("Saved energy_vs_time.png")
        
        final_error = abs(rel_errors[-1]) if rel_errors else 0
        print(f"Final energy error: {final_error:.6f}")

    # Create animation (FIXED: handle scatter plot sizes correctly)
    if len(xs) > 0 and len(xs[0]) > 0:
        print("Creating animation...")
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # FIXED: Use individual points instead of scatter for animation
            if fixed_star:
                colors = ["red"] + ["blue"] * (len(bodies) - 1)
                sizes = [8] + [2] * (len(bodies) - 1)  # Smaller sizes
            else:
                colors = ["blue"] * len(bodies)
                sizes = [2] * len(bodies)
            
            # FIXED: Use individual points instead of scatter for animation
            if fixed_star:
                colors = ["red"] + ["blue"] * (len(bodies) - 1)
                sizes = [8] + [2] * (len(bodies) - 1)  # Smaller sizes
            else:
                colors = ["blue"] * len(bodies)
                sizes = [2] * len(bodies)
            
            # Create individual points for each particle
            points = []
            for i in range(len(bodies)):
                color = colors[i] if i < len(colors) else "blue"
                size = sizes[i] if i < len(sizes) else 2
                point, = ax.plot([], [], 'o', color=color, markersize=size, alpha=0.7)
                points.append(point)
            
            # Set axis limits based on actual particle positions
            all_x = [x for particle_x in xs for x in particle_x]
            all_y = [y for particle_y in ys for y in particle_y]
            if all_x and all_y:
                margin = 0.1
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
                ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)
            else:
                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{solver.upper()} N-body simulation (N={len(bodies)})")

            def init():
                for point in points:
                    point.set_data([], [])
                return points

            def update(frame):
                if frame < len(xs[0]):
                    for i, point in enumerate(points):
                        if i < len(xs):
                            point.set_data([xs[i][frame]], [ys[i][frame]])
                return points

            ani = FuncAnimation(fig, update, frames=len(xs[0]),
                              init_func=init, blit=True, interval=50)
            ani.save(gif, writer=PillowWriter(fps=20))
            plt.close(fig)
            print(f"Saved {gif}")
            
        except Exception as e:
            print(f"Error creating animation: {e}")

# ════════════════════════════════════════════════════════════════════════════
#                  MENU 3 ― LIVE  ANIMATION (FIXED)
# ════════════════════════════════════════════════════════════════════════════
def live_animation():
    """Live animation with real-time integration - FIXED"""
    print("\n=== Live Animation ===")
    
    N = int(input("Number of particles [50]: ") or "50")
    solver = (input("Solver direct/bh/fmm [fmm]: ") or "fmm").lower()
    distribution = (input("Distribution disc/random/cluster [disc]: ") or "disc").lower()
    frames = int(input("Number of frames [500]: ") or "500")
    
    if solver not in SOLVERS:
        print(f"Unknown solver {solver}, using fmm")
        solver = "fmm"
    
    fixed_star = distribution == "disc"
    bodies = init_system(N, with_central=fixed_star, distribution=distribution)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # FIXED: Use individual points for live animation too
    if fixed_star:
        colors = ["red"] + ["blue"] * (len(bodies) - 1)
        sizes = [8] + [2] * (len(bodies) - 1)
    else:
        colors = ["blue"] * len(bodies)
        sizes = [2] * len(bodies)
    
    points = []
    for i in range(len(bodies)):
        color = colors[i] if i < len(colors) else "blue"
        size = sizes[i] if i < len(sizes) else 2
        point, = ax.plot([bodies[i].x], [bodies[i].y], 'o', 
                        color=color, markersize=size, alpha=0.7)
        points.append(point)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)})")

    def update(frame):
        try:
            leapfrog(bodies, solver, fixed_star=fixed_star)
            
            for i, point in enumerate(points):
                if i < len(bodies):
                    point.set_data([bodies[i].x], [bodies[i].y])
            
            # Update title with frame number
            ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)}, frame={frame})")
            return points
        except Exception as e:
            print(f"Error in live animation frame {frame}: {e}")
            return points

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Save animation
    gif_name = input("Save as GIF? (filename or Enter to skip): ").strip()
    if gif_name:
        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
        print(f"Saving animation to {gif_name}...")
        try:
            ani.save(gif_name, writer=PillowWriter(fps=20))
            print(f"Saved {gif_name}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
#                  MENU 3 ― LIVE  ANIMATION
# ════════════════════════════════════════════════════════════════════════════
def live_animation():
    """Live animation with real-time integration"""
    print("\n=== Live Animation ===")
    
    N = int(input("Number of particles [50]: ") or "50")
    solver = (input("Solver direct/bh/fmm [fmm]: ") or "fmm").lower()
    distribution = (input("Distribution disc/random/cluster [disc]: ") or "disc").lower()
    frames = int(input("Number of frames [500]: ") or "500")
    
    if solver not in SOLVERS:
        print(f"Unknown solver {solver}, using fmm")
        solver = "fmm"
    
    fixed_star = distribution == "disc"
    bodies = init_system(N, with_central=fixed_star, distribution=distribution)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if fixed_star:
        colors = ["red"] + ["blue"] * (len(bodies) - 1)
        sizes = [20] + [3] * (len(bodies) - 1)
    else:
        colors = ["blue"] * len(bodies)
        sizes = [3] * len(bodies)
    
    scat = ax.scatter([b.x for b in bodies], [b.y for b in bodies],
                     s=sizes, c=colors, alpha=0.7)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)})")

    def update(frame):
        try:
            leapfrog(bodies, solver, fixed_star=fixed_star)
            offsets = np.column_stack([[b.x for b in bodies], [b.y for b in bodies]])
            scat.set_offsets(offsets)
            
            # Update title with frame number
            ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)}, frame={frame})")
            return scat,
        except Exception as e:
            print(f"Error in live animation frame {frame}: {e}")
            return scat,

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Save animation
    gif_name = input("Save as GIF? (filename or Enter to skip): ").strip()
    if gif_name:
        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
        print(f"Saving animation to {gif_name}...")
        ani.save(gif_name, writer=PillowWriter(fps=20))
        print(f"Saved {gif_name}")
    
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
#            MENU 4 ― LARGE-N  SCALING  (CSV + PNG)
# ════════════════════════════════════════════════════════════════════════════
def scaling_test():
    """Large-N scaling test matching fmm_scaling_test.py"""
    print("\n=== Large-N Scaling Test ===")
    
    # Test parameters matching fmm_scaling_test.py
    Ns_small = [100, 300, 500, 750, 1000, 3000, 5000]
    Ns_large = [5000, 7000, 8500, 10000, 20000, 30000, 40000, 50000]
    
    choice = input("Test (1) small N with all methods or (2) large N with BH/FMM [1]: ").strip()
    
    if choice == "2":
        Ns = Ns_large
        methods = ["bh", "fmm"]
        print(f"Testing large N: {Ns}")
    else:
        Ns = Ns_small
        methods = ["direct", "bh", "fmm"]
        print(f"Testing small N: {Ns}")
    
    times = defaultdict(list)
    
    for N in Ns:
        print(f"\nN = {N}")
        bodies = init_system(N, with_central=False, distribution="random")
        
        for method in methods:
            if method == "direct" and N > 5000:
                print(f"  Skipping direct for N={N} (too slow)")
                continue
                
            try:
                t0 = time.time()
                compute_acc(bodies, method)
                elapsed = time.time() - t0
                times[method].append(elapsed)
                print(f"  {method.upper():6}: {elapsed:.4f} s")
            except Exception as e:
                print(f"  {method.upper():6}: ERROR - {e}")
                times[method].append(np.nan)

    # Save CSV
    csv_file = "scaling_largeN.csv" if choice == "2" else "scaling_smallN.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["N"] + [m.upper() for m in methods]
        writer.writerow(header)
        
        for i, N in enumerate(Ns):
            row = [N]
            for method in methods:
                if i < len(times[method]):
                    row.append(times[method][i])
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"Saved {csv_file}")

    # Plot scaling
    plt.figure(figsize=(10, 6))
    colors = {"direct": "red", "bh": "blue", "fmm": "green"}
    markers = {"direct": "o", "bh": "s", "fmm": "^"}
    
    for method in methods:
        if times[method]:
            valid_data = [(N, t) for N, t in zip(Ns, times[method]) if not np.isnan(t)]
            if valid_data:
                ns, ts = zip(*valid_data)
                plt.loglog(ns, ts, markers[method] + "-", 
                          label=method.upper(), color=colors[method], linewidth=2)
    
    # Add theoretical scaling lines
    if "fmm" in times and times["fmm"]:
        N_ref = Ns[0]
        t_ref = next(t for t in times["fmm"] if not np.isnan(t))
        plt.loglog(Ns, [t_ref * N / N_ref for N in Ns], 
                  "--", color="green", alpha=0.5, label="O(N)")
    
    if "bh" in times and times["bh"]:
        N_ref = Ns[0]
        t_ref = next(t for t in times["bh"] if not np.isnan(t))
        plt.loglog(Ns, [t_ref * N * np.log(N) / (N_ref * np.log(N_ref)) for N in Ns], 
                  "--", color="blue", alpha=0.5, label="O(N log N)")
    
    if "direct" in times and times["direct"]:
        N_ref = Ns[0]
        t_ref = next(t for t in times["direct"] if not np.isnan(t))
        plt.loglog(Ns, [t_ref * (N / N_ref)**2 for N in Ns], 
                  "--", color="red", alpha=0.5, label="O(N²)")
    
    plt.xlabel("Number of Particles")
    plt.ylabel("Computation Time (s)")
    plt.title("Scaling Comparison")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    png_file = "scaling_largeN.png" if choice == "2" else "scaling_smallN.png"
    plt.savefig(png_file, dpi=200)
    plt.show()
    print(f"Saved {png_file}")

# ════════════════════════════════════════════════════════════════════════════
#            MENU 5 ― ENERGY CONSERVATION TEST
# ════════════════════════════════════════════════════════════════════════════
def energy_conservation_test():
    """Test energy conservation for different solvers and time steps"""
    print("\n=== Energy Conservation Test ===")
    
    N = int(input("Number of particles [100]: ") or "100")
    steps = int(input("Integration steps [1000]: ") or "1000")
    
    bodies_init = init_system(N, with_central=True, distribution="disc")
    E0 = total_energy(bodies_init, include_central=True)
    
    solvers = ["direct", "bh", "fmm"]
    results = {}
    
    for solver in solvers:
        if solver == "direct" and N > 1000:
            print(f"Skipping direct solver for N={N} (too slow)")
            continue
            
        print(f"\nTesting {solver.upper()} solver...")
        bodies = [Body(b.x, b.y, b.m, b.vx, b.vy) for b in bodies_init]
        
        times = []
        energies = []
        errors = []
        
        try:
            for step in trange(steps, desc=f"{solver.upper()}"):
                if step % 25 == 0:
                    E = total_energy(bodies, include_central=True)
                    times.append(step * DT)
                    energies.append(E)
                    errors.append(abs(E - E0) / abs(E0) if E0 != 0 else 0)
                
                leapfrog(bodies, solver, fixed_star=True)
            
            results[solver] = (times, energies, errors)
            
        except Exception as e:
            print(f"Error testing {solver}: {e}")
    
    # Plot results
    if results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        colors = {"direct": "red", "bh": "blue", "fmm": "green"}
        
        for solver, (times, energies, errors) in results.items():
            ax1.plot(times, energies, label=f"{solver.upper()}", color=colors[solver])
            ax2.semilogy(times, errors, label=f"{solver.upper()}", color=colors[solver])
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Total Energy")
        ax1.set_title("Energy vs Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Relative Energy Error")
        ax2.set_title("Energy Conservation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("energy_conservation.png", dpi=200)
        plt.show()
        print("Saved energy_conservation.png")
        
        # Print final errors
        print("\nFinal energy errors:")
        for solver, (_, _, errors) in results.items():
            if errors:
                print(f"  {solver.upper():6}: {errors[-1]:.2e}")

# ────────────────────────────────────────────────────────────────────────────
# 8. MAIN MENU
# ────────────────────────────────────────────────────────────────────────────
def main_menu():
    """Interactive main menu"""
    print("\n" + "="*50)
    print("    2-D Parallel N-body Simulation")
    print("    Fixed version matching fmm_scaling_test.py")
    print("="*50)
    
    while True:
        print("\n=== Main Menu ===")
        print("1) Quick benchmark")
        print("2) Save trajectory + energy plot")
        print("3) Live animation")
        print("4) Large-N scaling test")
        print("5) Energy conservation test")
        print("q) Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        try:
            if choice == "1":
                quick_benchmark()
            elif choice == "2":
                save_trajectory()
            elif choice == "3":
                live_animation()
            elif choice == "4":
                scaling_test()
            elif choice == "5":
                energy_conservation_test()
            elif choice in ["q", "quit", "exit"]:
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or choose a different option.")

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main_menu()
