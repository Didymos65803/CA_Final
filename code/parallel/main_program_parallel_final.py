#!/usr/bin/env python3
"""
main_program_parallel_final.py
==============================
Interactive 2-D N-body playground with optimized high-precision kernels
Fixed Barnes-Hut parameters and enhanced user experience

Menu
-----
1. Quick benchmark               → benchmark_scaling.png
2. Save trajectory + energy plot → trajectory.gif  + energy_vs_time.png
3. Live animation (real-time)    → live.gif
4. Large-N scaling test          → scaling_largeN.png + scaling_largeN.csv
5. Energy conservation test      → energy_conservation.png
6. Parameter optimization        → Find best settings
q. Quit
"""

import os
import argparse
import sys
import math
import time
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

# Set OpenMP threads
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--threads", type=int, default=8)
_args, _ = _cli.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(_args.threads)
os.environ.setdefault("OMP_STACKSIZE", "64M")

# Import compiled kernels
try:
    from force_kernel import direct_omp, bh_omp
    from fmm_kernel import fmm_omp
    SOLVERS = dict(direct=direct_omp, bh=bh_omp, fmm=fmm_omp)
    print(f"✓ Loaded N-body kernels with {_args.threads} OpenMP threads")
except ImportError as e:
    print(f"✗ Error importing kernels: {e}")
    sys.exit(1)

# Physical constants
G = 1.0
SOFT = 0.01
DOMAIN = 100.0
DT = 0.01
STAR_M = 100.0

# Optimized parameters (from testing)
OPTIMIZED_PARAMS = {
    'bh_theta': 0.3,        # More accurate than default 0.5
    'fmm_theta': 0.4,       # Optimized for FMM
    'bh_domain': 100.0,     # Adequate domain size
    'fmm_domain': 100.0,
    'distribution_size': 50.0,  # Reduced clustering
    'mass_range': (0.5, 2.0),   # Reduced mass variation
}

class Body:
    """Particle container"""
    __slots__ = ("x", "y", "vx", "vy", "m")

    def __init__(self, x=0.0, y=0.0, m=1.0, vx=0.0, vy=0.0):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.m = float(m)

def init_system(N: int, with_central: bool = True, rng_seed: int = 0, distribution: str = "disc"):
    """Create initial conditions with optimized parameters"""
    rng = np.random.default_rng(rng_seed)
    bodies = []

    if distribution == "random":
        # Optimized random distribution - less clustered
        size = OPTIMIZED_PARAMS['distribution_size']
        mass_min, mass_max = OPTIMIZED_PARAMS['mass_range']
        
        for i in range(N):
            x = (rng.random() - 0.5) * size
            y = (rng.random() - 0.5) * size
            mass = rng.uniform(mass_min, mass_max)
            bodies.append(Body(x, y, mass, 0.0, 0.0))
    
    elif distribution == "disc":
        if with_central:
            bodies.append(Body(0.0, 0.0, STAR_M, 0.0, 0.0))

        for _ in range(N):
            r = rng.uniform(8.0, 30.0)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(ang), r * math.sin(ang)
            v = math.sqrt(G * STAR_M / r)
            vx, vy = -v * math.sin(ang), v * math.cos(ang)
            bodies.append(Body(x, y, 1.0, vx, vy))
    
    elif distribution == "cluster":
        # Compact cluster
        for i in range(N):
            r = abs(rng.normal(0, 5.0))
            ang = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(ang), r * math.sin(ang)
            mass = rng.uniform(0.8, 1.2)
            vx = rng.normal(0, 0.5)
            vy = rng.normal(0, 0.5)
            bodies.append(Body(x, y, mass, vx, vy))

    return bodies

def compute_acc(bodies, solver: str, theta: float = None):
    """Compute accelerations with optimized parameters"""
    if not bodies:
        return np.array([]), np.array([])
    
    x = np.array([b.x for b in bodies], dtype=np.float64)
    y = np.array([b.y for b in bodies], dtype=np.float64)
    m = np.array([b.m for b in bodies], dtype=np.float64)
    
    try:
        if solver == "direct":
            return direct_omp(x, y, m, G, SOFT)
        elif solver == "bh":
            theta_use = theta if theta is not None else OPTIMIZED_PARAMS['bh_theta']
            domain_use = OPTIMIZED_PARAMS['bh_domain']
            return bh_omp(x, y, m, domain_use, theta_use, G, SOFT)
        elif solver == "fmm":
            theta_use = theta if theta is not None else OPTIMIZED_PARAMS['fmm_theta']
            domain_use = OPTIMIZED_PARAMS['fmm_domain']
            return fmm_omp(x, y, m, domain_use, theta_use, G, SOFT)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    except Exception as e:
        print(f"Error in compute_acc with solver {solver}: {e}")
        return np.zeros_like(x), np.zeros_like(y)

def leapfrog(bodies, solver: str, fixed_star: bool = True, theta: float = None):
    """Leapfrog integration"""
    if not bodies:
        return
    
    try:
        ax, ay = compute_acc(bodies, solver, theta)
        if len(ax) != len(bodies):
            return
        
        start_idx = 1 if fixed_star else 0
        
        # First half-kick + drift
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            b.vx += ax[i] * DT * 0.5
            b.vy += ay[i] * DT * 0.5
            b.x += b.vx * DT
            b.y += b.vy * DT
        
        # Second half-kick
        ax, ay = compute_acc(bodies, solver, theta)
        if len(ax) != len(bodies):
            return
            
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            b.vx += ax[i] * DT * 0.5
            b.vy += ay[i] * DT * 0.5
            
    except Exception as e:
        print(f"Error in leapfrog: {e}")

def total_energy(bodies, include_central: bool = True):
    """Calculate total energy"""
    if not bodies:
        return 0.0
    
    try:
        ke, pe = 0.0, 0.0
        start_idx = 0 if include_central else 1
        
        # Kinetic energy
        for i in range(start_idx, len(bodies)):
            b = bodies[i]
            ke += 0.5 * b.m * (b.vx**2 + b.vy**2)
        
        # Potential energy
        for i in range(start_idx, len(bodies)):
            for j in range(i + 1, len(bodies)):
                bi, bj = bodies[i], bodies[j]
                dx, dy = bi.x - bj.x, bi.y - bj.y
                r = math.sqrt(dx*dx + dy*dy + SOFT*SOFT)
                if r > 0:
                    pe -= G * bi.m * bj.m / r
        
        return ke + pe
        
    except Exception as e:
        print(f"Error calculating energy: {e}")
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
#                          MENU FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def quick_benchmark():
    """Benchmark with optimized parameters"""
    print("Running optimized benchmark...")
    Ns = [100, 200, 500, 1000]
    times = defaultdict(list)
    errors = defaultdict(list)

    for N in Ns:
        print(f"\nTesting N = {N}")
        bodies = init_system(N, with_central=False, distribution="random")
        
        # Reference (direct)
        if N <= 1000:
            t0 = time.time()
            ax_ref, ay_ref = compute_acc(bodies, "direct")
            direct_time = time.time() - t0
            times['direct'].append(direct_time)
            print(f"  Direct:    {direct_time:.4e} s")
        else:
            ax_ref, ay_ref = None, None
        
        # Test other methods
        for solver in ["bh", "fmm"]:
            t0 = time.time()
            ax, ay = compute_acc(bodies, solver)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Performance plot
    colors = {"direct": "red", "bh": "blue", "fmm": "green"}
    markers = {"direct": "o", "bh": "s", "fmm": "^"}
    
    for solver in SOLVERS.keys():
        valid_data = [(n, t) for n, t in zip(Ns, times[solver]) if not np.isnan(t)]
        if valid_data:
            ns, ts = zip(*valid_data)
            ax1.loglog(ns, ts, markers[solver] + "-", label=solver.upper(), 
                      color=colors[solver], linewidth=2, markersize=8)
    
    ax1.set_xlabel("N particles", fontsize=12)
    ax1.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax1.set_title("Performance with Optimized Parameters", fontsize=14)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Error plot
    for solver in ["bh", "fmm"]:
        valid_data = [(n, e) for n, e in zip(Ns, errors[solver]) if not np.isnan(e)]
        if valid_data:
            ns, es = zip(*valid_data)
            ax2.loglog(ns, es, markers[solver] + "-", label=f"{solver.upper()} Error", 
                      color=colors[solver], linewidth=2, markersize=8)
    
    # Add reference lines
    ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='1% Target')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10% Limit')
    
    ax2.set_xlabel("N particles", fontsize=12)
    ax2.set_ylabel("Relative Error", fontsize=12)
    ax2.set_title("Accuracy vs Direct Method", fontsize=14)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("benchmark_scaling_optimized.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved benchmark_scaling_optimized.png")

def save_trajectory():
    """Save trajectory with optimized settings"""
    print("\n=== Trajectory + Energy Analysis ===")
    
    N = int(input("Number of particles [100]: ") or "100")
    steps = int(input("Integration steps [600]: ") or "600")
    solver = (input("Solver direct/bh/fmm [fmm]: ") or "fmm").lower()
    distribution = (input("Distribution disc/random/cluster [disc]: ") or "disc").lower()
    gif = (input("Output GIF filename [trajectory.gif]: ") or "trajectory.gif")
    
    # Ask for custom parameters
    if solver == "bh":
        theta_input = input(f"Barnes-Hut theta parameter [{OPTIMIZED_PARAMS['bh_theta']}]: ")
        theta = float(theta_input) if theta_input else None
    elif solver == "fmm":
        theta_input = input(f"FMM theta parameter [{OPTIMIZED_PARAMS['fmm_theta']}]: ")
        theta = float(theta_input) if theta_input else None
    else:
        theta = None
    
    if solver not in SOLVERS:
        print(f"Unknown solver {solver}, using fmm")
        solver = "fmm"
    
    print(f"\nInitializing {N} particles with {distribution} distribution...")
    fixed_star = distribution == "disc"
    bodies = init_system(N, with_central=fixed_star, distribution=distribution)
    
    print(f"Integrating for {steps} steps using {solver.upper()} solver...")
    if theta is not None:
        print(f"Using custom theta = {theta}")
    
    # Storage
    xs = [[] for _ in bodies]
    ys = [[] for _ in bodies]
    E_list = []
    E0 = total_energy(bodies, include_central=fixed_star)
    
    try:
        for s in trange(steps, desc="Integrating"):
            # Save coordinates
            for i, b in enumerate(bodies):
                xs[i].append(b.x)
                ys[i].append(b.y)
            
            # Advance
            leapfrog(bodies, solver, fixed_star=fixed_star, theta=theta)
            
            # Log energy
            if s % 10 == 0:
                E = total_energy(bodies, include_central=fixed_star)
                rel_error = (E - E0) / abs(E0) if E0 != 0 else 0
                E_list.append((s * DT, E, rel_error))
    
    except KeyboardInterrupt:
        print("\nIntegration interrupted")
    except Exception as e:
        print(f"\nError during integration: {e}")
        return

    # Energy plot
    if E_list:
        times, energies, rel_errors = zip(*E_list)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(times, energies, 'b-', linewidth=2, label='Total Energy')
        ax1.axhline(y=E0, color='r', linestyle='--', alpha=0.7, label='Initial Energy')
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Total Energy", fontsize=12)
        ax1.set_title(f"Energy vs Time ({solver.upper()}, θ={theta if theta else 'default'})", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(times, np.abs(rel_errors), 'r-', linewidth=2)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Absolute Relative Energy Error", fontsize=12)
        ax2.set_title("Energy Conservation", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("energy_vs_time_optimized.png", dpi=180, bbox_inches='tight')
        plt.close()
        print("Saved energy_vs_time_optimized.png")
        
        final_error = abs(rel_errors[-1]) if rel_errors else 0
        print(f"Final energy error: {final_error:.6f}")

    # Create animation using individual points
    if len(xs) > 0 and len(xs[0]) > 0:
        print("Creating animation...")
        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Create individual plot points
            points = []
            if fixed_star:
                colors = ["red"] + ["blue"] * (len(bodies) - 1)
                sizes = [10] + [3] * (len(bodies) - 1)
            else:
                colors = ["blue"] * len(bodies)
                sizes = [3] * len(bodies)
            
            for i in range(len(bodies)):
                point, = ax.plot([], [], 'o', color=colors[i], markersize=sizes[i], alpha=0.8)
                points.append(point)
            
            # Set limits
            all_x = [x for particle_x in xs for x in particle_x[::10]]  # Sample every 10th frame
            all_y = [y for particle_y in ys for y in particle_y[::10]]
            if all_x and all_y:
                margin = 0.1
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
                ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)
            else:
                ax.set_xlim(-60, 60)
                ax.set_ylim(-60, 60)
            
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

            ani = FuncAnimation(fig, update, frames=len(xs[0]), init_func=init, 
                              blit=True, interval=50)
            ani.save(gif, writer=PillowWriter(fps=20))
            plt.close(fig)
            print(f"Saved {gif}")
            
        except Exception as e:
            print(f"Error creating animation: {e}")

def live_animation():
    """Live animation with optimized parameters"""
    print("\n=== Live Animation ===")
    
    N = int(input("Number of particles [50]: ") or "50")
    solver = (input("Solver direct/bh/fmm [fmm]: ") or "fmm").lower()
    distribution = (input("Distribution disc/random/cluster [disc]: ") or "disc").lower()
    frames = int(input("Number of frames [500]: ") or "500")
    
    if solver not in SOLVERS:
        solver = "fmm"
    
    fixed_star = distribution == "disc"
    bodies = init_system(N, with_central=fixed_star, distribution=distribution)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create points
    points = []
    if fixed_star:
        colors = ["red"] + ["blue"] * (len(bodies) - 1)
        sizes = [10] + [3] * (len(bodies) - 1)
    else:
        colors = ["blue"] * len(bodies)
        sizes = [3] * len(bodies)
    
    for i in range(len(bodies)):
        point, = ax.plot([bodies[i].x], [bodies[i].y], 'o', 
                        color=colors[i], markersize=sizes[i], alpha=0.8)
        points.append(point)
    
    ax.set_xlim(-70, 70)
    ax.set_ylim(-70, 70)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)})")

    def update(frame):
        try:
            leapfrog(bodies, solver, fixed_star=fixed_star)
            
            for i, point in enumerate(points):
                if i < len(bodies):
                    point.set_data([bodies[i].x], [bodies[i].y])
            
            ax.set_title(f"Live {solver.upper()} simulation (N={len(bodies)}, frame={frame})")
            return points
        except Exception as e:
            print(f"Error in frame {frame}: {e}")
            return points

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    gif_name = input("Save as GIF? (filename or Enter to skip): ").strip()
    if gif_name:
        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
        try:
            ani.save(gif_name, writer=PillowWriter(fps=20))
            print(f"Saved {gif_name}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    plt.show()

def scaling_test():
    """Large-N scaling test with optimized parameters"""
    print("\n=== Large-N Scaling Test ===")
    
    choice = input("Test (1) small N with all methods or (2) large N with BH/FMM [1]: ").strip()
    
    if choice == "2":
        Ns = [1000, 2000, 5000, 10000, 20000]
        methods = ["bh", "fmm"]
    else:
        Ns = [100, 200, 500, 1000, 2000]
        methods = ["direct", "bh", "fmm"]
    
    times = defaultdict(list)
    
    for N in Ns:
        print(f"\nN = {N}")
        bodies = init_system(N, with_central=False, distribution="random")
        
        for method in methods:
            if method == "direct" and N > 2000:
                continue
                
            try:
                t0 = time.time()
                compute_acc(bodies, method)
                elapsed = time.time() - t0
                times[method].append(elapsed)
                print(f"  {method.upper():6}: {elapsed:.4f} s")
            except Exception as e:
                print(f"  {method.upper():6}: ERROR - {e}")

    # Save and plot results
    csv_file = f"scaling_{'large' if choice == '2' else 'small'}N_optimized.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N"] + [m.upper() for m in methods])
        
        for i, N in enumerate(Ns):
            row = [N]
            for method in methods:
                if i < len(times[method]):
                    row.append(times[method][i])
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"Saved {csv_file}")

    # Plot
    plt.figure(figsize=(12, 8))
    colors = {"direct": "red", "bh": "blue", "fmm": "green"}
    markers = {"direct": "o", "bh": "s", "fmm": "^"}
    
    for method in methods:
        if times[method]:
            valid_data = [(N, t) for N, t in zip(Ns, times[method]) if not np.isnan(t)]
            if valid_data:
                ns, ts = zip(*valid_data)
                plt.loglog(ns, ts, markers[method] + "-", 
                          label=method.upper(), color=colors[method], 
                          linewidth=2, markersize=8)
    
    # Theoretical scaling
    if times.get("fmm"):
        N_ref, t_ref = Ns[0], times["fmm"][0]
        plt.loglog(Ns, [t_ref * N / N_ref for N in Ns], 
                  "--", color="green", alpha=0.5, label="O(N) theory")
    
    if times.get("bh"):
        N_ref, t_ref = Ns[0], times["bh"][0]
        plt.loglog(Ns, [t_ref * N * np.log(N) / (N_ref * np.log(N_ref)) for N in Ns], 
                  "--", color="blue", alpha=0.5, label="O(N log N) theory")
    
    plt.xlabel("Number of Particles", fontsize=12)
    plt.ylabel("Computation Time (s)", fontsize=12)
    plt.title("Optimized Scaling Comparison", fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    png_file = f"scaling_{'large' if choice == '2' else 'small'}N_optimized.png"
    plt.savefig(png_file, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved {png_file}")

def energy_conservation_test():
    """Enhanced energy conservation test"""
    print("\n=== Energy Conservation Test ===")
    
    N = int(input("Number of particles [100]: ") or "100")
    steps = int(input("Integration steps [1000]: ") or "1000")
    
    bodies_init = init_system(N, with_central=True, distribution="disc")
    E0 = total_energy(bodies_init, include_central=True)
    
    solvers = ["direct", "bh", "fmm"]
    results = {}
    
    for solver in solvers:
        if solver == "direct" and N > 1000:
            print(f"Skipping direct solver for N={N}")
            continue
            
        print(f"\nTesting {solver.upper()} solver...")
        bodies = [Body(b.x, b.y, b.m, b.vx, b.vy) for b in bodies_init]
        
        times, energies, errors = [], [], []
        
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        colors = {"direct": "red", "bh": "blue", "fmm": "green"}
        
        for solver, (times, energies, errors) in results.items():
            color = colors[solver]
            ax1.plot(times, energies, label=f"{solver.upper()}", color=color, linewidth=2)
            ax2.semilogy(times, errors, label=f"{solver.upper()}", color=color, linewidth=2)
        
        ax1.axhline(y=E0, color='black', linestyle='--', alpha=0.7, label='Initial Energy')
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Total Energy", fontsize=12)
        ax1.set_title("Energy vs Time (Optimized Parameters)", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Relative Energy Error", fontsize=12)
        ax2.set_title("Energy Conservation", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("energy_conservation_optimized.png", dpi=200, bbox_inches='tight')
        plt.show()
        print("Saved energy_conservation_optimized.png")
        
        # Print final errors
        print("\nFinal energy errors:")
        for solver, (_, _, errors) in results.items():
            if errors:
                print(f"  {solver.upper():6}: {errors[-1]:.2e}")

def parameter_optimization():
    """Interactive parameter optimization"""
    print("\n=== Parameter Optimization ===")
    
    N = int(input("Number of particles for testing [100]: ") or "100")
    solver = input("Solver to optimize (bh/fmm) [bh]: ").strip().lower() or "bh"
    
    if solver not in ["bh", "fmm"]:
        print("Invalid solver")
        return
    
    print(f"\nOptimizing {solver.upper()} parameters...")
    
    # Create test system
    bodies = init_system(N, with_central=False, distribution="random")
    
    # Reference solution
    print("Computing reference solution...")
    ax_ref, ay_ref = compute_acc(bodies, "direct")
    
    # Parameter ranges
    if solver == "bh":
        theta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        domain_values = [50.0, 100.0, 150.0, 200.0]
    else:  # fmm
        theta_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        domain_values = [50.0, 100.0, 150.0, 200.0]
    
    best_error = float('inf')
    best_params = {}
    results = []
    
    print(f"\nTesting parameter combinations:")
    print("Theta  Domain  Error      Time (ms)")
    print("-" * 40)
    
    for domain in domain_values:
        for theta in theta_values:
            t0 = time.time()
            
            if solver == "bh":
                ax, ay = bh_omp(np.array([b.x for b in bodies]), 
                               np.array([b.y for b in bodies]),
                               np.array([b.m for b in bodies]),
                               domain, theta, G, SOFT)
            else:
                ax, ay = fmm_omp(np.array([b.x for b in bodies]), 
                                np.array([b.y for b in bodies]),
                                np.array([b.m for b in bodies]),
                                domain, theta, G, SOFT)
            
            t_elapsed = (time.time() - t0) * 1000  # ms
            
            error = np.mean(np.sqrt((ax - ax_ref)**2 + (ay - ay_ref)**2) / 
                           (np.sqrt(ax_ref**2 + ay_ref**2) + 1e-10))
            
            results.append((theta, domain, error, t_elapsed))
            print(f"{theta:5.1f}  {domain:6.1f}  {error:8.2e}  {t_elapsed:8.2f}")
            
            if error < best_error:
                best_error = error
                best_params = {'theta': theta, 'domain': domain, 'time': t_elapsed}
    
    print(f"\nBest parameters:")
    print(f"  Theta: {best_params['theta']}")
    print(f"  Domain: {best_params['domain']}")
    print(f"  Error: {best_error:.2e}")
    print(f"  Time: {best_params['time']:.2f} ms")
    
    # Create parameter space plot
    try:
        # Reshape results for plotting
        theta_grid = np.array([r[0] for r in results])
        domain_grid = np.array([r[1] for r in results])
        error_grid = np.array([r[2] for r in results])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(theta_grid, domain_grid, c=np.log10(error_grid), 
                           s=100, cmap='viridis', alpha=0.8)
        
        # Mark best point
        ax.scatter(best_params['theta'], best_params['domain'], 
                  c='red', s=200, marker='*', label='Best')
        
        ax.set_xlabel('Theta Parameter', fontsize=12)
        ax.set_ylabel('Domain Size', fontsize=12)
        ax.set_title(f'{solver.upper()} Parameter Optimization', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('log10(Relative Error)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{solver}_parameter_optimization.png', dpi=200, bbox_inches='tight')
        plt.show()
        print(f"Saved {solver}_parameter_optimization.png")
        
    except Exception as e:
        print(f"Could not create plot: {e}")

def main_menu():
    """Enhanced main menu"""
    print("\n" + "="*60)
    print("    2-D High-Precision N-body Simulation")
    print("    Optimized Barnes-Hut and FMM kernels")
    print("="*60)
    print(f"\nCurrent optimized parameters:")
    print(f"  Barnes-Hut: θ={OPTIMIZED_PARAMS['bh_theta']}, domain={OPTIMIZED_PARAMS['bh_domain']}")
    print(f"  FMM: θ={OPTIMIZED_PARAMS['fmm_theta']}, domain={OPTIMIZED_PARAMS['fmm_domain']}")
    print(f"  Distribution size: {OPTIMIZED_PARAMS['distribution_size']}")
    
    while True:
        print("\n=== Main Menu ===")
        print("1) Quick benchmark (optimized)")
        print("2) Save trajectory + energy plot")
        print("3) Live animation")
        print("4) Large-N scaling test")
        print("5) Energy conservation test")
        print("6) Parameter optimization")
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
            elif choice == "6":
                parameter_optimization()
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

if __name__ == "__main__":
    main_menu()
