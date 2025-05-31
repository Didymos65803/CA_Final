#!/usr/bin/env python3
"""
main_program_parallel.py
========================
Interactive 2-D N-body playground that calls the **parallel** C++ kernels
compiled from `force_kernel.cpp`  (direct_omp, bh_omp)
                          and `fmm_kernel.cpp`    (fmm_omp).

Menu
-----
1. Quick benchmark               → benchmark_scaling.png
2. Save trajectory + energy plot → trajectory.gif  + energy_vs_time.png
3. Live animation (real-time)    → live.gif
4. Large-N scaling test          → scaling_largeN.png + scaling_largeN.csv
q. Quit

CLI flags
---------
--threads N   cap the number of OpenMP threads (default 8)

All prompts are in English.
All figures / GIF / CSV are saved automatically in the working directory.
"""

# ────────────────────────────────────────────────────────────────────────────
# 0. SAFE OPENMP SETTINGS  (prevent seg-faults on big machines)
# ────────────────────────────────────────────────────────────────────────────
import os, argparse
_cli = argparse.ArgumentParser(add_help=False)
_cli.add_argument("--threads", type=int, default=8,
                  help="max OpenMP threads (default 8)")
_args, _ = _cli.parse_known_args()
os.environ["OMP_NUM_THREADS"] = str(_args.threads)   # hard cap
os.environ.setdefault("OMP_STACKSIZE", "64M")        # adequate per-thread stack

# ────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ────────────────────────────────────────────────────────────────────────────
import math, time, csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange            # progress bar

# compiled C++ kernels
from force_kernel import direct_omp, bh_omp
from fmm_kernel   import fmm_omp

SOLVERS = dict(direct=direct_omp, bh=bh_omp, fmm=fmm_omp)

# ────────────────────────────────────────────────────────────────────────────
# 2. PHYSICAL CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
G       = 1.0      # gravitation in code units
SOFT    = 0.03     # Plummer softening length
DOMAIN  = 100.0    # half-box size for BH / FMM kernels
DT      = 0.01     # time-step for leapfrog
STAR_M  = 100.0    # central massive star (kept fixed at origin)

# ────────────────────────────────────────────────────────────────────────────
# 3. SIMPLE PARTICLE CLASS
# ────────────────────────────────────────────────────────────────────────────
class Body:
    """minimal container: position (x,y), velocity (vx,vy), mass m"""
    __slots__ = ("x", "y", "vx", "vy", "m")

    def __init__(self, x=0.0, y=0.0, m=1.0, vx=0.0, vy=0.0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.m = m

# ────────────────────────────────────────────────────────────────────────────
# 4.  INITIAL CONDITIONS
# ────────────────────────────────────────────────────────────────────────────
def init_system(N: int, with_central: bool = True, rng_seed: int = 0):
    """
    Create a thin rotating disc around a fixed central star.
      • central star mass = STAR_M, fixed at (0,0)
      • N orbiter masses  = 1.0, random radius 8–30, circular velocity
    """
    rng = np.random.default_rng(rng_seed)
    bodies = []

    # central star (index 0) – we will NOT update its motion
    if with_central:
        bodies.append(Body(0.0, 0.0, STAR_M, 0.0, 0.0))

    # orbiters
    for _ in range(N):
        r   = rng.uniform(8.0, 30.0)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        x, y = r * math.cos(ang), r * math.sin(ang)
        v    = math.sqrt(G * STAR_M / r)        # circular speed
        vx, vy = -v * math.sin(ang),  v * math.cos(ang)
        bodies.append(Body(x, y, 1.0, vx, vy))

    return bodies

# ────────────────────────────────────────────────────────────────────────────
# 5. ACCELERATION COMPUTATION  (wrap C++ kernels)
# ────────────────────────────────────────────────────────────────────────────
def compute_acc(bodies, solver: str, theta: float = 0.5):
    """
    Return (ax, ay) for all bodies using the requested solver.
    Matches current C++ signatures – NO 'order' arg needed.
    """
    x = np.fromiter((b.x for b in bodies), dtype=np.float64)
    y = np.fromiter((b.y for b in bodies), dtype=np.float64)
    m = np.fromiter((b.m for b in bodies), dtype=np.float64)

    if solver == "direct":
        return direct_omp(x, y, m, G, SOFT)

    if solver == "bh":
        return bh_omp(x, y, m, DOMAIN, theta, G, SOFT)

    # FMM  →  x, y, m, domain, theta, G, soft, maxLeaf
    return fmm_omp(x, y, m, DOMAIN, theta, G, SOFT)

# ────────────────────────────────────────────────────────────────────────────
# 6.  LEAPFROG INTEGRATOR  (star kept fixed)
# ────────────────────────────────────────────────────────────────────────────
def leapfrog(bodies, solver: str):
    ax, ay = compute_acc(bodies, solver)
    # half-kick + drift  (skip index 0 → fixed star)
    for i, b in enumerate(bodies[1:], start=1):
        b.vx += ax[i] * DT * 0.5
        b.vy += ay[i] * DT * 0.5
        b.x  += b.vx * DT
        b.y  += b.vy * DT
    # second half-kick
    ax, ay = compute_acc(bodies, solver)
    for i, b in enumerate(bodies[1:], start=1):
        b.vx += ax[i] * DT * 0.5
        b.vy += ay[i] * DT * 0.5

# ────────────────────────────────────────────────────────────────────────────
# 7.  TOTAL ENERGY  (for drift check)
# ────────────────────────────────────────────────────────────────────────────
def total_energy(bodies):
    ke = sum(0.5 * b.m * (b.vx**2 + b.vy**2) for b in bodies[1:])   # skip star
    pe = 0.0
    for i, a in enumerate(bodies):
        for b in bodies[:i]:
            dx, dy = a.x - b.x, a.y - b.y
            r = math.hypot(dx, dy) + SOFT
            pe -= G * a.m * b.m / r
    return ke + pe

# ════════════════════════════════════════════════════════════════════════════
#                      MENU 1 ― QUICK  BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
def quick_benchmark():
    Ns = [100, 200, 500, 1_000, 2_000, 5_000]
    times = defaultdict(list)

    for N in Ns:
        bodies = init_system(N, with_central=False)
        for solv in SOLVERS:
            t0 = time.time()
            compute_acc(bodies, solv)
            times[solv].append(time.time() - t0)
            print(f"{solv:<6}  N={N:5d}   {times[solv][-1]:.4e} s")

    # plot
    plt.figure(figsize=(7, 5))
    for solv, marker in zip(SOLVERS, ("o", "s", "^")):
        plt.loglog(Ns, times[solv], marker + "-", label=solv.upper())
    plt.xlabel("N");  plt.ylabel("Wall-clock time (s)")
    plt.title("Solver scaling");  plt.grid(True, which="both");  plt.legend()
    plt.tight_layout();  plt.savefig("benchmark_scaling.png", dpi=200)
    plt.show()
    print("Saved  benchmark_scaling.png")

# ════════════════════════════════════════════════════════════════════════════
#                MENU 2 ― TRAJECTORY  + ENERGY  + GIF
# ════════════════════════════════════════════════════════════════════════════
def save_trajectory():
    N     = int(input("Number of orbiters [100] : ") or "100")
    steps = int(input("Integration steps [600] : ") or "600")
    solver= (input("Solver direct/bh/fmm [fmm]   : ") or "fmm").lower()
    gif   = (input("Output GIF filename [traj.gif] : ") or "traj.gif")

    bodies = init_system(N)
    xs = [[] for _ in bodies];  ys = [[] for _ in bodies]
    E_list = []

    for s in trange(steps, desc="Integrating"):
        # save coordinates for GIF
        for i, b in enumerate(bodies):
            xs[i].append(b.x)
            ys[i].append(b.y)
        # advance one time-step
        leapfrog(bodies, solver)
        # log total energy every 25 steps
        if s % 25 == 0:
            E_list.append((s * DT, total_energy(bodies)))

    # --- energy plot ---
    if E_list:
        t, E = zip(*E_list)
        plt.figure()
        plt.plot(t, E)
        plt.xlabel("time");  plt.ylabel("Total energy")
        plt.title("Energy vs time")
        plt.tight_layout()
        plt.savefig("energy_vs_time.png", dpi=180)
        plt.close()
        print("Saved energy_vs_time.png")

    # --- build GIF ---
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=3, c=["red"] + ["blue"] * N)
    ax.set_xlim(-50, 50);  ax.set_ylim(-50, 50)
    ax.set_aspect('equal');  ax.grid(True)

    def init():
        scat.set_offsets([]);  return scat,

    def update(i):
        scat.set_offsets(np.c_[ [x[i] for x in xs],
                                [y[i] for y in ys] ])
        return scat,

    ani = FuncAnimation(fig, update, frames=steps,
                        init_func=init, blit=True)
    ani.save(gif, writer=PillowWriter(fps=25))
    plt.close(fig)
    print("Saved", gif)

# ════════════════════════════════════════════════════════════════════════════
#                  MENU 3 ― LIVE  ANIMATION  +  GIF
# ════════════════════════════════════════════════════════════════════════════
def live_animation():
    N     = int(input("Number of orbiters [50]  : ") or "50")
    solver= (input("Solver direct/bh/fmm [fmm] : ") or "fmm").lower()
    gif   = (input("Output GIF filename [live.gif] : ") or "live.gif")

    bodies = init_system(N)
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([b.x for b in bodies],
                      [b.y for b in bodies],
                      s=3, c=["red"] + ["blue"] * N)
    ax.set_xlim(-50, 50);  ax.set_ylim(-50, 50)
    ax.set_aspect('equal');  ax.grid(True)

    def update(_):
        leapfrog(bodies, solver)
        scat.set_offsets([[b.x, b.y] for b in bodies])
        return scat,

    ani = FuncAnimation(fig, update, frames=600,
                        interval=30, blit=True)
    ani.save(gif, writer=PillowWriter(fps=30))
    plt.show()
    print("Saved", gif)

# ════════════════════════════════════════════════════════════════════════════
#            MENU 4 ― LARGE-N  SCALING  (CSV + PNG)
# ════════════════════════════════════════════════════════════════════════════
def scaling_test():
    Ns = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
    t_bh, t_fmm = [], []

    for N in Ns:
        bodies = init_system(N, with_central=False)
        # BH
        t0 = time.time(); compute_acc(bodies, "bh");  t_bh.append(time.time() - t0)
        # FMM
        t0 = time.time(); compute_acc(bodies, "fmm"); t_fmm.append(time.time() - t0)
        print(f"N={N:6d}   BH {t_bh[-1]:.3f}s    FMM {t_fmm[-1]:.3f}s")

    # save CSV
    with open("scaling_largeN.csv", "w", newline="") as f:
        wr = csv.writer(f);  wr.writerow(["N", "BH", "FMM"])
        wr.writerows(zip(Ns, t_bh, t_fmm))
    print("Saved scaling_largeN.csv")

    # plot
    plt.figure(figsize=(7, 5))
    plt.loglog(Ns, t_bh,  "s-", label="BH")
    plt.loglog(Ns, t_fmm, "^-", label="FMM")
    plt.xlabel("N"); plt.ylabel("time (s)")
    plt.title("Large-N scaling"); plt.grid(True, which="both"); plt.legend()
    plt.tight_layout(); plt.savefig("scaling_largeN.png", dpi=200); plt.show()
    print("Saved scaling_largeN.png")

# ────────────────────────────────────────────────────────────────────────────
# 8. MAIN MENU
# ────────────────────────────────────────────────────────────────────────────
def main_menu():
    while True:
        print("\n=== 2-D Parallel N-body ===")
        print("1) Quick benchmark")
        print("2) Save trajectory + energy")
        print("3) Live animation")
        print("4) Large-N scaling test")
        print("q) Quit")
        choice = input("Select option: ").strip().lower()

        if choice == "1": quick_benchmark()
        elif choice == "2": save_trajectory()
        elif choice == "3": live_animation()
        elif choice == "4": scaling_test()
        elif choice == "q": break
        else: print("Invalid choice.")

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main_menu()

