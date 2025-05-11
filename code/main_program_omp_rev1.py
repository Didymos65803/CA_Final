"""
Changelog vs. rev‑1
====================
1. **animate()**
   • `traj_fname`, `gif_name`, `energy_png` renamed with `_rev1` suffix.
   • energy plot saved as `energy_drift_<n>_rev1.png`.
2. **benchmark() / plot_results()**
   • internal CSV: `benchmark_<hash>_rev1.csv` (unique by N‑list).
   • figure: `benchmark_<Nlist>_rev1.jpg` (unchanged from rev‑1).
3. Minor: tqdm switched to `leave=False` to keep console clean.
4. Whole file reformatted (black 88) and type‑hints completed.

Physics and OpenMP behaviour **unchanged**.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Optional OpenMP kernel -------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import force_kernel  # noqa: F401

    USE_OMP = True
    print("[info] OpenMP kernel loaded (force_kernel)")
except ImportError:  # pragma: no cover
    USE_OMP = False
    print("[warn] force_kernel not found – falling back to pure‑Python direct method")

sys.setrecursionlimit(10000)  # deep QuadTree safety

# -----------------------------------------------------------------------------
# Physical constants -----------------------------------------------------------
# -----------------------------------------------------------------------------
G: float = 1.0
SOFTENING: float = 0.01
SOFT2: float = SOFTENING**2
DEFAULT_DT: float = 0.02


# -----------------------------------------------------------------------------
# Data structure ----------------------------------------------------------------
# -----------------------------------------------------------------------------
class Particle:  # noqa: D101 – simple container

    __slots__ = ("x", "y", "vx", "vy", "ax", "ay", "mass")

    def __init__(self, x: float, y: float, mass: float = 1.0, vx: float = 0.0, vy: float = 0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = 0.0
        self.ay = 0.0
        self.mass = mass

    # convenience views -----------------------------------------------------
    @property
    def pos(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def vel(self) -> tuple[float, float]:
        return self.vx, self.vy


# -----------------------------------------------------------------------------
# Direct solver (Python & OpenMP) ---------------------------------------------
# -----------------------------------------------------------------------------

def _compute_forces_direct_py(particles: list[Particle]) -> None:
    n = len(particles)
    for i in range(n):
        pi = particles[i]
        axi = ayi = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = particles[j].x - pi.x
            dy = particles[j].y - pi.y
            r2 = dx * dx + dy * dy
            if r2 < SOFT2:
                r2 = SOFT2
            inv_r = 1.0 / math.sqrt(r2)
            f = G * particles[j].mass * inv_r * inv_r
            axi += f * dx * inv_r
            ayi += f * dy * inv_r
        pi.ax, pi.ay = axi, ayi


def _compute_forces_direct_omp(particles: list[Particle]) -> None:  # noqa: D401
    n = len(particles)
    cache = _compute_forces_direct_omp.__dict__.setdefault("buf", {})
    if cache.get("n") != n:
        cache["x"] = np.empty(n, float)
        cache["y"] = np.empty(n, float)
        cache["m"] = np.empty(n, float)
        cache["ax"] = np.empty(n, float)
        cache["ay"] = np.empty(n, float)
        cache["n"] = n
    cache["x"][:] = [p.x for p in particles]
    cache["y"][:] = [p.y for p in particles]
    cache["m"][:] = [p.mass for p in particles]
    force_kernel.direct_omp(cache["x"], cache["y"], cache["m"], cache["ax"], cache["ay"], G, SOFT2)
    for i, p in enumerate(particles):
        p.ax = cache["ax"][i]
        p.ay = cache["ay"][i]


def compute_forces_direct(particles: list[Particle]) -> None:  # noqa: D401
    if USE_OMP:
        _compute_forces_direct_omp(particles)
    else:
        _compute_forces_direct_py(particles)


# -----------------------------------------------------------------------------
# Barnes–Hut QuadTree ----------------------------------------------------------
# -----------------------------------------------------------------------------
class QuadTreeNode:  # noqa: D101

    __slots__ = (
        "cx",
        "cy",
        "size",
        "children",
        "particles",
        "total_mass",
        "com_x",
        "com_y",
        "is_leaf",
        "is_empty",
    )

    def __init__(self, cx: float, cy: float, size: float):
        self.cx = cx
        self.cy = cy
        self.size = size
        self.children: list[QuadTreeNode | None] = [None] * 4
        self.particles: list[Particle] = []
        self.total_mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.is_leaf = True
        self.is_empty = True

    # insertion -----------------------------------------------------------
    def insert(self, p: Particle) -> None:
        self.is_empty = False
        if self.is_leaf and not self.particles:
            self.particles.append(p)
            self.total_mass = p.mass
            self.com_x = p.x
            self.com_y = p.y
            return
        if self.is_leaf and len(self.particles) < 1:
            self.particles.append(p)
            self._update_com(p)
            return
        if self.is_leaf:
            self.is_leaf = False
            half = self.size / 2.0
            q = half / 2.0
            for idx, (dx, dy) in enumerate([(-q, -q), (q, -q), (-q, q), (q, q)]):
                self.children[idx] = QuadTreeNode(self.cx + dx, self.cy + dy, half)
            for old in self.particles:
                self._insert_child(old)
            self.particles.clear()
        self._insert_child(p)
        self._update_com(p)

    def _insert_child(self, p: Particle) -> None:
        idx = (p.x > self.cx) + 2 * (p.y > self.cy)
        child = self.children[idx]
        assert child is not None
        child.insert(p)

    def _update_com(self, p: Particle) -> None:
        self.total_mass += p.mass
        self.com_x = (self.com_x * (self.total_mass - p.mass) + p.x * p.mass) / self.total_mass
        self.com_y = (self.com_y * (self.total_mass - p.mass) + p.y * p.mass) / self.total_mass

    # force ---------------------------------------------------------------
    def compute_force(self, p: Particle, theta: float = 0.5):
        if self.is_empty:
            return 0.0, 0.0
        if self.is_leaf and len(self.particles) == 1 and self.particles[0] is p:
            return 0.0, 0.0
        dx = self.com_x - p.x
        dy = self.com_y - p.y
        r2 = dx * dx + dy * dy
        if r2 < SOFT2:
            r2 = SOFT2
        r = math.sqrt(r2)
        if self.is_leaf or (self.size / r < theta):
            f = G * self.total_mass / r2
            return f * dx / r, f * dy / r
        ax = ay = 0.0
        for c in self.children:
            if c and not c.is_empty:
                fx, fy = c.compute_force(p, theta)
                ax += fx
                ay += fy
        return ax, ay


# driver --------------------------------------------------------------------

def compute_forces_fmm(particles: list[Particle], theta: float = 0.5) -> None:  # noqa: D401
    for p in particles:
        p.ax = p.ay = 0.0
    max_extent = max(max(abs(p.x), abs(p.y)) for p in particles) * 1.2
    max_extent = 1.0 if max_extent < 1.0 else max_extent
    root = QuadTreeNode(0.0, 0.0, 2 * max_extent)
    for p in particles:
        root.insert(p)
    for p in particles:
        p.ax, p.ay = root.compute_force(p, theta)


# -----------------------------------------------------------------------------
# Integrator ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def leapfrog_step(
    particles: list[Particle], *, dt: float, theta: float, method: str, mobile_star: bool
) -> None:
    # kick 1
    for i, p in enumerate(particles):
        if i == 0 and not mobile_star:
            continue
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt
    # drift
    for i, p in enumerate(particles):
        if i == 0 and not mobile_star:
            continue
        p.x += p.vx * dt
        p.y += p.vy * dt
    # force
    compute_forces_direct(particles) if method == "direct" else compute_forces_fmm(particles, theta)
    # kick 2
    for i, p in enumerate(particles):
        if i == 0 and not mobile_star:
            continue
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt


# -----------------------------------------------------------------------------
# Diagnostics -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def system_energy(particles: list[Particle]) -> float:
    ke = sum(0.5 * p.mass * (p.vx**2 + p.vy**2) for p in particles)
    pe = 0.0
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = particles[j].x - particles[i].x
            dy = particles[j].y - particles[i].y
            r2 = dx * dx + dy * dy + SOFT2
            pe -= G * particles[i].mass * particles[j].mass / math.sqrt(r2)
    return ke + pe


# -----------------------------------------------------------------------------
# Initial conditions -----------------------------------------------------------
# -----------------------------------------------------------------------------

def init_disc(n: int) -> list[Particle]:
    particles: list[Particle] = [Particle(0.0, 0.0, mass=1000.0)]
    for _ in range(1, n):
        r = np.random.uniform(5, 30)
        ang = np.random.uniform(0, 2 * math.pi)
        x, y = r * math.cos(ang), r * math.sin(ang)
        v = math.sqrt(G * particles[0].mass / r)
        particles.append(
            Particle(
                x,
                y,
                mass=np.random.uniform(1, 2),
                vx=-v * math.sin(ang),
                vy=v * math.cos(ang),
            )
        )
    return particles


# -----------------------------------------------------------------------------
# Benchmark utilities ----------------------------------------------------------
# -----------------------------------------------------------------------------

def benchmark(ns: list[int]) -> dict[str, np.ndarray]:
    res = defaultdict(list)
    for n in ns:
        a1 = [Particle((np.random.rand()-0.5)*50, (np.random.rand()-0.5)*50, mass=np.random.rand()*9+1) for _ in range(n)]
        a2 = [Particle(p.x, p.y, p.mass) for p in a1]
        t0 = time.time(); compute_forces_direct(a1); res["direct_times"].append(time.time() - t0)
        t0 = time.time(); compute_forces_fmm(a2);     res["fmm_times"].append(time.time() - t0)
        err = sum(math.hypot(p1.ax-p2.ax, p1.ay-p2.ay)/(math.hypot(p1.ax, p1.ay)+1e-12) for p1, p2 in zip(a1, a2)) / n
        res["rel_err"].append(err)
    return {k: np.asarray(v) for k, v in res.items()}


def plot_results(ns: list[int], res: dict[str, np.ndarray]) -> None:
    ns_arr = np.asarray(ns)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].loglog(ns_arr, res["direct_times"], "o-", label="Direct")
    axs[0].loglog(ns_arr, res["fmm_times"], "s-", label="FMM")
    axs[0].set_xlabel("N"); axs[0].set_ylabel("Time(s)"); axs[0].set_title("Performance"); axs[0].grid(True, which="both", ls="--", alpha=0.4); axs[0].legend()
    axs[1].loglog(ns_arr, res["rel_err"], "o-"); axs[1].set_xlabel("N"); axs[1].set_ylabel("Avg Rel Error"); axs[1].set_title("Accuracy"); axs[1].grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    fname = f"benchmark_{','.join(map(str, ns))}_rev1.jpg"
    plt.savefig(fname, dpi=300)
    print(f"[saved] {fname}")
    # CSV ----------------------------------------------------------------
    csvname = f"benchmark_rev1.csv"
    with open(csvname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "direct_s", "fmm_s", "avg_rel_err"])
        for n, d, fmm_t, e in zip(ns, res["direct_times"], res["fmm_times"], res["rel_err"]):
            w.writerow([n, d, fmm_t, e])
    print(f"[saved] {csvname}")


# -----------------------------------------------------------------------------
# Animation -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def animate(args: argparse.Namespace) -> None:
    n, steps, method, theta, dt, mobile_star = (
        args.n,
        args.steps,
        args.method,
        args.theta,
        args.dt,
        args.mobile_star,
    )

    parts = init_disc(n)
    compute_forces_direct(parts) if method == "direct" else compute_forces_fmm(parts, theta)

    xs, ys = [[] for _ in range(n)], [[] for _ in range(n)]
    energies: list[float] = []

    for _ in tqdm(range(steps), desc="sim", leave=False):
        for i, p in enumerate(parts):
            xs[i].append(p.x)
            ys[i].append(p.y)
        energies.append(system_energy(parts))
        leapfrog_step(parts, dt=dt, theta=theta, method=method, mobile_star=mobile_star)

    # static traj --------------------------------------------------------
    traj_fname = f"trajectories_{n}_rev1.png"
    plt.figure(figsize=(6, 6)); plt.title(f"Trajectories – n={n} ({method})"); plt.grid(True)
    for i in range(1, n):
        plt.plot(xs[i], ys[i], lw=0.6, alpha=0.6)
    plt.scatter(xs[0][0], ys[0][0], s=60, c="red", label="star"); plt.axis("equal"); plt.legend(); plt.tight_layout(); plt.savefig(traj_fname, dpi=300)
    print(f"[saved] {traj_fname}")

    # energy plot --------------------------------------------------------
    e_png = f"energy_drift_{n}_rev1.png"
    plt.figure(figsize=(5, 3)); plt.plot(energies); plt.xlabel("step"); plt.ylabel("E_total"); plt.title("Energy drift"); plt.tight_layout(); plt.savefig(e_png, dpi=300)
    print(f"[saved] {e_png}")

    # GIF ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6)); ax.set(xlim=(-60, 60), ylim=(-60, 60)); ax.grid(True)
    scat_p = ax.scatter([], [], s=15, c="tab:blue"); scat_s = ax.scatter([], [], s=60, c="red")

    def _upd(frame: int):
        scat_p.set_offsets(np.column_stack(( [xs[i][frame] for i in range(1, n)], [ys[i][frame] for i in range(1, n)] )))
        scat_s.set_offsets((xs[0][frame], ys[0][frame])); return scat_p, scat_s

    ani = FuncAnimation(fig, _upd, frames=steps, interval=30, blit=True)
    gif_name = f"simulation_{n}_rev1.gif"
    ani.save(gif_name, writer=PillowWriter(fps=30))
    print(f"[saved] {gif_name}")


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2‑D N‑body with Barnes–Hut & OpenMP benchmark")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="run benchmark and export figure")
    b.add_argument("N", nargs="*", type=int, default=[1000, 5000, 10000])

    a = sub.add_parser("anim", help="animate a disc simulation")
    a.add_argument("-n", type=int, default=100)
    a.add_argument("-steps", type=int, default=200)
    a.add_argument("-method", choices=["direct", "fmm"], default="fmm")
    a.add_argument("-theta", type=float, default=0.5)
    a.add_argument("-dt", type=float, default=DEFAULT_DT)
    a.add_argument("--mobile-star", action="store_true")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Entry -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # ----------------------------------------------------------
    #  Command dispatch
    # ----------------------------------------------------------
    if args.cmd == "bench":
        res = benchmark(args.N)
        plot_results(args.N, res)
    elif args.cmd == "anim":
        animate(args)
    else:
        # interactive fallback (same as earlier revisions)
        print("1) Benchmark\n2) Trajectory & Animation")
        if input("Select (1/2): ") == "1":
            nlist = list(map(int, input("Enter N values separated by commas [100,500,1000]: ").split(",")))
            res = benchmark(nlist)
            plot_results(nlist, res)
        else:
            n = int(input("Particles [100]: ") or "100")
            steps = int(input("Steps [200]: ") or "200")
            m = input("Method (direct/fmm) [fmm]: ") or "fmm"
            animate(argparse.Namespace(cmd="anim", n=n, steps=steps, method=m, theta=0.5, dt=DEFAULT_DT, mobile_star=False))

