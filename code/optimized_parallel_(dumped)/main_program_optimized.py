#!/usr/bin/env python3
"""main_program_optimized.py
================================
High‑level driver that demonstrates **performance**, **parallel scaling**,
**accuracy vs. θ** and **energy‑conservation** for the new kernels:

    • force_kernel_opt  – lock‑free O(N²) direct solver
    • fmm_kernel_opt    – parallel Barnes–Hut / FMM solver

Run without arguments for a compact benchmark table, or use flags:
    --scaling     parallel scaling graph
    --accuracy    θ‑sweep accuracy & timing plot
    --energy      simple leap‑frog energy test

All plots are saved into ./results_opt/ as PNGs.
"""
from __future__ import annotations
import os, sys, time, math, argparse, pathlib
import numpy as np
import matplotlib.pyplot as plt
import force_kernel_opt
import fmm_kernel_opt

OUT_DIR = pathlib.Path("rev2"); OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# OpenMP threads – default to physical cores unless user overrides.
# -----------------------------------------------------------------------------
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
print(f"[ OpenMP threads = {os.environ['OMP_NUM_THREADS']} ]")

# -----------------------------------------------------------------------------
# Kernel imports
# -----------------------------------------------------------------------------
try:
    import force_kernel_opt as force_kernel
    HAS_DIRECT = True
except ImportError as e:
    print("⚠  direct kernel missing:", e)
    HAS_DIRECT = False

try:
    import fmm_kernel_opt as fmm_kernel
    HAS_FMM = True
except ImportError as e:
    print("⚠  FMM kernel missing:", e)
    HAS_FMM = False

if not (HAS_DIRECT or HAS_FMM):
    sys.exit("No kernels available – build them with setup_optimized.py")

# -----------------------------------------------------------------------------
# Synthetic particle distribution (flat disk) for repeatable tests.
# -----------------------------------------------------------------------------
class Particles:
    def __init__(self, N, domain=50.0, seed=42):
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 2*math.pi, N)
        rad = domain * np.sqrt(rng.uniform(0, 1, N))
        self.x = np.ascontiguousarray(rad*np.cos(ang), dtype=np.float64)
        self.y = np.ascontiguousarray(rad*np.sin(ang), dtype=np.float64)
        self.m = np.ascontiguousarray(rng.uniform(0.8, 1.2, N), dtype=np.float64)
        self.domain = domain

# -----------------------------------------------------------------------------
# Thin wrappers
# -----------------------------------------------------------------------------

def direct(pset: Particles, eps=0.01):
    N = pset.x.size
    ax = np.zeros(N, dtype=np.float64); ay = np.zeros(N, dtype=np.float64)
    force_kernel.direct_force(pset.x, pset.y, pset.m, eps*eps, ax, ay)
    return ax, ay


def fmm(pset: Particles, theta=0.6, eps=0.01):
    N = pset.x.size
    ax = np.zeros(N, dtype=np.float64); ay = np.zeros(N, dtype=np.float64)
    fmm_kernel.fmm_force(pset.x, pset.y, pset.m, N,
                         pset.domain, theta, 12, eps, 1.0, ax, ay)
    return ax, ay

# -----------------------------------------------------------------------------
# 1) Performance benchmark table
# -----------------------------------------------------------------------------

def benchmark_table(sizes=(200,500,1000,2000), rep=3):
    print("\n*** Performance benchmark ***")
    hdr = f"{'N':>6}  {'Direct (s)':>12}  {'FMM (s)':>12}  {'Speed‑up':>8}"
    print(hdr); print("-"*len(hdr))

    rows = []
    for N in sizes:
        p = Particles(N)
        t_dir = t_fmm = np.nan

        if HAS_DIRECT and N<=3000:
            t0=time.perf_counter();
            for _ in range(rep): direct(p)
            t_dir=(time.perf_counter()-t0)/rep
        if HAS_FMM:
            t0=time.perf_counter();
            for _ in range(rep): fmm(p)
            t_fmm=(time.perf_counter()-t0)/rep

        speed = t_dir/t_fmm if np.isfinite(t_dir) and np.isfinite(t_fmm) else np.nan
        print(f"{N:6d}  {t_dir:12.6f}  {t_fmm:12.6f}  {speed:8.2f}")
        rows.append((N,t_dir,t_fmm,speed))
    return rows

# -----------------------------------------------------------------------------
# 2) Parallel‑scaling plot (threads vs time)
# -----------------------------------------------------------------------------

def scaling_plot(N=1000, theta=0.6, eps=0.01):
    if not HAS_FMM:
        print("Scaling test needs FMM."); return
    p=Particles(N)
    thread_counts=[1,2,4,8,16][:int(os.cpu_count()*1.3)]
    times=[]
    for t in thread_counts:
        os.environ["OMP_NUM_THREADS"]=str(t); time.sleep(0.05)
        t0=time.perf_counter();
        for _ in range(3): fmm(p,theta,eps)
        times.append((time.perf_counter()-t0)/3)
        print(f"  {t:2d} threads : {times[-1]:.4f} s")

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(thread_counts,times,'o-');
    plt.xlabel('# threads'); plt.ylabel('time [s]')
    plt.title(f'FMM scaling  N={N}')
    plt.grid(alpha=.3)
    path=OUT_DIR/"thread_scaling.png"; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print(f"✓ scaling plot saved {path}")

# -----------------------------------------------------------------------------
# 3) Accuracy vs θ plot
# -----------------------------------------------------------------------------

def accuracy_plot(N=300, thetas=(0.3,0.5,0.7,1.0), eps=0.01):
    if not (HAS_DIRECT and HAS_FMM):
        print("Accuracy test needs both kernels."); return
    p=Particles(N,seed=123)
    ax_ref, ay_ref = direct(p,eps)
    ref = np.hypot(ax_ref,ay_ref)

    errs=[]; tms=[]
    for th in thetas:
        t0=time.perf_counter(); ax,ay=fmm(p,th,eps); tm=time.perf_counter()-t0
        err=np.mean(np.abs(np.hypot(ax,ay)-ref)/(ref+1e-12))
        errs.append(err); tms.append(tm)
        print(f"θ={th:.2f}  rel‑err={err:.3e}  t={tm:.5f}s")

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))
    ax1.semilogy(thetas,errs,'o-'); ax1.set_xlabel('θ'); ax1.set_ylabel('relative error')
    ax2.plot(thetas,tms,'o-');      ax2.set_xlabel('θ'); ax2.set_ylabel('time [s]')
    fig.suptitle(f'FMM accuracy / performance  N={N}')
    path=OUT_DIR/"accuracy_theta.png"; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print(f"✓ accuracy plot saved {path}")

# -----------------------------------------------------------------------------
# 4) Simple energy‑conservation monitoring (leap‑frog)
# -----------------------------------------------------------------------------

def energy_test(N=256, steps=1000, dt=0.01, theta=0.6):
    if not HAS_FMM: print("energy test needs FMM"); return
    p=Particles(N,seed=7)
    vx=np.zeros(N); vy=np.zeros(N)
    ax,ay=fmm(p,theta)

    def total_E():
        KE=.5*np.sum(p.m*(vx*vx+vy*vy)); PE=0.
        for i in range(N):
            dx=p.x[i]-p.x[i+1:]; dy=p.y[i]-p.y[i+1:]
            r=np.hypot(dx,dy)+1e-8; PE-=np.sum(p.m[i]*p.m[i+1:]/r)
        return KE+PE
    E0=total_E(); ts=[]; Es=[]
    for s in range(steps):
        vx+=.5*dt*ax; vy+=.5*dt*ay
        p.x+=dt*vx;   p.y+=dt*vy
        ax,ay=fmm(p,theta)
        vx+=.5*dt*ax; vy+=.5*dt*ay
        if (s+1)%20==0: ts.append((s+1)*dt); Es.append(total_E())
    rel=np.abs((np.array(Es)-E0)/E0)
    plt.figure(figsize=(6,4)); plt.plot(ts,rel); plt.yscale('log')
    plt.xlabel('time'); plt.ylabel('|ΔE|/E0'); plt.title('Energy drift')
    path=OUT_DIR/"energy_drift.png"; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print(f"✓ energy plot saved {path}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser(description="Optimised N‑body benchmark driver")
    ap.add_argument('--scaling', action='store_true', help='thread‑scaling plot')
    ap.add_argument('--accuracy', action='store_true', help='θ‑sweep plot')
    ap.add_argument('--energy', action='store_true', help='leap‑frog energy test')
    args=ap.parse_args()

    benchmark_table()
    if args.scaling:  scaling_plot()
    if args.accuracy: accuracy_plot()
    if args.energy:   energy_test()

if __name__=='__main__':
    main()

