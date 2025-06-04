# benchmark_fmm.py – complete benchmark + plotting utility
# =========================================================
# Requires:
#   • compiled fmm_openmp module (fmm_openmp.so)
#   • matplotlib, numpy
#
# It will:
#   1. measure direct vs FMM times for several N
#   2. produce a thread‑scaling plot for one representative N
#   3. sweep θ to show accuracy vs runtime
#   4. save three PNGs into ./results_bench/
#
# Usage:
#   python benchmark_fmm.py               # default settings
#   python benchmark_fmm.py --sizes 500 1000 2000 --threads 1 2 4 8 --theta 0.3 0.5 0.7
# =========================================================
from __future__ import annotations
import os, time, math, argparse, pathlib
import numpy as np
import matplotlib.pyplot as plt
import fmm_openmp as fm

OUT = pathlib.Path("results_bench"); OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------
# reference direct vs FMM times over sizes
# ---------------------------------------------------------
def benchmark_sizes(sizes, threads=4, soft=0.01):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    soft2 = soft*soft
    ref, fmm = [], []

    for N in sizes:
        rng = np.random.default_rng(42)
        x = rng.uniform(-50,50,N).astype(np.float64)
        y = rng.uniform(-50,50,N).astype(np.float64)
        m = np.ones(N, dtype=np.float64)
        ax = np.zeros(N); ay = np.zeros(N)

        # direct
        t0=time.perf_counter(); fm.direct_force(x,y,m,soft2,ax,ay); td=time.perf_counter()-t0
        # fmm
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,100.0,ax,ay); tf=time.perf_counter()-t0
        ref.append(td); fmm.append(tf)
        print(f"N={N:6}  direct={td:.4e}s  fmm={tf:.4e}s  speedup={td/tf:.2f}")

    plt.figure(figsize=(6,4))
    plt.loglog(sizes, ref,'o-',label='Direct O(N²)')
    plt.loglog(sizes, fmm,'s-',label='FMM O(N log N)')
    plt.xlabel('N'); plt.ylabel('time [s]'); plt.title(f'Timing @ {threads} threads')
    plt.grid(alpha=.3); plt.legend()
    path=OUT/'size_sweep.png'; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print('✓ size plot ->',path)

# ---------------------------------------------------------
# thread scaling for fixed N
# ---------------------------------------------------------

def scaling(N, thread_list, soft=0.01):
    rng = np.random.default_rng(1)
    x = rng.uniform(-50,50,N).astype(np.float64)
    y = rng.uniform(-50,50,N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    ax = np.zeros(N); ay = np.zeros(N)
    soft2 = soft*soft

    times=[]
    for t in thread_list:
        os.environ['OMP_NUM_THREADS']=str(t); time.sleep(0.1)
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,100.0,ax,ay); times.append(time.perf_counter()-t0)
        print(f"{t} threads : {times[-1]:.5f}s")

    speed = [times[0]/tt for tt in times]
    plt.figure(figsize=(6,4))
    plt.plot(thread_list, times,'o-'); plt.xlabel('#threads'); plt.ylabel('time [s]');
    plt.twinx(); plt.plot(thread_list, speed,'s--',color='orange'); plt.ylabel('speed‑up')
    plt.title(f'FMM scaling N={N}')
    path=OUT/'thread_scaling.png'; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print('✓ scaling plot ->',path)

# ---------------------------------------------------------
# θ sweep to show accuracy/performance tradeoff
# ---------------------------------------------------------

def theta_sweep(N, thetas, soft=0.01):
    rng = np.random.default_rng(7)
    x = rng.uniform(-50,50,N).astype(np.float64)
    y = rng.uniform(-50,50,N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    ax=np.zeros(N); ay=np.zeros(N)
    soft2=soft*soft

    # reference
    fm.direct_force(x,y,m,soft2,ax,ay)
    ref=np.hypot(ax,ay)

    errs=[]; times=[]
    for th in thetas:
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,100.0,ax,ay); dt=time.perf_counter()-t0
        force=np.hypot(ax,ay)
        errs.append(np.mean(np.abs(force-ref)/(ref+1e-12)))
        times.append(dt)
        print(f"θ={th:.2f}  time={dt:.4e}s  rel‑err={errs[-1]:.3e}")

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
    ax1.semilogy(thetas,errs,'o-'); ax1.set_xlabel('θ'); ax1.set_ylabel('relative error')
    ax2.plot(thetas,times,'s-');    ax2.set_xlabel('θ'); ax2.set_ylabel('time [s]')
    fig.suptitle(f'Accuracy vs θ  (N={N})')
    path=OUT/'theta_sweep.png'; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print('✓ θ sweep plot ->',path)

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='+', type=int, default=[500,1000,2000,4000])
    ap.add_argument('--threads', nargs='+', type=int, default=[1,2,4,8])
    ap.add_argument('--theta',  nargs='+', type=float, default=[0.3,0.5,0.7,1.0])
    ap.add_argument('--soft', type=float, default=0.01)
    args=ap.parse_args()

    benchmark_sizes(args.sizes, threads=args.threads[-1], soft=args.soft)
    scaling(args.sizes[len(args.sizes)//2], args.threads, soft=args.soft)
    theta_sweep(min(args.sizes), args.theta, soft=args.soft)
    print('\nAll plots saved to', OUT)

