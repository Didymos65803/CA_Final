# benchmark_fmm.py — in‑depth OpenMP & algorithmic speed‑up analysis
# ================================================================
#  This script produces **four** figures:
#   1. size_vs_time.png      – direct vs FMM timing with O(N²) & O(N log N) guidelines
#   2. size_vs_speedup.png   – algorithmic speed‑up (direct / FMM)
#   3. thread_scaling.png    – wall‑time & parallel speed‑up vs thread count
#   4. theta_tradeoff.png    – accuracy vs runtime for several θ values
#
#  All plots are written into ./results_bench/  (created on the fly).
#
#  Requirements:  fmm_openmp (compiled), numpy, matplotlib
#
#  Usage examples
#  --------------
#  $ python benchmark_fmm.py                         # defaults
#  $ python benchmark_fmm.py --sizes 1e3 2e3 4e3 8e3 --threads 1 2 4 8 16 \
#        --theta 0.4 0.6 0.8                        # custom sweep
#
#  Tip: set   OMP_PROC_BIND=spread   OMP_PLACES=cores   in your shell to
#       reduce thread contention when scaling.
# ================================================================
from __future__ import annotations
import os, time, math, argparse, pathlib, sys
import numpy as np, matplotlib.pyplot as plt

try:
    import fmm_openmp as fm
except ImportError:
    sys.exit("fmm_openmp module not found – compile fmm_openmp.cpp first!")

OUT = pathlib.Path("results_bench_rev2"); OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
_rng = np.random.default_rng(42)

def random_system(N: int, domain=50.0):
    x = _rng.uniform(-domain, domain, N).astype(np.float64)
    y = _rng.uniform(-domain, domain, N).astype(np.float64)
    m = np.ones(N, dtype=np.float64)
    return x, y, m

# ---------------------------------------------------------------
# 1)  size sweep (direct vs FMM, with theoretical guidelines)
# ---------------------------------------------------------------

def run_size_sweep(Ns, threads, soft2, domain):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    direct_t, fmm_t = [], []

    for N in Ns:
        x, y, m = random_system(N)
        ax = np.zeros(N); ay = np.zeros(N)

        t0 = time.perf_counter(); fm.direct_force(x, y, m, soft2, ax, ay); direct_t.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); fm.fmm_force   (x, y, m, soft2, domain, ax, ay); fmm_t   .append(time.perf_counter()-t0)
        print(f"N={N:6d}  direct={direct_t[-1]:.4g}s  fmm={fmm_t[-1]:.4g}s  speed‑up={direct_t[-1]/fmm_t[-1]:.2f}")

    # theoretical lines through first data point
    N0 = Ns[0]
    th_n2   = [direct_t[0]*(N/N0)**2           for N in Ns]
    th_nlog = [fmm_t[0]*(N/N0)*math.log2(N)/math.log2(N0) for N in Ns]

    # plot times -------------------------------------------------
    plt.figure(figsize=(6.4,4.2))
    plt.loglog(Ns, direct_t,'o-', label='Direct O(N²)')
    plt.loglog(Ns, fmm_t,   's-', label='FMM  O(N log N)')
    plt.loglog(Ns, th_n2,  '--', color='C0', alpha=.4)
    plt.loglog(Ns, th_nlog,':',  color='C1', alpha=.4)
    plt.title(f'Timing @ {threads} threads'); plt.xlabel('N'); plt.ylabel('time [s]')
    plt.legend(); plt.grid(alpha=.3)
    path_time = OUT/'size_vs_time.png'; plt.tight_layout(); plt.savefig(path_time,dpi=300); plt.close()

    # plot speed‑up ---------------------------------------------
    speed = np.array(direct_t)/np.array(fmm_t)
    plt.figure(figsize=(6.4,4.2))
    plt.loglog(Ns, speed,'o-')
    plt.xlabel('N'); plt.ylabel('direct / FMM'); plt.title('Algorithmic speed‑up')
    plt.grid(alpha=.3)
    path_speed = OUT/'size_vs_speedup.png'; plt.tight_layout(); plt.savefig(path_speed,dpi=300); plt.close()

    print('✓ saved', path_time, 'and', path_speed)

# ---------------------------------------------------------------
# 2) thread scaling for one N
# ---------------------------------------------------------------

def run_scaling(N, thread_list, soft2, domain):
    x, y, m = random_system(N)
    ax = np.zeros(N); ay = np.zeros(N)

    times=[]
    for t in thread_list:
        os.environ['OMP_NUM_THREADS']=str(t); time.sleep(0.05)
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,domain,ax,ay); times.append(time.perf_counter()-t0)
        print(f" {t:3d} threads : {times[-1]:.6f}s")

    speed = [times[0]/tt for tt in times]
    plt.figure(figsize=(6.4,4.2))
    plt.plot(thread_list, times,'o-', label='wall‑time')
    plt.xlabel('#threads'); plt.ylabel('time [s]')
    ax2 = plt.gca().twinx()
    ax2.plot(thread_list, speed,'s--',color='C1', label='speed‑up')
    ax2.set_ylabel('speed‑up')
    plt.title(f'FMM scaling  N={N}')
    plt.grid(alpha=.3)
    plt.tight_layout();
    path = OUT/'thread_scaling.png'; plt.savefig(path,dpi=300); plt.close()
    print('✓ saved', path)

# ---------------------------------------------------------------
# 3) θ trade‑off (accuracy vs time)
# ---------------------------------------------------------------

def run_theta(N, thetas, soft2, domain):
    x,y,m = random_system(N)
    ax = np.zeros(N); ay = np.zeros(N)

    fm.direct_force(x,y,m,soft2,ax,ay); ref = np.hypot(ax,ay)
    errs, tms = [], []
    for th in thetas:
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,domain,ax,ay); dt=time.perf_counter()-t0
        err = np.mean(np.abs(np.hypot(ax,ay)-ref)/(ref+1e-12))
        errs.append(err); tms.append(dt)
        print(f" θ={th:.2f} : time={dt:.3e}s  rel‑err={err:.3e}")

    fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
    a1.semilogy(thetas, errs,'o-'); a1.set_xlabel('θ'); a1.set_ylabel('relative error')
    a2.plot(thetas, tms,'s-');     a2.set_xlabel('θ'); a2.set_ylabel('time [s]')
    fig.suptitle(f'Accuracy vs runtime  (N={N})')
    path=OUT/'theta_tradeoff.png'; plt.tight_layout(); plt.savefig(path,dpi=300); plt.close()
    print('✓ saved', path)

# ===== CLI wrapper ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='FMM vs Direct performance analyser')
    p.add_argument('--sizes',   type=float, nargs='+', default=[1e3,2e3,4e3,8e3], help='particle counts for algorithmic sweep')
    p.add_argument('--threads', type=int,   nargs='+', default=[1,2,4,8], help='thread counts for scaling')
    p.add_argument('--theta',   type=float, nargs='+', default=[0.4,0.6,0.8], help='θ values for accuracy trade‑off')
    p.add_argument('--soft',    type=float, default=0.01, help='softening length')
    p.add_argument('--domain',  type=float, default=100.0, help='simulation box half‑width')
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes]
    print('\n=== Size sweep ============================================')
    run_size_sweep(sizes, threads=max(args.threads), soft2=args.soft**2, domain=args.domain)

    print('\n=== Thread scaling ========================================')
    midN = sizes[len(sizes)//2]
    run_scaling(midN, args.threads, args.soft**2, args.domain)

    print('\n=== θ trade‑off ===========================================')
    run_theta(min(sizes), args.theta, args.soft**2, args.domain)

    print('\nAll figures written to', OUT)

