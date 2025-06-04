#!/usr/bin/env python3
"""benchmark_fmm.py  –  Direct (O(N²)) vs. FMM (O(N log N)) with OpenMP"""

from __future__ import annotations
import os, sys, time, math, argparse, pathlib, importlib.util
import numpy as np, matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))

def _dl(name: str, short: str):
    try: return importlib.import_module(name)
    except ModuleNotFoundError:
        so = here/f"{short}.so"
        spec = importlib.util.spec_from_file_location(name, so)
        mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
fm   = _dl("fmm_openmp",   "fmm_openmp")
direct = _dl("force_openmp","force_openmp")

OUT = here/"results_bench_rev6"; OUT.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)

def random_sys(N:int):
    return (_rng.uniform(-100,100,N),
            _rng.uniform(-100,100,N),
            np.ones(N))

# ------------------------------------------------------------------ size sweep
def run_size(Ns, threads, eps2, domain, theta):
    os.environ["OMP_NUM_THREADS"]=str(threads)
    dt,ft=[],[]
    for N in Ns:
        print(f"\n--- N={N} ---", flush=True)
        x,y,m = map(np.asarray, random_sys(N))
        ax=np.zeros(N); ay=np.zeros(N)

        t=time.perf_counter(); direct.direct_symm(x,y,m,eps2,ax,ay)
        dt.append(time.perf_counter()-t); print(f"direct  {dt[-1]:.4g}s")

        t=time.perf_counter(); fm.fmm_force_theta(x,y,m,eps2,domain,theta,ax,ay)
        ft.append(time.perf_counter()-t); print(f"fmm     {ft[-1]:.4g}s")

        print(f"speed-up {dt[-1]/ft[-1]:.2f}")

    np.savetxt(OUT/"speed.tsv", np.c_[Ns, np.array(dt)/ft], header="N speedup")
    plt.figure(); plt.loglog(Ns, dt,'o-',label='Direct'); plt.loglog(Ns,ft,'s-',label='FMM')
    plt.legend(); plt.xlabel('N'); plt.ylabel('time [s]')
    plt.savefig(OUT/"size_vs_time.png",dpi=300); plt.close()

# -------------------------------------------------------------- thread scaling
def run_scale(N, thread_list, eps2, domain, theta):
    x,y,m = map(np.asarray, random_sys(N))
    dt=[]; ft=[]
    for th in thread_list:
        os.environ["OMP_NUM_THREADS"]=str(th)
        ax=np.zeros(N); ay=np.zeros(N)
        t=time.perf_counter(); direct.direct_symm(x,y,m,eps2,ax,ay)
        dt.append(time.perf_counter()-t)
        t=time.perf_counter(); fm.fmm_force_theta(x,y,m,eps2,domain,theta,ax,ay)
        ft.append(time.perf_counter()-t)
        print(f"thr={th:2d}  direct={dt[-1]:.4g}s  fmm={ft[-1]:.4g}s")
    np.savetxt(OUT/"scale.tsv", np.c_[thread_list, dt, ft], header="thr direct fmm")

# ------------------------------------------------------------------------- main
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--sizes",nargs="+",type=int,
        default=[2_000,10_000,50_000])
    p.add_argument("--threads",nargs="+",type=int, default=[1,2,4,8,16])
    p.add_argument("--theta_base",type=float,default=0.6); p.add_argument(
        "--theta",nargs="+",type=float,default=[0.3,0.5,0.7,1.0])
    p.add_argument("--soft",type=float,default=1.0); p.add_argument(
        "--domain",type=float,default=100.0)
    a=p.parse_args(); eps2=a.soft*a.soft

    run_size(a.sizes,max(a.threads),eps2,a.domain,a.theta_base)
    run_scale(max(a.sizes),a.threads,eps2,a.domain,a.theta_base)
    print("Figures & tables in", OUT)

