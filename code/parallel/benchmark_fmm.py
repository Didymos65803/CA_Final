#!/usr/bin/env python3
"""benchmark_fmm.py  –  Direct vs FMM + OpenMP   (makes a threads plot)"""

from __future__ import annotations
import os, sys, time, math, argparse, pathlib, importlib.util
import numpy as np, matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))

def load_so(name, short):
    try: return importlib.import_module(name)
    except ModuleNotFoundError:
        so = here/f"{short}.so"
        spec = importlib.util.spec_from_file_location(name, so)
        mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
fm  = load_so("fmm_openmp","fmm_openmp")
direct = load_so("force_openmp","force_openmp")

OUT = here/"results_bench_rev6"; OUT.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)

def rand_sys(N:int):
    return (_rng.uniform(-100,100,N),
            _rng.uniform(-100,100,N))

def run_scale(N:int, threads:list[int], eps2:float, domain:float, theta:float):
    x,y = map(np.asarray, rand_sys(N))
    direct_t, fmm_t = [], []
    for th in threads:
        os.environ["OMP_NUM_THREADS"]=str(th)
        ax=np.zeros(N); ay=np.zeros(N)
        t=time.perf_counter(); direct.direct_symm(x,y,np.ones(N),eps2,ax,ay)
        direct_t.append(time.perf_counter()-t)

        ax=np.zeros(N); ay=np.zeros(N)
        t=time.perf_counter(); fm.fmm_force_theta(x,y,np.ones(N),eps2,domain,theta,ax,ay)
        fmm_t.append(time.perf_counter()-t)
        print(f"thr={th:2d}  direct={direct_t[-1]:.4g}s  fmm={fmm_t[-1]:.4g}s")

    # ---- plot FMM wall-time vs threads
    plt.figure(figsize=(4.5,3.2))
    plt.plot(threads, fmm_t,'o-')
    plt.xlabel('# threads'); plt.ylabel('FMM wall-time [s]')
    plt.title(f'N = {N:,}  (θ={theta})')
    plt.tight_layout()
    plt.savefig(OUT/"threads_plot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",nargs="+",type=int,default=[200000])
    p.add_argument("--threads",nargs="+",type=int,default=[1,2,4,8,16])
    p.add_argument("--theta_base",type=float,default=0.6)
    p.add_argument("--soft",type=float,default=1.0)
    p.add_argument("--domain",type=float,default=100.0)
    a=p.parse_args()

    run_scale(a.sizes[0], a.threads, a.soft*a.soft, a.domain, a.theta_base)
    print("threads_plot.png written to", OUT.resolve())

