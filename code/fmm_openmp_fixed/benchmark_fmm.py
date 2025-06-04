# benchmark_fmm.py — in‑depth OpenMP & algorithmic speed‑up analysis
# (REV‑2 – richer thread‑scaling diagnostics)
# ================================================================
#  ‑ size_vs_time.png      – direct vs FMM timings + theory
#  ‑ size_vs_speedup.png   – algorithmic speed‑up (direct / FMM)
#  ‑ thread_scaling.png    – *both* Direct & FMM wall‑time + speed‑up curves
#  ‑ theta_tradeoff.png    – θ accuracy vs runtime
# ================================================================
from __future__ import annotations
import os, time, math, argparse, pathlib, sys
import numpy as np, matplotlib.pyplot as plt
try:
    import fmm_openmp as fm
except ImportError:
    sys.exit("fmm_openmp module not found – compile fmm_openmp.cpp first!")

OUT = pathlib.Path("results_bench_rev3"); OUT.mkdir(exist_ok=True)
_rng = np.random.default_rng(42)

def random_system(N:int, domain=50.):
    return (_rng.uniform(-domain,domain,N).astype(np.float64),
            _rng.uniform(-domain,domain,N).astype(np.float64),
            np.ones(N,dtype=np.float64))
# ------------------------------------------------------------------
# 1) size sweep (unchanged)
# ------------------------------------------------------------------

def run_size_sweep(Ns, threads, soft2, domain):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    direct_t, fmm_t = [], []
    for N in Ns:
        x,y,m = random_system(N); ax=np.zeros(N); ay=np.zeros(N)
        t=time.perf_counter(); fm.direct_force(x,y,m,soft2,ax,ay); direct_t.append(time.perf_counter()-t)
        t=time.perf_counter(); fm.fmm_force(x,y,m,soft2,domain,ax,ay); fmm_t.append(time.perf_counter()-t)
        print(f"N={N:6d} direct={direct_t[-1]:.3g}s  fmm={fmm_t[-1]:.3g}s  speed‑up={direct_t[-1]/fmm_t[-1]:.2f}")
    N0=Ns[0]
    th_n2=[direct_t[0]*(N/N0)**2 for N in Ns]
    th_nl=[fmm_t[0]*(N/N0)*math.log2(N)/math.log2(N0) for N in Ns]
    plt.figure(figsize=(6.2,4)); plt.loglog(Ns,direct_t,'o-',label='Direct O(N²)'); plt.loglog(Ns,fmm_t,'s-',label='FMM O(N log N)')
    plt.loglog(Ns,th_n2,'--',color='C0',alpha=.35); plt.loglog(Ns,th_nl,':',color='C1',alpha=.35)
    plt.title(f'Timing @ {threads} threads'); plt.xlabel('N'); plt.ylabel('time [s]'); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); (OUT/'size_vs_time.png').write_bytes(plt.gcf().canvas.buffer_rgba()); plt.close()
    plt.figure(figsize=(6.2,4)); plt.loglog(Ns,np.array(direct_t)/np.array(fmm_t),'o-')
    plt.xlabel('N'); plt.ylabel('direct / FMM'); plt.title('Algorithmic speed‑up'); plt.grid(alpha=.3)
    plt.tight_layout(); (OUT/'size_vs_speedup.png').write_bytes(plt.gcf().canvas.buffer_rgba()); plt.close()
# ------------------------------------------------------------------
# 2) rich thread scaling (both algos)
# ------------------------------------------------------------------

def run_scaling(N, thread_list, soft2, domain):
    x,y,m=random_system(N); ax=np.zeros(N); ay=np.zeros(N)
    d_times=[]; f_times=[]
    for t in thread_list:
        os.environ['OMP_NUM_THREADS']=str(t); time.sleep(0.05)
        t0=time.perf_counter(); fm.direct_force(x,y,m,soft2,ax,ay); d_times.append(time.perf_counter()-t0)
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,domain,ax,ay);   f_times.append(time.perf_counter()-t0)
        print(f" {t:3d} thr  direct={d_times[-1]:.6f}s  fmm={f_times[-1]:.6f}s")
    speed_d=np.array(d_times[0])/np.array(d_times)
    speed_f=np.array(f_times[0])/np.array(f_times)
    fig,ax1=plt.subplots(figsize=(6.4,4.2))
    ax1.plot(thread_list,d_times,'o-',label='Direct time'); ax1.plot(thread_list,f_times,'s-',label='FMM time')
    ax1.set_xlabel('#threads'); ax1.set_ylabel('wall‑time [s]'); ax1.grid(alpha=.3)
    ax2=ax1.twinx(); ax2.plot(thread_list,speed_d,'o--',color='C0',label='Direct speed‑up')
    ax2.plot(thread_list,speed_f,'s--',color='C1',label='FMM speed‑up'); ax2.set_ylabel('speed‑up')
    lns=ax1.get_lines()+ax2.get_lines(); ax1.legend(lns,[l.get_label() for l in lns],fontsize=8)
    plt.title(f'Thread scaling  N={N}'); plt.tight_layout(); plt.savefig(OUT/'thread_scaling.png',dpi=300); plt.close()
# ------------------------------------------------------------------
# 3) θ trade‑off (same as before but L2 error)
# ------------------------------------------------------------------

def run_theta(N, thetas, soft2, domain):
    x,y,m=random_system(N); ax=np.zeros(N); ay=np.zeros(N)
    fm.direct_force(x,y,m,soft2,ax,ay); ref=np.vstack((ax,ay)).T
    errs=[]; tms=[]
    for th in thetas:
        t0=time.perf_counter(); fm.fmm_force(x,y,m,soft2,domain,ax,ay); tms.append(time.perf_counter()-t0)
        err=np.linalg.norm(np.vstack((ax,ay)).T-ref)/np.linalg.norm(ref); errs.append(err)
        print(f" θ={th:.2f}  t={tms[-1]:.3e}s  L2‑err={errs[-1]:.2e}")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
    a1.semilogy(thetas,errs,'o-'); a1.set_xlabel('θ'); a1.set_ylabel('L2 relative error')
    a2.plot(thetas,tms,'s-'); a2.set_xlabel('θ'); a2.set_ylabel('time [s]')
    fig.suptitle(f'Accuracy vs runtime  N={N}'); plt.tight_layout(); plt.savefig(OUT/'theta_tradeoff.png',dpi=300); plt.close()
# ------------------------------------------------------------------
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--sizes',type=float,nargs='+',default=[2e3,4e3,8e3,1.6e4])
    ap.add_argument('--threads',type=int,nargs='+',default=[1,2,4,8,16]); ap.add_argument('--theta',type=float,nargs='+',default=[0.3,0.5,0.7,1.0])
    ap.add_argument('--soft',type=float,default=0.01); ap.add_argument('--domain',type=float,default=100.0); args=ap.parse_args()
    sizes=[int(s) for s in args.sizes]
    print('\n=== Size sweep ==='); run_size_sweep(sizes,max(args.threads),args.soft**2,args.domain)
    print('\n=== Thread scaling ==='); run_scaling(sizes[len(sizes)//2],args.threads,args.soft**2,args.domain)
    print('\n=== θ trade‑off ==='); run_theta(sizes[1],args.theta,args.soft**2,args.domain)
    print('\nPlots →',OUT)

