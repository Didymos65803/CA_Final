"""Time comparison of direct_omp (O N²) vs bh_omp (O N log N).
Outputs scaling.png and timings.csv"""
import numpy as np, time, csv, matplotlib.pyplot as plt, matplotlib
matplotlib.use('Agg')
from force_kernel import bh_omp
from force_kernel import bh_omp as _   # placeholder to ensure import (direct_omp will come from force_kernel.cpp)
from force_kernel import direct_omp     # compiled in force_kernel.cpp

N_list = [100, 200, 400, 800, 1600]
THETA, DOMAIN, G, SOFT = 0.5, 60.0, 1.0, 0.05

timings = []
for N in N_list:
    x = np.random.uniform(-20,20,N)
    y = np.random.uniform(-20,20,N)
    m = np.ones(N)/N

    # direct
    t0 = time.perf_counter()
    ax, ay = direct_omp(x,y,m,G,SOFT)        # assumes direct_omp signature
    t1 = time.perf_counter()
    t_direct = t1-t0

    # BH
    t0 = time.perf_counter()
    ax, ay = bh_omp(x,y,m,DOMAIN,THETA,G,SOFT)
    t1 = time.perf_counter()
    t_bh = t1-t0

    timings.append((N,t_direct,t_bh))
    print(f"N={N:5d}  direct={t_direct:.4f}s  BH={t_bh:.4f}s  speedup={t_direct/t_bh:.1f}×")

# save CSV
with open('timings.csv','w',newline='') as f:
    csv.writer(f).writerows([('N','direct','BH')]+timings)

# plot scaling
N, Tdir, Tbh = zip(*timings)
plt.figure(figsize=(6,4))
plt.loglog(N,Tdir,'o-',label='direct O(N²)')
plt.loglog(N,Tbh,'s-',label='BH O(N log N)')
plt.xlabel('N'); plt.ylabel('wall‑clock time [s]')
plt.legend(); plt.title('Direct vs Barnes–Hut scaling')
plt.grid(True, which='both', ls=':')
plt.savefig('scaling.png', dpi=300)
print('timings.csv and scaling.png saved')