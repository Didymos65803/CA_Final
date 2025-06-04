import os, time, numpy as np, fmm_openmp as fm

N = 10000                 # enough to see scaling
x = np.random.uniform(-50, 50, N).astype(np.float64)
y = np.random.uniform(-50, 50, N).astype(np.float64)
m = np.ones(N, dtype=np.float64)
ax = np.zeros(N); ay = np.zeros(N)
soft2 = 0.01**2

print("Threads   FMM time (s)")
for t in (1, 2, 4, 8):
    os.environ["OMP_NUM_THREADS"] = str(t)
    time.sleep(0.1)
    t0 = time.time()
    fm.fmm_force(x, y, m, soft2, 100.0, ax, ay)
    print(f"{t:>6} : {time.time()-t0:.4f}")

