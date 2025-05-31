# benchmark_fmm_vs_bh_vs_direct.py
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
from force_kernel import bh_omp, direct_omp
from fmm_kernel import fmm_omp   # ← now implemented below

Ns = [100, 200, 400, 800, 1600]
G = 1.0
soft = 0.05
L = 1.0

results = []

for N in tqdm(Ns, desc="Benchmarking"):
    rng = np.random.default_rng(42)
    x = rng.uniform(-L/2, L/2, N)
    y = rng.uniform(-L/2, L/2, N)
    m = rng.uniform(0.5, 1.5, N)

    # --- Direct ---
    t0 = time.time()
    ax_d, ay_d = direct_omp(x, y, m, G, soft)
    t1 = time.time()
    t_direct = t1 - t0

    # --- BH ---
    t0 = time.time()
    ax_b, ay_b = bh_omp(x, y, m, domain=L, theta=0.5, G=G, soft=soft)
    t1 = time.time()
    t_bh = t1 - t0

    # --- FMM ---
    t0 = time.time()
    ax_f, ay_f = fmm_omp(x, y, m, domain=L, G=G, soft=soft, order=4)
    t1 = time.time()
    t_fmm = t1 - t0

    results.append(dict(N=N, Direct=t_direct, BH=t_bh, FMM=t_fmm))

# Save
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)

plt.figure()
plt.loglog(df.N, df.Direct, 'o-', label='Direct')
plt.loglog(df.N, df.BH,     's-', label='Barnes–Hut')
plt.loglog(df.N, df.FMM,    '^-', label='FMM')
plt.xlabel("N")
plt.ylabel("Runtime (s)")
plt.title("Gravity Solver Runtime Scaling")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("benchmark_scaling.png")
plt.show()
