#!/usr/bin/env python3
"""
benchmark_fmm.py

A simple benchmark script for the fully‐parallel Barnes–Hut FMM
(via the fmm_omp module). Measures wall‐time for various problem sizes.

Usage:
    python3 benchmark_fmm.py

The script will loop over a predefined list of N (number of bodies),
generate random positions and unit masses, then call
fmm_omp.fmm_force_theta(...) and print the elapsed time for each N.

Before running, ensure you have:
  • Built and installed the fmm_omp extension (fmm_omp*.so)
    by running: python3 setup_openmp.py build_ext --inplace
  • NumPy installed in your Python environment.

"""

import numpy as np
import time
import fmm_omp

def benchmark_fmm(sizes, eps2, domain, theta):
    """
    For each N in `sizes`, generate N random bodies in [0,1]²,
    call fmm_force_theta, and print the elapsed time.
    """
    print(f"{'N':>8}    {'FMM time (s)':>15}")
    print("-" * 26)
    for N in sizes:
        # Generate random positions in [0,1]²
        x = np.random.rand(N).astype(np.float64)
        y = np.random.rand(N).astype(np.float64)
        # Use unit masses for simplicity
        m = np.ones(N, dtype=np.float64)
        # Prepare output arrays
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)

        # Warm‐up call (optional, can help with any lazy initialization)
        fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)

        # Benchmark the FMM call
        t0 = time.time()
        fmm_omp.fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
        t1 = time.time()

        elapsed = t1 - t0
        print(f"{N:8d}    {elapsed:15.6f}")

if __name__ == "__main__":
    # List of problem sizes to test:
    sizes = [2000, 4000, 8000, 16000, 32000, 64000]

    # Domain bounding‐box: [xmin, xmax, ymin, ymax]
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)

    # Softening length squared (eps2) and opening angle theta
    eps2 = 1e-6
    theta = 0.6

    benchmark_fmm(sizes, eps2, domain, theta)

