# main_program_parallel_final.py
# =====================================
#
# Interactive 2D N-Body Playground (Parallel, High-Precision)
# Forces OpenMP to use exactly 8 threads by setting OMP_NUM_THREADS=8
#
# Copy this entire file over your existing main_program_parallel_final.py,
# then run `python3 main_program_parallel_final.py` as usual.
#
# -----------------------------------------------------

import os

# Force OpenMP to use 8 threads (regardless of os.cpu_count())
os.environ["OMP_NUM_THREADS"] = "8"

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Now that OMP_NUM_THREADS is set, importing the extensions will pick it up.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import force_kernel
import fmm_kernel

# -----------------------------------------------------
# Global output directory
# -----------------------------------------------------
OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -----------------------------------------------------
# Utility: initialize N particles randomly in a disk
# -----------------------------------------------------
def initialize_particles(N, domain_size):
    angles = np.random.rand(N) * 2.0 * math.pi
    radii  = domain_size * np.sqrt(np.random.rand(N))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    # Give each mass = 1/N
    m = np.ones(N) * (1.0 / N)
    return x, y, m

# -----------------------------------------------------
# Leapfrog integrator step using a chosen kernel
# -----------------------------------------------------
def leapfrog_step(x, y, vx, vy, m, dt, kernel_fn,
                  soft, G, domain_size, theta, maxLeaf):
    N = len(x)
    # (1) Half‐kick
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    kernel_fn(x, y, m, soft, ax, ay) if kernel_fn == force_kernel.direct_force \
      else kernel_fn(x.tolist(), y.tolist(), m.tolist(),
                     N, domain_size, theta, maxLeaf, soft, G, ax.tolist(), ay.tolist()) or \
           (ax := np.array(ax), ay := np.array(ay))

    vx += 0.5 * dt * ax
    vy += 0.5 * dt * ay

    # (2) Drift
    x += dt * vx
    y += dt * vy

    # (3) Full‐kick
    ax.fill(0.0)
    ay.fill(0.0)
    kernel_fn(x, y, m, soft, ax, ay) if kernel_fn == force_kernel.direct_force \
      else kernel_fn(x.tolist(), y.tolist(), m.tolist(),
                     N, domain_size, theta, maxLeaf, soft, G, ax.tolist(), ay.tolist()) or \
           (ax := np.array(ax), ay := np.array(ay))

    vx += 0.5 * dt * ax
    vy += 0.5 * dt * ay

    return x, y, vx, vy

# -----------------------------------------------------
# Option (1): Quick Benchmark Scaling
# -----------------------------------------------------
# (In main_program_parallel_final.py, replace the entire quick_benchmark() with:)

def quick_benchmark():
    print("\nQuick Benchmark Scaling\n-----------------------")
    print("Choose mode:")
    print("  1) small‐N (50 → 2000)")
    print("  2) large‐N (600 → 4000)")
    print(" q) back to main menu")
    choice = input("Select: ").strip().lower()

    if choice == '1':
        Ns = [50, 100, 200, 500, 1000, 2000]
    elif choice == '2':
        Ns = [600, 1000, 2000, 3000, 4000]
    else:
        return

    dt = 1e-3
    steps = 10
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    nthreads_used = 8
    print(f"Using {nthreads_used} threads (OpenMP).")

    results = []  # will store (N, t_direct, t_BH, t_FMM)

    for N in Ns:
        print(f"\nN = {N}")
        # Generate random ICs as NumPy arrays
        x, y, m = initialize_particles(N, domain_size)
        vx = np.zeros(N, dtype=np.float64)
        vy = np.zeros(N, dtype=np.float64)

        # 1) Direct
        t0 = time.time()
        for _ in range(steps):
            ax = np.zeros(N, dtype=np.float64)
            ay = np.zeros(N, dtype=np.float64)
            force_kernel.direct_force(x, y, m, soft*soft, ax, ay)
        t_direct = (time.time() - t0) / steps
        print(f"  Direct:      {t_direct:.6f} s")

        # 2) Barnes‐Hut (FMM with maxLeaf=1)
        t0 = time.time()
        for _ in range(steps):
            bx = np.zeros(N, dtype=np.float64)
            by = np.zeros(N, dtype=np.float64)
            fmm_kernel.fmm_force(x, y, m,
                                 N, domain_size, theta, 1,
                                 soft, G, bx, by)
        t_bh = (time.time() - t0) / steps
        print(f"  Barnes‐Hut:  {t_bh:.6f} s (θ={theta})")

        # 3) FMM (maxLeaf=8)
        t0 = time.time()
        for _ in range(steps):
            fx = np.zeros(N, dtype=np.float64)
            fy = np.zeros(N, dtype=np.float64)
            fmm_kernel.fmm_force(x, y, m,
                                 N, domain_size, theta, maxLeaf,
                                 soft, G, fx, fy)
        t_fmm = (time.time() - t0) / steps
        print(f"  FMM:         {t_fmm:.6f} s")

        results.append((N, t_direct, t_bh, t_fmm))

    # Save CSV
    outfile = os.path.join(OUTPUT_DIR, "scaling_quick.csv")
    with open(outfile, "w") as fd:
        fd.write("N,Direct,BH,FMM\n")
        for (N, d, bh, f) in results:
            fd.write(f"{N},{d:.8e},{bh:.8e},{f:.8e}\n")
    print(f"\n✓ Saved quick‐benchmark CSV to {outfile}")

    # Plot log‐log
    Ns_plot = [r[0] for r in results]
    direct_plot = [r[1] for r in results]
    bh_plot = [r[2] for r in results]
    fmm_plot = [r[3] for r in results]

    plt.figure(figsize=(6,4))
    plt.loglog(Ns_plot, direct_plot, 'o-r', label="Direct O(N²)")
    plt.loglog(Ns_plot, bh_plot,     's-b', label="BH O(N log N)")
    plt.loglog(Ns_plot, fmm_plot,    '^-g', label="FMM O(N)")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time per Step (s)")
    plt.title("Quick Scaling Comparison (8 threads)")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, "scaling_quick.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"Saved plot to {pngfile}")

# -----------------------------------------------------
# Option (2): Save Trajectory + Energy Plot
# -----------------------------------------------------
def save_trajectory_and_energy():
    print("\nSave Trajectory + Energy Plot\n-----------------------------")
    N = int(input("Enter N (e.g. 200): ").strip())
    domain_size = float(input("Enter domain radius (e.g. 50.0): ").strip())
    theta = float(input("Enter θ (e.g. 0.5): ").strip())
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    dt = 1e-3
    nsteps = 200

    solver = input("Solver (direct/bh/fmm): ").strip().lower()
    if solver not in ("direct","bh","fmm"):
        print("Invalid solver. Returning to menu.")
        return

    # Initialize particles
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    x[:], y[:], m = initialize_particles(N, domain_size)

    times = []
    energies = []
    traj_x = np.zeros((nsteps, N), dtype=np.float64)
    traj_y = np.zeros((nsteps, N), dtype=np.float64)

    for tstep in range(nsteps):
        # Compute current energy (kinetic + potential)
        pot = 0.0
        if solver == "direct":
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = math.sqrt(dx*dx + dy*dy + soft*soft)
                    if dist > 0:
                        pot -= G * m[i] * m[j] / dist
        else:
            # Use direct‐sum potential to track relative error baseline
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = math.sqrt(dx*dx + dy*dy + soft*soft)
                    if dist > 0:
                        pot -= G * m[i] * m[j] / dist

        kin = 0.0
        for i in range(N):
            kin += 0.5 * m[i] * (vx[i]*vx[i] + vy[i]*vy[i])
        total_energy = kin + pot
        times.append(tstep*dt)
        energies.append(total_energy)

        traj_x[tstep, :] = x[:]
        traj_y[tstep, :] = y[:]

        # Leapfrog integration
        if solver == "direct":
            ax_arr = np.zeros(N, dtype=np.float64)
            ay_arr = np.zeros(N, dtype=np.float64)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr
            x += dt * vx
            y += dt * vy
            ax_arr.fill(0.0)
            ay_arr.fill(0.0)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr

        else:
            fx = [0.0]*N
            fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)
            x += dt * vx
            y += dt * vy
            fx = [0.0]*N
            fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)

    # 2.a) Save trajectory GIF
    fig = plt.figure(figsize=(5,5))
    axplt = plt.subplot(111)
    scat = axplt.scatter(traj_x[0,:], traj_y[0,:], s=5, c='b')
    axplt.set_xlim(-domain_size, domain_size)
    axplt.set_ylim(-domain_size, domain_size)
    axplt.set_title(f"Trajectory ({solver.upper()}, N={N}, threads=8)")

    def animate(frame):
        scat.set_offsets(np.vstack((traj_x[frame,:], traj_y[frame,:])).T)
        return scat,

    ani = animation.FuncAnimation(fig, animate, frames=nsteps, interval=50, blit=True)
    giffile = os.path.join(OUTPUT_DIR, f"trajectory_{solver}_{N}_8.gif")
    ani.save(giffile, writer='pillow', fps=20)
    plt.close()
    print(f"✓ Saved trajectory GIF to {giffile}")

    # 2.b) Save energy vs time
    plt.figure(figsize=(5,3))
    plt.plot(times, energies, '-k', linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"Energy vs Time ({solver.upper()}, N={N}, threads=8)")
    plt.grid(True, which='both', ls='--', lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, f"energy_{solver}_{N}_8.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"✓ Saved energy plot to {pngfile}")

# -----------------------------------------------------
# Option (3): Live Simulation Animation
# -----------------------------------------------------
def live_simulation():
    print("\nLive Simulation Animation\n-------------------------")
    N = int(input("Enter N (e.g. 200): ").strip())
    domain_size = float(input("Enter domain radius (e.g. 50.0): ").strip())
    theta = float(input("Enter θ (e.g. 0.5): ").strip())
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    dt = 5e-4
    nsteps = 400

    solver = input("Solver (direct/bh/fmm): ").strip().lower()
    if solver not in ("direct","bh","fmm"):
        print("Invalid solver. Returning to menu.")
        return

    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    x[:], y[:], m = initialize_particles(N, domain_size)

    fig = plt.figure(figsize=(5,5))
    axplt = plt.subplot(111)
    scat = axplt.scatter(x, y, s=5, c='b')
    axplt.set_xlim(-domain_size, domain_size)
    axplt.set_ylim(-domain_size, domain_size)
    axplt.set_title(f"Live Simulation ({solver.upper()}, N={N}, threads=8)")

    def update_frame(frame):
        nonlocal x, y, vx, vy
        if solver == "direct":
            ax_arr = np.zeros(N, dtype=np.float64)
            ay_arr = np.zeros(N, dtype=np.float64)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr
            x += dt * vx
            y += dt * vy
            ax_arr.fill(0.0)
            ay_arr.fill(0.0)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr

        else:
            fx = [0.0]*N
            fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)
            x += dt * vx
            y += dt * vy
            fx = [0.0]*N
            fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)

        scat.set_offsets(np.vstack((x, y)).T)
        return scat,

    ani = animation.FuncAnimation(fig, update_frame, frames=nsteps, interval=30, blit=True)
    giffile = os.path.join(OUTPUT_DIR, f"live_{solver}_{N}_8.gif")
    ani.save(giffile, writer='pillow', fps=30)
    plt.close()
    print(f"✓ Saved live‐simulation GIF to {giffile}")

# -----------------------------------------------------
# Option (4): Large‐N Scaling Test
# -----------------------------------------------------
def largeN_scaling():
    print("\nLarge‐N Scaling Test\n--------------------")
    Ns = [600, 1000, 2000, 3000, 4000]
    dt = 1e-3
    steps = 10
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    print(f"Using 8 threads.")

    results = []
    for N in Ns:
        print(f"\nN = {N}")
        x, y, m = initialize_particles(N, domain_size)

        t0 = time.time()
        for _ in range(steps):
            force_kernel.direct_force(x, y, m, 1e-4,
                                      ax := [0.0]*N,
                                      ay := [0.0]*N)
        t_direct = (time.time() - t0) / steps
        print(f"  Direct: {t_direct:.6f}")

        t0 = time.time()
        for _ in range(steps):
            bx = [0.0]*N
            by = [0.0]*N
            fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                 N, domain_size, theta, 1,
                                 soft, G, bx, by)
        t_bh = (time.time() - t0) / steps
        print(f"  BH:     {t_bh:.6f}")

        t0 = time.time()
        for _ in range(steps):
            fx = [0.0]*N
            fy = [0.0]*N
            fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                 N, domain_size, theta, maxLeaf,
                                 soft, G, fx, fy)
        t_fmm = (time.time() - t0) / steps
        print(f"  FMM:    {t_fmm:.6f}")

        results.append((N, t_direct, t_bh, t_fmm))

    # Save CSV
    outfile = os.path.join(OUTPUT_DIR, "scaling_largeN.csv")
    with open(outfile, "w") as fd:
        fd.write("N,Direct,BH,FMM\n")
        for (N, d, bh, f) in results:
            fd.write(f"{N},{d:.8e},{bh:.8e},{f:.8e}\n")
    print(f"\n✓ Saved largeN CSV to {outfile}")

    # Plot
    Ns_plot = [r[0] for r in results]
    direct_plot = [r[1] for r in results]
    bh_plot = [r[2] for r in results]
    fmm_plot = [r[3] for r in results]

    plt.figure(figsize=(6,4))
    plt.loglog(Ns_plot, direct_plot, 'o-r', label="Direct O(N²)")
    plt.loglog(Ns_plot, bh_plot,     's-b', label="BH O(N log N)")
    plt.loglog(Ns_plot, fmm_plot,    '^-g', label="FMM O(N)")
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time per Step (s)")
    plt.title("Large‐N Scaling Comparison (8 threads)")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, "scaling_largeN.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"✓ Saved plot to {pngfile}")

# -----------------------------------------------------
# Option (5): Energy Conservation Test
# -----------------------------------------------------
def energy_conservation_test():
    print("\nEnergy Conservation Test\n------------------------")
    N = int(input("Enter N (e.g. 200): ").strip())
    domain_size = float(input("Enter domain radius (e.g. 50.0): ").strip())
    theta = float(input("Enter θ (e.g. 0.5): ").strip())
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    dt = 5e-4
    nsteps = 500

    solver = input("Solver (direct/bh/fmm): ").strip().lower()
    if solver not in ("direct","bh","fmm"):
        print("Invalid solver. Returning.")
        return

    x, y, m = initialize_particles(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)

    # Compute initial energy E0 via direct‐sum
    E0 = 0.0
    for i in range(N):
        E0 += 0.5 * m[i] * (vx[i]*vx[i] + vy[i]*vy[i])
    for i in range(N):
        for j in range(i+1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = math.sqrt(dx*dx + dy*dy + soft*soft)
            if dist > 0:
                E0 -= G * m[i] * m[j] / dist

    times = []
    rel_errors = []

    for tstep in range(nsteps):
        if solver == "direct":
            ax_arr = np.zeros(N, dtype=np.float64)
            ay_arr = np.zeros(N, dtype=np.float64)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr
            x += dt * vx
            y += dt * vy
            ax_arr.fill(0.0); ay_arr.fill(0.0)
            force_kernel.direct_force(x, y, m, 1e-4, ax_arr, ay_arr)
            vx += 0.5 * dt * ax_arr
            vy += 0.5 * dt * ay_arr

        else:
            fx = [0.0]*N
            fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)
            x += dt * vx
            y += dt * vy
            fx = [0.0]*N; fy = [0.0]*N
            if solver == "bh":
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, 1,
                                     soft, G, fx, fy)
            else:
                fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                     N, domain_size, theta, maxLeaf,
                                     soft, G, fx, fy)
            vx += 0.5 * dt * np.array(fx, dtype=np.float64)
            vy += 0.5 * dt * np.array(fy, dtype=np.float64)

        # Compute total energy E(t)
        E = 0.0
        for i in range(N):
            E += 0.5 * m[i] * (vx[i]*vx[i] + vy[i]*vy[i])
        pot = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = math.sqrt(dx*dx + dy*dy + soft*soft)
                if dist > 0:
                    pot -= G * m[i] * m[j] / dist
        E += pot

        times.append(tstep*dt)
        rel_errors.append(abs(E - E0) / abs(E0 + 1e-16))

    # Plot relative energy error
    plt.figure(figsize=(6,4))
    plt.semilogy(times, rel_errors, '-k', linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy Conservation (N={N}, threads=8)")
    plt.grid(True, which='both', ls='--', lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, f"energy_conservation_{solver}_{N}_8.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"✓ Saved energy-conservation plot to {pngfile}")

# -----------------------------------------------------
# Option (6): Parameter Optimization (N=100)
# -----------------------------------------------------
def parameter_optimization():
    print("\nParameter Optimization (vary θ)\n--------------------------------")
    N = 100
    domain_size = 50.0
    thetas = [0.1, 0.3, 0.5, 0.7, 1.0]
    maxLeaf = 8
    soft = 0.01
    G = 1.0

    # Compute “truth” forces via direct once
    x, y, m = initialize_particles(N, domain_size)
    fx_truth = np.zeros(N, dtype=np.float64)
    fy_truth = np.zeros(N, dtype=np.float64)
    force_kernel.direct_force(x, y, m, 1e-4, fx_truth, fy_truth)

    bh_errors = []
    fmm_errors = []

    for theta in thetas:
        # BH (maxLeaf=1)
        fx_bh = [0.0]*N
        fy_bh = [0.0]*N
        fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                             N, domain_size, theta, 1,
                             soft, G, fx_bh, fy_bh)

        # FMM (maxLeaf=8)
        fx_fm = [0.0]*N
        fy_fm = [0.0]*N
        fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                             N, domain_size, theta, maxLeaf,
                             soft, G, fx_fm, fy_fm)

        # Compute relative L2 errors
        bh_err = 0.0
        fm_err = 0.0
        norm_truth = 0.0
        for i in range(N):
            tx = fx_truth[i]
            ty = fy_truth[i]
            norm_truth += tx*tx + ty*ty
            bx = fx_bh[i]
            by = fy_bh[i]
            bh_err  += (bx - tx)**2 + (by - ty)**2
            fmx = fx_fm[i]
            fmy = fy_fm[i]
            fm_err  += (fmx - tx)**2 + (fmy - ty)**2

        bh_errors.append(math.sqrt(bh_err / norm_truth))
        fmm_errors.append(math.sqrt(fm_err / norm_truth))
        print(f"θ = {theta:.1f} → BH error = {bh_errors[-1]:.3e}, FMM error = {fmm_errors[-1]:.3e}")

    # Plot errors vs θ
    plt.figure(figsize=(6,4))
    plt.loglog(thetas, bh_errors, 's-b', label="BH error (maxLeaf=1)")
    plt.loglog(thetas, fmm_errors, '^-g', label="FMM error (maxLeaf=8)")
    plt.xlabel("θ (opening angle)")
    plt.ylabel("Relative Force Error")
    plt.title("Parameter Optimization (N=100, threads=8)")
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, "parameter_optimization.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"✓ Saved parameter-opt plot to {pngfile}")

# -----------------------------------------------------
# Option (7): OpenMP Thread Benchmark (N=10000)
# -----------------------------------------------------
def openmp_thread_benchmark():
    print("\nOpenMP Thread Benchmark\n-----------------------")
    N = 10000
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    soft = 0.01
    G = 1.0
    dt = 1e-3
    steps = 5

    thread_counts = [1, 2, 4, 8]
    times = []

    # Pre-warm: build tree once
    x, y, m = initialize_particles(N, domain_size)
    _ = [0]*N; _ = [0]*N
    fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                         N, domain_size, theta, maxLeaf,
                         soft, G, [0]*N, [0]*N)

    for nt in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(nt)
        time.sleep(0.1)
        t0 = time.time()
        for _ in range(steps):
            fmm_kernel.fmm_force(x.tolist(), y.tolist(), m.tolist(),
                                 N, domain_size, theta, maxLeaf,
                                 soft, G, [0]*N, [0]*N)
        t_avg = (time.time() - t0) / steps
        print(f"Threads = {nt:>2} → Time = {t_avg:.6f} s")
        times.append(t_avg)

    sp = [times[0]/t for t in times]
    plt.figure(figsize=(6,4))
    plt.plot(thread_counts, sp, 'o-r', label="Measured Speedup")
    plt.plot(thread_counts, thread_counts, '--k', label="Ideal Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (Relative to 1 thread)")
    plt.title("OpenMP Thread Benchmark (FMM, N=10000)")
    plt.xticks(thread_counts)
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    pngfile = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.png")
    plt.savefig(pngfile, dpi=200)
    plt.close()
    print(f"✓ Saved thread-benchmark plot to {pngfile}")

    # Restore OMP_NUM_THREADS back to 8
    os.environ["OMP_NUM_THREADS"] = "8"

# -----------------------------------------------------
# Option (8): System Information
# -----------------------------------------------------
def system_information():
    print("\nSystem Information\n------------------")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Forced threads : 8")
    try:
        import platform
        print(f"Platform       : {platform.platform()}")
    except:
        pass
    try:
        import multiprocessing
        print(f"Logical cores  : {multiprocessing.cpu_count()}")
    except:
        pass
    # Query OpenMP max threads
    try:
        import ctypes
        libgomp = ctypes.CDLL(None)
        omp_get_max_threads = libgomp.omp_get_max_threads
        omp_get_max_threads.restype = ctypes.c_int
        print("OpenMP max threads:", omp_get_max_threads())
    except:
        print("OpenMP max threads: N/A")

# -----------------------------------------------------
# Main menu
# -----------------------------------------------------
if __name__ == "__main__":
    while True:
        print("\n=== 2D N-Body Playground (Parallel, High-Precision) ===")
        print("  1) Quick benchmark scaling")
        print("  2) Save trajectory + energy plot")
        print("  3) Live simulation animation")
        print("  4) Large-N scaling test")
        print("  5) Energy conservation test")
        print("  6) Parameter optimization")
        print("  7) OpenMP thread benchmark")
        print("  8) System information")
        print("  q) Quit")
        print("=======================================================")
        choice = input("Enter choice: ").strip().lower()

        if choice == '1':
            quick_benchmark()
        elif choice == '2':
            save_trajectory_and_energy()
        elif choice == '3':
            live_simulation()
        elif choice == '4':
            largeN_scaling()
        elif choice == '5':
            energy_conservation_test()
        elif choice == '6':
            parameter_optimization()
        elif choice == '7':
            openmp_thread_benchmark()
        elif choice == '8':
            system_information()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

