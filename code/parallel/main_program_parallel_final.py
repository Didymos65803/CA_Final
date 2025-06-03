#!/usr/bin/env python3
# main_program_parallel_final.py
# Fixed version for new C++ kernel interface

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Force OpenMP to use 8 threads
os.environ["OMP_NUM_THREADS"] = "8"

# Set path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import C++ modules
try:
    import force_kernel
    HAS_DIRECT = True
    print("✓ force_kernel loaded successfully")
except ImportError as e:
    HAS_DIRECT = False
    print(f"✗ force_kernel not available: {e}")

try:
    import fmm_kernel
    HAS_FMM = True
    print("✓ fmm_kernel loaded successfully")
except ImportError as e:
    HAS_FMM = False
    print(f"✗ fmm_kernel not available: {e}")

# Global settings
OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def initialize_particles(N, domain_size):
    """Initialize N particles uniformly in a disk"""
    angles = np.random.rand(N) * 2.0 * math.pi
    radii = domain_size * np.sqrt(np.random.rand(N))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    m = np.ones(N, dtype=np.float64)
    return x, y, m

def safe_direct_force(x, y, m, eps2):
    """Safe wrapper for direct force calculation using new interface"""
    N = len(x)
    
    # Ensure inputs are contiguous NumPy arrays
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    m_arr = np.ascontiguousarray(m, dtype=np.float64)
    
    # Pre-allocate output arrays
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_DIRECT:
        try:
            # New interface: direct_force(x, y, m, eps2, ax, ay)
            force_kernel.direct_force(x_arr, y_arr, m_arr, eps2, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"Direct force calculation failed: {e}")
            return None, None
    else:
        return None, None

def safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G):
    """Safe wrapper for FMM force calculation using new interface"""
    
    # Ensure inputs are contiguous NumPy arrays
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    m_arr = np.ascontiguousarray(m, dtype=np.float64)
    
    # Pre-allocate output arrays
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)
    
    if HAS_FMM:
        try:
            # New interface: fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G, ax, ay)
            fmm_kernel.fmm_force(x_arr, y_arr, m_arr, N, domain_size, 
                               theta, maxLeaf, eps, G, ax, ay)
            return ax, ay
        except Exception as e:
            print(f"FMM force calculation failed: {e}")
            return None, None
    else:
        return None, None

def quick_benchmark():
    """Quick benchmark scaling"""
    print("\nQuick Benchmark Scaling")
    print("=" * 50)
    
    Ns = [50, 100, 200, 500]  # 減少測試規模避免卡住
    steps = 3  # 減少步數
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    
    print(f"Using 8 threads (OpenMP)")
    print(f"Testing particle counts: {Ns}")
    
    results = []
    
    for N in Ns:
        print(f"\nTesting N = {N}")
        
        # Initialize particles
        x, y, m = initialize_particles(N, domain_size)
        
        # Test direct method
        t_direct = None
        if HAS_DIRECT:
            try:
                print("  Testing direct method...")
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                    if ax is None:
                        raise Exception("Direct force failed")
                t_direct = (time.time() - t0) / steps
                print(f"  ✓ Direct method: {t_direct:.6f} seconds")
            except Exception as e:
                print(f"  ✗ Direct method: Failed ({e})")
                t_direct = float('nan')
        else:
            print("  ✗ Direct method: Not available")
            t_direct = float('nan')
        
        # Test Barnes-Hut (using FMM with maxLeaf=1)
        t_bh = None
        if HAS_FMM:
            try:
                print("  Testing Barnes-Hut method...")
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                    if ax is None:
                        raise Exception("BH force failed")
                t_bh = (time.time() - t0) / steps
                print(f"  ✓ Barnes-Hut: {t_bh:.6f} seconds")
            except Exception as e:
                print(f"  ✗ Barnes-Hut: Failed ({e})")
                t_bh = float('nan')
        else:
            print("  ✗ Barnes-Hut: Not available")
            t_bh = float('nan')
        
        # Test FMM
        t_fmm = None
        if HAS_FMM:
            try:
                print("  Testing FMM method...")
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                    if ax is None:
                        raise Exception("FMM force failed")
                t_fmm = (time.time() - t0) / steps
                print(f"  ✓ FMM: {t_fmm:.6f} seconds")
            except Exception as e:
                print(f"  ✗ FMM: Failed ({e})")
                t_fmm = float('nan')
        else:
            print("  ✗ FMM: Not available")
            t_fmm = float('nan')
        
        results.append((N, t_direct, t_bh, t_fmm))
    
    # Save results
    save_scaling_results(results, "scaling_quick")
    print("\n✓ Quick benchmark completed!")

def save_trajectory_and_energy():
    """Save trajectory + energy plot"""
    print("\nSave Trajectory + Energy Plot")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 200): "))
        method = input("Choose method (direct/bh/fmm): ").strip().lower()
        steps = int(input("Enter number of steps (e.g., 100): "))
    except ValueError:
        print("Invalid input. Using default values.")
        N = 200
        method = "fmm"
        steps = 100
    
    if method not in ["direct", "bh", "fmm"]:
        print("Invalid method. Using FMM.")
        method = "fmm"
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    # Store trajectory and energy
    trajectory = []
    energies = []
    
    print(f"Running simulation with {method} method...")
    
    for step in range(steps):
        if step % 20 == 0:
            print(f"  Step {step}/{steps}")
        
        # Calculate forces
        if method == "direct" and HAS_DIRECT:
            ax, ay = safe_direct_force(x, y, m, eps*eps)
        elif method == "bh" and HAS_FMM:
            ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
        elif method == "fmm" and HAS_FMM:
            ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
        else:
            print(f"Method {method} not available")
            return
        
        if ax is None:
            print("Force calculation failed")
            return
        
        # Leapfrog integration
        vx += 0.5 * dt * ax
        vy += 0.5 * dt * ay
        x += dt * vx
        y += dt * vy
        
        # Calculate energy (kinetic + potential)
        ke = 0.5 * np.sum(m * (vx**2 + vy**2))
        pe = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r = math.sqrt(dx*dx + dy*dy + eps*eps)
                pe -= G * m[i] * m[j] / r
        
        trajectory.append((x.copy(), y.copy()))
        energies.append(ke + pe)
    
    # Create and save plots
    create_trajectory_animation(trajectory, method, N)
    create_energy_plot(energies, dt, method, N)
    print("✓ Trajectory and energy plots saved!")

def live_simulation_animation():
    """Live simulation animation"""
    print("\nLive Simulation Animation")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 100): "))
        method = input("Choose method (direct/bh/fmm): ").strip().lower()
    except ValueError:
        print("Invalid input. Using default values.")
        N = 100
        method = "fmm"
    
    if method not in ["direct", "bh", "fmm"]:
        print("Invalid method. Using FMM.")
        method = "fmm"
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    frames = 200
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    vx = np.zeros(N, dtype=np.float64)
    vy = np.zeros(N, dtype=np.float64)
    
    print(f"Creating animation with {method} method...")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-domain_size*1.2, domain_size*1.2)
    ax.set_ylim(-domain_size*1.2, domain_size*1.2)
    scat = ax.scatter(x, y, s=10, c='blue')
    ax.set_title(f"Live Simulation ({method.upper()}, N={N})")
    
    def update(frame):
        nonlocal x, y, vx, vy
        
        # Calculate forces
        if method == "direct" and HAS_DIRECT:
            ax_arr, ay_arr = safe_direct_force(x, y, m, eps*eps)
        elif method == "bh" and HAS_FMM:
            ax_arr, ay_arr = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
        elif method == "fmm" and HAS_FMM:
            ax_arr, ay_arr = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
        else:
            return scat,
        
        if ax_arr is None:
            return scat,
        
        # Update positions
        vx += dt * ax_arr
        vy += dt * ay_arr
        x += dt * vx
        y += dt * vy
        
        scat.set_offsets(np.column_stack((x, y)))
        return scat,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    # Save animation
    gif_path = os.path.join(OUTPUT_DIR, f"live_simulation_{method}_{N}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close()
    print(f"✓ Live simulation saved to {gif_path}")

def large_n_scaling():
    """Large-N scaling test"""
    print("\nLarge-N Scaling Test")
    print("=" * 50)
    
    Ns = [500, 1000, 2000]  # 減少測試規模
    steps = 3
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    
    print(f"Testing large N values: {Ns}")
    
    results = []
    
    for N in Ns:
        print(f"\nTesting N = {N}")
        
        x, y, m = initialize_particles(N, domain_size)
        
        # Skip direct method for large N
        t_direct = float('nan') if N > 1000 else None
        if t_direct is None and HAS_DIRECT:
            try:
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                t_direct = (time.time() - t0) / steps
                print(f"  Direct method: {t_direct:.6f} seconds")
            except:
                t_direct = float('nan')
        
        # Test other methods
        t_bh = float('nan')
        t_fmm = float('nan')
        
        if HAS_FMM:
            try:
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                t_bh = (time.time() - t0) / steps
                print(f"  Barnes-Hut: {t_bh:.6f} seconds")
            except:
                pass
            
            try:
                t0 = time.time()
                for _ in range(steps):
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                t_fmm = (time.time() - t0) / steps
                print(f"  FMM: {t_fmm:.6f} seconds")
            except:
                pass
        
        results.append((N, t_direct, t_bh, t_fmm))
    
    save_scaling_results(results, "scaling_large")
    print("\n✓ Large-N scaling test completed!")

def energy_conservation_test():
    """Energy conservation test"""
    print("\nEnergy Conservation Test")
    print("=" * 50)
    
    try:
        N = int(input("Enter number of particles (e.g., 200): "))
    except ValueError:
        N = 200
    
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    dt = 0.001
    steps = 500
    
    methods = []
    if HAS_DIRECT:
        methods.append(("Direct", "direct"))
    if HAS_FMM:
        methods.append(("Barnes-Hut", "bh"))
        methods.append(("FMM", "fmm"))
    
    plt.figure(figsize=(10, 6))
    
    for method_name, method in methods:
        print(f"\nTesting {method_name}...")
        
        # Initialize particles
        x, y, m = initialize_particles(N, domain_size)
        vx = np.zeros(N, dtype=np.float64)
        vy = np.zeros(N, dtype=np.float64)
        
        # Calculate initial energy
        E0 = calculate_total_energy(x, y, vx, vy, m, G, eps)
        
        times = []
        rel_errors = []
        
        for step in range(0, steps, 5):  # Record every 5 steps
            # Calculate forces
            if method == "direct":
                ax, ay = safe_direct_force(x, y, m, eps*eps)
            elif method == "bh":
                ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
            elif method == "fmm":
                ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
            
            if ax is None:
                break
            
            # Integrate for 5 steps
            for _ in range(5):
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
                x += dt * vx
                y += dt * vy
                
                # Recalculate forces
                if method == "direct":
                    ax, ay = safe_direct_force(x, y, m, eps*eps)
                elif method == "bh":
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
                elif method == "fmm":
                    ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
                
                vx += 0.5 * dt * ax
                vy += 0.5 * dt * ay
            
            # Calculate current energy and relative error
            E = calculate_total_energy(x, y, vx, vy, m, G, eps)
            rel_error = abs(E - E0) / abs(E0)
            
            times.append(step * dt)
            rel_errors.append(rel_error)
        
        plt.semilogy(times, rel_errors, label=method_name)
    
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title(f"Energy Conservation Test (N={N})")
    plt.legend()
    plt.grid(True)
    
    energy_path = os.path.join(OUTPUT_DIR, "energy_conservation.png")
    plt.savefig(energy_path, dpi=300)
    plt.close()
    print(f"✓ Energy conservation plot saved to {energy_path}")

def parameter_optimization():
    """Parameter optimization"""
    print("\nParameter Optimization")
    print("=" * 50)
    
    N = 100
    domain_size = 50.0
    eps = 0.01
    G = 1.0
    
    # Initialize particles
    x, y, m = initialize_particles(N, domain_size)
    
    # Get reference solution
    if not HAS_DIRECT:
        print("Direct method not available for reference")
        return
    
    ax_ref, ay_ref = safe_direct_force(x, y, m, eps*eps)
    if ax_ref is None:
        print("Failed to compute reference solution")
        return
    
    thetas = [0.1, 0.3, 0.5, 0.7, 1.0]
    bh_errors = []
    fmm_errors = []
    
    for theta in thetas:
        # Test Barnes-Hut
        if HAS_FMM:
            ax_bh, ay_bh = safe_fmm_force(x, y, m, N, domain_size, theta, 1, eps, G)
            if ax_bh is not None:
                bh_error = np.sqrt(np.sum((ax_bh - ax_ref)**2 + (ay_bh - ay_ref)**2)) / np.sqrt(np.sum(ax_ref**2 + ay_ref**2))
                bh_errors.append(bh_error)
            else:
                bh_errors.append(float('nan'))
            
            # Test FMM
            ax_fmm, ay_fmm = safe_fmm_force(x, y, m, N, domain_size, theta, 8, eps, G)
            if ax_fmm is not None:
                fmm_error = np.sqrt(np.sum((ax_fmm - ax_ref)**2 + (ay_fmm - ay_ref)**2)) / np.sqrt(np.sum(ax_ref**2 + ay_ref**2))
                fmm_errors.append(fmm_error)
            else:
                fmm_errors.append(float('nan'))
        else:
            bh_errors.append(float('nan'))
            fmm_errors.append(float('nan'))
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.semilogy(thetas, bh_errors, 's-', label="Barnes-Hut Error")
    plt.semilogy(thetas, fmm_errors, '^-', label="FMM Error")
    plt.xlabel("Theta (opening angle)")
    plt.ylabel("Relative Force Error")
    plt.title(f"Parameter Optimization (N={N})")
    plt.legend()
    plt.grid(True)
    
    param_path = os.path.join(OUTPUT_DIR, "parameter_optimization.png")
    plt.savefig(param_path, dpi=300)
    plt.close()
    print(f"✓ Parameter optimization plot saved to {param_path}")

def openmp_thread_benchmark():
    """OpenMP thread benchmark"""
    print("\nOpenMP Thread Benchmark")
    print("=" * 50)
    
    if not HAS_FMM:
        print("FMM not available for thread benchmark")
        return
    
    N = 500
    domain_size = 50.0
    theta = 0.5
    maxLeaf = 8
    eps = 0.01
    G = 1.0
    
    thread_counts = [1, 2, 4, 8]
    times = []
    
    x, y, m = initialize_particles(N, domain_size)
    
    for threads in thread_counts:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        time.sleep(0.1)  # Allow environment to update
        
        t0 = time.time()
        for _ in range(10):  # Average over multiple runs
            ax, ay = safe_fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G)
        avg_time = (time.time() - t0) / 10
        
        times.append(avg_time)
        print(f"Threads: {threads}, Time: {avg_time:.6f} seconds")
    
    # Calculate speedup
    speedup = [times[0] / t for t in times]
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(thread_counts, speedup, 'o-', label="Measured Speedup")
    plt.plot(thread_counts, thread_counts, '--', label="Ideal Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"OpenMP Thread Benchmark (FMM, N={N})")
    plt.legend()
    plt.grid(True)
    
    thread_path = os.path.join(OUTPUT_DIR, "openmp_thread_benchmark.png")
    plt.savefig(thread_path, dpi=300)
    plt.close()
    print(f"✓ Thread benchmark plot saved to {thread_path}")
    
    # Restore default thread count
    os.environ["OMP_NUM_THREADS"] = "8"

def system_information():
    """System information"""
    print("\nSystem Information")
    print("=" * 50)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OpenMP threads: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    print(f"Direct method module: {'Available' if HAS_DIRECT else 'Not available'}")
    print(f"FMM module: {'Available' if HAS_FMM else 'Not available'}")
    
    try:
        import platform
        print(f"Operating system: {platform.platform()}")
    except:
        pass
    
    try:
        import multiprocessing
        print(f"CPU cores: {multiprocessing.cpu_count()}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except:
        pass

# Helper functions
def calculate_total_energy(x, y, vx, vy, m, G, eps):
    """Calculate total energy (kinetic + potential)"""
    N = len(x)
    
    # Kinetic energy
    ke = 0.5 * np.sum(m * (vx**2 + vy**2))
    
    # Potential energy
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = math.sqrt(dx*dx + dy*dy + eps*eps)
            pe -= G * m[i] * m[j] / r
    
    return ke + pe

def save_scaling_results(results, filename):
    """Save scaling results to CSV and create plot"""
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    with open(csv_path, "w") as f:
        f.write("N,Direct,BH,FMM\n")
        for N, t_direct, t_bh, t_fmm in results:
            f.write(f"{N},{t_direct},{t_bh},{t_fmm}\n")
    
    # Create plot
    Ns = [r[0] for r in results]
    times_direct = [r[1] for r in results if not math.isnan(r[1])]
    times_bh = [r[2] for r in results if not math.isnan(r[2])]
    times_fmm = [r[3] for r in results if not math.isnan(r[3])]
    
    plt.figure(figsize=(8, 6))
    
    if times_direct:
        Ns_direct = [r[0] for r in results if not math.isnan(r[1])]
        plt.loglog(Ns_direct, times_direct, 'o-', label="Direct O(N²)")
    
    if times_bh:
        Ns_bh = [r[0] for r in results if not math.isnan(r[2])]
        plt.loglog(Ns_bh, times_bh, 's-', label="Barnes-Hut O(N log N)")
    
    if times_fmm:
        Ns_fmm = [r[0] for r in results if not math.isnan(r[3])]
        plt.loglog(Ns_fmm, times_fmm, '^-', label="FMM O(N)")
    
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time per Step (seconds)")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(True)
    
    png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    print(f"✓ Results saved to {csv_path} and {png_path}")

def create_trajectory_animation(trajectory, method, N):
    """Create trajectory animation"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set limits based on trajectory
    all_x = np.concatenate([pos[0] for pos in trajectory])
    all_y = np.concatenate([pos[1] for pos in trajectory])
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    ax.set_xlim(all_x.min() - margin*x_range, all_x.max() + margin*x_range)
    ax.set_ylim(all_y.min() - margin*y_range, all_y.max() + margin*y_range)
    ax.set_title(f"Trajectory ({method.upper()}, N={N})")
    
    scat = ax.scatter(trajectory[0][0], trajectory[0][1], s=10, c='blue')
    
    def animate(frame):
        x, y = trajectory[frame]
        scat.set_offsets(np.column_stack((x, y)))
        return scat,
    
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=50, blit=True)
    
    gif_path = os.path.join(OUTPUT_DIR, f"trajectory_{method}_{N}.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close()
    print(f"✓ Trajectory animation saved to {gif_path}")

def create_energy_plot(energies, dt, method, N):
    """Create energy vs time plot"""
    times = np.arange(len(energies)) * dt
    
    plt.figure(figsize=(8, 6))
    plt.plot(times, energies, '-')
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title(f"Energy vs Time ({method.upper()}, N={N})")
    plt.grid(True)
    
    energy_path = os.path.join(OUTPUT_DIR, f"energy_{method}_{N}.png")
    plt.savefig(energy_path, dpi=300)
    plt.close()
    print(f"✓ Energy plot saved to {energy_path}")

def main_menu():
    """Main menu with 8 options"""
    while True:
        print("\n" + "=" * 60)
        print("2D N-Body Problem Simulation Platform")
        print("(Parallel High-Precision Version)")
        print("=" * 60)
        print("Select function:")
        print(" 1) Quick benchmark scaling")
        print(" 2) Save trajectory + energy plot")
        print(" 3) Live simulation animation")
        print(" 4) Large-N scaling test")
        print(" 5) Energy conservation test")
        print(" 6) Parameter optimization")
        print(" 7) OpenMP thread benchmark")
        print(" 8) System information")
        print(" q) Exit program")
        print("=" * 60)
        
        choice = input("Please enter your choice: ").strip().lower()
        
        if choice == '1':
            quick_benchmark()
        elif choice == '2':
            save_trajectory_and_energy()
        elif choice == '3':
            live_simulation_animation()
        elif choice == '4':
            large_n_scaling()
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

if __name__ == "__main__":
    print("2D N-Body Problem Simulation Platform starting...")
    
    # Check module availability
    if not HAS_DIRECT and not HAS_FMM:
        print("Error: No available computation modules!")
        print("Please ensure force_kernel and fmm_kernel modules are properly compiled.")
        print("Run: python setup.py build_ext --inplace")
        sys.exit(1)
    
    main_menu()

