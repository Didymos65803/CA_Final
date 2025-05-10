import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

# Try loading OpenMP kernel
try:
    import force_kernel
    USE_OMP = True
    print("[info] OpenMP kernel loaded (force_kernel)")
except ImportError:
    USE_OMP = False
    print("[warn] force_kernel not found; using pure Python direct method")

import sys
# Increase recursion limit for deep quad-tree
sys.setrecursionlimit(10000)

# Constants
G = 1.0
softening = 0.01
soft2 = softening * softening
DT = 0.1  # time step

# Particle class
class Particle:
    def __init__(self, x, y, mass=1.0, vx=0.0, vy=0.0):
        self.x = x; self.y = y
        self.mass = mass
        self.vx = vx; self.vy = vy
        self.ax = 0.0; self.ay = 0.0

# Pure Python direct O(n^2)
def compute_forces_direct_py(particles):
    n = len(particles)
    for i in range(n):
        p_i = particles[i]
        axi = ayi = 0.0
        for j in range(n):
            if i == j: continue
            dx = particles[j].x - p_i.x
            dy = particles[j].y - p_i.y
            r2 = dx*dx + dy*dy
            if r2 < soft2: r2 = soft2
            inv_r = 1.0/math.sqrt(r2)
            f = G * particles[j].mass * inv_r * inv_r
            axi += f * dx * inv_r
            ayi += f * dy * inv_r
        p_i.ax, p_i.ay = axi, ayi

# OpenMP direct via C++ kernel
def compute_forces_direct_omp(particles):
    n = len(particles)
    x = np.fromiter((p.x for p in particles), float, count=n)
    y = np.fromiter((p.y for p in particles), float, count=n)
    m = np.fromiter((p.mass for p in particles), float, count=n)
    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)
    force_kernel.direct_omp(x, y, m, ax, ay, G, soft2)
    for i, p in enumerate(particles):
        p.ax = ax[i]; p.ay = ay[i]

# Combined direct method dispatcher
def compute_forces_direct(particles):
    if USE_OMP:
        compute_forces_direct_omp(particles)
    else:
        compute_forces_direct_py(particles)

# QuadTree and FMM
class QuadTreeNode:
    def __init__(self, cx, cy, size):
        self.cx, self.cy, self.size = cx, cy, size
        self.children = [None]*4
        self.particles = []
        self.total_mass = 0.0
        self.com_x = 0.0; self.com_y = 0.0
        self.is_leaf = True; self.is_empty = True

    def insert(self, p):
        self.is_empty = False
        if self.is_leaf and len(self.particles) == 0:
            self.particles.append(p)
            self.total_mass = p.mass
            self.com_x, self.com_y = p.x, p.y
            return
        if self.is_leaf and len(self.particles) < 1:
            self.particles.append(p)
            self.total_mass += p.mass
            self.com_x = (self.com_x*(self.total_mass-p.mass)+p.x*p.mass)/self.total_mass
            self.com_y = (self.com_y*(self.total_mass-p.mass)+p.y*p.mass)/self.total_mass
            return
        if self.is_leaf:
            self.is_leaf = False
            half = self.size/2; q = half/2
            xs = [(-q,-q),(q,-q),(-q,q),(q,q)]
            for idx,(dx,dy) in enumerate(xs):
                self.children[idx] = QuadTreeNode(self.cx+dx, self.cy+dy, half)
            for old in self.particles:
                self._insert_child(old)
            self.particles.clear()
        self._insert_child(p)
        self.total_mass += p.mass
        self.com_x = (self.com_x*(self.total_mass-p.mass)+p.x*p.mass)/self.total_mass
        self.com_y = (self.com_y*(self.total_mass-p.mass)+p.y*p.mass)/self.total_mass

    def _insert_child(self, p):
        idx = (p.x>self.cx) + 2*(p.y>self.cy)
        self.children[idx].insert(p)

    def compute_force(self, p, theta=0.5):
        if self.is_empty: return (0.0,0.0)
        dx = self.com_x - p.x; dy = self.com_y - p.y
        r2 = dx*dx+dy*dy
        if r2 < soft2: r2 = soft2
        r = math.sqrt(r2)
        if self.is_leaf or (self.size/r<theta):
            f = G*self.total_mass/r2
            return (f*dx/r, f*dy/r)
        ax=ay=0.0
        for c in self.children:
            if c and not c.is_empty:
                fx,fy = c.compute_force(p, theta)
                ax+=fx; ay+=fy
        return (ax,ay)

def compute_forces_fmm(particles, domain=100.0, theta=0.5):
    for p in particles: p.ax=p.ay=0.0
    root = QuadTreeNode(0.0,0.0,domain)
    for p in particles: root.insert(p)
    for p in particles:
        p.ax,p.ay = root.compute_force(p, theta)

# Update positions

def update_positions(particles, dt=DT):
    for p in particles:
        p.vx += 0.5*p.ax*dt; p.vy += 0.5*p.ay*dt
        p.x  += p.vx*dt*0.5; p.y  += p.vy*dt*0.5

# Benchmark
def benchmark(n_list, method='both'):
    results=defaultdict(list)
    for n in n_list:
        arr=[Particle((np.random.rand()-0.5)*50,(np.random.rand()-0.5)*50,mass=np.random.rand()*9+1) for _ in range(n)]
        arr_d=[Particle(p.x,p.y,p.mass) for p in arr]
        if method in ['direct','both']:
            t0=time.time(); compute_forces_direct(arr_d); results['direct_times'].append(time.time()-t0)
        if method in ['fmm','both']:
            t0=time.time(); compute_forces_fmm(arr);     results['fmm_times'].append(time.time()-t0)
        if method=='both':
            err=0.0
            for a,b in zip(arr_d,arr): err+=math.hypot(a.ax-b.ax,a.ay-b.ay)/(math.hypot(a.ax,a.ay)+1e-12)
            results['errors'].append(err/n)
    return results

# Plot results and save

def plot_results(n_list, res, fname='benchmark.png'):
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    axs[0].loglog(n_list,res['direct_times'],'o-',label='Direct')
    axs[0].loglog(n_list,res['fmm_times'],'s-',label='FMM')
    axs[0].set(xlabel='N',ylabel='Time(s)',title='Performance'); axs[0].legend(); axs[0].grid(True)
    x=np.array(n_list)
    axs[1].loglog(n_list,res['errors'],'o-')
    axs[1].set(xlabel='N',ylabel='Avg Rel Error',title='Accuracy'); axs[1].grid(True)
    plt.tight_layout(); plt.savefig(fname); print(f"Saved {fname}")

# Trajectory & Animation

def simulate_and_animate(n=100, steps=200, method='fmm', theta=0.5,
                         traj_fname='trajectories.png', anim_fname='simulation.gif'):
    from tqdm import tqdm
    # 1) Initialize particles
    arr = [Particle(0, 0, mass=1000.0)]
    for _ in range(1, n):
        r, ang = np.random.uniform(5, 30), np.random.uniform(0, 2*np.pi)
        x, y = r*np.cos(ang), r*np.sin(ang)
        v = math.sqrt(G*arr[0].mass/r)
        arr.append(Particle(x, y, mass=np.random.uniform(1, 2),
                            vx=-v*math.sin(ang), vy=v*math.cos(ang)))
    # 2) Record trajectories with progress bar
    xs = [[] for _ in range(n)]
    ys = [[] for _ in range(n)]
    for _ in tqdm(range(steps), desc="Simulating steps"):
        for i, p in enumerate(arr):
            xs[i].append(p.x)
            ys[i].append(p.y)
        if method == 'direct':
            compute_forces_direct(arr)
        else:
            compute_forces_fmm(arr, theta=theta)
        update_positions(arr)
    # 3) Static trajectory plot
    plt.figure(figsize=(6,6))
    plt.title(f'Trajectories ({method.upper()})')
    plt.grid(True)
    for i in range(n):
        plt.plot(xs[i], ys[i], alpha=0.5, lw=0.5)
    plt.axis('equal')
    plt.savefig(traj_fname)
    print(f"Saved static trajectories ▶ {traj_fname}")

    # 4) Build animation (using recorded positions)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set(xlim=(-60,60), ylim=(-60,60))
    ax.grid(True)
    scat = ax.scatter(xs[0][0:1], ys[0][0:1], c='blue')  # init one point

    def update(frame):
        offsets = np.column_stack(( [xs[i][frame] for i in range(n)],
                                    [ys[i][frame] for i in range(n)] ))
        scat.set_offsets(offsets)
        return (scat,)

    anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

    # 5) Save GIF with progress bar
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=30)
    with writer.saving(fig, anim_fname, dpi=200):
        for frame in tqdm(range(steps), desc="Saving GIF"):
            update(frame)
            writer.grab_frame()
    print(f"Saved animation ▶ {anim_fname}")

# Main menu

def main():
    print("1) Benchmark")
    print("2) Trajectory & Animation")
    mode=input("Select (1/2): ")
    if mode=='1':
        nlist=list(map(int,input("Enter N values separated by comma [100,500,1000]: ").split(',')))
        res=benchmark(nlist,'both'); plot_results(nlist,res)
    else:
        n=int(input("Particles [100]: ") or "100")
        steps=int(input("Steps [200]: ") or "200")
        m=input("Method(direct/fmm) [fmm]: ") or "fmm"
        simulate_and_animate(n,steps,m)

if __name__=='__main__':
    main()

