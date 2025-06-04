#!/usr/bin/env python3
"""
simulate_orbit.py  -- 2-D N-body demo with Direct / BH / FMM

示例:
  python simulate_orbit.py -N 800   -m fmm     --scenario ring    -o ring.gif
  python simulate_orbit.py -N 10    -m bh      --scenario central -o central.gif
"""
import argparse, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange
from force_kernel import bh_omp, direct_omp
from fmm_kernel   import fmm_omp

methods = dict(direct=direct_omp, bh=bh_omp, fmm=fmm_omp)

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument('-N', type=int, default=800, help='number of orbiters (not counting central mass)')
ap.add_argument('-m', '--method', choices=methods, default='fmm')
ap.add_argument('-s', '--steps', type=int, default=400)
ap.add_argument('-o', '--output', default='orbit.gif')
ap.add_argument('--threads', type=int, default=None)
ap.add_argument('--scenario', choices=['ring', 'central'], default='ring',
                help="ring = uniform circle (default); central = big red star + small bodies")
args = ap.parse_args()
if args.threads: os.environ['OMP_NUM_THREADS'] = str(args.threads)

# ---------- constants ----------
G       = 1.0
DT      = 0.02
DOMAIN  = 60.0
SOFT    = 0.05      # Plummer softening
STAR_M  = 100.0     # mass of central red star (central scenario)
R0      = 20.0

# ---------- initialise ----------
N = args.N
px, py, vx, vy, mass = [], [], [], [], []

if args.scenario == 'ring':
    for i in range(N):
        ang = 2*np.pi*i/N
        px.append(R0*np.cos(ang)); py.append(R0*np.sin(ang))
        v   = np.sqrt(G/R0)
        vx.append(-v*np.sin(ang)); vy.append( v*np.cos(ang))
        mass.append(1.0/N)

elif args.scenario == 'central':
    # 0) central massive star (固定在原點，速度 = 0)
    px.append(0.0); py.append(0.0); vx.append(0.0); vy.append(0.0); mass.append(STAR_M)
    # 1) N small bodies randomly distributed
    rng = np.random.default_rng(0)
    for _ in range(N):
        r   = rng.uniform(10, 30)           # 隨機半徑
        ang = rng.uniform(0, 2*np.pi)
        px.append(r*np.cos(ang)); py.append(r*np.sin(ang))
        v_k = np.sqrt(G*STAR_M/r)           # Kepler 圓軌
        vx.append(-v_k*np.sin(ang)); vy.append( v_k*np.cos(ang))
        mass.append(1.0)

# ---------- choose kernel ----------
kernel = methods[args.method]

def compute_acc(x, y, m):
    if args.method == 'direct':
        return kernel(x, y, m, G, SOFT)
    else:
        return kernel(x, y, m, DOMAIN, G=G, soft=SOFT)

# ---------- animation ----------
fig, ax = plt.subplots(figsize=(6,6))
sc = ax.scatter(px, py, s=4)
if args.scenario == 'central':  # 將中央質量標紅
    sc.set_facecolors(np.array([[1,0,0]] + [[0.2,0.6,0.9]]*N))

ax.set_xlim(-DOMAIN, DOMAIN); ax.set_ylim(-DOMAIN, DOMAIN)
ax.set_aspect('equal'); ax.set_title(f"{args.method} N={args.N}")

def leapfrog():
    x = np.array(px); y = np.array(py); m = np.array(mass)
    ax_, ay_ = compute_acc(x, y, m)
    for i in range(len(px)):
        # 若想讓 central 質量靜止，可跳過 i==0 更新
        if args.scenario=='central' and i==0: continue
        vx[i] += ax_[i]*DT*0.5; vy[i] += ay_[i]*DT*0.5
        px[i] += vx[i]*DT;      py[i] += vy[i]*DT
    x = np.array(px); y = np.array(py)
    ax_, ay_ = compute_acc(x, y, m)
    for i in range(len(px)):
        if args.scenario=='central' and i==0: continue
        vx[i] += ax_[i]*DT*0.5; vy[i] += ay_[i]*DT*0.5

ani = FuncAnimation(fig,
                    lambda f: (leapfrog(), sc.set_offsets(np.c_[px,py])),
                    frames=args.steps, blit=False)
ani.save(args.output, writer='pillow', fps=30)
print("Saved", args.output)

