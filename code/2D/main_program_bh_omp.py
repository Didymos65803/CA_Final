"""Barnes–Hut 2‑D demo (head‑less):
• tqdm progress‑bar
• save final snapshot + energy‑vs‑step PNG
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')  # head‑less backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from force_kernel import bh_omp

# ---------- sim params ----------
G      = 1.0
THETA  = 0.5
DT     = 0.02
STEPS  = 400
DOMAIN = 60.0   # half‑width of root square
N      = 800
SOFT   = 0.05

class P:
    __slots__ = ("x","y","vx","vy","m")
    def __init__(self,x,y,vx,vy,m): self.x,self.y,self.vx,self.vy,self.m = x,y,vx,vy,m

# ring initial condition
ring_r  = 20.0
ring_v  = np.sqrt(G*1.0/ring_r)
parts = [P(ring_r*np.cos(a), ring_r*np.sin(a),
           -ring_v*np.sin(a), ring_v*np.cos(a), 1/N)
         for a in np.linspace(0,2*np.pi,N,endpoint=False)]

energy = []

for step in tqdm(range(STEPS), desc="BH‑OMP"):
    # numpy views for kernel
    x = np.fromiter((p.x for p in parts), float, N)
    y = np.fromiter((p.y for p in parts), float, N)
    m = np.fromiter((p.m for p in parts), float, N)

    ax, ay = bh_omp(x, y, m, DOMAIN, THETA, G, SOFT)

    # kick‑drift‑kick (leapfrog)
    for i,p in enumerate(parts):
        p.vx += ax[i]*DT*0.5;  p.vy += ay[i]*DT*0.5
        p.x  += p.vx*DT;       p.y  += p.vy*DT
    x[...] = [p.x for p in parts]; y[...] = [p.y for p in parts]
    ax, ay = bh_omp(x, y, m, DOMAIN, THETA, G, SOFT)
    for i,p in enumerate(parts):
        p.vx += ax[i]*DT*0.5;  p.vy += ay[i]*DT*0.5

    # store total energy every step
    ke = 0.5*np.sum(m*(x*0+x*0))  # placeholder; replace with real KE if需要
    energy.append(ke)

# --------- save plots ----------
# 1) final snapshot
plt.figure(figsize=(6,6))
plt.scatter([p.x for p in parts], [p.y for p in parts], s=2)
plt.gca().set_aspect('equal'); plt.title('Final snapshot')
plt.savefig('snapshot.png', dpi=300)

# 2) energy‑vs‑step (dummy KE)
plt.figure()
plt.plot(energy)
plt.xlabel('step'); plt.ylabel('Kinetic‑Energy (arb)')
plt.title('Energy vs step')
plt.savefig('energy_vs_step.png', dpi=300)
print('Saved snapshot.png and energy_vs_step.png')

"""Minimal test harness for the Barnes–Hut OpenMP kernel."""
import numpy as np
import matplotlib.pyplot as plt
from force_kernel import bh_omp

G      = 1.0
THETA  = 0.5
DT     = 0.02
STEPS  = 400
DOMAIN = 60.0   # must enclose all particles; units = half‑width
N      = 800

class Particle:
    __slots__ = ("x","y","vx","vy","m")
    def __init__(self,x,y,vx,vy,m): self.x,self.y,self.vx,self.vy,self.m = x,y,vx,vy,m

# --- simple ring initial condition ---
ring_r  = 20.0
ring_v  = np.sqrt(G*1.0/ring_r)
particles = [Particle(ring_r*np.cos(a), ring_r*np.sin(a),
                      -ring_v*np.sin(a), ring_v*np.cos(a),
                      1/N)
             for a in np.linspace(0,2*np.pi,N,endpoint=False)]

for step in range(STEPS):
    x = np.array([p.x for p in particles]);   y = np.array([p.y for p in particles])
    m = np.array([p.m for p in particles])
    ax, ay = bh_omp(x,y,m,DOMAIN,THETA,G,soft=0.05)
    for i,p in enumerate(particles):
        p.vx += ax[i]*DT/2; p.vy += ay[i]*DT/2   # kick
        p.x  += p.vx*DT;   p.y  += p.vy*DT       # drift
    x[:] = [p.x for p in particles]; y[:] = [p.y for p in particles]
    ax, ay = bh_omp(x,y,m,DOMAIN,THETA,G,soft=0.05)           # new accel
    for i,p in enumerate(particles):
        p.vx += ax[i]*DT/2; p.vy += ay[i]*DT/2
    if step%50==0:
        print(f"step {step}/{STEPS}")

# quick & dirty scatter
plt.figure(figsize=(6,6))
plt.scatter([p.x for p in particles],[p.y for p in particles],s=2)
plt.gca().set_aspect('equal'); plt.title('Final snapshot')
plt.show()
