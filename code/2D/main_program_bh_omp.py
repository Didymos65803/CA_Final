"""Barnes–Hut 2‑D demo (remote / head‑less)
• tqdm 進度條  ✔
• 正確的動能 (KE) 與總能量 (KE+PE) 曲線  ✔
• 自動存圖 (snapshot.png, energy_vs_step.png)  ✔
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')  # 背景模式
import matplotlib.pyplot as plt
from tqdm import tqdm
from force_kernel import bh_omp

# ---------- simulation parameters ----------
G      = 1.0      # gravitational constant (code‑units)
THETA  = 0.5      # opening‑angle criterion
DT     = 0.02
STEPS  = 400
DOMAIN = 60.0     # half‑width of root square (must cover all particles)
N      = 800
SOFT   = 0.05

class P:
    __slots__ = ("x","y","vx","vy","m")
    def __init__(self,x,y,vx,vy,m):
        self.x,self.y,self.vx,self.vy,self.m = x,y,vx,vy,m

# --- ring initial condition ---------------------------------------------
R   = 20.0
V_0 = np.sqrt(G*1.0/R)          # circular velocity in code‑units (central mass=1)
parts = [P(R*np.cos(a), R*np.sin(a),
           -V_0*np.sin(a), V_0*np.cos(a), 1.0/N)
         for a in np.linspace(0,2*np.pi,N,endpoint=False)]

# containers for diagnostics
ke_list, pe_list, totE_list = [], [], []

# helper – compute PE by direct O(N²) every ENERGY_FREQ steps (cheap for N=800)
ENERGY_FREQ = 10
soft2 = SOFT*SOFT

def potential_energy(xs, ys, ms):
    N = len(xs)
    U = 0.0
    for i in range(N-1):
        dx = xs[i+1:] - xs[i]
        dy = ys[i+1:] - ys[i]
        r2 = dx*dx + dy*dy + soft2
        U -= G * ms[i] * np.sum(ms[i+1:] / np.sqrt(r2))
    return U

# --------------------------- main loop ----------------------------------
for step in tqdm(range(STEPS), desc="BH‑OMP"):
    # gather arrays for kernel
    x = np.fromiter((p.x for p in parts), float, N)
    y = np.fromiter((p.y for p in parts), float, N)
    m = np.fromiter((p.m for p in parts), float, N)

    ax, ay = bh_omp(x, y, m, DOMAIN, THETA, G, SOFT)

    # leap‑frog: kick‑drift‑kick
    for i,p in enumerate(parts):
        p.vx += ax[i]*DT*0.5; p.vy += ay[i]*DT*0.5   # first half‑kick
        p.x  += p.vx*DT;      p.y  += p.vy*DT        # drift
    # update arrays
    x[...] = [p.x for p in parts]; y[...] = [p.y for p in parts]
    ax, ay = bh_omp(x, y, m, DOMAIN, THETA, G, SOFT)  # new accel
    for i,p in enumerate(parts):
        p.vx += ax[i]*DT*0.5; p.vy += ay[i]*DT*0.5   # second half‑kick

    # ------------- diagnostics -------------
    if step % ENERGY_FREQ == 0:
        v2 = np.fromiter((p.vx**2 + p.vy**2 for p in parts), float, N)
        KE = 0.5*np.sum(m * v2)
        PE = potential_energy(x, y, m)
        ke_list.append(KE); pe_list.append(PE); totE_list.append(KE+PE)

# ---------------- save plots ----------------
# 1) final snapshot
plt.figure(figsize=(6,6))
plt.scatter([p.x for p in parts], [p.y for p in parts], s=2)
plt.gca().set_aspect('equal'); plt.title('Final snapshot (N = %d)'%N)
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('snapshot.png', dpi=300)

# 2) energy vs step
steps_axis = np.arange(0, STEPS, ENERGY_FREQ)
plt.figure(figsize=(7,4))
plt.plot(steps_axis, ke_list,  label='KE')
plt.plot(steps_axis, pe_list,  label='PE')
plt.plot(steps_axis, totE_list, label='Total')
plt.xlabel('step'); plt.ylabel('Energy')
plt.title('Energy conservation test')
plt.legend()
plt.tight_layout()
plt.savefig('energy_vs_step.png', dpi=300)

print('Saved snapshot.png and energy_vs_step.png')
```python
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
