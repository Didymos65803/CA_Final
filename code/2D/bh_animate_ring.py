"""Animate the ring evolution using the existing bh_omp kernel.
Output: orbit.gif (default 200 frames)."""
import numpy as np, matplotlib.pyplot as plt, matplotlib.animation as anim, matplotlib
from force_kernel import bh_omp
matplotlib.use('Agg')

# ---------- parameters ----------
N, R, DT, STEPS = 800, 20.0, 0.02, 400
DOMAIN, THETA, G, SOFT = 60.0, 0.5, 1.0, 0.05
frames = 200  # number of frames in animation
skip   = STEPS//frames

class P: __slots__ = ("x","y","vx","vy","m")
parts = [P() for _ in range(N)]
for i,p in enumerate(parts):
    a = 2*np.pi*i/N; v = np.sqrt(G/R)
    p.x, p.y = R*np.cos(a), R*np.sin(a)
    p.vx, p.vy = -v*np.sin(a), v*np.cos(a)
    p.m = 1.0/N

def array(attr):
    return np.fromiter((getattr(p,attr) for p in parts), float, N)

afig, ax = plt.subplots(figsize=(6,6))
dots,   = ax.plot([], [], 'o', ms=2)
ax.set_xlim(-DOMAIN,DOMAIN); ax.set_ylim(-DOMAIN,DOMAIN)
ax.set_aspect('equal'); ax.set_title('Barnes–Hut ring')

step = 0

def update(frame):
    global step
    for _ in range(skip):
        x = array('x'); y = array('y'); m = array('m')
        ax_, ay_ = bh_omp(x,y,m,DOMAIN,THETA,G,SOFT)
        for i,p in enumerate(parts):
            p.vx += ax_[i]*DT*0.5; p.vy += ay_[i]*DT*0.5
            p.x  += p.vx*DT;       p.y  += p.vy*DT
        x[:] = array('x'); y[:] = array('y')
        ax_, ay_ = bh_omp(x,y,m,DOMAIN,THETA,G,SOFT)
        for i,p in enumerate(parts):
            p.vx += ax_[i]*DT*0.5; p.vy += ay_[i]*DT*0.5
        step += 1
    dots.set_data(array('x'), array('y'))
    ax.set_title(f'step {step}/{STEPS}')
    return dots,

ani = anim.FuncAnimation(afig, update, frames=frames, blit=True)
ani.save('orbit.gif', writer='pillow', fps=30)
print('orbit.gif saved')