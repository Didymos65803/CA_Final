#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt
from tqdm import trange
from force_kernel import bh_omp, direct_omp
from fmm_kernel import fmm_omp

methods=dict(direct=direct_omp,bh=bh_omp,fmm=fmm_omp)

parser=argparse.ArgumentParser("2D N‑body orbit demo")
parser.add_argument('-N', type=int, default=800)
parser.add_argument('-m','--method', choices=methods.keys(), default='bh')
parser.add_argument('-s','--steps', type=int, default=400)
parser.add_argument('-o','--output', default='orbit.gif')
parser.add_argument('--threads', type=int, default=None)
args=parser.parse_args()

if args.threads: import os; os.environ['OMP_NUM_THREADS']=str(args.threads)

G, DT, DOMAIN, SOFT = 1.0, 0.02, 60.0, 0.05
N=args.N
R=20.0; v0=np.sqrt(G/R)
px=[R*np.cos(2*np.pi*i/N) for i in range(N)]
py=[R*np.sin(2*np.pi*i/N) for i in range(N)]
vx=[-v0*np.sin(2*np.pi*i/N) for i in range(N)]
vy=[ v0*np.cos(2*np.pi*i/N) for i in range(N)]
m =[1.0/N]*N

fig,ax=plt.subplots(figsize=(6,6))
sc=ax.scatter(px,py,s=2)
ax.set_xlim(-DOMAIN,DOMAIN);ax.set_ylim(-DOMAIN,DOMAIN);ax.set_aspect('equal')
plt.tight_layout()
plt.title(f"{args.method} N={N}")
plt.savefig('init.png')

kernel=methods[args.method]
from matplotlib.animation import FuncAnimation

def step_system():
    x=np.array(px); y=np.array(py); mm=np.array(m)
    if args.method=='direct': ax_,ay_=kernel(x,y,mm,G,SOFT)
    else: ax_,ay_=kernel(x,y,mm,DOMAIN,G=G,soft=SOFT)
    for i in range(N):
        vx[i]+=ax_[i]*DT*0.5; vy[i]+=ay_[i]*DT*0.5
        px[i]+=vx[i]*DT;      py[i]+=vy[i]*DT
    x=np.array(px); y=np.array(py)
    if args.method=='direct': ax_,ay_=kernel(x,y,mm,G,SOFT)
    else: ax_,ay_=kernel(x,y,mm,DOMAIN,G=G,soft=SOFT)
    for i in range(N):
        vx[i]+=ax_[i]*DT*0.5; vy[i]+=ay_[i]*DT*0.5

ani=FuncAnimation(fig, lambda f: (step_system(), sc.set_offsets(np.c_[px,py])), frames=args.steps, blit=False)
ani.save(args.output, writer='pillow', fps=30)
print('Saved',args.output)
