#!/usr/bin/env python3
"""
Interactive 2‑D N‑body playground (parallel C++ cores)
Menu
 1. Quick benchmark              -> benchmark_scaling.png
 2. Save trajectory + energy     -> trajectory.gif + energy_vs_time.png
 3. Live animation               -> live.gif
 4. Large‑N scaling test         -> scaling_largeN.png + scaling.csv
CLI flag
 --threads N   limit OpenMP threads (default 8)
"""
import os, time, math, csv, argparse, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange
from collections import defaultdict
# ---------- safe OpenMP defaults ----------
cli = argparse.ArgumentParser(add_help=False)
cli.add_argument('--threads', type=int, default=8, help='OMP threads')
args, _ = cli.parse_known_args()
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ.setdefault('OMP_STACKSIZE', '64M')
# ---------- import kernels ----------
from force_kernel import direct_omp, bh_omp
from fmm_kernel   import fmm_omp
SOLVERS = dict(direct=direct_omp, bh=bh_omp, fmm=fmm_omp)
# ---------- constants ----------
G, SOFT, DOMAIN, DT, STAR_M = 1.0, 0.03, 100.0, 0.01, 100.0
class B: __slots__=('x','y','vx','vy','m')
# ---------- init ----------
def init_sys(N, star=True):
    rng=np.random.default_rng(0); bodies=[]
    if star: bodies.append(B()); bodies[0].x=bodies[0].y=0; bodies[0].m=STAR_M; bodies[0].vx=bodies[0].vy=0
    for _ in range(N):
        r=rng.uniform(8,30); a=rng.uniform(0,2*np.pi)
        x,y=r*math.cos(a), r*math.sin(a); v=math.sqrt(G*STAR_M/r)
        bodies.append(B()); b=bodies[-1]; b.x,b.y,b.m,b.vx,b.vy=x,y,1.0,-v*math.sin(a),v*math.cos(a)
    return bodies
# ---------- force ----------
def acc(bodies, m):
    x=np.fromiter((b.x for b in bodies),float)
    y=np.fromiter((b.y for b in bodies),float)
    mass=np.fromiter((b.m for b in bodies),float)
    if m=='direct':return direct_omp(x,y,mass,G,SOFT)
    if m=='bh':    return bh_omp(x,y,mass,DOMAIN,0.5,G,SOFT)
    return fmm_omp(x,y,mass,DOMAIN,G=G,soft=SOFT)
# ---------- leapfrog ----------
def step(bodies,m):
    ax,ay=acc(bodies,m)
    for i,b in enumerate(bodies[1:]): b.vx+=ax[i+1]*DT*0.5; b.vy+=ay[i+1]*DT*0.5; b.x+=b.vx*DT; b.y+=b.vy*DT
    ax,ay=acc(bodies,m)
    for i,b in enumerate(bodies[1:]): b.vx+=ax[i+1]*DT*0.5; b.vy+=ay[i+1]*DT*0.5
# ---------- energy ----------
def E(bodies):
    ke=sum(0.5*b.m*(b.vx**2+b.vy**2) for b in bodies[1:])
    pe=0
    for i,a in enumerate(bodies):
        for b in bodies[:i]:
            dx,dy=a.x-b.x,a.y-b.y; r=math.hypot(dx,dy)+SOFT; pe-=G*a.m*b.m/r
    return ke+pe
# ---------- menu funcs ----------
def bench():
    Ns=[100,200,500,1_000,2_000,5_000]
    t=defaultdict(list)
    for N in Ns:
        bodies=init_sys(N,False)
        for m in SOLVERS:
            s=time.time(); acc(bodies,m); t[m].append(time.time()-s)
    for m in SOLVERS: plt.loglog(Ns,t[m],'o-',label=m)
    plt.xlabel('N');plt.ylabel('s');plt.legend();plt.savefig('benchmark_scaling.png');plt.show()

def traj():
    N=int(input('orbiters [100]:') or '100'); steps=int(input('steps[600]:') or '600'); m=input('solver[bh]:')or 'bh'; gif=input('gif[traj.gif]:')or'traj.gif'
    bodies=init_sys(N)
    xs=[[] for _ in bodies];ys=[[] for _ in bodies]; Elist=[]
    for s in trange(steps):
        for i,b in enumerate(bodies): xs[i].append(b.x);ys[i].append(b.y)
        step(bodies,m);
        if s%25==0: Elist.append((s*DT,E(bodies)))
    if Elist:
        t,e=zip(*Elist);plt.plot(t,e);plt.xlabel('t');plt.ylabel('E');plt.savefig('energy_vs_time.png');plt.close()
    fig,ax=plt.subplots();sc=ax.scatter([],[],s=3,c=['red']+['blue']*N);ax.set_aspect('equal');ax.set_xlim(-50,50);ax.set_ylim(-50,50)
    def up(i): sc.set_offsets(np.c_[ [x[i] for x in xs], [y[i] for y in ys] ]);return sc,
    FuncAnimation(fig,up,frames=steps).save(gif,writer=PillowWriter(fps=25))
    print('saved',gif)

def live():
    N=int(input('orbiters [50]:')or'50'); m=input('solver[fmm]:')or'fmm'; gif=input('gif[live.gif]:')or'live.gif'
    bodies=init_sys(N)
    fig,ax=plt.subplots();sc=ax.scatter([b.x for b in bodies],[b.y for b in bodies],s=3,c=['red']+['blue']*N)
    ax.set_aspect('equal');ax.set_xlim(-50,50);ax.set_ylim(-50,50)
    def up(_): step(bodies,m); sc.set_offsets([[b.x,b.y] for b in bodies]);return sc,
    FuncAnimation(fig,up,frames=600,interval=30).save(gif,writer=PillowWriter(fps=30));plt.show()

def scale():
    Ns=[1_000,2_000,4_000,8_000,16_000]
    with open('scaling.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['N','BH','FMM'])
        for N in Ns:
            bodies=init_sys(N,False)
            tb,time_b=time.time(),acc(bodies,'bh');tb=time.time()-tb
            tf,time_f=time.time(),acc(bodies,'fmm');tf=time.time()-tf
            w.writerow([N,tb,tf]);print(N,tb,tf)
    print('saved scaling.csv')
# ---------- menu ----------
while True:
    print('
1) benchmark 2) trajectory 3) live 4) scaling  q) quit')
    c=input('> ').lower()
    if c=='1': bench()
    elif c=='2': traj()
    elif c=='3': live()
    elif c=='4': scale()
    elif c=='q': break
