import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
from scipy.stats import linregress

# 匯入已編譯好的 C++ OpenMP 模組
import fmm_omp

# Constants for Python 版 direct / BH（保留原本的 direct、Barnes-Hut，以便比較）
G = 1.0  # gravitational constant
softening = 0.01  # softening parameter to avoid singularities
domain_size = 100.0  # size of the domain for the quad tree

class Particle:
    def __init__(self, x, y, mass=1.0, vx=0.0, vy=0.0):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.ax = 0.0
        self.ay = 0.0
        self.f = 0.0

# ---------------------------------------------------------------------------
#  Direct N-body method (O(n²))
# ---------------------------------------------------------------------------
def compute_forces_direct(particles):
    n = len(particles)
    for i in range(n):
        particles[i].ax = 0.0
        particles[i].ay = 0.0
        
        for j in range(n):
            if i != j:
                dx = particles[j].x - particles[i].x
                dy = particles[j].y - particles[i].y
                r = np.sqrt(dx*dx + dy*dy)
                if r < softening:
                    r = softening
                    
                # Gravitational force
                particles[i].f = G * particles[i].mass * particles[j].mass / r**2
                # Acceleration components
                particles[i].ax += particles[i].f * dx / (r * particles[i].mass)
                particles[i].ay += particles[i].f * dy / (r * particles[i].mass)

# ---------------------------------------------------------------------------
#  Barnes-Hut QuadTree 節點
# ---------------------------------------------------------------------------
class QuadTreeNode:
    global_level_registry = {}  # class-level registry of nodes per level
    level_hash = {}

    def __init__(self, cx, cy, size, level=0, max_level=20, parent=None):
        self.cx = cx      # Center x-coordinate
        self.cy = cy      # Center y-coordinate
        self.size = size  # Size of the square
        self.level = level  # Current level in the tree

        # --- grid key of this box ---------------------------------
        grid_i = int((self.cx + domain_size/2) / self.size)
        grid_j = int((self.cy + domain_size/2) / self.size)
        self._grid_key = (grid_i, grid_j)
        
        # --- hash table -------------------------------------------
        lvl_hash = QuadTreeNode.level_hash.setdefault(self.level, {})
        lvl_hash[self._grid_key] = self
        # ----------------------------------------------------------

        self.max_level = max_level
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.parent = parent
        self.particles = []
        self.total_mass = 0.0
        self.com_x = 0.0  # Center of mass x
        self.com_y = 0.0  # Center of mass y
        self.is_leaf = True
        self.is_empty = True
        
        self.multipole = None  # Placeholder for multipole expansion (unused in BH)
        self.local = None      # Placeholder for local expansion (unused in BH)
        self.p = 16            # Number of terms in multipole/local expansions (unused here)

        # Register this node in global level registry
        if level not in QuadTreeNode.global_level_registry:
            QuadTreeNode.global_level_registry[level] = []
        QuadTreeNode.global_level_registry[level].append(self)

    def insert(self, particle):
        self.is_empty = False

        if self.is_leaf:
            if len(self.particles) == 0 or self.level >= self.max_level:
                # Accept particle in this leaf
                self.particles.append(particle)
                self.total_mass += particle.mass
                self.com_x = (self.com_x * (self.total_mass - particle.mass) + particle.x * particle.mass) / self.total_mass
                self.com_y = (self.com_y * (self.total_mass - particle.mass) + particle.y * particle.mass) / self.total_mass
                return
            else:
                # Need to subdivide
                self.is_leaf = False
                old_particles = self.particles
                self.particles = []

                # Create children
                half = self.size / 2
                quarter = half / 2
                self.children[0] = QuadTreeNode(self.cx - quarter, self.cy - quarter, half, self.level + 1, self.max_level, parent=self)
                self.children[1] = QuadTreeNode(self.cx + quarter, self.cy - quarter, half, self.level + 1, self.max_level, parent=self)
                self.children[2] = QuadTreeNode(self.cx - quarter, self.cy + quarter, half, self.level + 1, self.max_level, parent=self)
                self.children[3] = QuadTreeNode(self.cx + quarter, self.cy + quarter, half, self.level + 1, self.max_level, parent=self)

                for p in old_particles:
                    self._insert_to_child(p)

        # Insert the current particle into the correct child
        self._insert_to_child(particle)

        # Update center of mass
        self.total_mass += particle.mass
        self.com_x = (self.com_x * (self.total_mass - particle.mass) + particle.x * particle.mass) / self.total_mass
        self.com_y = (self.com_y * (self.total_mass - particle.mass) + particle.y * particle.mass) / self.total_mass
        
    def _insert_to_child(self, particle):
        # Determine which quadrant the particle belongs to
        index = 0
        if particle.x > self.cx:
            index += 1  # East
        if particle.y > self.cy:
            index += 2  # South
            
        self.children[index].insert(particle)

    def compute_force_barnes_hut(self, particle, theta=0.3):
        if self.is_empty:
            return 0.0, 0.0
        
        dx = self.com_x - particle.x
        dy = self.com_y - particle.y
        r_squared = dx*dx + dy*dy
        if r_squared < softening**2:
            r_squared = softening**2
        r = np.sqrt(r_squared)
        
        if self.is_leaf or (self.size / r < theta):
            if r > 0:  # Avoid self-interactions
                f = G * particle.mass * self.total_mass / r_squared
                ax = f * dx / (r * particle.mass)
                ay = f * dy / (r * particle.mass)
                return ax, ay
            return 0.0, 0.0
        
        ax_total = 0.0
        ay_total = 0.0
        for child in self.children:
            if child is not None and not child.is_empty:
                ax, ay = child.compute_force_barnes_hut(particle, theta)
                ax_total += ax
                ay_total += ay
                
        return ax_total, ay_total

# ---------------------------------------------------------------------------
#  CUDA/pybind11 多核心 FMM：已改成用 C++ + OpenMP fmm_omp.fmm_force_theta
#  下面的 compute_forces_fmm 將直接呼叫 C++ 版本
# ---------------------------------------------------------------------------
def compute_forces_fmm(particles, domain_size=100.0, theta=0.6):
    """
    這裡不再用 Python 寫 multipole，而改成呼叫已編譯好的 C++ OpenMP 版本：
        fmm_omp.fmm_force_theta(...)
    輸入：
      particles  : 一個包含 Particle 物件的 list (每個 Particle.x, y, mass 已設定)
      domain_size: 範圍大小 (在 C++ 端我們只當作 domain = [  -domain_size/2, +domain_size/2 ]×[  -domain_size/2, +domain_size/2 ] 使用)
      theta      : Barnes–Hut 開啟角參數
    輸出：
      修改 particles[i].ax, particles[i].ay
    """
    N = len(particles)
    if N == 0:
        return

    # 建立 numpy 陣列：x, y, m
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    m = np.zeros(N, dtype=np.float64)
    for i, p in enumerate(particles):
        x[i] = p.x
        y[i] = p.y
        m[i] = p.mass
        # 重置加速度
        particles[i].ax = 0.0
        particles[i].ay = 0.0

    # domain = [xmin, xmax, ymin, ymax]，這裡我們假設 domain_size 為寬度
    half = domain_size / 2.0
    domain = np.array([-half, half, -half, half], dtype=np.float64)

    # 準備輸出 numpy array
    ax = np.zeros(N, dtype=np.float64)
    ay = np.zeros(N, dtype=np.float64)

    # 呼叫 C++ OpenMP 版本的 fmm_force_theta
    fmm_omp.fmm_force_theta(x, y, m, softening*softening, domain, theta, ax, ay)

    # 把結果回寫到 particles 裡
    for i in range(N):
        particles[i].ax = ax[i]
        particles[i].ay = ay[i]

# ---------------------------------------------------------------------------
#  Performance comparison
# ---------------------------------------------------------------------------
def performance_comparison(n_particles_list, method='all'):
    results = defaultdict(list)
    bh_error_array = []
    fmm_error_array = []
    direct_array = []

    for n in n_particles_list:
        print(f"\nNumber of particles: {n}")

        # 隨機產生 n 顆粒子
        particles = []
        np.random.seed(0)  # 為了可重現
        for i in range(n):
            x = (np.random.random() - 0.5) * domain_size
            y = (np.random.random() - 0.5) * domain_size
            mass = np.random.uniform(1.0, 3.0)
            particles.append(Particle(x, y, mass))

        # 深拷貝給不同方法
        particles_direct = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        particles_bh     = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        particles_fmm_py = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]

        # 直接 O(N²)
        if method in ['direct', 'all'] and n < 5000:  # n 若太大，就跳過 direct
            start_time = time.time()
            compute_forces_direct(particles_direct)
            direct_time = time.time() - start_time
            results['direct_times'].append(direct_time)
            print(f" Direct method: {direct_time:.6f} seconds")

        # Barnes-Hut O(N log N)
        if method in ['bh', 'all']:
            root = QuadTreeNode(0.0, 0.0, domain_size)
            start_time = time.time()
            for p in particles_bh:
                root.insert(p)
            for p in particles_bh:
                ax, ay = root.compute_force_barnes_hut(p)
                p.ax, p.ay = ax, ay
            bh_time = time.time() - start_time
            results['bh_times'].append(bh_time)
            print(f" Barnes-Hut: {bh_time:.6f} seconds")

        # FMM (改成呼叫 C++ OpenMP 版本)
        if method in ['fmm', 'all']:
            start_time = time.time()
            compute_forces_fmm(particles_fmm_py, domain_size=domain_size, theta=0.6)
            fmm_time = time.time() - start_time
            results['fmm_times'].append(fmm_time)
            print(f" FMM (C++ OpenMP): {fmm_time:.6f} seconds")

        # 如果同時做 direct, BH, FMM，就算誤差 (僅在 n<5000 時)
        if method == 'all' and n < 5000:
            bh_max_error = 0.0
            fmm_max_error = 0.0
            for i in range(n):
                p_direct = particles_direct[i]
                p_bh     = particles_bh[i]
                p_fmm_py = particles_fmm_py[i]

                direct_acc = np.sqrt(p_direct.ax**2 + p_direct.ay**2)
                
                # BH error
                ax_error_bh = p_bh.ax - p_direct.ax
                ay_error_bh = p_bh.ay - p_direct.ay
                bh_err = np.sqrt(ax_error_bh**2 + ay_error_bh**2)
                if direct_acc > 0:
                    bh_rel = bh_err / direct_acc
                else:
                    bh_rel = 0.0
                bh_error_array.append(bh_rel)
                bh_max_error = max(bh_max_error, bh_rel)

                # FMM error
                ax_error_fmm = p_fmm_py.ax - p_direct.ax
                ay_error_fmm = p_fmm_py.ay - p_direct.ay
                fmm_err = np.sqrt(ax_error_fmm**2 + ay_error_fmm**2)
                if direct_acc > 0:
                    fmm_rel = fmm_err / direct_acc
                else:
                    fmm_rel = 0.0
                fmm_error_array.append(fmm_rel)
                fmm_max_error = max(fmm_max_error, fmm_rel)

                direct_array.append(direct_acc)

            results['bh_max_errors'].append(bh_max_error)
            results['fmm_max_errors'].append(fmm_max_error)
            print(f" Barnes-Hut max relative error: {bh_max_error:.6f}")
            print(f" FMM max relative error: {fmm_max_error:.6f}")
        
    if method == 'all':
        return results, direct_array, bh_error_array, fmm_error_array
    elif method == 'fmm':
        return results
    else:
        return results

# ---------------------------------------------------------------------------
#  繪圖函式 (保留原版)
# ---------------------------------------------------------------------------
def plot_results(n_particles_list, results, direct_array, bh_error_array, fmm_error_array):
    
    plt.figure(figsize=(6, 5))
    plt.title('Relative error against direct N-body method values')
    plt.scatter(direct_array, bh_error_array, label='Barnes-Hut Error', color='blue', alpha=0.7)
    plt.scatter(direct_array, fmm_error_array, label='FMM Error', color='green', alpha=0.7)
    plt.xlabel('Direct N-body Method Values')
    plt.ylabel('Relative Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    # Performance comparison
    plt.subplot(1, 2, 1)
    if 'direct_times' in results and results['direct_times']:
        n_direct = n_particles_list[:len(results['direct_times'])]
        plt.plot(n_direct, results['direct_times'], 'o-', label='Direct N-body', color='red')
        scale_direct = results['direct_times'][0] / (n_direct[0]**2)
        plt.plot(n_direct, scale_direct * np.array(n_direct)**2, '--', label='O(N²) reference', color='red', alpha=0.5)
    if 'bh_times' in results and results['bh_times']:
        n_bh = n_particles_list[:len(results['bh_times'])]
        plt.plot(n_bh, results['bh_times'], 'o-', label='Barnes-Hut', color='blue')
        scale_bh = results['bh_times'][0] / (n_bh[0] * np.log(n_bh[0]))
        plt.plot(n_bh, scale_bh * np.array(n_bh) * np.log(n_bh), '--', label='O(N log N) reference', color='blue', alpha=0.5)
    if 'fmm_times' in results and results['fmm_times']:
        plt.plot(n_particles_list, results['fmm_times'], 's-', label='FMM (C++ OpenMP)', color='green')
        scale_fmm = results['fmm_times'][0] / n_particles_list[0]
        plt.plot(n_particles_list, scale_fmm * np.array(n_particles_list), '--', label='O(N) reference', color='green', alpha=0.5)
    
    plt.xlabel('Number of Particles')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    # Speedup comparison (如果 direct_times, bh_times, fmm_times 都存在)
    if 'direct_times' in results and 'bh_times' in results and 'fmm_times' in results:
        plt.subplot(1, 2, 2)
        n_speedup = n_particles_list[:len(results['direct_times'])]
        bh_speedup = [results['direct_times'][i] / results['bh_times'][i] for i in range(len(results['direct_times']))]
        fmm_speedup = [results['direct_times'][i] / results['fmm_times'][i] for i in range(len(results['direct_times']))]
        
        plt.loglog(n_speedup, bh_speedup, 's-', label='Barnes-Hut Speedup', color='blue')
        plt.loglog(n_speedup, fmm_speedup, '^-', label='FMM Speedup', color='green')
        plt.xlabel('Number of Particles')
        plt.ylabel('Speedup vs Direct')
        plt.title('Speedup Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Analysis', fontsize=16)
    plt.show()
    
# ---------------------------------------------------------------------------
#  修正版 plot_fmm_scaling：只在結果中存在 bh_times 時才嘗試繪製 Barnes-Hut
# ---------------------------------------------------------------------------
def plot_fmm_scaling(n_particles_large, results_large):

    plt.figure(figsize=(10, 6))
    # 繪製 FMM 時間
    plt.loglog(n_particles_large, results_large['fmm_times'], '^-', label='FMM (C++ OpenMP) O(N)', color='green', linewidth=2)
    
    # 如果有 Barnes-Hut 的時間，才繪製
    if 'bh_times' in results_large and results_large['bh_times']:
        plt.loglog(n_particles_large[:len(results_large['bh_times'])], results_large['bh_times'], 's-', 
                   label='Barnes-Hut O(N log N)', color='blue', linewidth=2)

        # 理論 O(N log N) 線
        scale_bh = results_large['bh_times'][0] / (n_particles_large[0] * np.log(n_particles_large[0]))
        plt.loglog(n_particles_large[:len(results_large['bh_times'])], 
                   [scale_bh * n * np.log(n) for n in n_particles_large[:len(results_large['bh_times'])]], 
                   '--', color='blue', alpha=0.7, label='Theoretical O(N log N)')

        # 用最早幾個點估算 Barnes-Hut 絕對階 (若需要)
        log_n = np.log(n_particles_large[:len(results_large['bh_times'])])
        log_times = np.log(results_large['bh_times'])
        slope, intercept, r_value, p_value, std_err = linregress(log_n, log_times)
        print(f"\nEmpirical Barnes-Hut scaling: O(N^{slope:.2f}), R² = {r_value**2:.3f}")
    
    # 繪製理論 O(N) 線
    scale_fmm = results_large['fmm_times'][0] / n_particles_large[0]
    plt.loglog(n_particles_large, [scale_fmm * n for n in n_particles_large], 
               '--', color='green', alpha=0.7, label='Theoretical O(N)')

    # 用所有 FMM 點估算階
    log_n_fmm = np.log(n_particles_large)
    log_times_fmm = np.log(results_large['fmm_times'])
    slope_fmm, intercept_fmm, r_val_fmm, p_val_fmm, std_err_fmm = linregress(log_n_fmm, log_times_fmm)
    print(f"\nEmpirical FMM scaling: O(N^{slope_fmm:.2f}), R² = {r_val_fmm**2:.3f}")

    plt.xlabel('Number of Particles')
    plt.ylabel('Computation Time (seconds)')
    plt.title('FMM Scaling for Large N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":    
    # 小規模測試 (direct)
    n_particles_small = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
    # 大規模測試 (FMM vs Barnes-Hut)
    n_particles_large = [5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]
    
    print("Performance comparison with direct method (small N):")
    results_small, darray, bharray, farray = performance_comparison(n_particles_small, method='all')
    
    print("\nPerformance comparison for large N (FMM vs Barnes-Hut):")
    results_large = performance_comparison(n_particles_large, method='fmm')
    
    # 繪圖
    plot_results(n_particles_small, results_small, darray, bharray, farray)
    plot_fmm_scaling(n_particles_large, results_large)

