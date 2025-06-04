import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
from scipy.stats import linregress

# Constants
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

# Direct N-body method (O(n²))
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
        
        self.multipole = None  # Placeholder for multipole expansion
        self.local = None  # Placeholder for local expansion
        self.p = 16  # Number of terms in multipole/local expansions

        # Register this node in the global level registry
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
        
        # Distance between particle and center of mass
        dx = self.com_x - particle.x
        dy = self.com_y - particle.y
        r_squared = dx*dx + dy*dy
        if r_squared < softening**2:
            r_squared = softening**2
        r = np.sqrt(r_squared)
        
        # If it's a leaf with a single particle or the node is sufficiently far away
        if self.is_leaf or (self.size / r < theta):
            if r > 0:  # Avoid self-interactions
                # Gravitational force
                f = G * particle.mass * self.total_mass / r_squared
                # Acceleration components                
                ax = f * dx / (r * particle.mass)
                ay = f * dy / (r * particle.mass)
                return ax, ay
            return 0.0, 0.0
        
        # Otherwise, recursively compute forces from children
        ax_total = 0.0
        ay_total = 0.0
        for child in self.children:
            if child is not None and not child.is_empty:
                ax, ay = child.compute_force_barnes_hut(particle, theta)
                ax_total += ax
                ay_total += ay
                
        return ax_total, ay_total

    def compute_multipole_expansion(self):
        # Compute multipole expansion for this node 
        if self.is_empty:
            return
        
        self.multipole = np.zeros(self.p, dtype=complex) # Initialize multipole expansion coefficients
        self.multipole[0] = self.total_mass  # a_0 is the total mass

        if self.is_leaf:
            #print(f"Leaf at ({self.cx:.1f}, {self.cy:.1f}) multipole:", self.multipole)
            # For leaf nodes, compute multipole expansion from particles
            
            for particle in self.particles:
                z_rel = complex(particle.x - self.cx, particle.y - self.cy)
                # Q is already computed as self.total_mass
                for l in range(1, self.p): # l from 1 to p-1 for a_l
                    self.multipole[l] -= particle.mass * (z_rel ** l) / l

        else:
            # For internal nodes, translate multipole expansions from children
            for child in self.children:
                if child is not None and not child.is_empty:
                    child.compute_multipole_expansion()
                    # Translate child's multipole expansion to this node's center
                    z0 = complex(child.cx - self.cx, child.cy - self.cy)
                    self._translate_multipole_to_multipole(child.multipole, z0)

    def _translate_multipole_to_multipole(self, child_expansion, z0):
        if self.multipole is None:
            self.multipole = np.zeros(self.p, dtype=complex)

        for l in range(self.p):
            if l == 0:
                self.multipole[0] += child_expansion[0]
                continue

            self.multipole[l] += -child_expansion[0] * (z0**l) / l

            # Σ_{k=1..l} a_k C(l−1,k−1) z₀^{l−k}
            for k in range(1, min(l, self.p-1)+1):
                self.multipole[l] += (
                    child_expansion[k] *
                    math.comb(l-1, k-1) *
                    (z0 ** (l-k))
                )
    
    def compute_local_expansion(self, parent_local=None):
        #Compute local expansion for this node (downward pass)
        if self.is_empty:
            return
            
        self.local = np.zeros(self.p, dtype=complex)
        
        # Add contribution from parent's local expansion
        if parent_local is not None and self.parent is not None:
            z0 = complex(self.cx - self.parent.cx, self.cy - self.parent.cy)
            self._translate_local_to_local(parent_local, z0)
        
        # Add contributions from well-separated nodes (interaction list)
        interaction_list = self._get_interaction_list()
        for node in interaction_list:
            if node.multipole is not None:
                self._multipole_to_local(node)
        #print(f"Node at ({self.cx:.1f}, {self.cy:.1f}) local expansion:", self.local)

    def _translate_local_to_local(self, parent_local, z0):
        #Translate local expansion from parent to child centered at z0 (L2L)
        if self.local is None:
            self.local = np.zeros(self.p, dtype=complex)

        # Make a copy to not modify parent_local in-place
        b = parent_local.copy()

        # Horner's scheme to translate local expansion to new center
        for k in range(self.p - 2, -1, -1):
            b[k] += z0 * b[k + 1]

        self.local += b

    def _multipole_to_local(self, source_node):
        #print(f"M2L: from ({source_node.cx:.1f}, {source_node.cy:.1f}) → ({self.cx:.1f}, {self.cy:.1f})")
        #print("source multipole:", source_node.multipole)
        #Convert source multipole expansion to local expansion at target (M2L)
        z0 = complex(source_node.cx - self.cx, source_node.cy - self.cy)
        if abs(z0) < 1e-12: # Avoid division by zero if centers are too close
            return

        if self.local is None:
            self.local = np.zeros(self.p, dtype=complex)

        source_multipoles = source_node.multipole # These are M_k or a_k

        # L_l = sum_{k=0}^{p-1} M_k * (-1)^k * C(l+k, k) / (z0^(l+k+1))
        for l_idx in range(self.p): # l for L_l
            term_sum_for_L_l = 0j
            for k_idx in range(self.p): # k for M_k
                if source_multipoles[k_idx] == 0: # Skip if source term is zero
                    continue

                binom_coeff = math.comb(l_idx + k_idx, k_idx)
                denominator = z0**(l_idx + k_idx + 1)

                term = ((-1)**k_idx) * source_multipoles[k_idx] * binom_coeff / denominator
                term_sum_for_L_l += term
            self.local[l_idx] += term_sum_for_L_l

    def are_neighbors(self, other):
        # Check if another box is a neighbor (shares edge or corner).
        dx = abs(self.cx - other.cx)
        dy = abs(self.cy - other.cy)
        dist = self.size
        return dx <= dist and dy <= dist
    
    def get_neighbors(self):
        if hasattr(self, '_neighbor_cache'):
            return self._neighbor_cache

        i, j = self._grid_key
        h     = QuadTreeNode.level_hash[self.level]
        nbrs = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                n = h.get((i+di, j+dj))
                if n is not None:
                    nbrs.append(n)

        self._neighbor_cache = nbrs     # cache for future calls
        return nbrs
    
    def _get_interaction_list(self):
        # empty root -> nothing to do
        if self.parent is None:
            return []

        interaction = []
        # parent plus all its neighbours (U-list in Greengard’s jargon)
        for nbr in self.parent.get_neighbors() + [self.parent]:

            # boxes that touch B are excluded
            if self.are_neighbors(nbr):
                continue

            if nbr.is_leaf:
                # (1) neighbour itself, already a leaf – ADD IT
                interaction.append(nbr)
            else:
                # (2) neighbour is internal – test each child
                for child in nbr.children:
                    if child is not None and not self.are_neighbors(child):
                        interaction.append(child)

        return interaction

    def evaluate_local_expansion(self, particle):
        if self.local is None:
            return 0.0, 0.0

        z = complex(particle.x - self.cx, particle.y - self.cy)
        force_complex = 0.0 + 0.0j
        z_power = 1.0 + 0.0j                # z⁰
        for k in range(1, self.p):          # start at k = 1
            force_complex += self.local[k] * k * z_power
            z_power *= z                    # increment z^{k}

        ax = -force_complex.real
        ay =  force_complex.imag
        return ax, ay

# Barnes-Hut method (O(n log n))
def compute_forces_barnes_hut(particles, domain_size=100.0, theta=0.3):
    # Reset accelerations
    for p in particles:
        p.ax = 0.0
        p.ay = 0.0
    
    # Build the quad tree
    root = QuadTreeNode(0.0, 0.0, domain_size)
    for p in particles:
        root.insert(p)

    # Compute forces for each particle
    for p in particles:
        ax, ay = root.compute_force_barnes_hut(p, theta)
        p.ax += ax
        p.ay += ay

# Fast Multipole Method (O(N))
def compute_forces_fmm(particles, domain_size=100.0, max_level=20):
    QuadTreeNode.global_level_registry.clear()  # Clear global registry for new run
    QuadTreeNode.level_hash.clear() 
    # Reset accelerations
    for p in particles:
        p.ax = 0.0
        p.ay = 0.0
    
    if len(particles) == 0:
        return
    
    # Build the quad tree with limited depth for FMM
    #target_leaf = 16
    #max_level = math.ceil(math.log(max(len(particles)/target_leaf, 1), 4))
    root = QuadTreeNode(0.0, 0.0, domain_size, max_level=max_level)
    for p in particles:
        root.insert(p)
    
    # Upward pass: compute multipole expansions
    root.compute_multipole_expansion()
    
    # Downward pass: compute local expansions
    def downward_pass(node, parent_local=None):
        if node.is_empty:
            return
            
        node.compute_local_expansion(parent_local)
        
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    downward_pass(child, node.local)
    
    downward_pass(root)
    
    # Evaluate forces at particle positions
    def evaluate_forces(node):
        if node.is_empty:
            return
        # ---------- leaf -----------------------------------------------------
        if node.is_leaf:
            for particle in node.particles:

                # far-field part (local expansion)
                ax_far, ay_far = node.evaluate_local_expansion(particle)

                # near-field part (self-leaf + all neighbours, whether leaf or not)
                ax_near, ay_near = 0.0, 0.0
                soft2 = softening * softening

                # helper *inside* the loop so it can see “particle”
                def accumulate_direct(src_node):
                    nonlocal ax_near, ay_near
                    if src_node.is_leaf:
                        for other in src_node.particles:
                            if other is particle:
                                continue
                            dx = other.x - particle.x
                            dy = other.y - particle.y
                            r2 = dx*dx + dy*dy + soft2
                            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                            ax_near += G * other.mass * dx * inv_r3
                            ay_near += G * other.mass * dy * inv_r3
                    else:
                        for child in src_node.children:
                            if child is not None:
                                accumulate_direct(child)

                # apply to self-leaf and every neighbour
                accumulate_direct(node)
                for neighbour in node.get_neighbors():
                    accumulate_direct(neighbour)

                # store total acceleration
                particle.ax += ax_far + ax_near
                particle.ay += ay_far + ay_near
        # ---------- internal node -------------------------------------------
        else:
            for child in node.children:
                if child is not None:
                    evaluate_forces(child)
    
    evaluate_forces(root)

def performance_comparison(n_particles_list, method='all'):
    results = defaultdict(list)
    bh_error_array = []
    fmm_error_array = []
    direct_array = []
    for n in n_particles_list:
        print(f"\nNumber of particles: {n}")

        # Create particles randomly distributed in a square
        particles = []
        np.random.seed(0)  # For reproducibility
        # Create a massive particle at the center
        #particles.append(Particle(0.0, 0.0, mass=100.0))  # Central massive particle
        for i in range(n):
            x = (np.random.random() - 0.5) * 100.0
            y = (np.random.random() - 0.5) * 100.0
            mass = np.random.uniform(1.0, 3.0)  # Mass between 1 and 5
            particles.append(Particle(x, y, mass))
        
        # Make a deep copy for comparing the two methods
        particles_direct = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        particles_bh = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        particles_fmm = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        
        # Benchmark direct method
        if method in ['direct', 'all'] and n < 5000: #Skip direct for large n
            start_time = time.time()
            compute_forces_direct(particles_direct)
            direct_time = time.time() - start_time
            results['direct_times'].append(direct_time)
            print(f" Direct method: {direct_time:.6f} seconds")
        
        # Benchmark Barnes-Hut method
        if method in ['fmm', 'all']:
            start_time = time.time()
            compute_forces_barnes_hut(particles_bh)
            bh_time = time.time() - start_time
            results['bh_times'].append(bh_time)
            print(f" Barnes-Hut: {bh_time:.6f} seconds")

        # Benchmark FMM method
        if method in ['fmm', 'all']:
            start_time = time.time()
            compute_forces_fmm(particles_fmm)
            #compute_forces_fmm(particles_fmm, domain_size=100, max_level=max_level)
            fmm_time = time.time() - start_time
            results['fmm_times'].append(fmm_time)
            print(f" FMM: {fmm_time:.6f} seconds")
        
        # Calculate discrapency between methods (if all methods are used)
        if method == 'all' and n < 5000:
            bh_max_error = 0.0
            fmm_max_error = 0.0
            for i in range(n):
                p_direct = particles_direct[i]
                p_bh = particles_bh[i]
                p_fmm = particles_fmm[i]

                # BH error
                ax_error = p_bh.ax - p_direct.ax
                ay_error = p_bh.ay - p_direct.ay
                bh_error_magnitude = np.sqrt(ax_error**2 + ay_error**2)
                direct = np.sqrt(p_direct.ax**2 + p_direct.ay**2)
                direct_array.append(direct)
                bh_error_array.append(bh_error_magnitude / direct)
                bh_max_error = max(bh_max_error, bh_error_magnitude / direct)
                
                # FMM error
                ax_error = p_fmm.ax - p_direct.ax
                ay_error = p_fmm.ay - p_direct.ay
                fmm_error_magnitude = np.sqrt(ax_error**2 + ay_error**2)
                fmm_max_error = max(fmm_max_error, fmm_error_magnitude / direct)
                fmm_error_array.append(fmm_error_magnitude / direct)
            
            results['bh_max_errors'].append(bh_max_error)
            results['fmm_max_errors'].append(fmm_max_error)
            print(f" Barnes-Hut max error: {bh_max_error:.6f}")
            print(f" FMM max error: {fmm_max_error:.6f}")
        
    if method == 'all':
        return results, direct_array, bh_error_array, fmm_error_array
    if method == 'fmm':
        return results

def plot_results(n_particles_list, results, direct_array, bh_error_array, fmm_error_array):
    
    # Error analysis
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
        plt.plot(n_particles_list, results['fmm_times'], 's-', label='FMM', color='green')
        scale_fmm = results['fmm_times'][0] / n_particles_list[0]
        plt.plot(n_particles_list, scale_fmm * np.array(n_particles_list), '--', label='O(N) reference', color='green', alpha=0.5)
    
    plt.xlabel('Number of Particles')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    # Speedup comparison
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
    
def plot_fmm_scaling(n_particles_large, results_large):

    plt.figure(figsize=(10, 6))
    plt.loglog(n_particles_large, results_large['fmm_times'], '^-', label='FMM O(N)', color='green', linewidth=2)
    plt.loglog(n_particles_large, results_large['bh_times'], 's-', label='Barnes-Hut O(N log N)', color='blue', linewidth=2)

    # Theoretical O(N) scaling
    scale_fmm = results_large['fmm_times'][0] / n_particles_large[0]
    scale_bh = results_large['bh_times'][0] / (n_particles_large[0] * np.log(n_particles_large[0]))
    plt.loglog(n_particles_large, [scale_fmm * n for n in n_particles_large], '--', color='green', alpha=0.7, label='Theoretical O(N)')
    plt.loglog(n_particles_large, [scale_bh * n * np.log(n) for n in n_particles_large], '--', color='blue', alpha=0.7, label='Theoretical O(N log N)')
    
    # Fit and display empirical scaling
    log_n = np.log(n_particles_large)
    log_times = np.log(results_large['fmm_times'])
    slope, intercept, r_value, p_value, std_err = linregress(log_n, np.log(results_large['bh_times']))
    print(f"\nEmpirical Barnes-Hut scaling: O(N log N) {slope:.2f} with R² = {r_value**2:.3f}")
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_times)
    print(f"\nEmpirical FMM scaling: O(N^{slope:.2f}) with R² = {r_value**2:.3f}")
    
    plt.xlabel('Number of Particles')
    plt.ylabel('Computation Time (seconds)')
    plt.title('FMM Scaling for Large N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":    
    # Test with different particle counts
    n_particles_small = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]  # Smaller for direct comparison
    n_particles_large = [5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]  # Larger for FMM scaling
    
    print("Performance comparison with direct method (smaller N):")
    results_small, darray, bharray, farray = performance_comparison(n_particles_small, 'all')
    
    print("\nPerformance comparison for large N (FMM vs Barnes-Hut):")
    results_large = performance_comparison(n_particles_large, method='fmm')
    
    # Plot results
    plot_results(n_particles_small, results_small, darray, bharray, farray)
    plot_fmm_scaling(n_particles_large, results_large)