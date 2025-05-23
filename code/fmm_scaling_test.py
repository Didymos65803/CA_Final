import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
from scipy.stats import linregress

# Constants
G = 1.0  # gravitational constant
softening = 0.01  # softening parameter to avoid singularities

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
    def __init__(self, cx, cy, size, level=0, max_level=10, parent=None):
        self.cx = cx      # Center x-coordinate
        self.cy = cy      # Center y-coordinate
        self.size = size  # Size of the square
        self.level = level  # Current level in the tree
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
        self.p = 8  # Number of terms in multipole/local expansions

    def insert(self, particle):
        self.is_empty = False
        
        # If this is a leaf node with no particles or has reached capacity and can be subdivided
        if len(self.particles) == 0:
            self.particles.append(particle)
            self.total_mass = particle.mass
            self.com_x = particle.x
            self.com_y = particle.y
            return
        
        # If this is a leaf node with fewer particles than capacity or has reached max level
        if self.is_leaf and len(self.particles) < 1 or self.level >= self.max_level: 
            self.particles.append(particle)
            # Update center of mass
            old_mass = self.total_mass
            self.total_mass += particle.mass
            if self.total_mass > 0:
                self.com_x = (self.com_x * old_mass + particle.x * particle.mass) / self.total_mass
                self.com_y = (self.com_y * old_mass + particle.y * particle.mass) / self.total_mass
            return
            
        # Otherwise, we need to subdivide and push particles down
        if self.is_leaf:
            self.is_leaf = False
            existing_particles = self.particles
            self.particles = []
            
            # Create the four children
            half_size = self.size / 2
            quarter_size = half_size / 2
            
            self.children[0] = QuadTreeNode(self.cx - quarter_size, self.cy - quarter_size, half_size, self.level + 1, self.max_level)  # NW
            self.children[1] = QuadTreeNode(self.cx + quarter_size, self.cy - quarter_size, half_size, self.level + 1, self.max_level)  # NE
            self.children[2] = QuadTreeNode(self.cx - quarter_size, self.cy + quarter_size, half_size, self.level + 1, self.max_level)  # SW
            self.children[3] = QuadTreeNode(self.cx + quarter_size, self.cy + quarter_size, half_size, self.level + 1, self.max_level)  # SE
            
            # Re-insert existing particles
            for p in existing_particles:
                self._insert_to_child(p)
                
        # Insert the new particle into the appropriate child
        self._insert_to_child(particle)
        
        # Update center of mass
        old_mass = self.total_mass
        self.total_mass += particle.mass
        if self.total_mass > 0:
            # Update center of mass for the node
            self.com_x = (self.com_x * old_mass + particle.x * particle.mass) / self.total_mass
            self.com_y = (self.com_y * old_mass + particle.y * particle.mass) / self.total_mass
        
    def _insert_to_child(self, particle):
        # Determine which quadrant the particle belongs to
        index = 0
        if particle.x > self.cx:
            index += 1  # East
        if particle.y > self.cy:
            index += 2  # South
            
        self.children[index].insert(particle)

    def compute_force_barnes_hut(self, particle, theta=0.5):
        if self.is_empty:
            return 0.0, 0.0
        
        # Distance between particle and center of mass
        dx = self.com_x - particle.x
        dy = self.com_y - particle.y
        r_squared = dx*dx + dy*dy + softening*softening
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
        
        self.multipole = np.zeros(self.p, dtype=complex)

        if self.is_leaf:
            # For leaf nodes, compute multipole expansion from particles
            for particle in self.particles:
                # Position relative to node center
                z = complex(particle.x - self.cx, particle.y - self.cy)
                if abs(z) < 1e-10:  # Avoid division by zero
                    z = complex(1e-10, 1e-10)
                
                # Multipole moments: a_k = q * z^k / k! (simplified)
                for k in range(self.p):
                    if k == 0:
                        self.multipole[k] += particle.mass
                    else:
                        self.multipole[k] += particle.mass * (z ** k) / math.factorial(k)
        else:
            # For internal nodes, translate multipole expansions from children
            for child in self.children:
                if child is not None and not child.is_empty:
                    child.compute_multipole_expansion()
                    # Translate child's multipole expansion to this node's center
                    z0 = complex(child.cx - self.cx, child.cy - self.cy)
                    self._translate_multipole_to_multipole(child.multipole, z0)

    def _translate_multipole_to_multipole(self, child_expansion, z0):
        #Translate multipole expansion from child to parent
        for k in range(self.p):
            for j in range(len(child_expansion)):
                if k + j < self.p:
                    # Translation formula
                    coeff = math.comb(k + j, j) if k + j >= j else 0
                    self.multipole[k + j] += child_expansion[j] * coeff * (z0 ** k)

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

    def _translate_local_to_local(self, parent_local, z0):
        #Translate local expansion from parent to child
        for k in range(len(parent_local)):
            for j in range(min(k + 1, self.p)):
                # Translation formula for local expansions
                coeff = math.comb(k, j) if k >= j else 0
                self.local[j] += parent_local[k] * coeff * (z0 ** (k - j))

    def _multipole_to_local(self, source_node):
        #Convert multipole expansion to local expansion
        z0 = complex(source_node.cx - self.cx, source_node.cy - self.cy)
        if abs(z0) < 1e-10:
            return
            
        # M2L translation (simplified)
        for k in range(len(source_node.multipole)):
            for j in range(self.p):
                if k + j + 1 < 20:  # Avoid overflow
                    coeff = (-1) ** k / (z0 ** (k + j + 1))
                    if k == 0:
                        self.local[j] += source_node.multipole[k] * coeff
                    else:
                        self.local[j] += source_node.multipole[k] * coeff * math.factorial(k + j) / (math.factorial(k) * math.factorial(j))

    def _get_interaction_list(self):
        #Get interaction list for this node (simplified)
        # This is a simplified interaction list - in a full FMM implementation,
        # this would be more sophisticated
        return []

    def evaluate_local_expansion(self, particle):
        #Evaluate local expansion at particle position
        if self.local is None:
            return 0.0, 0.0
            
        z = complex(particle.x - self.cx, particle.y - self.cy)
        
        # Evaluate potential and its derivative
        potential = 0.0
        force_complex = 0.0
        
        for k in range(len(self.local)):
            if k == 0:
                potential += self.local[k].real
            else:
                potential += self.local[k].real * (z ** k).real / math.factorial(k)
                force_complex += self.local[k] * k * (z ** (k-1)) / math.factorial(k)
        
        # Convert complex force to acceleration components
        ax = -G * particle.mass * force_complex.real / particle.mass
        ay = -G * particle.mass * force_complex.imag / particle.mass
        
        return ax, ay

# Barnes-Hut method (O(n log n))
def compute_forces_barnes_hut(particles, domain_size=100.0, theta=0.5):
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
def compute_forces_fmm(particles, domain_size=100.0, max_level=6):
    # Reset accelerations
    for p in particles:
        p.ax = 0.0
        p.ay = 0.0
    
    if len(particles) == 0:
        return
    
    # Build the quad tree with limited depth for FMM
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
            
        if node.is_leaf:
            # For leaf nodes, evaluate local expansion and add near-field interactions
            for particle in node.particles:
                # Far-field contribution from local expansion
                ax_far, ay_far = node.evaluate_local_expansion(particle)
                
                # Near-field interactions within the same leaf
                ax_near, ay_near = 0.0, 0.0
                for other in node.particles:
                    if particle != other:
                        dx = other.x - particle.x
                        dy = other.y - particle.y
                        r_squared = dx*dx + dy*dy + softening*softening
                        r = np.sqrt(r_squared)
                        
                        if r > 0:
                            f = G * particle.mass * other.mass / r_squared
                            ax_near += f * dx / (r * particle.mass)
                            ay_near += f * dy / (r * particle.mass)
                
                particle.ax += ax_far + ax_near
                particle.ay += ay_far + ay_near
        else:
            for child in node.children:
                if child is not None:
                    evaluate_forces(child)
    
    evaluate_forces(root)

def update_positions(particles, dt=0.1):
    for p in particles:
        
        # Update velocities (half-step)
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt
        
        # Update positions
        p.x += p.vx * 0.5 * dt
        p.y += p.vy * 0.5 * dt


def performance_comparison(n_particles_list, method='all'):
    results = defaultdict(list)

    for n in n_particles_list:
        print(f"\nNumber of particles: {n}")

        # Create particles randomly distributed in a square
        particles = []
        np.random.seed(0)  # For reproducibility
        for i in range(n):
            x = (np.random.random() - 0.5) * 100.0
            y = (np.random.random() - 0.5) * 100.0
            mass = np.random.uniform(1.0, 5.0)  # Mass between 1 and 5
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
            fmm_time = time.time() - start_time
            results['fmm_times'].append(fmm_time)
            print(f" FMM: {fmm_time:.6f} seconds")
        
        # Calculate discrapency between methods (if all methods are used)
        if method == 'all' and n < 5000:
            bh_error = 0.0
            fmm_error = 0.0
            for i in range(n):
                p_direct = particles_direct[i]
                p_bh = particles_bh[i]
                p_fmm = particles_fmm[i]

                # BH error
                ax_error = abs(p_bh.ax - p_direct.ax)
                ay_error = abs(p_bh.ay - p_direct.ay)
                error = np.sqrt(ax_error**2 + ay_error**2)
                direct = np.sqrt(p_direct.ax**2 + p_direct.ay**2) + 1e-10 # Avoid division by zero
                bh_error += error / direct
                # FMM error
                ax_error = abs(p_fmm.ax - p_direct.ax)
                ay_error = abs(p_fmm.ay - p_direct.ay)
                error = np.sqrt(ax_error**2 + ay_error**2)
                fmm_error += error / direct
            
            results['bh_errors'].append(bh_error / n)
            results['fmm_errors'].append(fmm_error / n)
            print(f" Barnes-Hut relative error: {bh_error / n:.6f}")
            print(f" FMM relative error: {fmm_error / n:.6f}")
        
    return results

def plot_results(n_particles_list, results):
    
    plt.figure(figsize=(15, 6))

    # Performance comparison
    plt.subplot(1, 3, 1)
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
    
    # Error comparison
    if 'bh_errors' in results and results['bh_errors']:
        plt.subplot(1, 3, 2)
        n_error = n_particles_list[:len(results['bh_errors'])]
        plt.loglog(n_error, results['bh_errors'], 's-', label='Barnes-Hut Error', color='blue')
        if 'fmm_errors' in results:
            plt.loglog(n_error, results['fmm_errors'], '^-', label='FMM Error', color='green')
        plt.xlabel('Number of Particles')
        plt.ylabel('Relative Error')
        plt.title('Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Speedup comparison
    if 'direct_times' in results and 'bh_times' in results and 'fmm_times' in results:
        plt.subplot(1, 3, 3)
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
    
    plt.tight_layout()
    plt.show()

def plot_fmm_scaling(n_particles, results):

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
    n_particles_small = [100, 300, 500, 750, 1000, 2000, 5000]  # Smaller for direct comparison
    n_particles_large = [5000, 7000, 8500, 10000, 20000, 30000, 40000, 50000, 100000]  # Larger for FMM scaling
    
    print("Performance comparison with direct method (smaller N):")
    results_small = performance_comparison(n_particles_small, 'all')
    
    print("\nPerformance comparison for large N (FMM vs Barnes-Hut):")
    results_large = performance_comparison(n_particles_large, method='fmm')
    
    # Plot results
    plot_results(n_particles_small, results_small)
    plot_fmm_scaling(n_particles_large, results_large)
