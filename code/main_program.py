import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
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
                #f = G * particles[i].mass * particles[j].mass / r**2
                particles[i].f = G * particles[i].mass * particles[j].mass / r**2
                
                # Acceleration components
                particles[i].ax += particles[i].f * dx / (r * particles[i].mass)
                particles[i].ay += particles[i].f * dy / (r * particles[i].mass)

class QuadTreeNode:
    def __init__(self, cx, cy, size):
        self.cx = cx      # Center x-coordinate
        self.cy = cy      # Center y-coordinate
        self.size = size  # Size of the square
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.particles = []
        self.total_mass = 0.0
        self.com_x = 0.0  # Center of mass x
        self.com_y = 0.0  # Center of mass y
        self.is_leaf = True
        self.is_empty = True

    def insert(self, particle):
        self.is_empty = False
        
        # If this is a leaf node with no particles or has reached capacity and can be subdivided
        if len(self.particles) == 0:
            self.particles.append(particle)
            self.total_mass = particle.mass
            self.com_x = particle.x
            self.com_y = particle.y
            return
        
        # If this is a leaf node with fewer particles than capacity
        if self.is_leaf and len(self.particles) < 1:  # Just one particle per leaf for simplicity
            self.particles.append(particle)
            # Update center of mass
            self.total_mass += particle.mass
            self.com_x = (self.com_x * (self.total_mass - particle.mass) + particle.x * particle.mass) / self.total_mass
            self.com_y = (self.com_y * (self.total_mass - particle.mass) + particle.y * particle.mass) / self.total_mass
            return
            
        # Otherwise, we need to subdivide and push particles down
        if self.is_leaf:
            self.is_leaf = False
            existing_particles = self.particles
            self.particles = []
            
            # Create the four children
            half_size = self.size / 2
            quarter_size = half_size / 2
            
            self.children[0] = QuadTreeNode(self.cx - quarter_size, self.cy - quarter_size, half_size)  # NW
            self.children[1] = QuadTreeNode(self.cx + quarter_size, self.cy - quarter_size, half_size)  # NE
            self.children[2] = QuadTreeNode(self.cx - quarter_size, self.cy + quarter_size, half_size)  # SW
            self.children[3] = QuadTreeNode(self.cx + quarter_size, self.cy + quarter_size, half_size)  # SE
            
            # Re-insert existing particles
            for p in existing_particles:
                self._insert_to_child(p)
                
        # Insert the new particle into the appropriate child
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

    def compute_force(self, particle, theta=0.5):
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
                ax, ay = child.compute_force(particle, theta)
                ax_total += ax
                ay_total += ay
                
        return ax_total, ay_total

# Fast Multipole Method (FMM) implementation
def compute_forces_fmm(particles, domain_size=100.0, theta=0.5):
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
        ax, ay = root.compute_force(p, theta)
        p.ax += ax
        p.ay += ay



def update_positions(particles, dt=0.1):
    for p in particles:
        
        # Update velocities (half-step)
        p.vx += 0.5 * p.ax * dt
        p.vy += 0.5 * p.ay * dt
        
        # Update positions
        p.x += p.vx * 0.5 * dt
        p.y += p.vy * 0.5 * dt


def performance_comparison(n_particles_list, method='both'):
    results = defaultdict(list)
    positions_x = []
    positions_y = []
    plt.figure(figsize=(24, 4))

    for n in n_particles_list:
        # Create particles randomly distributed in a square
        particles = []
        for i in range(n):
            x = (np.random.random() - 0.5) * 100.0
            y = (np.random.random() - 0.5) * 100.0
            mass = np.random.uniform(1.0, 5.0)  # Mass between 1 and 5
            particles.append(Particle(x, y, mass))
        
        # Make a deep copy for comparing the two methods
        particles_direct = [Particle(p.x, p.y, p.mass, p.vx, p.vy) for p in particles]
        
        # Benchmark direct method
        if method in ['direct', 'both']:
            start_time = time.time()
            compute_forces_direct(particles_direct)
            direct_time = time.time() - start_time
            results['direct_times'].append(direct_time)
            print(f"Direct method with {n} particles: {direct_time:.6f} seconds")
        
        # Benchmark FMM
        if method in ['fmm', 'both']:
            start_time = time.time()
            compute_forces_fmm(particles)
            fmm_time = time.time() - start_time
            results['fmm_times'].append(fmm_time)
            print(f"FMM with {n} particles: {fmm_time:.6f} seconds")
        
        # Calculate discrapency between two methods
        if method == 'both':
            total_error = 0.0
            for i in range(n):
                p_direct = particles_direct[i]
                p_fmm = particles[i]
                ax_error = abs(p_direct.ax - p_fmm.ax)
                ay_error = abs(p_direct.ay - p_fmm.ay)
                # Relative error norm
                error = math.sqrt(ax_error**2 + ay_error**2) / (math.sqrt(p_direct.ax**2 + p_direct.ay**2) + 1e-10)
                #error = abs(p_direct.f - p_fmm.f) / (abs(p_direct.f) + 1e-10)
                total_error += error
            avg_error = total_error / n
            results['errors'].append(avg_error)
            print(f"Average relative error: {avg_error:.6f}")
        
        plt.subplot(1, len(n_particles_list), n_particles_list.index(n)+1)
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.grid(True)
        positions_x.clear()
        positions_y.clear()
        positions_x.append([p.x for p in particles])
        positions_y.append([p.y for p in particles])
        plt.scatter(positions_x, positions_y, s=particles[i].mass * 3, c='red', alpha=0.7)
        #plt.axis('equal')
        plt.title('Particle Initial Positions for n = {}'.format(n))

    # Particle positions
    #plt.tight_layout()
    plt.show()
    return results

def plot_results(n_particles_list, results):
    
    # fitting the data in log-log
    log_n = np.log(n_particles_list)
    log_direct = np.log(results['direct_times'])
    log_fmm = np.log(results['fmm_times'])
    slope_direct, *_ = linregress(log_n, log_direct)
    slope_fmm, *_ = linregress(log_n, log_fmm)
    print(f"Empirical slope (Direct): ~{slope_direct:.2f}")
    print(f"Empirical slope (FMM): ~{slope_fmm:.2f}")

    plt.figure(figsize=(12, 6))
    
    # Performance comparison
    x = np.array(n_particles_list)
    # Scale the theoretical curves to match the actual data
    scale_factor_direct = results['direct_times'][0] / (n_particles_list[0]**2)
    scale_factor_fmm = results['fmm_times'][0] / (n_particles_list[0] * np.log(n_particles_list[0]))
    #scale_factor_fmm = results['fmm_times'][0] / (np.log2(n_particles_list[0]))
    #scale_factor_fmm = np.mean([results['fmm_times'][i] / (n * np.log2(n))for i, n in enumerate(n_particles_list)])
    
    plt.plot(x, scale_factor_direct * x**2, '--', label='O(n²) reference')
    plt.plot(x, scale_factor_fmm * x * np.log(x), '--', label='O(n log n) reference')
    plt.plot(n_particles_list, results['direct_times'], 'o-', label='Direct N-body')
    plt.plot(n_particles_list, results['fmm_times'], 's-', label='FMM')
    
    plt.xlabel('Number of Particles')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Performance Comparison and Scaling Analysis')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    '''
    if 'errors' in results:
        plt.figure(figsize=(8, 6))
        plt.plot(n_particles_list, results['errors'], 'o-')
        plt.xlabel('Number of Particles')
        plt.ylabel('Average Relative Error')
        plt.title('FMM Accuracy vs. Direct Method')
        plt.grid(True)
        plt.xscale('log')
    '''
    plt.tight_layout()
    plt.show()


def simulate_and_animate(n_particles=100, steps=200, method='fmm', dt=0.1, theta=0.5):
    # Create particles
    particles = []
    # Create a central massive particle
    particles.append(Particle(0, 0, mass=100.0))
    
    # Create orbiting particles
    for i in range(1, n_particles):
        distance = np.random.uniform(5.0, 30.0)
        angle = np.random.uniform(0, 2 * np.pi)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        # Calculate orbital velocity for a circular orbit
        v = np.sqrt(G * particles[0].mass / distance)
        vx = -v * np.sin(angle)  # Tangential velocity
        vy = v * np.cos(angle)
        
        particles.append(Particle(x, y, mass=np.random.uniform(1.0, 2.0), vx=vx, vy=vy))
    
    # Store particle positions for static visualization
    positions_x = []
    positions_y = []
    
    # Run simulation for specified steps
    print(f"Running {method.upper()} simulation with {n_particles} particles...")
    
    # Setup visualization
    plt.figure(figsize=(10, 10))
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.grid(True)
    
    # Run simulation steps
    for step in range(steps):
        # Store current positions
        positions_x.append([p.x for p in particles])
        positions_y.append([p.y for p in particles])
        
        # Update forces
        if method == 'direct':
            compute_forces_direct(particles)
        else:
            compute_forces_fmm(particles, domain_size=100.0, theta=theta)
        
        # Update positions
        update_positions(particles, dt)
        
        # Print progress
        if step % 20 == 0:
            print(f"Step {step}/{steps} completed")
    
    # Plot particle trajectories
    for i in range(n_particles):
        # Extract this particle's trajectory
        x_traj = [positions_x[step][i] for step in range(steps)]
        y_traj = [positions_y[step][i] for step in range(steps)]
        
        # Plot with different style for central particle
        if i == 0:  # Central particle
            plt.plot(x_traj, y_traj, 'r-', linewidth=1, alpha=0.3)
            plt.scatter(x_traj[-1], y_traj[-1], s=particles[i].mass * 10, c='red', 
                      label='Central Mass')
        else:
            plt.plot(x_traj, y_traj, '-', linewidth=0.5, alpha=0.3)
            plt.scatter(x_traj[-1], y_traj[-1], s=particles[i].mass * 5, alpha=0.7)
    
    plt.title(f'2D Gravity Simulation ({method.upper()} method) - Particle Trajectories')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
 
# Real-time simulation showing particle movements frame by frame
def run_real_time_simulation(n_particles=100, method='fmm', dt=0.01, theta=0.5, max_steps=1000):

    # Create particles
    particles = []
    
    # Create a massive particle at the center
    particles.append(Particle(0, 0, mass=100.0))
    
    # Create orbiting particles
    for i in range(1, n_particles):
        distance = np.random.uniform(5.0, 30.0)
        angle = np.random.uniform(0, 2 * np.pi)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        # Calculate orbital velocity for a circular orbit
        v = np.sqrt(G * particles[0].mass / distance) 
        vx = -v * np.sin(angle)
        vy = v * np.cos(angle)
        
        particles.append(Particle(x, y, mass=np.random.uniform(1.0, 2.0), vx=vx, vy=vy))
        #particles.append(Particle(x, y, mass=np.random.uniform(1.0, 2.0), vx=0, vy=0))

    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Initial scatter plot
    masses = [p.mass * 5 for p in particles]
    colors = ['red' if i == 0 else 'blue' for i in range(n_particles)]
    scatter = ax.scatter([p.x for p in particles], [p.y for p in particles], 
                         s=masses, c=colors, alpha=0.7)
    
    # Add a title with current method
    plt.title = ax.set_title(f'2D Gravity Simulation - {method.upper()} method')
    
    # Add step counter text
    step_text = ax.text(0.02, 0.98, 'Step: 0', transform=ax.transAxes,
                        verticalalignment='top')
    
    # Trails (past positions)
    trails = [ax.plot([], [], '-', lw=0.5, alpha=0.3)[0] for _ in range(n_particles)]
    trail_history = [[] for _ in range(n_particles)]
    max_trail_length = 50  # Maximum number of points in trail
    
    # Function to initialize the animation
    def init():
        for trail in trails:
            trail.set_data([], [])
        return [scatter] + trails + [step_text]
    
    # Animation update function
    def update(frame):
        # Calculate forces
        if method == 'direct':
            compute_forces_direct(particles)
        else:
            compute_forces_fmm(particles, domain_size=100.0, theta=theta)
        
        # Update positions
        update_positions(particles, dt)
        update_positions(particles, dt)
        
        # Update scatter plot
        scatter.set_offsets(np.column_stack([[p.x for p in particles], [p.y for p in particles]]))
        
        # Update trails
        for i, p in enumerate(particles):
            trail_history[i].append((p.x, p.y))
            # Limit trail length
            if len(trail_history[i]) > max_trail_length:
                trail_history[i] = trail_history[i][-max_trail_length:]
            
            if trail_history[i]:
                x_trail, y_trail = zip(*trail_history[i])
                trails[i].set_data(x_trail, y_trail)
        
        # Update step counter
        step_text.set_text(f'Step: {frame}')
        
        return [scatter] + trails + [step_text]
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, 
                        interval=50, blit=True)
    
    plt.tight_layout()
    print(f"Running {method} simulation with {n_particles} particles. Close the plot window to stop.")
    plt.show()

def main():
    print("2D Gravity Simulation: Fast Multipole Method vs Direct N-body")
    print("\n1. Gravitational force computing performance comparison")
    print("2. Simulation result with trajectories")
    print("3. Real-time simulation animation")
    choice = input("Select an option (1/2/3): ")
    
    if choice == '1':
        # Performance comparison
        #n_particles_list = [100, 500, 1000, 5000, 10000, 20000]
        #n_particles_list = [100, 200, 500, 1000, 2000, 4000, 5000]
        n_particles_list = [1000, 2000, 5000, 7000, 10000]
        print("\nComputing in both methods...")
        results = performance_comparison(n_particles_list, 'both')
        plot_results(n_particles_list, results)
        
    elif choice == '2':
        # Run simulation with trajectories
        print("\n1. Direct N-body method")
        print("2. Fast Multipole Method (FMM)")
        sim_choice = input("Select simulation method (1/2): ")
        
        n = int(input("Number of particles [default=100]: ") or "100")
        steps = int(input("Number of simulation steps [default=200]: ") or "200")
        
        if sim_choice == '1':
            simulate_and_animate(n_particles=n, method='direct', steps=steps)
        else:
            theta = float(input("FMM accuracy parameter theta (0.1-1.0) [default=0.5]: ") or "0.5")
            simulate_and_animate(n_particles=n, method='fmm', theta=theta, steps=steps)
    
    elif choice == '3':
        # Run real-time simulation
        print("\n1. Direct N-body method")
        print("2. Fast Multipole Method (FMM)")
        sim_choice = input("Select simulation method (1/2): ")
        
        n = int(input("Number of particles [default=100]: ") or "100")
        
        if sim_choice == '1':
            run_real_time_simulation(n_particles=n, method='direct')
        else:
            theta = float(input("FMM accuracy parameter theta (0.1-1.0) [default=0.5]: ") or "0.5")
            run_real_time_simulation(n_particles=n, method='fmm', theta=theta)
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()