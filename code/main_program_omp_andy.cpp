// nbody_simulation.cpp
// Compile with:
// g++ nbody_simulation.cpp -o nbody_sim -O3 -std=c++17 -fopenmp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm> // For std::max_element, std::min_element if needed for bounds
#include <memory> // For std::unique_ptr

// OpenMP include
#ifdef _OPENMP
#include <omp.h>
#endif

// --- Physical Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;

// --- Data Structures ---
struct Particle {
    int id;
    double x, y;
    double vx, vy;
    double ax, ay;
    double mass;

    Particle(int _id, double _x, double _y, double _mass, double _vx = 0.0, double _vy = 0.0)
        : id(_id), x(_x), y(_y), mass(_mass), vx(_vx), vy(_vy), ax(0.0), ay(0.0) {}
};

class QuadTreeNode {
public:
    double cx, cy, size;
    std::vector<std::unique_ptr<QuadTreeNode>> children; // Pointers to children
    std::vector<Particle*> node_particles; // Particles in this leaf node (if it's a leaf)
    
    double total_mass;
    double com_x, com_y; // Center of mass
    
    bool is_leaf;
    bool is_empty;
    static const int MAX_PARTICLES_PER_LEAF = 1; // As per Python logic

    QuadTreeNode(double center_x, double center_y, double s)
        : cx(center_x), cy(center_y), size(s), total_mass(0.0), com_x(0.0), com_y(0.0), 
          is_leaf(true), is_empty(true) {
        children.resize(4); // NW, NE, SW, SE or 0, 1, 2, 3
    }

    void insert(Particle* p) {
        is_empty = false;

        if (is_leaf) {
            if (node_particles.size() < MAX_PARTICLES_PER_LEAF && node_particles.empty()) { // Simplified: Python logic was more like if first particle, or if still capacity
                 // Update COM incrementally before adding (or after)
                double old_total_mass = total_mass;
                total_mass += p->mass;
                if (old_total_mass == 0.0) { // First particle
                    com_x = p->x;
                    com_y = p->y;
                } else {
                    com_x = (com_x * old_total_mass + p->x * p->mass) / total_mass;
                    com_y = (com_y * old_total_mass + p->y * p->mass) / total_mass;
                }
                node_particles.push_back(p);
                return;
            } else { // Leaf is full or has particles and needs to insert another
                subdivide();
                // Re-insert existing particles from this node into children
                for (Particle* old_p : node_particles) {
                    insert_into_child(old_p);
                }
                node_particles.clear(); // Particles are now in children
                // Insert the new particle into the appropriate child
                insert_into_child(p);
            }
        } else { // Not a leaf, insert into appropriate child
            insert_into_child(p);
        }
        
        // Update COM for internal nodes based on new particle p
        double old_total_mass = total_mass;
        total_mass += p->mass;
         if (old_total_mass == 0.0 && is_empty) { // Should not happen if is_empty was set for new particle
            com_x = p->x;
            com_y = p->y;
        } else if (old_total_mass > 0.0) { // Check to avoid division by zero if it's the first overall particle in this branch
            com_x = (com_x * old_total_mass + p->x * p->mass) / total_mass;
            com_y = (com_y * old_total_mass + p->y * p->mass) / total_mass;
        } else { // This internal node was empty, now gets its first particle (indirectly)
             com_x = p->x;
             com_y = p->y;
        }
    }

    void subdivide() {
        is_leaf = false;
        double half_size = size / 2.0;
        double quarter_size = size / 4.0; // Offset for new centers

        // Children: 0: (-q, -q) NW, 1: (q, -q) NE, 2: (-q, q) SW, 3: (q, q) SE (relative to current center)
        // This matches Python: (p.x > self.cx) + 2 * (p.y > self.cy)
        // Child 0 (NW): cx - quarter_size, cy - quarter_size
        // Child 1 (NE): cx + quarter_size, cy - quarter_size
        // Child 2 (SW): cx - quarter_size, cy + quarter_size
        // Child 3 (SE): cx + quarter_size, cy + quarter_size
        children[0] = std::make_unique<QuadTreeNode>(cx - quarter_size, cy - quarter_size, half_size); // NW
        children[1] = std::make_unique<QuadTreeNode>(cx + quarter_size, cy - quarter_size, half_size); // NE
        children[2] = std::make_unique<QuadTreeNode>(cx - quarter_size, cy + quarter_size, half_size); // SW
        children[3] = std::make_unique<QuadTreeNode>(cx + quarter_size, cy + quarter_size, half_size); // SE
    }

    void insert_into_child(Particle* p) {
        int child_idx = (p->x > cx) + 2 * (p->y > cy);
        if (!children[child_idx]) { // Should not happen if subdivided properly
             std::cerr << "Error: Child not initialized before insertion." << std::endl;
             return;
        }
        children[child_idx]->insert(p);
    }
    
    // Stub for force calculation
    void compute_force(const Particle* target_p, double& force_x, double& force_y, double theta) const {
        if (is_empty || (is_leaf && !node_particles.empty() && node_particles[0] == target_p)) { // Don't act on self
            return;
        }

        double dx = com_x - target_p->x;
        double dy = com_y - target_p->y;
        double r2 = dx * dx + dy * dy;

        if (r2 < SOFT2) r2 = SOFT2;
        double r = std::sqrt(r2);

        if (is_leaf || (size / r < theta)) {
            double f_over_r = G_CONST * total_mass / (r2 * r); // G*M/r^2 * (1/r)
            force_x += f_over_r * dx;
            force_y += f_over_r * dy;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, theta);
                }
            }
        }
    }
};


// --- Force Calculation ---
void compute_forces_direct(std::vector<Particle>& particles) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        double acc_x_local = 0.0;
        double acc_y_local = 0.0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double r2 = dx * dx + dy * dy;
            if (r2 < SOFT2) {
                r2 = SOFT2;
            }
            double inv_r = 1.0 / std::sqrt(r2);
            // f = G * m_j / r^2. acc = f / m_i. Here we calculate acceleration directly.
            // Force from j on i is G * m_i * m_j / r^2. acc_i = G * m_j / r^2
            // The Python code's `f = G * particles[j].mass * inv_r * inv_r` is force magnitude if particle[i] mass is 1,
            // or part of acceleration calculation where particle[i].mass would divide it.
            // The python code computes `axi += f * dx * inv_r` where f is G * m_j / r^2, so it's directly acceleration.
            double f_factor = G_CONST * particles[j].mass * inv_r * inv_r * inv_r; // G * m_j / r^3
            acc_x_local += f_factor * dx;
            acc_y_local += f_factor * dy;
        }
        particles[i].ax = acc_x_local;
        particles[i].ay = acc_y_local;
    }
}

void compute_forces_fmm(std::vector<Particle>& particles, double theta) {
    if (particles.empty()) return;

    // 1. Determine domain boundaries
    double min_x = particles[0].x, max_x = particles[0].x;
    double min_y = particles[0].y, max_y = particles[0].y;
    for (const auto& p : particles) {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }
    double domain_size = std::max(max_x - min_x, max_y - min_y) * 1.2; // Add some padding
    domain_size = std::max(domain_size, 1.0); // Minimum size
    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    
    QuadTreeNode root(center_x, center_y, domain_size);

    // 2. Build QuadTree
    for (auto& p : particles) {
        // Check if particle is within root bounds, simplistic for now
        if (p.x >= root.cx - root.size/2 && p.x <= root.cx + root.size/2 &&
            p.y >= root.cy - root.size/2 && p.y <= root.cy + root.size/2) {
            root.insert(&p);
        } else {
            // Handle particles outside initial root bounds - expand root or error
            // For simplicity, we assume init_disc places them reasonably
        }
    }
    
    // 3. Compute forces
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        double force_x_total = 0.0;
        double force_y_total = 0.0;
        root.compute_force(&particles[i], force_x_total, force_y_total, theta);
        particles[i].ax = force_x_total; // Assuming compute_force gives acceleration if target mass is 1 (or it's G*M/r^2)
        particles[i].ay = force_y_total; // The Barnes-Hut force calc in python was: f = G * self.total_mass / r2; return f * dx / r, f * dy / r
                                      // This IS acceleration.
    }
}


// --- Integrator ---
void leapfrog_step(std::vector<Particle>& particles, double dt, const std::string& method, double theta, bool mobile_star) {
    // Kick 1 (update velocities by 0.5 * dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
    }

    // Drift (update positions by dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }

    // Force calculation
    if (method == "direct") {
        compute_forces_direct(particles);
    } else if (method == "fmm") {
        compute_forces_fmm(particles, theta);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        compute_forces_direct(particles); // Default to direct
    }

    // Kick 2 (update velocities by 0.5 * dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
    }
}

// --- Diagnostics ---
double system_energy(const std::vector<Particle>& particles) {
    double ke = 0.0;
    double pe = 0.0;

    for (const auto& p : particles) {
        ke += 0.5 * p.mass * (p.vx * p.vx + p.vy * p.vy);
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double r2 = dx * dx + dy * dy + SOFT2; // Apply softening for PE calc too
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions ---
std::vector<Particle> init_disc(int n_particles) {
    std::vector<Particle> particles;
    std::mt19937 rng(std::random_device{}()); // Mersenne Twister random number generator
    
    // Central star
    particles.emplace_back(0, 0.0, 0.0, 1000.0, 0.0, 0.0);

    std::uniform_real_distribution<double> r_dist(5.0, 30.0);
    std::uniform_real_distribution<double> ang_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> mass_dist(1.0, 2.0);

    for (int i = 1; i < n_particles; ++i) {
        double r_val = r_dist(rng);
        double ang_val = ang_dist(rng);
        double x = r_val * std::cos(ang_val);
        double y = r_val * std::sin(ang_val);
        
        double v_mag_sq = G_CONST * particles[0].mass / r_val; // v^2 = GM/r
        if (v_mag_sq < 0) v_mag_sq = 0; // Should not happen if r_val > 0
        double v_mag = std::sqrt(v_mag_sq);
        
        double vx = -v_mag * std::sin(ang_val);
        double vy = v_mag * std::cos(ang_val);
        double mass = mass_dist(rng);
        particles.emplace_back(i, x, y, mass, vx, vy);
    }
    return particles;
}

// --- Main Simulation ---
int main(int argc, char* argv[]) {
    // Default parameters
    int n_particles = 1000;
    int steps = 200;
    std::string method = "fmm";
    double theta = 0.5;
    double dt = 0.02;
    bool mobile_star = false;
    std::string traj_fname_base = "trajectories";
    std::string energy_fname_base = "energy_drift";


    // Simple command-line argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "--mobile-star") mobile_star = true;
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0] << " [-n N_PARTICLES] [-steps N_STEPS] [-method direct|fmm] [-theta THETA_VAL] [-dt DT_VAL] [--mobile-star] [-traj_out TRAJ_FNAME] [-energy_out ENERGY_FNAME]\n";
            return 1;
        }
    }
    
    std::string traj_fname = traj_fname_base + "_" + std::to_string(n_particles) + "_rev1.csv";
    std::string energy_fname = energy_fname_base + "_" + std::to_string(n_particles) + "_rev1.csv";

    std::ofstream traj_file(traj_fname);
    std::ofstream energy_file(energy_fname);

    if (!traj_file.is_open()) {
        std::cerr << "Error: Could not open trajectory file " << traj_fname << std::endl;
        return 1;
    }
    if (!energy_file.is_open()) {
        std::cerr << "Error: Could not open energy file " << energy_fname << std::endl;
        return 1;
    }

    traj_file << "step,particle_id,x,y,vx,vy\n"; // CSV Header for trajectories
    energy_file << "step,total_energy\n";   // CSV Header for energy

    std::vector<Particle> particles = init_disc(n_particles);

    // Initial force calculation for the first leapfrog step
    if (method == "direct") {
        compute_forces_direct(particles);
    } else {
        compute_forces_fmm(particles, theta);
    }
    
    std::cout << "Starting simulation: N=" << n_particles << ", Steps=" << steps 
              << ", Method=" << method << ", dt=" << dt 
              << (mobile_star ? ", Mobile Star" : ", Fixed Star") << std::endl;
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;


    for (int step = 0; step < steps; ++step) {
        // Output data for the current step BEFORE advancing
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.vx << "," << p.vy << "\n";
        }
        energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles) << "\n";

        if (step % (steps/10 > 0 ? steps/10 : 1) == 0 || step == steps -1) { // Progress update
             std::cout << "Step " << step << "/" << steps << " | Energy: " << system_energy(particles) << std::endl;
        }
        
        leapfrog_step(particles, dt, method, theta, mobile_star);
    }
    
    // Output data for the final step if needed (current loop outputs BEFORE step)
    // If you want data AFTER the last step, add one more output block here.
    // For consistency with python (which records state then steps), this is fine.

    std::cout << "Simulation finished." << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}
