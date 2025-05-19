// nbody_simulation_3d.cpp
// Compile with:
// g++ nbody_simulation_3d.cpp -o nbody_sim_3d -O3 -std=c++17 -fopenmp -Wall
#include <iostream>
#include <vector>
#include <string>
#include <cmath>     // For std::sqrt, std::cos, std::sin, M_PI (may need -D_USE_MATH_DEFINES on Windows/MSVC)
#include <random>    // For std::mt19937, std::uniform_real_distribution, std::random_device
#include <fstream>   // For std::ofstream
#include <iomanip>   // For std::fixed, std::setprecision
#include <algorithm> // For std::max, std::min, std::max({})
#include <memory>    // For std::unique_ptr, std::make_unique

// OpenMP include
#ifdef _OPENMP
#include <omp.h>
#endif

// Define M_PI if not defined (e.g., on some systems with strict C++ standard)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Physical Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;

// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z;    // 3D coordinates
    double vx, vy, vz; // 3D velocities
    double ax, ay, az; // 3D accelerations
    double mass;

    Particle(int _id, double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
        : id(_id),
          x(_x), y(_y), z(_z),
          vx(_vx), vy(_vy), vz(_vz),
          ax(0.0), ay(0.0), az(0.0),
          mass(_mass) {}
};

class OctreeNode {
public:
    double cx, cy, cz, size; // Center and size of the cubic node
    std::vector<std::unique_ptr<OctreeNode>> children;
    std::vector<Particle*> node_particles; // Particles if this is a leaf node

    double total_mass;
    double com_x, com_y, com_z; // Center of Mass

    bool is_leaf;
    bool is_empty;
    static const int MAX_PARTICLES_PER_LEAF = 1; // Subdivide if more than this many particles

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s),
          total_mass(0.0), com_x(0.0), com_y(0.0), com_z(0.0),
          is_leaf(true), is_empty(true) {
        children.resize(8); // 8 children for an octree
    }

    void subdivide() {
        is_leaf = false; // This node is no longer a leaf
        double child_size = size / 2.0;
        double offset = size / 4.0; // Offset from parent center to child center

        int child_idx_counter = 0;
        for (int i = -1; i <= 1; i += 2) {     // z_factor relative to parent center (+offset or -offset)
            for (int j = -1; j <= 1; j += 2) { // y_factor
                for (int k = -1; k <= 1; k += 2) { // x_factor
                    children[child_idx_counter++] = std::make_unique<OctreeNode>(
                        cx + k * offset, cy + j * offset, cz + i * offset, child_size);
                }
            }
        }
    }

    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1; // Bit 0 for X (+cx direction)
        if (p->y > cy) index |= 2; // Bit 1 for Y (+cy direction)
        if (p->z > cz) index |= 4; // Bit 2 for Z (+cz direction)
        return index;
    }
    
    void insert_into_child(Particle* p) {
        int child_idx = get_child_index(p);
        if (!children[child_idx]) {
             // This should ideally not happen if subdivide is called correctly.
             // If it does, it implies a logic error where subdivide wasn't called,
             // or children were not created. For robustness, one might create it here,
             // but it's better to ensure subdivide() has been called.
             std::cerr << "Error: Octree child not initialized during insert_into_child. "
                       << "This indicates a prior logic flaw in subdivision." << std::endl;
             // Attempting to create children now might be too late or hide the root cause.
             // For now, we assume subdivide() was correctly called if 'is_leaf' is false.
             return; 
        }
        children[child_idx]->insert(p);
    }

    void insert(Particle* p) {
        is_empty = false; // Mark this node (and its path from root) as not empty

        if (is_leaf) {
            if (node_particles.empty()) { // MAX_PARTICLES_PER_LEAF is 1
                node_particles.push_back(p);
                // This leaf now contains its first and only particle 'p'.
                // Its COM is p's position, and its mass is p's mass.
                total_mass = p->mass;
                com_x = p->x;
                com_y = p->y;
                com_z = p->z;
                return; // Crucial: This node's properties are set. No further update in this call.
            } else { // Leaf is full (already has 1 particle), must subdivide.
                Particle* existing_particle = node_particles[0]; // Get the particle already in this leaf.
                node_particles.clear(); // This node will no longer hold particles directly.
                
                subdivide(); // Convert this leaf to an internal node and create children.

                // Re-insert the existing particle into the appropriate new child.
                insert_into_child(existing_particle); 
                // Insert the new particle 'p' into the appropriate new child.
                insert_into_child(p);
                // Now, 'this' node is an internal node. Its COM and total_mass need to be updated based on 'p'
                // (and implicitly 'existing_particle' through the recursive calls to children).
                // The flow will continue to the common COM update section below.
            }
        } else { // Not a leaf (already an internal node).
            insert_into_child(p);
            // Particle 'p' has been passed to a child.
            // The flow will continue to the common COM update section below to update 'this' node.
        }

        // Common COM and total_mass update for 'this' node.
        // This section is reached if:
        // 1. 'this' was a leaf that just subdivided (its old COM/mass are stale).
        // 2. 'this' was already an internal node and 'p' was passed to a child.
        // In both cases, 'p' has been added to the subtree of 'this'.
        // We perform an incremental update of 'this' node's properties.
        double old_node_total_mass = this->total_mass; // Mass of this node's subtree BEFORE 'p' is accounted for AT THIS LEVEL.
                                                       // (Children's properties are already updated recursively)

        if (old_node_total_mass == 0.0) { 
            // If this node (or the part of its subtree relevant before 'p') had no mass,
            // its COM becomes p's position, and its mass becomes p's mass.
            // This case is important if 'this' node just subdivided and its 'old_node_total_mass' was from a single particle now in a child.
            // Or if 'this' internal node's children were all empty before 'p' arrived in one of them.
            this->com_x = p->x;
            this->com_y = p->y;
            this->com_z = p->z;
        } else {
            // Incrementally update COM: weighted average of old COM and p's position.
            this->com_x = (this->com_x * old_node_total_mass + p->x * p->mass) / (old_node_total_mass + p->mass);
            this->com_y = (this->com_y * old_node_total_mass + p->y * p->mass) / (old_node_total_mass + p->mass);
            this->com_z = (this->com_z * old_node_total_mass + p->z * p->mass) / (old_node_total_mass + p->mass);
        }
        this->total_mass = old_node_total_mass + p->mass; // Update total mass for this node.
    }


    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) {
            return;
        }

        if (is_leaf) {
            // If it's a leaf node, sum forces from all particles it directly contains (if any).
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) {
                    continue; // Skip self-interaction
                }
                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < SOFT2) {
                    r2 = SOFT2;
                }
                double r = std::sqrt(r2);
                if (r == 0) continue; // Should be caught by SOFT2, but as a safeguard
                double f_over_r = G_CONST * p_in_node->mass / (r2 * r); // G*m/r^3
                force_x += f_over_r * dx;
                force_y += f_over_r * dy;
                force_z += f_over_r * dz;
            }
            return; // All particles in this leaf processed
        }

        // Not a leaf, so it's an internal node.
        // Calculate distance from target particle to this node's Center of Mass (COM).
        double dx_com = com_x - target_p->x;
        double dy_com = com_y - target_p->y;
        double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;

        if (r2_com < SOFT2) { // If target_p is very close to/at the COM of this internal node
            // This case needs careful handling. Softening helps, but if r_com is effectively zero,
            // an alternative is to recurse anyway or sum children's contributions directly.
            // For now, applying softening to r2_com if it's too small.
             r2_com = SOFT2;
        }
        double r_com = std::sqrt(r2_com);
        if (r_com == 0) { // Should be caught by SOFT2, safeguard for division by zero.
             // If still zero (e.g. target_p is exactly at COM and total_mass > 0),
             // we should probably recurse to avoid infinite force.
             // For simplicity now, just return (no force from this problematic COM).
             // A better way: always recurse if r_com is too small.
             for (int i = 0; i < 8; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
            return;
        }


        if ((size / r_com) < theta) {
            // Node is sufficiently far away, approximate it as a single point mass at its COM.
            double f_over_r_com = G_CONST * total_mass / (r2_com * r_com); // G*M_node/r_com^3
            force_x += f_over_r_com * dx_com;
            force_y += f_over_r_com * dy_com;
            force_z += f_over_r_com * dz_com;
        } else {
            // Node is too close or too large relative to its distance, recurse into its children.
            for (int i = 0; i < 8; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
        }
    }
};

// --- Force Calculation ---
void compute_forces_direct(std::vector<Particle>& particles) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0; particles[i].ay = 0.0; particles[i].az = 0.0;
        double acc_x_local = 0.0, acc_y_local = 0.0, acc_z_local = 0.0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < SOFT2) {
                r2 = SOFT2;
            }
            double inv_r = 1.0 / std::sqrt(r2);
            double f_factor = G_CONST * particles[j].mass * inv_r * inv_r * inv_r; // G*m_j/r^3
            acc_x_local += f_factor * dx;
            acc_y_local += f_factor * dy;
            acc_z_local += f_factor * dz;
        }
        particles[i].ax = acc_x_local;
        particles[i].ay = acc_y_local;
        particles[i].az = acc_z_local;
    }
}

void compute_forces_octree(std::vector<Particle>& particles, double theta) {
    if (particles.empty()) return;

    double min_x = particles[0].x, max_x = particles[0].x;
    double min_y = particles[0].y, max_y = particles[0].y;
    double min_z = particles[0].z, max_z = particles[0].z;

    for (size_t i = 1; i < particles.size(); ++i) {
        const auto& p = particles[i];
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }

    double dx_domain = max_x - min_x;
    double dy_domain = max_y - min_y;
    double dz_domain = max_z - min_z;

    double domain_size = std::max({dx_domain, dy_domain, dz_domain}) * 1.2; // Initial domain guess
    domain_size = std::max(domain_size, 1.0); // Ensure a minimum practical size

    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    double center_z = (min_z + max_z) / 2.0;
    
    OctreeNode root(center_x, center_y, center_z, domain_size);

    for (auto& p : particles) {
        // Basic check if particle is within the initial root domain.
        // A more robust implementation might dynamically expand the root if particles are outside.
        if (std::abs(p.x - root.cx) <= root.size / 2.0 &&
            std::abs(p.y - root.cy) <= root.size / 2.0 &&
            std::abs(p.z - root.cz) <= root.size / 2.0) {
            root.insert(&p);
        } else {
             //std::cerr << "Warning: Particle " << p.id << " (" << p.x << "," << p.y << "," << p.z 
             //          << ") is outside initial root bounds (" << root.cx << "," << root.cy << "," << root.cz 
             //          << ", size=" << root.size << "). Skipping for Octree insertion." << std::endl;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0; particles[i].ay = 0.0; particles[i].az = 0.0;
        double force_x_total = 0.0, force_y_total = 0.0, force_z_total = 0.0;
        root.compute_force(&particles[i], force_x_total, force_y_total, force_z_total, theta);
        particles[i].ax = force_x_total;
        particles[i].ay = force_y_total;
        particles[i].az = force_z_total;
    }
}

// --- Integrator ---
void leapfrog_step(std::vector<Particle>& particles, double dt, const std::string& method, double theta, bool mobile_star) {
    // Kick 1 (update velocities by 0.5 * dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star && particles[0].mass > 0) continue; // Assuming particle 0 is the star if central_mass > 0
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    // Drift (update positions by dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star && particles[0].mass > 0) continue;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }

    // Force calculation
    if (method == "direct") {
        compute_forces_direct(particles);
    } else if (method == "fmm" || method == "octree") { // Allow "fmm" for Octree
        compute_forces_octree(particles, theta);
    } else {
        std::cerr << "Unknown method: " << method << ". Defaulting to direct." << std::endl;
        compute_forces_direct(particles);
    }

    // Kick 2 (update velocities by 0.5 * dt)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star && particles[0].mass > 0) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

// --- Diagnostics ---
double system_energy(const std::vector<Particle>& particles) {
    double ke = 0.0;
    double pe = 0.0;

    for (const auto& p : particles) {
        ke += 0.5 * p.mass * (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r2 = dx * dx + dy * dy + dz * dz + SOFT2; // Apply softening for PE calc too
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions ---
std::vector<Particle> init_3d_disc_orbiting_central_mass(
    int n_orbiting_particles,   // Number of orbiting particles
    double central_mass_val,    // Mass of the central object
    double min_radius,          // Inner radius of the disc
    double max_radius,          // Outer radius of the disc
    double disc_thickness,      // Thickness of the disc (z-extent)
    double particle_mass_min = 0.1, 
    double particle_mass_max = 1.0 
) {
    std::vector<Particle> particles_vec; // Renamed to avoid conflict with global 'particles' if any
    std::mt19937 rng(std::random_device{}()); 

    int current_id = 0;

    if (central_mass_val > 0) {
        particles_vec.emplace_back(current_id++, 0.0, 0.0, 0.0, central_mass_val, 0.0, 0.0, 0.0); // Central object at origin
    }

    std::uniform_real_distribution<double> radius_dist(min_radius, max_radius);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> z_offset_dist(-disc_thickness / 2.0, disc_thickness / 2.0);
    std::uniform_real_distribution<double> mass_dist(particle_mass_min, particle_mass_max);
    std::uniform_real_distribution<double> slight_vel_perturb_dist(-0.05, 0.05); // For small velocity perturbations

    for (int i = 0; i < n_orbiting_particles; ++i) {
        double r = radius_dist(rng);
        double angle = angle_dist(rng);
        double particle_mass = mass_dist(rng);

        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        double z = z_offset_dist(rng);

        double vx = 0.0, vy = 0.0, vz = 0.0;
        if (central_mass_val > 0 && r > 1e-5) { // Avoid division by zero or very small r
            double orbital_speed_mag = std::sqrt(G_CONST * central_mass_val / r);
            vx = -orbital_speed_mag * std::sin(angle);
            vy =  orbital_speed_mag * std::cos(angle);
            
            // Add slight perturbation to make orbits less perfect and more "belt-like"
            vx += orbital_speed_mag * slight_vel_perturb_dist(rng) * 0.1; // e.g., up to 10% of orbital speed
            vy += orbital_speed_mag * slight_vel_perturb_dist(rng) * 0.1;
        }
        vz = slight_vel_perturb_dist(rng) * 0.2; // Small random z velocity

        particles_vec.emplace_back(current_id++, x, y, z, particle_mass, vx, vy, vz);
    }

    return particles_vec;
}


// --- Main Simulation ---
int main(int argc, char* argv[]) {
    // Default parameters for an asteroid belt-like scenario
    int n_orbiting_particles = 100; // Number of orbiting N-bodies
    int steps = 200;                // Number of simulation steps
    std::string method = "fmm";     // Use "fmm" (which maps to Octree) or "direct"
    double theta = 0.5;             // Barnes-Hut opening angle parameter
    double dt = 0.02;               // Time step
    bool mobile_star = false;       // Is the central star (particle 0) mobile?
    
    // Central mass and disc parameters
    double central_mass = 1000.0;   // Mass of the central object (set to 0 for no central object)
    double disc_min_radius = 5.0;
    double disc_max_radius = 30.0;
    double disc_thickness = 1.0;    // z-height of the disc / 2

    // Output file names
    std::string traj_fname_base = "trajectories_3d_disc";
    std::string energy_fname_base = "energy_drift_3d_disc";

    // Command-line argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_orbiting_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "-cmass" && i + 1 < argc) central_mass = std::stod(argv[++i]);
        else if (arg == "--mobile-star") mobile_star = true;
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        // TODO: Add command line args for disc_min_radius, disc_max_radius, disc_thickness if desired
        else {
            std::cerr << "Usage: " << argv[0] 
                      << " [-n N_ORBITING_PARTICLES (default: " << n_orbiting_particles << ")]"
                      << " [-steps N_STEPS (default: " << steps << ")]"
                      << " [-method direct|fmm (default: " << method << ")]"
                      << " [-theta THETA_VAL (default: " << theta << ")]"
                      << " [-dt DT_VAL (default: " << dt << ")]"
                      << " [-cmass CENTRAL_MASS (default: " << central_mass << ")]"
                      << " [--mobile-star (default: " << (mobile_star ? "true" : "false") << ")]"
                      << " [-traj_out TRAJ_FNAME_BASE (default: " << traj_fname_base << ")]"
                      << " [-energy_out ENERGY_FNAME_BASE (default: " << energy_fname_base << ")]"
                      << std::endl;
            return 1;
        }
    }
    
    int total_particles_in_sim = n_orbiting_particles + (central_mass > 0 ? 1 : 0);

    std::string traj_fname = traj_fname_base + "_" + std::to_string(total_particles_in_sim) + "_rev1.csv";
    std::string energy_fname = energy_fname_base + "_" + std::to_string(total_particles_in_sim) + "_rev1.csv";

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

    traj_file << "step,particle_id,x,y,z,vx,vy,vz\n"; // CSV Header for trajectories
    energy_file << "step,total_energy\n";   // CSV Header for energy

    std::vector<Particle> particles = init_3d_disc_orbiting_central_mass(
        n_orbiting_particles, 
        central_mass, 
        disc_min_radius, 
        disc_max_radius, 
        disc_thickness
    );
    
    if (particles.empty() && total_particles_in_sim > 0) {
        std::cerr << "Error: Particle initialization resulted in zero particles when non-zero expected." << std::endl;
        return 1;
    }
     if (particles.size() != static_cast<size_t>(total_particles_in_sim) ) {
         std::cerr << "Warning: Number of initialized particles (" << particles.size() 
                   << ") does not match expected (" << total_particles_in_sim << ")." << std::endl;
         // Could be an issue if central_mass is zero but n_orbiting_particles is also zero.
         if (particles.empty()) return 1; // Exit if truly no particles to simulate.
    }


    // Initial force calculation, only if particles exist
    if (!particles.empty()) {
        if (method == "direct") {
            compute_forces_direct(particles);
        } else { // fmm or octree
            compute_forces_octree(particles, theta);
        }
    }
    
    std::cout << "Starting 3D disc simulation: Orbiting N=" << n_orbiting_particles
              << ", Central Mass=" << central_mass
              << ", Total Particles in Sim=" << particles.size()
              << ", Steps=" << steps << ", Method=" << method << ", dt=" << dt 
              << (mobile_star && central_mass > 0 ? ", Mobile Star" : (central_mass > 0 ? ", Fixed Star" : ", No Star")) 
              << std::endl;
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;


    for (int step = 0; step < steps; ++step) {
        if (particles.empty()) break; 

        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z 
                      << "," << p.vx << "," << p.vy << "," << p.vz << "\n";
        }
        energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles) << "\n";

        if (step % std::max(1, steps / 10) == 0 || step == steps - 1) { // Progress update
             std::cout << "Step " << step + 1 << "/" << steps << " | Energy: " << system_energy(particles) << std::endl;
        }
        
        leapfrog_step(particles, dt, method, theta, mobile_star);
    }
    
    std::cout << "Simulation finished." << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}