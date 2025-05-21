// nbody_simulation_3d.cpp
// Compile with:
// g++ nbody_simulation_3d.cpp -o nbody_sim_3d -O3 -std=c++17 -fopenmp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm> // For std::max, std::min
#include <memory>    // For std::unique_ptr, std::make_unique

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
    double x, y, z;    // 3D coordinates
    double vx, vy, vz; // 3D velocities
    double ax, ay, az; // 3D accelerations
    double mass;

    Particle(int _id, double _x, double _y, double _z, double _mass,
         double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
    : id(_id), x(_x), y(_y), z(_z),                       //  1-4
      vx(_vx), vy(_vy), vz(_vz),                         // 5-7
      ax(0.0), ay(0.0), az(0.0),                       // 8-10 
      mass(_mass) {}                                    // 11
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
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0; // Offset from parent center to child center

        int child_idx = 0;
        for (int i = -1; i <= 1; i += 2) {     // z_factor (-1, 1)
            for (int j = -1; j <= 1; j += 2) { // y_factor (-1, 1)
                for (int k = -1; k <= 1; k += 2) { // x_factor (-1, 1)
                    children[child_idx++] = std::make_unique<OctreeNode>(
                        cx + k * offset, cy + j * offset, cz + i * offset, child_size);
                }
            }
        }
    }

    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1; // Right half (X-axis)
        if (p->y > cy) index |= 2; // Top half   (Y-axis)
        if (p->z > cz) index |= 4; // Front half (Z-axis) - adjust convention as needed
        return index;
    }
    
    void insert_into_child(Particle* p) {
        int child_idx = get_child_index(p);
        if (!children[child_idx]) { // Should be pre-initialized by subdivide
             std::cerr << "Error: Octree child not initialized before insertion." << std::endl;
             // This might happen if subdivide wasn't called when it should have been.
             // As a fallback, create it, though this indicates a logic flaw elsewhere.
             subdivide(); // Should not be needed if logic is correct
        }
        children[child_idx]->insert(p);
    }

    void insert(Particle* p) {
        is_empty = false;

        if (is_leaf) {
            if (node_particles.empty()) { // MAX_PARTICLES_PER_LEAF is 1
                node_particles.push_back(p);
                // This leaf now contains particle p, update its COM and mass directly
                total_mass = p->mass;
                com_x = p->x;
                com_y = p->y;
                com_z = p->z;
                return; // Crucial: COM for this leaf is set, no further updates needed in this call
            } else { // Leaf is full (has 1 particle), must subdivide
                // Current particle(s) in node_particles need to be re-inserted
                Particle* existing_particle = node_particles[0]; // Since MAX_PARTICLES_PER_LEAF = 1
                node_particles.clear(); // Clear before subdividing and re-inserting

                subdivide(); // Now 'this' is an internal node

                insert_into_child(existing_particle); // Re-insert the old particle
                insert_into_child(p);                 // Insert the new particle
                // Fall through to update this (now internal) node's COM based on p
                // (and existing_particle, which was handled by its child's recursive insert)
            }
        } else { // Not a leaf (already an internal node)
            insert_into_child(p);
            // Fall through to update this internal node's COM based on p
        }

        // General COM update for 'this' node (internal, or leaf that just became internal)
        // This part is reached if:
        // - 'this' was a leaf that just subdivided. Its total_mass & COM are now stale (were from its single particle).
        // - 'this' was already an internal node. Its total_mass & COM are from particles previously in its subtree.
        // We are now accounting for particle 'p' being added to the subtree of 'this'.
        double old_node_total_mass = total_mass;
        // If the node was just subdivided, its 'total_mass' is that of the single particle it held.
        // If it was an internal node, 'total_mass' is the sum of masses in its children *before* p.
        // The recursive calls to insert_into_child will update children's masses.
        // For the incremental update of *this* node, we consider the effect of particle 'p'.
        
        // A robust way to update internal node COM/Mass after children have processed 'p':
        // Re-calculate from children (after all insertions into children are done for 'p' and any 'old_p's)
        // total_mass = 0; com_x = 0; com_y = 0; com_z = 0;
        // for(const auto& child : children) {
        //     if(child && !child->is_empty) {
        //         total_mass += child->total_mass;
        //         com_x += child->com_x * child->total_mass;
        //         com_y += child->com_y * child->total_mass;
        //         com_z += child->com_z * child->total_mass;
        //     }
        // }
        // if (total_mass > 0) { com_x /= total_mass; com_y /= total_mass; com_z /= total_mass; }
        // This re-calculation is more robust but more expensive than perfect incremental.

        // Sticking to incremental update similar to 2D version:
        // 'total_mass' here is the mass of this node's subtree *before* p's mass is added AT THIS LEVEL.
        // (children's masses have already been updated to include p if p went into them)
        // So, we just add p's mass to this node's total_mass and update COM.
        if (old_node_total_mass == 0.0) { // If this node's subtree was effectively empty mass-wise before p
            com_x = p->x;
            com_y = p->y;
            com_z = p->z;
        } else { // Node already had mass, incrementally update COM by adding p
            // (com_x * old_node_total_mass) is the moment before p
            // (p->x * p->mass) is p's moment
            // (total_mass) here is the new total_mass after adding p
            com_x = (com_x * old_node_total_mass + p->x * p->mass) / (old_node_total_mass + p->mass);
            com_y = (com_y * old_node_total_mass + p->y * p->mass) / (old_node_total_mass + p->mass);
            com_z = (com_z * old_node_total_mass + p->z * p->mass) / (old_node_total_mass + p->mass);
        }
        total_mass = old_node_total_mass + p->mass; // Update total mass *after* using old_node_total_mass in COM calculation
    }


    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) return;
        if (is_leaf) { // If leaf, contains actual particles (or is empty but not "is_empty" overall if it HAD particles)
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) continue; // Skip self interaction

                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < SOFT2) r2 = SOFT2;
                double r = std::sqrt(r2);
                double f_over_r = G_CONST * p_in_node->mass / (r2 * r);
                force_x += f_over_r * dx;
                force_y += f_over_r * dy;
                force_z += f_over_r * dz;
            }
            return; // Processed all particles in this leaf
        }

        // Not a leaf, so it's an internal node or an empty subdivided node
        double dx_com = com_x - target_p->x;
        double dy_com = com_y - target_p->y;
        double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;

        if (r2_com < SOFT2) r2_com = SOFT2; // Soften distance to COM as well
        double r_com = std::sqrt(r2_com);

        if (size / r_com < theta) { // Barnes-Hut criterion: use COM approximation
            double f_over_r_com = G_CONST * total_mass / (r2_com * r_com);
            force_x += f_over_r_com * dx_com;
            force_y += f_over_r_com * dy_com;
            force_z += f_over_r_com * dz_com;
        } else { // Node is too close or too large, recurse into children
            for (int i = 0; i < 8; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
        }
    }
};

// --- Elastic Collision ---


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
            double f_factor = G_CONST * particles[j].mass * inv_r * inv_r * inv_r;
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
    if (p.x < min_x) { min_x = p.x; }
    if (p.x > max_x) { max_x = p.x; }
    if (p.y < min_y) { min_y = p.y; }
    if (p.y > max_y) { max_y = p.y; }
    if (p.z < min_z) { min_z = p.z; }
    if (p.z > max_z) { max_z = p.z; }
}

    double dx_domain = max_x - min_x;
    double dy_domain = max_y - min_y;
    double dz_domain = max_z - min_z;

    double domain_size = std::max({dx_domain, dy_domain, dz_domain}) * 1.2;
    domain_size = std::max(domain_size, 1.0); // Ensure a minimum size

    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    double center_z = (min_z + max_z) / 2.0;
    
    OctreeNode root(center_x, center_y, center_z, domain_size);

    for (auto& p : particles) {
         // A simple check; robust applications might need to expand the root or handle outliers.
        if (std::abs(p.x - center_x) <= domain_size / 2.0 &&
            std::abs(p.y - center_y) <= domain_size / 2.0 &&
            std::abs(p.z - center_z) <= domain_size / 2.0) {
            root.insert(&p);
        } else {
            // std::cerr << "Warning: Particle " << p.id << " is outside initial root bounds. Skipping for FMM." << std::endl;
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
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }

    if (method == "direct") {
        compute_forces_direct(particles);
    } else if (method == "fmm" || method == "octree") { // Allow "fmm" for backward compatibility
        compute_forces_octree(particles, theta);
    } else {
        std::cerr << "Unknown method: " << method << ". Defaulting to direct." << std::endl;
        compute_forces_direct(particles);
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && !mobile_star) continue;
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
            double r2 = dx * dx + dy * dy + dz * dz + SOFT2;
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions ---
std::vector<Particle> init_random_cube(int n_particles, double cube_half_side = 20.0, double central_mass = 1000.0) {
    std::vector<Particle> particles;
    std::mt19937 rng(std::random_device{}());
    
    if (central_mass > 0) {
        particles.emplace_back(0, 0.0, 0.0, 0.0, central_mass, 0.0, -10.0, 0.0); // Central object
    }

    std::uniform_real_distribution<double> pos_dist(-cube_half_side, cube_half_side);
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);
    // Initial velocities can be zero or some random distribution, or based on orbit around central mass if applicable
    std::uniform_real_distribution<double> vel_dist(-0.1, 0.1); 


    int start_id = particles.size(); // If central mass was added, IDs start from 1
    for (int i = 0; i < n_particles - start_id; ++i) {
        double x = pos_dist(rng);
        double y = pos_dist(rng);
        double z = pos_dist(rng);
        double mass = mass_dist(rng);
        // For simplicity, small random initial velocities. 
        // A more physical setup might involve a virial equilibrium or specific orbital parameters.
        double vx = vel_dist(rng); 
        double vy = vel_dist(rng);
        double vz = vel_dist(rng);
        particles.emplace_back(start_id + i, x, y, z, mass, vx, vy, vz);
    }
    return particles;
}


// --- Main Simulation ---
int main(int argc, char* argv[]) {
    int n_actual_particles = 100; // This is the N for non-central particles
    int steps = 200;
    std::string method = "fmm"; // "fmm" will now map to octree
    double theta = 0.5;
    double dt = 0.02;
    bool mobile_star = false; // Assuming the central object (if any) is particle 0
    std::string traj_fname_base = "trajectories_3d";
    std::string energy_fname_base = "energy_drift_3d";
    double init_cube_half_side = 30.0;
    double central_mass = 0.0; // Set to 0 for no central mass

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_actual_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "--mobile-star") mobile_star = true;
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0] << " [-n N_PARTICLES] [-steps N_STEPS] [-method direct|fmm] ...\n";
            return 1;
        }
    }
    
    int total_particles_in_sim = n_actual_particles + (central_mass > 0 ? 1 : 0);

    std::string traj_fname = traj_fname_base + "_" + std::to_string(total_particles_in_sim) + "_rev1.csv";
    std::string energy_fname = energy_fname_base + "_" + std::to_string(total_particles_in_sim) + "_rev1.csv";

    std::ofstream traj_file(traj_fname);
    std::ofstream energy_file(energy_fname);

    if (!traj_file.is_open() || !energy_file.is_open()) {
        std::cerr << "Error opening output files." << std::endl;
        return 1;
    }

    traj_file << "step,particle_id,x,y,z,vx,vy,vz\n"; // Added z, vz
    energy_file << "step,total_energy\n";

    std::vector<Particle> particles = init_random_cube(total_particles_in_sim, init_cube_half_side, central_mass);
    if (particles.empty() && total_particles_in_sim > 0) { // Basic check
        std::cerr << "Error: Particle initialization failed or resulted in zero particles when non-zero expected." << std::endl;
        return 1;
    }
    if (particles.size() != static_cast<size_t>(total_particles_in_sim) ) {
         std::cerr << "Warning: Number of initialized particles (" << particles.size() 
                   << ") does not match expected (" << total_particles_in_sim << ")." << std::endl;
    }


    // Initial force calculation
    if (method == "direct") {
        compute_forces_direct(particles);
    } else {
        compute_forces_octree(particles, theta);
    }
    
    std::cout << "Starting 3D simulation: N=" << total_particles_in_sim 
              << " (Actual moving: " << n_actual_particles << ")"
              << ", Steps=" << steps << ", Method=" << method << ", dt=" << dt 
              << (mobile_star && central_mass > 0 ? ", Mobile Star" : (central_mass > 0 ? ", Fixed Star" : ", No Star")) 
              << std::endl;
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;

    for (int step = 0; step < steps; ++step) {
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z 
                      << "," << p.vx << "," << p.vy << "," << p.vz << "\n";
        }
        energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles) << "\n";

        if (step % std::max(1, steps / 10) == 0 || step == steps - 1) {
             std::cout << "Step " << step << "/" << steps << " | Energy: " << system_energy(particles) << std::endl;
        }
        
        leapfrog_step(particles, dt, method, theta, mobile_star);
    }
    
    std::cout << "Simulation finished." << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}