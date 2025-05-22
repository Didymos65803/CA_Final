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
#include <algorithm> // For std::max, std::min, std::sort
#include <memory>    // For std::unique_ptr, std::make_unique

// OpenMP include
#ifdef _OPENMP
#include <omp.h>
#endif

// --- Physical Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01; // Also used as collision radius for elastic collisions
const double SOFT2 = SOFTENING * SOFTENING;
const double ACCRETION_RADIUS_FACTOR = 100; // How many 'SOFTENING' lengths for accretion radius
const double ACCRETION_RADIUS = ACCRETION_RADIUS_FACTOR * SOFTENING;
const double ACCRETION_RADIUS_SQ = ACCRETION_RADIUS * ACCRETION_RADIUS;


// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z;    // 3D coordinates
    double vx, vy, vz; // 3D velocities
    double ax, ay, az; // 3D accelerations
    double mass;

    Particle(int _id, double _x, double _y, double _z, double _mass,
         double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
    : id(_id), x(_x), y(_y), z(_z),
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
    static const int MAX_PARTICLES_PER_LEAF = 1;

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s),
          total_mass(0.0), com_x(0.0), com_y(0.0), com_z(0.0),
          is_leaf(true), is_empty(true) {
        children.resize(8);
    }

    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;

        int child_idx = 0;
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                for (int k = -1; k <= 1; k += 2) {
                    children[child_idx++] = std::make_unique<OctreeNode>(
                        cx + k * offset, cy + j * offset, cz + i * offset, child_size);
                }
            }
        }
    }

    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1;
        if (p->y > cy) index |= 2;
        if (p->z > cz) index |= 4;
        return index;
    }

    void insert_into_child(Particle* p) {
        int child_idx = get_child_index(p);
        if (!children[child_idx]) {
             std::cerr << "Error: Octree child not initialized before insertion (should not happen)." << std::endl;
             subdivide(); // Fallback, though indicative of a flaw if reached
        }
        children[child_idx]->insert(p);
    }

    void insert(Particle* p) {
        is_empty = false;

        if (is_leaf) {
            if (node_particles.empty()) {
                node_particles.push_back(p);
                total_mass = p->mass;
                com_x = p->x;
                com_y = p->y;
                com_z = p->z;
                return;
            } else {
                Particle* existing_particle = node_particles[0];
                node_particles.clear();
                subdivide();
                insert_into_child(existing_particle);
                insert_into_child(p);
            }
        } else {
            insert_into_child(p);
        }

        // Incremental COM update for internal nodes
        double old_node_total_mass = total_mass;
        if (old_node_total_mass == 0.0) {
            com_x = p->x;
            com_y = p->y;
            com_z = p->z;
        } else {
            com_x = (com_x * old_node_total_mass + p->x * p->mass) / (old_node_total_mass + p->mass);
            com_y = (com_y * old_node_total_mass + p->y * p->mass) / (old_node_total_mass + p->mass);
            com_z = (com_z * old_node_total_mass + p->z * p->mass) / (old_node_total_mass + p->mass);
        }
        total_mass = old_node_total_mass + p->mass;
    }

    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) return;
        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) continue;
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
            return;
        }

        double dx_com = com_x - target_p->x;
        double dy_com = com_y - target_p->y;
        double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;

        if (r2_com < SOFT2) r2_com = SOFT2;
        double r_com = std::sqrt(r2_com);

        if (size / r_com < theta) {
            double f_over_r_com = G_CONST * total_mass / (r2_com * r_com);
            force_x += f_over_r_com * dx_com;
            force_y += f_over_r_com * dy_com;
            force_z += f_over_r_com * dz_com;
        } else {
            for (int i = 0; i < 8; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
        }
    }
};

// --- Collision and Accretion Handling ---
void handle_elastic_collisions_and_accretion(std::vector<Particle>& particles,
                                           bool particle_0_is_central_star_rules,
                                           bool mobile_star_setting) {
    if (particles.empty()) return;

    std::vector<int> accreted_indices; // Store original indices of particles to be removed

    bool use_central_star_rules_for_accretion = false;
    if (particle_0_is_central_star_rules && !particles.empty()) {
        // This implies particles[0] was initialized as the central massive object
        use_central_star_rules_for_accretion = true;
    }

    // --- Accretion Check ---
    if (use_central_star_rules_for_accretion && particles.size() > 1) {
        Particle& central_particle = particles[0]; // particles[0] is the accretor

        for (size_t i = 1; i < particles.size(); ++i) { // Check peripheral particles
            // Skip if particle 'i' is already marked for accretion by a previous iteration
            // (not strictly necessary with current single pass, but good for robustness if logic changes)
            bool already_marked = false;
            for (int acc_idx : accreted_indices) {
                if (static_cast<size_t>(acc_idx) == i) {
                    already_marked = true;
                    break;
                }
            }
            if (already_marked) continue;

            Particle& p_outer = particles[i];
            double dx = central_particle.x - p_outer.x;
            double dy = central_particle.y - p_outer.y;
            double dz = central_particle.z - p_outer.z;
            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < ACCRETION_RADIUS_SQ && r2 > 1e-9) { // r2 > epsilon to avoid issues if coincident
                accreted_indices.push_back(i);
                if (mobile_star_setting) { // If central star is mobile, conserve momentum
                    double combined_mass = central_particle.mass + p_outer.mass;
                    if (combined_mass > 1e-9) { // Avoid division by zero if masses are tiny
                        central_particle.vx = (central_particle.mass * central_particle.vx + p_outer.mass * p_outer.vx) / combined_mass;
                        central_particle.vy = (central_particle.mass * central_particle.vy + p_outer.mass * p_outer.vy) / combined_mass;
                        central_particle.vz = (central_particle.mass * central_particle.vz + p_outer.mass * p_outer.vz) / combined_mass;
                    }
                }
                central_particle.mass += p_outer.mass;
                // p_outer will be removed later.
                // std::cout << "Particle " << p_outer.id << " accreted by particle " << central_particle.id << ". New central mass: " << central_particle.mass << std::endl;
            }
        }
    }

    // Sort indices in descending order for safe removal from vector
    std::sort(accreted_indices.rbegin(), accreted_indices.rend());
    for (int original_idx : accreted_indices) {
        particles.erase(particles.begin() + original_idx);
    }
    // particles.size() is now updated.

    // --- Elastic Collisions ---
    size_t collision_loop_start_idx = 0;
    if (use_central_star_rules_for_accretion) { // If particles[0] is central star, it doesn't elastically collide
        collision_loop_start_idx = 1;
    }

    for (size_t i = collision_loop_start_idx; i < particles.size(); ++i) {
        Particle& p1 = particles[i];
        for (size_t j = i + 1; j < particles.size(); ++j) {
            Particle& p2 = particles[j];

            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double dz = p2.z - p1.z;
            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < SOFT2 && r2 > 1e-9) { // Collision if closer than SOFTENING, r2 > epsilon
                // Check if approaching (v_rel dot x_rel < 0)
                double rel_vx = p1.vx - p2.vx;
                double rel_vy = p1.vy - p2.vy;
                double rel_vz = p1.vz - p2.vz;

                double x1_minus_x2_x = p1.x - p2.x; // = -dx
                double x1_minus_x2_y = p1.y - p2.y; // = -dy
                double x1_minus_x2_z = p1.z - p2.z; // = -dz

                double dot_v_x = rel_vx * x1_minus_x2_x + rel_vy * x1_minus_x2_y + rel_vz * x1_minus_x2_z;

                if (dot_v_x < 0) { // Only collide if approaching
                    // std::cout << "Collision between particle " << p1.id << " and " << p2.id << std::endl;
                    double m1 = p1.mass;
                    double m2 = p2.mass;
                    double M_sum = m1 + m2;

                    if (M_sum < 1e-9) continue; // Avoid division by zero with tiny/zero mass particles

                    // v1' = v1 - (2m2/(m1+m2)) * <v1-v2, x1-x2> / ||x1-x2||^2 * (x1-x2)
                    double common_factor_p1 = (2 * m2 / M_sum) * dot_v_x / r2;
                    p1.vx -= common_factor_p1 * x1_minus_x2_x;
                    p1.vy -= common_factor_p1 * x1_minus_x2_y;
                    p1.vz -= common_factor_p1 * x1_minus_x2_z;

                    // v2' = v2 - (2m1/(m1+m2)) * <v2-v1, x2-x1> / ||x2-x1||^2 * (x2-x1)
                    // <v2-v1, x2-x1> = dot_v_x (same dot product as above)
                    // (x2-x1) = (dx, dy, dz)
                    double common_factor_p2 = (2 * m1 / M_sum) * dot_v_x / r2;
                    p2.vx -= common_factor_p2 * (p2.x - p1.x); // dx
                    p2.vy -= common_factor_p2 * (p2.y - p1.y); // dy
                    p2.vz -= common_factor_p2 * (p2.z - p1.z); // dz

                    // Post-collision separation to prevent sticking/overlap
                    double r = std::sqrt(r2);
                    double overlap = SOFTENING - r;
                    if (overlap > 1e-9 && r > 1e-9) { // If actual overlap and r is not zero
                        double norm_dx = dx / r;
                        double norm_dy = dy / r;
                        double norm_dz = dz / r;
                        
                        double move_dist = overlap * 0.501; // Move slightly more than half overlap each

                        p1.x -= norm_dx * move_dist;
                        p1.y -= norm_dy * move_dist;
                        p1.z -= norm_dz * move_dist;

                        p2.x += norm_dx * move_dist;
                        p2.y += norm_dy * move_dist;
                        p2.z += norm_dz * move_dist;
                    }
                }
            }
        }
    }
}


// --- Force Calculation ---
void compute_forces_direct(std::vector<Particle>& particles) {
    if (particles.empty()) return;
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
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
        if (p.z < min_z) min_z = p.z;
        if (p.z > max_z) max_z = p.z;
    }

    double dx_domain = max_x - min_x;
    double dy_domain = max_y - min_y;
    double dz_domain = max_z - min_z;

    double domain_size = std::max({dx_domain, dy_domain, dz_domain}) * 1.2; // Add padding
    domain_size = std::max(domain_size, SOFTENING * 10); // Ensure a minimum sensible size

    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    double center_z = (min_z + max_z) / 2.0;
    
    OctreeNode root(center_x, center_y, center_z, domain_size);

    for (auto& p : particles) {
        if (std::abs(p.x - center_x) <= domain_size / 2.0 + SOFTENING && // Add tolerance for particles near boundary
            std::abs(p.y - center_y) <= domain_size / 2.0 + SOFTENING &&
            std::abs(p.z - center_z) <= domain_size / 2.0 + SOFTENING) {
            root.insert(&p);
        } else {
             // Fallback: if particle is outside, compute its force directly against all others (less efficient)
             // This situation should be rare if domain_size is chosen well.
             // For simplicity now, we are skipping particles that are too far out of the main cluster
             // for octree calculation. This is a simplification. A robust FMM would expand the root.
            // std::cerr << "Warning: Particle " << p.id << " (" << p.x << "," << p.y << "," << p.z 
            //           << ") is outside root bounds (" << center_x << "," << center_y << "," << center_z << " size " << domain_size 
            //           << "). Consider increasing domain_size padding or dynamic root expansion." << std::endl;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0; particles[i].ay = 0.0; particles[i].az = 0.0;
        double force_x_total = 0.0, force_y_total = 0.0, force_z_total = 0.0;
        // Check if particle was inserted into tree, otherwise direct sum (simplification: assuming it was for now)
        root.compute_force(&particles[i], force_x_total, force_y_total, force_z_total, theta);
        particles[i].ax = force_x_total;
        particles[i].ay = force_y_total;
        particles[i].az = force_z_total;
    }
}

// --- Integrator ---
void leapfrog_step(std::vector<Particle>& particles, double dt, const std::string& method, double theta,
                   bool mobile_star, bool particle_0_is_special_central_star) {
    if (particles.empty()) return;

    // First half-kick for velocities
    for (size_t i = 0; i < particles.size(); ++i) {
        if (particle_0_is_special_central_star && i == 0 && !mobile_star) continue; // Skip fixed central star
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    // Drift (update positions)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (particle_0_is_special_central_star && i == 0 && !mobile_star) continue;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }

    // --- Handle Collisions and Accretion ---
    // This function might change particles.size() due to accretion.
    handle_elastic_collisions_and_accretion(particles, particle_0_is_special_central_star, mobile_star);

    // Re-calculate forces with (potentially) new particle set and positions
    if (particles.empty()) return; // All particles might have been accreted

    if (method == "direct") {
        compute_forces_direct(particles);
    } else if (method == "fmm" || method == "octree") {
        compute_forces_octree(particles, theta);
    } else {
        std::cerr << "Unknown method: " << method << ". Defaulting to direct." << std::endl;
        compute_forces_direct(particles);
    }

    // Second half-kick for velocities
    for (size_t i = 0; i < particles.size(); ++i) {
        if (particle_0_is_special_central_star && i == 0 && !mobile_star) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

// --- Diagnostics ---
double system_energy(const std::vector<Particle>& particles) {
    if (particles.empty()) return 0.0;
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
            double r2 = dx * dx + dy * dy + dz * dz + SOFT2; // SOFT2 added in PE to match force softening
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions ---
std::vector<Particle> init_random_cube(int n_total_particles_to_init, // Renamed for clarity
                                     double cube_half_side = 20.0,
                                     double central_obj_mass = 1000.0) { // Matched arg name
    std::vector<Particle> particles_vec; // Renamed for clarity
    std::mt19937 rng(std::random_device{}());
    
    int current_id = 0;
    if (central_obj_mass > 0) {
        // Central object (ID 0)
        particles_vec.emplace_back(current_id++, 0.0, 0.0, 0.0, central_obj_mass, 0.0, 0.0, 0.0); // Example initial V
        // The prompt had vz = -10.0 for central star, can be adjusted. Setting to 0,0,0 for simplicity here.
    }

    // Number of peripheral particles to generate
    int n_peripheral_to_generate = n_total_particles_to_init - particles_vec.size();

    std::uniform_real_distribution<double> pos_dist(-cube_half_side, cube_half_side);
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);
    std::uniform_real_distribution<double> vel_dist(-0.1, 0.1); 

    for (int i = 0; i < n_peripheral_to_generate; ++i) {
        double x = pos_dist(rng);
        double y = pos_dist(rng);
        double z = pos_dist(rng);
        // Ensure not too close to origin if central mass exists
        if (central_obj_mass > 0) {
            while (x*x + y*y + z*z < (ACCRETION_RADIUS * 1.5)*(ACCRETION_RADIUS * 1.5) ) { // Avoid immediate accretion
                x = pos_dist(rng); y = pos_dist(rng); z = pos_dist(rng);
            }
        }
        double mass = mass_dist(rng);
        double vx = vel_dist(rng); 
        double vy = vel_dist(rng);
        double vz = vel_dist(rng);
        particles_vec.emplace_back(current_id++, x, y, z, mass, vx, vy, vz);
    }
    return particles_vec;
}


// --- Main Simulation ---
int main(int argc, char* argv[]) {
    int n_peripheral_particles = 100; // Number of particles *excluding* the central one if it exists
    int steps = 200;
    std::string method = "fmm";
    double theta = 0.5;
    double dt = 0.02;
    bool mobile_star_cli = false; // Is the central star (if any) mobile?
    std::string traj_fname_base = "trajectories_3d";
    std::string energy_fname_base = "energy_drift_3d";
    double init_cube_half_side = 30.0;
    double central_mass_cli = 1000; // Mass of the central object. 0 means no central object.

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_peripheral_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "--mobile-star") mobile_star_cli = true;
        else if (arg == "-central_mass" && i + 1 < argc) central_mass_cli = std::stod(argv[++i]);
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0] << " [-n N_PERIPHERAL_PARTICLES] [-steps N_STEPS] [-method direct|fmm] \n"
                      << "       [-theta THETA_FMM] [-dt DT] [--mobile-star] [-central_mass MASS] \n"
                      << "       [-traj_out TRAJ_FNAME_BASE] [-energy_out ENERGY_FNAME_BASE]\n";
            return 1;
        }
    }
    
    int total_initial_particles = n_peripheral_particles + (central_mass_cli > 0 ? 1 : 0);
    bool particle_0_is_special_central_star = (central_mass_cli > 0.0);

    std::string suffix = "_" + std::to_string(total_initial_particles) + "_coll_accr.csv";
    std::string traj_fname = traj_fname_base + suffix;
    std::string energy_fname = energy_fname_base + suffix;

    std::ofstream traj_file(traj_fname);
    std::ofstream energy_file(energy_fname);

    if (!traj_file.is_open() || !energy_file.is_open()) {
        std::cerr << "Error opening output files." << std::endl;
        return 1;
    }

    traj_file << "step,particle_id,x,y,z,vx,vy,vz,mass\n"; // Added mass
    energy_file << "step,total_energy,num_particles\n"; // Added num_particles

    std::vector<Particle> particles = init_random_cube(total_initial_particles, init_cube_half_side, central_mass_cli);
    
    if (particles.empty() && total_initial_particles > 0) {
        std::cerr << "Error: Particle initialization failed or resulted in zero particles when non-zero expected." << std::endl;
        return 1;
    }
    if (particles.size() != static_cast<size_t>(total_initial_particles) ) {
         std::cerr << "Warning: Number of initialized particles (" << particles.size() 
                   << ") does not match expected (" << total_initial_particles << ")." << std::endl;
    }


    // Initial force calculation
    if (!particles.empty()) {
        if (method == "direct") {
            compute_forces_direct(particles);
        } else {
            compute_forces_octree(particles, theta);
        }
    }
    
    std::cout << "Starting 3D simulation: Initial N=" << total_initial_particles
              << " (Peripheral: " << n_peripheral_particles << ")"
              << ", Steps=" << steps << ", Method=" << method << ", dt=" << dt
              << (particle_0_is_special_central_star ? (mobile_star_cli ? ", Mobile Star" : ", Fixed Star") : ", No Central Star")
              << std::endl;
    if (particle_0_is_special_central_star) {
        std::cout << "Central Star Mass: " << central_mass_cli 
                  << ", Accretion Radius: " << ACCRETION_RADIUS << std::endl;
    }
    std::cout << "Collision Radius (SOFTENING): " << SOFTENING << std::endl;
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;

    for (int step = 0; step < steps; ++step) {
        if (particles.empty()) {
            std::cout << "Step " << step << "/" << steps << " | All particles accreted or simulation empty. Ending." << std::endl;
            break;
        }
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << ","
                      << p.x << "," << p.y << "," << p.z << ","
                      << p.vx << "," << p.vy << "," << p.vz << ","
                      << p.mass << "\n";
        }
        double current_energy = system_energy(particles);
        energy_file << step << "," << std::fixed << std::setprecision(8) << current_energy 
                    << "," << particles.size() << "\n";

        if (step % std::max(1, steps / 10) == 0 || step == steps - 1) {
             std::cout << "Step " << step << "/" << steps 
                       << " | Particles: " << particles.size()
                       << " | Energy: " << current_energy << std::endl;
        }
        
        leapfrog_step(particles, dt, method, theta, mobile_star_cli, particle_0_is_special_central_star);
    }
    
    std::cout << "Simulation finished. Final particle count: " << particles.size() << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}