// nbody_simulation_3d.cpp
// Compile with:
// g++ nbody_simulation_3d.cpp -o nbody_sim_3d_disc -O3 -std=c++17 -fopenmp -Wall
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
const double SOFTENING = 0.01; // Also used as collision radius for elastic collisions
const double SOFT2 = SOFTENING * SOFTENING;
const double ACCRETION_RADIUS_FACTOR = 100.0; // From your working version
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
        : id(_id),
          x(_x), y(_y), z(_z),
          vx(_vx), vy(_vy), vz(_vz),
          ax(0.0), ay(0.0), az(0.0),
          mass(_mass) {}
};

class OctreeNode {
public:
    double cx, cy, cz, size; 
    std::vector<std::unique_ptr<OctreeNode> > children;
    std::vector<Particle*> node_particles; 

    double total_mass;
    double com_x, com_y, com_z; 

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

        int child_idx_counter = 0;
        for (int i = -1; i <= 1; i += 2) {     
            for (int j = -1; j <= 1; j += 2) { 
                for (int k = -1; k <= 1; k += 2) { 
                    children[child_idx_counter++] = std::make_unique<OctreeNode>(
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
             std::cerr << "Error: Octree child not initialized during insert_into_child. "
                       << "This indicates a prior logic flaw in subdivision." << std::endl;
             // Attempting to recover by subdividing, though this points to a logic issue.
             // If this node isn't a leaf, it should have children from a previous subdivide.
             if (is_leaf) subdivide(); // Should not be a leaf if we are in insert_into_child from an internal node.
                                       // If it *is* a leaf here, it means subdivide wasn't called when it should've been.
             if (!children[child_idx]) { // If still no child, something is very wrong.
                std::cerr << "Critical Error: Failed to ensure child exists for insertion." << std::endl;
                return;
             }
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
        
        // Common COM and total_mass update from your working code
        double old_node_total_mass = this->total_mass; 
        if (old_node_total_mass == 0.0) { 
            this->com_x = p->x;
            this->com_y = p->y;
            this->com_z = p->z;
        } else {
            this->com_x = (this->com_x * old_node_total_mass + p->x * p->mass) / (old_node_total_mass + p->mass);
            this->com_y = (this->com_y * old_node_total_mass + p->y * p->mass) / (old_node_total_mass + p->mass);
            this->com_z = (this->com_z * old_node_total_mass + p->z * p->mass) / (old_node_total_mass + p->mass);
        }
        this->total_mass = old_node_total_mass + p->mass; 
    }

    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) {
            return;
        }

        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) {
                    continue; 
                }
                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < SOFT2) {
                    r2 = SOFT2;
                }
                double r = std::sqrt(r2);
                if (r < 1e-9) continue; // Avoid division by zero if somehow r is still 0
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

        if (r2_com < SOFT2) {
             r2_com = SOFT2;
        }
        double r_com = std::sqrt(r2_com);
        
        if (r_com < 1e-9) { // Target is at the COM of a non-leaf node
             // Must recurse to get individual particle contributions.
            for (int i = 0; i < 8; ++i) {
                if (children[i] && !children[i]->is_empty) {
                    children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
            return;
        }

        if ((size / r_com) < theta) {
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


// --- Collision and Accretion Handling (from your working version) ---
void handle_elastic_collisions_and_accretion(std::vector<Particle>& particles,
                                           bool particle_0_is_central_star_rules,
                                           bool mobile_star_setting) {
    if (particles.empty()) return;

    std::vector<int> accreted_indices; 

    bool use_central_star_rules_for_accretion = false;
    if (particle_0_is_central_star_rules && !particles.empty()) {
        use_central_star_rules_for_accretion = true;
    }

    if (use_central_star_rules_for_accretion && particles.size() > 1) {
        Particle& central_particle = particles[0]; 

        for (size_t i = 1; i < particles.size(); ++i) { 
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

            if (r2 < ACCRETION_RADIUS_SQ && r2 > 1e-9) { 
                accreted_indices.push_back(i);
                if (mobile_star_setting) { 
                    double combined_mass = central_particle.mass + p_outer.mass;
                    if (combined_mass > 1e-9) { 
                        central_particle.vx = (central_particle.mass * central_particle.vx + p_outer.mass * p_outer.vx) / combined_mass;
                        central_particle.vy = (central_particle.mass * central_particle.vy + p_outer.mass * p_outer.vy) / combined_mass;
                        central_particle.vz = (central_particle.mass * central_particle.vz + p_outer.mass * p_outer.vz) / combined_mass;
                    }
                }
                central_particle.mass += p_outer.mass;
            }
        }
    }

    std::sort(accreted_indices.rbegin(), accreted_indices.rend());
    for (int original_idx : accreted_indices) {
        particles.erase(particles.begin() + original_idx);
    }

    size_t collision_loop_start_idx = 0;
    if (use_central_star_rules_for_accretion) { 
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

            if (r2 < SOFT2 && r2 > 1e-9) { 
                double rel_vx = p1.vx - p2.vx;
                double rel_vy = p1.vy - p2.vy;
                double rel_vz = p1.vz - p2.vz;

                double x1_minus_x2_x = -dx; 
                double x1_minus_x2_y = -dy; 
                double x1_minus_x2_z = -dz; 

                double dot_v_x = rel_vx * x1_minus_x2_x + rel_vy * x1_minus_x2_y + rel_vz * x1_minus_x2_z;

                if (dot_v_x < 0) { 
                    double m1 = p1.mass;
                    double m2 = p2.mass;
                    double M_sum = m1 + m2;

                    if (M_sum < 1e-9) continue; 

                    double common_factor_p1 = (2 * m2 / M_sum) * dot_v_x / r2;
                    p1.vx -= common_factor_p1 * x1_minus_x2_x;
                    p1.vy -= common_factor_p1 * x1_minus_x2_y;
                    p1.vz -= common_factor_p1 * x1_minus_x2_z;
                    
                    double common_factor_p2 = (2 * m1 / M_sum) * dot_v_x / r2; 
                    p2.vx -= common_factor_p2 * (p2.x - p1.x); 
                    p2.vy -= common_factor_p2 * (p2.y - p1.y); 
                    p2.vz -= common_factor_p2 * (p2.z - p1.z); 

                    double r_val = std::sqrt(r2); // Use r_val to avoid re-calculating sqrt
                    double overlap = SOFTENING - r_val;
                    if (overlap > 1e-9 && r_val > 1e-9) { 
                        double norm_dx = dx / r_val;
                        double norm_dy = dy / r_val;
                        double norm_dz = dz / r_val;
                        
                        double move_dist = overlap * 0.501; 

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
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }

    double dx_domain = max_x - min_x;
    double dy_domain = max_y - min_y;
    double dz_domain = max_z - min_z;

    double domain_size = std::max({dx_domain, dy_domain, dz_domain, SOFTENING * 10.0}) * 1.2; 
    domain_size = std::max(domain_size, 1.0); 

    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    double center_z = (min_z + max_z) / 2.0;
    
    OctreeNode root(center_x, center_y, center_z, domain_size);

    for (auto& p : particles) {
        if (std::abs(p.x - root.cx) <= root.size / 2.0 + SOFTENING && // Add tolerance
            std::abs(p.y - root.cy) <= root.size / 2.0 + SOFTENING &&
            std::abs(p.z - root.cz) <= root.size / 2.0 + SOFTENING) {
            root.insert(&p);
        } else {
            // std::cerr << "Warning: Particle " << p.id << " outside Octree root. Check domain sizing." << std::endl;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0; particles[i].ay = 0.0; particles[i].az = 0.0;
        double force_x_total = 0.0, force_y_total = 0.0, force_z_total = 0.0;
        // Only compute force if particle was likely inserted (is within domain).
        // This is a simplification; ideally, all particles should have forces computed.
        if (std::abs(particles[i].x - root.cx) <= root.size / 2.0 + SOFTENING &&
            std::abs(particles[i].y - root.cy) <= root.size / 2.0 + SOFTENING &&
            std::abs(particles[i].z - root.cz) <= root.size / 2.0 + SOFTENING) {
            root.compute_force(&particles[i], force_x_total, force_y_total, force_z_total, theta);
        } else {
            // Fallback for particles outside the tree: direct summation (expensive, not fully implemented here for brevity)
            // For now, particles far outside the tree might get zero Octree force.
            // This indicates the tree's domain might need to be larger or adaptive.
        }
        particles[i].ax = force_x_total;
        particles[i].ay = force_y_total;
        particles[i].az = force_z_total;
    }
}

// --- Integrator ---
void leapfrog_step(std::vector<Particle>& particles, double dt, const std::string& method, double theta, 
                   bool mobile_star, bool particle_0_is_special_central_star, int current_step) {
    if (particles.empty()) return;

    for (size_t i = 0; i < particles.size(); ++i) {
        if (particle_0_is_special_central_star && i == 0 && !mobile_star) continue; 
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        if (particle_0_is_special_central_star && i == 0 && !mobile_star) continue;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
    
    // Pass current_step if your handle_elastic_collisions_and_accretion uses it for debugging
    handle_elastic_collisions_and_accretion(particles, particle_0_is_special_central_star, mobile_star);


    if (particles.empty()) return; 

    if (method == "direct") {
        compute_forces_direct(particles);
    } else if (method == "fmm" || method == "octree") { 
        compute_forces_octree(particles, theta);
    } else {
        std::cerr << "Unknown method: " << method << ". Defaulting to direct." << std::endl;
        compute_forces_direct(particles);
    }

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
            double r2 = dx * dx + dy * dy + dz * dz + SOFT2; 
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions ---
std::vector<Particle> init_3d_disc_orbiting_central_mass(
    int n_orbiting_particles,   
    double central_mass_val,    
    double min_radius,          
    double max_radius,          
    double disc_thickness,      
    double particle_mass_min = 0.1, 
    double particle_mass_max = 1.0 
) {
    std::vector<Particle> particles_vec; 
    std::mt19937 rng(std::random_device{}()); 

    int current_id = 0;

    if (central_mass_val > 0) {
        particles_vec.emplace_back(current_id++, 0.0, 0.0, 0.0, central_mass_val, 0.0, 0.0, 0.0); 
    }

    std::uniform_real_distribution<double> radius_dist(min_radius, max_radius);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> z_offset_dist(-disc_thickness / 2.0, disc_thickness / 2.0);
    std::uniform_real_distribution<double> mass_dist(particle_mass_min, particle_mass_max);
    std::uniform_real_distribution<double> slight_vel_perturb_dist(-0.05, 0.05); 

    for (int i = 0; i < n_orbiting_particles; ++i) {
        double r = radius_dist(rng);
        // Ensure r is not too small, especially if min_radius can be very close to 0
        r = std::max(r, SOFTENING * 2.0); // Ensure particles start outside basic softening/collision distance
        // Also ensure particles start outside the accretion radius if it's larger
        if (central_mass_val > 0) {
             r = std::max(r, ACCRETION_RADIUS * 1.1); // Start just outside accretion zone
        }


        double angle = angle_dist(rng);
        double particle_mass = mass_dist(rng);

        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        double z = z_offset_dist(rng);

        double vx = 0.0, vy = 0.0, vz = 0.0;
        if (central_mass_val > 0 && r > 1e-5) { 
            double orbital_speed_mag = std::sqrt(G_CONST * (central_mass_val + particle_mass) / r); // Use M+m for stability
            vx = -orbital_speed_mag * std::sin(angle);
            vy =  orbital_speed_mag * std::cos(angle);
            
            vx += orbital_speed_mag * slight_vel_perturb_dist(rng) * 0.1; 
            vy += orbital_speed_mag * slight_vel_perturb_dist(rng) * 0.1;
        }
        vz = slight_vel_perturb_dist(rng) * 0.2 * std::sqrt(G_CONST * central_mass_val / max_radius) ; // Scale z-velocity perturbations with typical orbital speeds

        particles_vec.emplace_back(current_id++, x, y, z, particle_mass, vx, vy, vz);
    }

    return particles_vec;
}


// --- Main Simulation ---
int main(int argc, char* argv[]) {
    int n_orbiting_particles = 10000; 
    int steps = 200; // Increased steps for disc evolution               
    std::string method = "fmm";     
    double theta = 0.5;             
    double dt = 0.01; // Potentially smaller dt for disc stability              
    bool mobile_star = false;       
    
    double central_mass = 1000.0;   
    double disc_min_radius = 5.0;   // Make sure this is > ACCRETION_RADIUS
    double disc_max_radius = 20.0;  // Reduced max_radius for denser disc initially
    double disc_thickness = 0.5;    // Thinner disc initially

    std::string traj_fname_base = "trajectories_3d_disc";
    std::string energy_fname_base = "energy_drift_3d_disc";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_orbiting_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "-cmass" && i + 1 < argc) central_mass = std::stod(argv[++i]); // Use -cmass
        else if (arg == "--mobile-star") mobile_star = true;
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        else if (arg == "-min_r" && i + 1 < argc) disc_min_radius = std::stod(argv[++i]);
        else if (arg == "-max_r" && i + 1 < argc) disc_max_radius = std::stod(argv[++i]);
        else if (arg == "-thick_z" && i + 1 < argc) disc_thickness = std::stod(argv[++i]);
        else {
            std::cerr << "Usage: " << argv[0] 
                      << " [-n N_ORBITING (def: " << n_orbiting_particles << ")]"
                      << " [-steps N_STEPS (def: " << steps << ")]"
                      << " [-method direct|fmm (def: " << method << ")]"
                      << " [-dt DT (def: " << dt << ")]"
                      << " [-cmass CM (def: " << central_mass << ")]"
                      << " [--mobile-star (def: " << (mobile_star ? "true" : "false") << ")]"
                      << " [-min_r MIN_R (def: " << disc_min_radius << ")]"
                      << " [-max_r MAX_R (def: " << disc_max_radius << ")]"
                      << " [-thick_z THICK (def: " << disc_thickness << ")]"
                      << std::endl;
            return 1;
        }
    }
    
    // Ensure min_radius is outside accretion radius
    if (central_mass > 0 && disc_min_radius <= ACCRETION_RADIUS) {
        std::cout << "Warning: disc_min_radius (" << disc_min_radius 
                  << ") is too close or inside ACCRETION_RADIUS (" << ACCRETION_RADIUS 
                  << "). Adjusting disc_min_radius to " << ACCRETION_RADIUS * 1.2 << std::endl;
        disc_min_radius = ACCRETION_RADIUS * 1.2;
        if (disc_min_radius >= disc_max_radius) {
            disc_max_radius = disc_min_radius * 1.5; // Ensure max_r > min_r
             std::cout << "Adjusting disc_max_radius to " << disc_max_radius << std::endl;
        }
    }

    int total_particles_in_sim = n_orbiting_particles + (central_mass > 0 ? 1 : 0);
    bool particle_0_is_special_central_star = (central_mass > 0.0);


    std::string suffix = "_" + std::to_string(total_particles_in_sim)  + ".csv";
    std::string traj_fname = traj_fname_base + suffix;
    std::string energy_fname = energy_fname_base + suffix;

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

    traj_file << "step,particle_id,x,y,z,vx,vy,vz,mass\n"; // Added mass
    energy_file << "step,total_energy,num_particles\n";   // Added num_particles

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
         if (particles.empty()) return 1;
    }

    if (!particles.empty()) {
        if (method == "direct") {
            compute_forces_direct(particles);
        } else { 
            compute_forces_octree(particles, theta);
        }
    }
    
    std::cout << "Starting 3D disc simulation: Orbiting N=" << n_orbiting_particles
              << ", Central Mass=" << central_mass
              << ", Total Initial Particles=" << particles.size()
              << ", Steps=" << steps << ", Method=" << method << ", dt=" << dt 
              << (particle_0_is_special_central_star ? (mobile_star ? ", Mobile Star" : ", Fixed Star") : ", No Star") 
              << std::endl;
    std::cout << "Disc params: min_r=" << disc_min_radius << ", max_r=" << disc_max_radius << ", thickness=" << disc_thickness << std::endl;
    if (particle_0_is_special_central_star) {
         std::cout << "Accretion Radius: " << ACCRETION_RADIUS << ", Collision Radius (Softening): " << SOFTENING << std::endl;
    }
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;
    if (!particle_0_is_special_central_star && n_orbiting_particles > 0) {
        std::cout << "WARNING: No central mass specified (central_mass is 0). Particles will not orbit in a stable disc." << std::endl;
    }


    for (int step = 0; step < steps; ++step) {
        if (particles.empty()) {
             std::cout << "Step " << step + 1 << "/" << steps << " | All particles processed or simulation empty. Ending." << std::endl;
            break;
        }

        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z 
                      << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass << "\n";
        }
        double current_energy = system_energy(particles);
        energy_file << step << "," << std::fixed << std::setprecision(8) << current_energy 
                    << "," << particles.size() << "\n";

        if (step % std::max(1, steps / 20) == 0 || step == steps - 1) { 
             std::cout << "Step " << step + 1 << "/" << steps 
                       << " | Particles: " << particles.size()
                       << " | Energy: " << current_energy << std::endl;
        }
        
        // Pass current_step for potential debug prints within handle_elastic_collisions_and_accretion
        leapfrog_step(particles, dt, method, theta, mobile_star, particle_0_is_special_central_star, step);
    }
    
    std::cout << "Simulation finished. Final particle count: " << particles.size() << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}
