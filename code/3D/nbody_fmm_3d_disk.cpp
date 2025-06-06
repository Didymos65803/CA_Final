// nbody_disk_sim.cpp
// Simulates a rotating disc of particles around a central mass,
// with particle-particle collision and accretion.
// COMPILE WITH:
// g++ nbody_disk_sim.cpp -o nbody_disk_sim -O3 -std=c++17 -fopenmp -Wall
#include <iostream>
#include <vector>
#include <string>
#include <cmath>    // For std::sqrt, std::cbrt, M_PI
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm> // For std::min, std::max, std::sort, std::remove_if
#include <memory>    // For std::unique_ptr, std::make_unique

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Physical Constants ---
const double G_CONST = 1.0;                 // Gravitational constant
const double SOFTENING = 0.01;              // Softening length for gravity calculation
const double SOFT2 = SOFTENING * SOFTENING; // Softening length squared
const double ACCRETION_RADIUS_FACTOR = 1.5; // Factor for central star's special accretion radius
const double ACCRETION_RADIUS = ACCRETION_RADIUS_FACTOR * SOFTENING; // Star's special accretion radius
const double ACCRETION_RADIUS_SQ = ACCRETION_RADIUS * ACCRETION_RADIUS; // Squared

// --- Tree Parameters ---
const int MAX_PARTICLES_PER_LEAF = 16; // Max particles in a leaf node before subdividing

// --- Collision/Accretion Parameters ---
const double PARTICLE_DENSITY_PROXY = 10.0; // Proxy for density to calculate radius from mass
                                           // Adjust this to change particle sizes relative to mass
const double MIN_PARTICLE_RADIUS = 0.001;  // Minimum radius for any particle
const double STAR_RADIUS_FACTOR = 2.0;     // If star's radius is dynamic, factor to make it larger

// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z, mass;
    double vx, vy, vz;
    double ax, ay, az;
    double radius; // Physical radius of the particle for collision detection
    bool active;   // Flag to mark if the particle is active or has been accreted/merged

    Particle(int _id, double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
        : id(_id), x(_x), y(_y), z(_z), mass(_mass),
          vx(_vx), vy(_vy), vz(_vz),
          ax(0.0), ay(0.0), az(0.0), active(true) {
        // Calculate radius based on mass (e.g., assuming constant density: V ~ m, r ~ m^(1/3))
        if (mass > 1e-9) {
            // For the central star (id 0), potentially assign a larger or fixed radius
            if (id == 0 && mass > 100.0) { // Assuming central star is massive
                 // Example: make star radius larger or use a different logic
                radius = STAR_RADIUS_FACTOR * ACCRETION_RADIUS_FACTOR * SOFTENING; 
                // Or based on its mass with a different density/factor
                // radius = 0.05 * std::cbrt(mass / PARTICLE_DENSITY_PROXY); 
            } else {
                radius = 0.02 * std::cbrt(mass / PARTICLE_DENSITY_PROXY); // Factor for planetesimals
            }
            radius = std::max(radius, MIN_PARTICLE_RADIUS);
        } else {
            radius = MIN_PARTICLE_RADIUS;
        }
    }
};

class OctreeNode {
public:
    double cx, cy, cz, size;
    std::vector<std::unique_ptr<OctreeNode>> children;
    std::vector<Particle*> node_particles;
    double total_mass;
    double com_x, com_y, com_z;
    bool is_leaf;
    bool is_empty;

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s),
          total_mass(0.0), com_x(0.0), com_y(0.0), com_z(0.0),
          is_leaf(true), is_empty(true) {
        children.resize(8); // Initializes unique_ptrs to nullptr
    }

    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;
        int child_idx_counter = 0;
        for (int i = -1; i <= 1; i += 2) { // z-offset
            for (int j = -1; j <= 1; j += 2) { // y-offset
                for (int k = -1; k <= 1; k += 2) { // x-offset
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

    void insert(Particle* p) {
        if (!p->active) return; // Do not insert inactive particles into the tree

        if (is_empty) {
            com_x = p->x;
            com_y = p->y;
            com_z = p->z;
            total_mass = p->mass;
            is_empty = false;
        } else {
            double new_total_mass = total_mass + p->mass;
            if (new_total_mass > 1e-12) {
                 com_x = (com_x * total_mass + p->x * p->mass) / new_total_mass;
                 com_y = (com_y * total_mass + p->y * p->mass) / new_total_mass;
                 com_z = (com_z * total_mass + p->z * p->mass) / new_total_mass;
            } else {
                 com_x = p->x; com_y = p->y; com_z = p->z; // Fallback if masses are tiny
            }
            total_mass = new_total_mass;
        }

        if (is_leaf) {
            node_particles.push_back(p);
            if (node_particles.size() > MAX_PARTICLES_PER_LEAF && size > 2.0 * SOFTENING * 10.0) { // Avoid over-subdividing
                subdivide();
                std::vector<Particle*> particles_to_move = node_particles;
                node_particles.clear();
                for (Particle* particle_to_move : particles_to_move) {
                    if (particle_to_move->active) { // Ensure only active particles are moved
                        children[get_child_index(particle_to_move)]->insert(particle_to_move);
                    }
                }
            }
        } else {
            children[get_child_index(p)]->insert(p);
        }
    }

    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta_sq) const {
        if (is_empty || total_mass < 1e-12 || !target_p->active) return;

        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (!p_in_node->active || p_in_node == target_p) continue;

                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < 1e-12) continue;
                double r_soft_inv = 1.0 / std::sqrt(r2 + SOFT2);
                double f_over_r3 = G_CONST * p_in_node->mass * r_soft_inv * r_soft_inv * r_soft_inv;
                force_x += f_over_r3 * dx;
                force_y += f_over_r3 * dy;
                force_z += f_over_r3 * dz;
            }
            return;
        }

        double dx_com = com_x - target_p->x;
        double dy_com = com_y - target_p->y;
        double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;

        if (size * size < theta_sq * r2_com || r2_com < SOFT2 ) {
            if (r2_com < 1e-12) return;
            double r_com_soft_inv = 1.0 / std::sqrt(r2_com + SOFT2);
            double f_over_r3_com = G_CONST * total_mass * r_com_soft_inv * r_com_soft_inv * r_com_soft_inv;
            force_x += f_over_r3_com * dx_com;
            force_y += f_over_r3_com * dy_com;
            force_z += f_over_r3_com * dz_com;
        } else {
            for (const auto& child : children) {
                if (child && !child->is_empty) {
                    child->compute_force(target_p, force_x, force_y, force_z, theta_sq);
                }
            }
        }
    }
};

// --- Interaction Handling ---

// Handles special accretion onto the central star (particles[0]) based on a fixed ACCRETION_RADIUS
void handle_star_accretion(std::vector<Particle>& particles, bool fixed_star) {
    if (particles.empty() || !particles[0].active || particles[0].mass <= 0) return;

    Particle& central_star = particles[0];
    // Iterate from 1 as particles[0] is the star
    for (size_t i = 1; i < particles.size(); ++i) {
        if (!particles[i].active) continue;

        double dx = central_star.x - particles[i].x;
        double dy = central_star.y - particles[i].y;
        double dz = central_star.z - particles[i].z;
        double r2 = dx * dx + dy * dy + dz * dz;

        if (r2 < ACCRETION_RADIUS_SQ) { // Particle is within the star's special accretion radius
            if (!fixed_star) { // Conserve momentum if star is mobile
                double combined_mass = central_star.mass + particles[i].mass;
                if (combined_mass > 1e-9) {
                    central_star.vx = (central_star.vx * central_star.mass + particles[i].vx * particles[i].mass) / combined_mass;
                    central_star.vy = (central_star.vy * central_star.mass + particles[i].vy * particles[i].mass) / combined_mass;
                    central_star.vz = (central_star.vz * central_star.mass + particles[i].vz * particles[i].mass) / combined_mass;
                }
            }
            central_star.mass += particles[i].mass;
            particles[i].active = false; // Mark particle as inactive
            particles[i].mass = 0.0;     // Set mass to zero

            // Update central star's radius after accretion
            if (central_star.id == 0 && central_star.mass > 100.0) {
                 central_star.radius = STAR_RADIUS_FACTOR * ACCRETION_RADIUS_FACTOR * SOFTENING;
            } else {
                central_star.radius = 0.02 * std::cbrt(central_star.mass / PARTICLE_DENSITY_PROXY);
            }
            central_star.radius = std::max(central_star.radius, MIN_PARTICLE_RADIUS);
        }
    }
}

// Handles general particle-particle collisions and mergers based on physical radii
void handle_particle_collisions(std::vector<Particle>& particles) {
    if (particles.size() < 2) return; // Need at least two particles to collide

    // Iterate through all unique pairs of active particles
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;

        for (size_t j = i + 1; j < particles.size(); ++j) {
            if (!particles[j].active) continue;

            Particle& p1 = particles[i];
            Particle& p2 = particles[j];

            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double dz = p2.z - p1.z;
            double r_sq = dx * dx + dy * dy + dz * dz;
            double sum_radii = p1.radius + p2.radius;

            if (r_sq < sum_radii * sum_radii) { // Collision detected (particles overlap)
                // Perform perfectly inelastic collision (merge)
                // Assume the more massive particle accretes the less massive one
                Particle* accretor = &p1;
                Particle* accreted = &p2;
                if (p2.mass > p1.mass) {
                    std::swap(accretor, accreted);
                }
                
                // Prevent star from being accreted by a much smaller particle if IDs are not strictly managed for star
                // This check is more robust if star is always particles[0] or has a unique massive property
                if (accreted->id == 0 && accreted->mass > accretor->mass * 10.0) { // If p[0] is somehow the smaller one but still massive
                    std::swap(accretor, accreted); // Ensure star (id=0) is the accretor
                }


                double combined_mass = accretor->mass + accreted->mass;
                if (combined_mass < 1e-9) { // Both particles are effectively massless
                    accretor->active = false; // Deactivate both if their combined mass is negligible
                    accreted->active = false;
                    accretor->mass = 0;
                    accreted->mass = 0;
                    continue;
                }

                // Conserve momentum
                accretor->vx = (accretor->vx * accretor->mass + accreted->vx * accreted->mass) / combined_mass;
                accretor->vy = (accretor->vy * accretor->mass + accreted->vy * accreted->mass) / combined_mass;
                accretor->vz = (accretor->vz * accretor->mass + accreted->vz * accreted->mass) / combined_mass;
                
                // Update position to center of mass of the two colliding particles (optional, can keep accretor's position)
                // accretor->x = (accretor->x * accretor->mass + accreted->x * accreted->mass) / combined_mass;
                // accretor->y = (accretor->y * accretor->mass + accreted->y * accreted->mass) / combined_mass;
                // accretor->z = (accretor->z * accretor->mass + accreted->z * accreted->mass) / combined_mass;

                accretor->mass = combined_mass;
                
                // Update accretor's radius
                if (accretor->id == 0 && accretor->mass > 100.0) { // If it's the central star
                     accretor->radius = STAR_RADIUS_FACTOR * ACCRETION_RADIUS_FACTOR * SOFTENING;
                } else {
                    accretor->radius = 0.02 * std::cbrt(accretor->mass / PARTICLE_DENSITY_PROXY);
                }
                accretor->radius = std::max(accretor->radius, MIN_PARTICLE_RADIUS);

                accreted->active = false; // Mark the smaller particle as inactive
                accreted->mass = 0.0;     // Set its mass to zero
            }
        }
    }
}

// Utility function to remove inactive particles from the vector
void remove_inactive_particles(std::vector<Particle>& particles) {
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
                       [](const Particle& p) { return !p.active; }),
        particles.end());
}


void compute_forces_octree(std::vector<Particle>& particles, double theta_val) {
    if (particles.empty()) return;

    double min_x = particles[0].x, max_x = particles[0].x;
    double min_y = particles[0].y, max_y = particles[0].y;
    double min_z = particles[0].z, max_z = particles[0].z;
    bool first = true;
    for (const auto& p : particles) { // Iterate only over active particles for bounds
        if (!p.active) continue;
        if (first) {
            min_x = max_x = p.x;
            min_y = max_y = p.y;
            min_z = max_z = p.z;
            first = false;
        } else {
            min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
        }
    }
     if (first) return; // No active particles found

    double domain_cx = (max_x + min_x) / 2.0;
    double domain_cy = (max_y + min_y) / 2.0;
    double domain_cz = (max_z + min_z) / 2.0;
    double domain_side = std::max({max_x - min_x, max_y - min_y, max_z - min_z, 1.0});
    
    OctreeNode root(domain_cx, domain_cy, domain_cz, domain_side * 1.2);
    for (auto& p : particles) { // Pass pointers to active particles
        if (p.active) {
            root.insert(&p);
        }
    }

    double theta_sq = theta_val * theta_val;

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;
        particles[i].ax = 0; particles[i].ay = 0; particles[i].az = 0;
        root.compute_force(&particles[i], particles[i].ax, particles[i].ay, particles[i].az, theta_sq);
    }
}

void leapfrog_step(std::vector<Particle>& particles, double dt, double theta, bool mobile_star) {
    if (particles.empty()) return;
    
    // Determine which particles to update (all if star is mobile or no star)
    // Note: particles[0] might not be the star if it got accreted, or if star_initially_exists was false.
    // A more robust way would be to check particles[0].id == 0 AND particles[0].active
    int start_index = 0; // Default to updating all active particles
    if (!mobile_star && !particles.empty() && particles[0].active && particles[0].id == 0) {
        start_index = 1; // If star is fixed, active, and is particles[0], skip its dynamics
    }

    // --- First half-kick for velocities ---
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;
        if (i < (size_t)start_index && particles[i].id == 0) continue; // Skip fixed star
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    // --- Drift (update positions) ---
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;
        if (i < (size_t)start_index && particles[i].id == 0) continue; // Skip fixed star
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
    
    // --- Handle interactions (accretion and collisions) ---
    // 1. Special accretion onto the central star (if it exists and is active)
    if (!particles.empty() && particles[0].active && particles[0].id == 0) {
         handle_star_accretion(particles, !mobile_star);
    }
    
    // 2. General particle-particle collisions and mergers
    handle_particle_collisions(particles);

    // 3. Remove inactive particles from the simulation vector
    remove_inactive_particles(particles);
    
    // --- Update forces based on new positions and particle list ---
    if (!particles.empty()) {
      compute_forces_octree(particles, theta);
    } else { // All particles might have been removed
        return; 
    }

    // Determine start_index again in case particles[0] changed due to removal
    start_index = 0; 
    if (!mobile_star && !particles.empty() && particles[0].active && particles[0].id == 0) {
        start_index = 1;
    }

    // --- Second half-kick for velocities ---
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue; // Should not happen if remove_inactive_particles worked
        if (i < (size_t)start_index && particles[i].id == 0) continue; // Skip fixed star
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

double system_energy(const std::vector<Particle>& particles) {
    double ke = 0.0, pe = 0.0;
    #pragma omp parallel for reduction(+:ke)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;
        ke += 0.5 * particles[i].mass * (particles[i].vx*particles[i].vx + particles[i].vy*particles[i].vy + particles[i].vz*particles[i].vz);
    }
    
    #pragma omp parallel for reduction(+:pe)
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i].active) continue;
        for(size_t j = i + 1; j < particles.size(); ++j) {
            if (!particles[j].active) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > 1e-12) { // Avoid division by zero if particles are at same spot (though SOFT2 handles this)
                 pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2 + SOFT2);
            }
        }
    }
    return ke + pe;
}

std::vector<Particle> init_3d_disc_orbiting_central_mass(
    int n_orbiting_particles, double central_mass_val, double min_radius_domain,
    double max_radius_domain, double disc_thickness, bool& star_exists_flag) {
    std::vector<Particle> particles_vec;
    std::mt19937 rng(std::random_device{}());
    int current_id = 0;
    star_exists_flag = false;

    if (central_mass_val > 1e-9) {
        particles_vec.emplace_back(current_id++, 0.0, 0.0, 0.0, central_mass_val, 0.0, 0.0, 0.0);
        star_exists_flag = true;
    }

    std::uniform_real_distribution<double> radius_dist(min_radius_domain, max_radius_domain);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> z_offset_dist(-disc_thickness / 2.0, disc_thickness / 2.0);
    std::uniform_real_distribution<double> mass_dist_factor(0.0001, 0.001); // Factor relative to central mass
    std::uniform_real_distribution<double> vel_perturb_dist(-0.05, 0.05);

    for (int i = 0; i < n_orbiting_particles; ++i) {
        double r_pos = radius_dist(rng);
        double angle = angle_dist(rng);
        
        double particle_mass_val = 0.01; // Default mass if no star
        if (star_exists_flag) {
            particle_mass_val = mass_dist_factor(rng) * central_mass_val;
        } else { // If no central star, generate some base mass particles
            std::uniform_real_distribution<double> base_mass_dist(0.01, 0.1);
            particle_mass_val = base_mass_dist(rng);
        }
        if (particle_mass_val < 1e-6) particle_mass_val = 1e-6;

        double x = r_pos * std::cos(angle);
        double y = r_pos * std::sin(angle);
        double z = z_offset_dist(rng);

        double orbital_speed = 0.0;
        if (star_exists_flag && r_pos > 1e-6) { // Calculate orbital speed only if star exists and radius is valid
            // Approximate orbital speed based on central mass only for stability
            orbital_speed = std::sqrt(G_CONST * central_mass_val / r_pos);
        }
        
        double vx = -orbital_speed * std::sin(angle);
        double vy =  orbital_speed * std::cos(angle);
        
        vx *= (1.0 + vel_perturb_dist(rng));
        vy *= (1.0 + vel_perturb_dist(rng));
        double vz = vel_perturb_dist(rng) * orbital_speed * 0.1 * (star_exists_flag ? 1.0 : 0.0);

        particles_vec.emplace_back(current_id++, x, y, z, particle_mass_val, vx, vy, vz);
    }
    return particles_vec;
}

int main(int argc, char* argv[]) {
    int n_orbiting_particles = 2000;
    int steps = 1000;
    double theta = 0.5;
    double dt = 0.01;
    bool mobile_star = false;
    double central_mass = 1000.0;
    double disc_min_radius = 5.0;
    double disc_max_radius = 20.0;
    double disc_thickness = 0.5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_orbiting_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "-mobile_star") mobile_star = true;
        else if (arg == "-fixed_star") mobile_star = false;
        else if (arg == "-central_mass" && i + 1 < argc) central_mass = std::stod(argv[++i]);
        else if (arg == "-min_r" && i + 1 < argc) disc_min_radius = std::stod(argv[++i]);
        else if (arg == "-max_r" && i + 1 < argc) disc_max_radius = std::stod(argv[++i]);
        else if (arg == "-thickness" && i + 1 < argc) disc_thickness = std::stod(argv[++i]);
         else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    bool star_initially_exists;
    std::vector<Particle> particles = init_3d_disc_orbiting_central_mass(
        n_orbiting_particles, central_mass, disc_min_radius, disc_max_radius, disc_thickness, star_initially_exists);

    if (particles.empty()){
        std::cout << "No particles initialized. Exiting." << std::endl;
        return 0;
    }
    if (!star_initially_exists && !particles.empty()) { // If no star was created but particles exist
        mobile_star = true; // All particles must be mobile
        std::cout << "No central star created (mass <= 0), all particles are mobile." << std::endl;
    } else if (!star_initially_exists && mobile_star && !particles.empty()) {
         std::cout << "Warning: Star is set to fixed, but no central mass > 0 was specified. Treating system as fully mobile." << std::endl;
    }


    int total_initial_particles = particles.size();
    std::string traj_fname = "trajectories_disc_coll_" + std::to_string(total_initial_particles) + ".csv";
    std::string energy_fname = "energy_disc_coll_" + std::to_string(total_initial_particles) + ".csv";

    std::ofstream traj_file(traj_fname);
    traj_file << "step,particle_id,x,y,z,vx,vy,vz,mass,radius\n"; // Added radius to output
    std::ofstream energy_file(energy_fname);
    energy_file << "step,total_energy,num_particles\n";

    std::cout << "Starting 3D disc simulation with collisions. Initial particles: " << particles.size() 
              << ", Theta: " << theta << ", dt: " << dt 
              << ", Mobile Star: " << (mobile_star ? "Yes" : "No") << std::endl;
    
    if (!particles.empty()) {
        compute_forces_octree(particles, theta);
    }

    for (int step = 0; step < steps; ++step) {
        if (particles.empty()) {
            std::cout << "All particles lost/accreted at step " << step << std::endl;
            break;
        }

        if (step % 20 == 0) {
            for (const auto& p : particles) { // This will iterate over all particles, including those marked inactive if not yet removed.
                                             // However, remove_inactive_particles should have cleaned them up.
                if(p.active) { // Only log active particles
                    traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z
                              << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass 
                              << "," << p.radius << "\n";
                }
            }
            energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles)
                        << "," << std::count_if(particles.begin(), particles.end(), [](const Particle&p){ return p.active;}) 
                        << "\n";
            
            long active_particle_count = std::count_if(particles.begin(), particles.end(), [](const Particle&p){ return p.active;});
            std::cout << "Step " << step << "/" << steps << " | Active Particles: " << active_particle_count
                      << " | Energy: " << system_energy(particles) << std::endl;
        }

        leapfrog_step(particles, dt, theta, mobile_star);
    }
    // Write final state
    if (!particles.empty() && steps % 20 != 0) { // If simulation ended not on a reporting step
         for (const auto& p : particles) {
             if(p.active){
                traj_file << (steps-1) << "," << p.id << "," << p.x << "," << p.y << "," << p.z
                          << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass 
                          << "," << p.radius << "\n";
             }
        }
        energy_file << (steps-1) << "," << std::fixed << std::setprecision(8) << system_energy(particles)
                    << "," << std::count_if(particles.begin(), particles.end(), [](const Particle&p){ return p.active;}) << "\n";
    }

    std::cout << "Simulation finished. Output files:\n" << traj_fname << "\n" << energy_fname << std::endl;
    traj_file.close();
    energy_file.close();
    return 0;
}