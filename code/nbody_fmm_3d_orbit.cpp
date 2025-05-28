// nbody_solarsystem_sim_v2.cpp
// A robust version for simulating an asteroid belt perturbed by a Jupiter-like planet.
// COMPILE WITH:
// g++ nbody_solarsystem_sim_v2.cpp -o nbody_solarsystem_sim -O3 -std=c++17 -fopenmp -Wall
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Physical Constants & Simulation Params ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;
const double ACCRETION_RADIUS = 0.05; 
const double ACCRETION_RADIUS_SQ = ACCRETION_RADIUS * ACCRETION_RADIUS;

// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z, mass;
    double vx, vy, vz;
    double ax, ay, az;

    Particle(int _id, double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
        : id(_id), x(_x), y(_y), z(_z), mass(_mass),
          vx(_vx), vy(_vy), vz(_vz),
          ax(0.0), ay(0.0), az(0.0) {}
};

class OctreeNode {
public:
    double cx, cy, cz, size;
    std::vector<std::unique_ptr<OctreeNode>> children;
    std::vector<Particle*> node_particles;
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    bool is_leaf = true;
    bool is_empty = true;

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s) {
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

    void insert(Particle* p) {
        if (is_leaf) {
            if (node_particles.empty()) {
                node_particles.push_back(p);
            } else {
                Particle* existing_particle = node_particles[0];
                node_particles.clear();
                subdivide();
                children[get_child_index(existing_particle)]->insert(existing_particle);
                children[get_child_index(p)]->insert(p);
            }
        } else {
            children[get_child_index(p)]->insert(p);
        }
        is_empty = false;
        double new_total_mass = total_mass + p->mass;
        com_x = (com_x * total_mass + p->x * p->mass) / new_total_mass;
        com_y = (com_y * total_mass + p->y * p->mass) / new_total_mass;
        com_z = (com_z * total_mass + p->z * p->mass) / new_total_mass;
        total_mass = new_total_mass;
    }

    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) return;
        
        // If leaf node, compute direct interaction
        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) continue;
                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                double inv_r = 1.0 / std::sqrt(r2 + SOFT2);
                double f_factor = G_CONST * p_in_node->mass * inv_r * inv_r * inv_r;
                force_x += f_factor * dx;
                force_y += f_factor * dy;
                force_z += f_factor * dz;
            }
            return;
        }

        double dx_com = com_x - target_p->x;
        double dy_com = com_y - target_p->y;
        double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;

        // Barnes-Hut criterion: if node is far enough away, treat it as a single body
        if ((size * size / (r2_com + SOFT2)) < (theta * theta)) {
            // *** THE FIX IS HERE ***
            // This calculation is now robust and correctly softened to prevent division by zero.
            double inv_r_softened = 1.0 / std::sqrt(r2_com + SOFT2);
            double f_factor = G_CONST * total_mass * inv_r_softened * inv_r_softened * inv_r_softened;
            force_x += f_factor * dx_com;
            force_y += f_factor * dy_com;
            force_z += f_factor * dz_com;
        } else { // Otherwise, recurse into children
            for (const auto& child : children) {
                if (child && !child->is_empty) {
                    child->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
        }
    }
};

// --- Physics Engine ---
void handle_interactions(std::vector<Particle>& particles) {
    if (particles.size() < 2) return;
    std::vector<int> to_remove_indices;

    // IMPROVEMENT: Only check for accretion between the Sun (ID 0) and asteroids (ID > 1)
    Particle& central_star = particles[0];
    for (size_t i = 1; i < particles.size(); ++i) {
        if (particles[i].id == 1) continue; // Skip Jupiter
        
        double dx = central_star.x - particles[i].x;
        double dy = central_star.y - particles[i].y;
        double dz = central_star.z - particles[i].z;
        double r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < ACCRETION_RADIUS_SQ) {
            to_remove_indices.push_back(i);
        }
    }
    
    // Remove accreted particles safely
    std::sort(to_remove_indices.rbegin(), to_remove_indices.rend());
    for (int index : to_remove_indices) {
        particles[0].mass += particles[index].mass; // Conserve mass
        particles.erase(particles.begin() + index);
    }
}

void compute_forces_octree(std::vector<Particle>& particles, double theta, double domain_size) {
    if (particles.empty()) return;

    OctreeNode root(0, 0, 0, domain_size);
    for (auto& p : particles) {
        if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) < domain_size / 2.0) {
            root.insert(&p);
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0; particles[i].ay = 0; particles[i].az = 0;
        root.compute_force(&particles[i], particles[i].ax, particles[i].ay, particles[i].az, theta);
    }
}

void leapfrog_step(std::vector<Particle>& particles, double dt, double theta, double domain_size) {
    if (particles.empty()) return;
    const size_t start_index = 1; // Sun at index 0 is fixed

    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
    handle_interactions(particles);
    compute_forces_octree(particles, theta, domain_size);
    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

double system_energy(const std::vector<Particle>& particles) { /* ... unchanged ... */ return 0.0;}

// --- Initial Conditions for Solar System ---
std::vector<Particle> init_solar_system(
    double central_mass, int num_asteroids, double belt_min_r,
    double belt_max_r, double jupiter_mass, double jupiter_r) {
    std::vector<Particle> particles;
    std::mt19937 rng(std::random_device{}());
    int id_counter = 0;

    particles.emplace_back(id_counter++, 0.0, 0.0, 0.0, central_mass);

    if (jupiter_mass > 0) {
        double jupiter_speed = std::sqrt(G_CONST * (central_mass + jupiter_mass) / jupiter_r);
        particles.emplace_back(id_counter++, jupiter_r, 0.0, 0.0, jupiter_mass, 0.0, jupiter_speed, 0.0);
    }

    std::uniform_real_distribution<double> radius_dist(belt_min_r, belt_max_r);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> mass_dist(0.00001, 0.0001); // Very low mass asteroids

    for (int i = 0; i < num_asteroids; ++i) {
        double r = radius_dist(rng);
        double angle = angle_dist(rng);
        double asteroid_mass = mass_dist(rng);
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        double orbital_speed = std::sqrt(G_CONST * central_mass / r);
        double vx = -orbital_speed * std::sin(angle);
        double vy = orbital_speed * std::cos(angle);
        particles.emplace_back(id_counter++, x, y, 0.0, asteroid_mass, vx, vy, 0.0);
    }
    std::cout << "Initialized Solar System with " << id_counter << " bodies." << std::endl;
    return particles;
}

// --- Main Simulation ---
int main(int argc, char* argv[]) {
    int n_asteroids = 4000;
    int steps = 10000; // Longer simulation needed to see gaps form
    double dt = 0.004; // Smaller timestep for higher accuracy

    double central_mass = 1000.0;
    double jupiter_mass = 2.0; 
    double jupiter_radius = 25.0;
    double belt_min_r = 10.0;
    double belt_max_r = 21.0;
    
    // The 2:1 resonance with Jupiter at r=25 is at ~15.75
    // The 3:1 resonance is at ~12.0
    // A belt from 10 to 21 is perfect for seeing these gaps.
    
    double domain_size = jupiter_radius * 2.5; // IMPROVEMENT: Fixed domain size

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_asteroids = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-j_r" && i + 1 < argc) jupiter_radius = std::stod(argv[++i]);
    }

    int total_initial_particles = n_asteroids + 2;
    std::string traj_fname = "trajectories_solarsystem_" + std::to_string(total_initial_particles) + ".csv";
    std::string energy_fname = "energy_solarsystem_" + std::to_string(total_initial_particles) + ".csv";

    std::ofstream traj_file(traj_fname);
    traj_file << "step,particle_id,x,y,z,vx,vy,vz,mass\n";
    std::ofstream energy_file(energy_fname);
    energy_file << "step,total_energy,num_particles\n";

    std::vector<Particle> particles = init_solar_system(central_mass, n_asteroids, belt_min_r, belt_max_r, jupiter_mass, jupiter_radius);

    std::cout << "Starting VALID Solar System simulation..." << std::endl;
    std::cout << "Jupiter at r=" << jupiter_radius << ". Asteroid belt from r=" << belt_min_r << " to " << belt_max_r << "." << std::endl;
    
    compute_forces_octree(particles, 0.5, domain_size);

    for (int step = 0; step < steps; ++step) {
        if (step % 200 == 0) { // Write data less frequently
            for (const auto& p : particles) {
                traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z
                          << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass << "\n";
            }
        }
        if (step % 100 == 0) {
            energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles)
                        << "," << particles.size() << "\n";
            std::cout << "Step " << step << "/" << steps << " | Particles: " << particles.size() << std::endl;
        }
        leapfrog_step(particles, dt, 0.5, domain_size);
    }
    
    // Write final state for analysis
    for (const auto& p : particles) {
        traj_file << steps << "," << p.id << "," << p.x << "," << p.y << "," << p.z
                    << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass << "\n";
    }

    std::cout << "Simulation finished. Output files:\n" << traj_fname << "\n" << energy_fname << std::endl;
    traj_file.close();
    energy_file.close();
    return 0;
}