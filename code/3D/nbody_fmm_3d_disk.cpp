// nbody_disk_sim.cpp
// Simulates a rotating disc of particles around a central mass.
// COMPILE WITH:
// g++ nbody_disk_sim.cpp -o nbody_disk_sim -O3 -std=c++17 -fopenmp -Wall
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

// --- Physical Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;
const double ACCRETION_RADIUS_FACTOR = 1.5;
const double ACCRETION_RADIUS = ACCRETION_RADIUS_FACTOR * SOFTENING;
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
    double total_mass;
    double com_x, com_y, com_z;
    bool is_leaf;
    bool is_empty;

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
        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) continue;
                double dx = p_in_node->x - target_p->x;
                double dy = p_in_node->y - target_p->y;
                double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < SOFT2) r2 = SOFT2;
                double r = std::sqrt(r2);
                if (r < 1e-9) continue;
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

        if (size / std::sqrt(r2_com + SOFT2) < theta) {
            double r_com = std::sqrt(r2_com);
            double f_over_r_com = G_CONST * total_mass / (r2_com * r_com + SOFT2);
            force_x += f_over_r_com * dx_com;
            force_y += f_over_r_com * dy_com;
            force_z += f_over_r_com * dz_com;
        } else {
            for (const auto& child : children) {
                if (child && !child->is_empty) {
                    child->compute_force(target_p, force_x, force_y, force_z, theta);
                }
            }
        }
    }
};

// --- Collision, Accretion, and Integration from previous version ---
void handle_interactions(std::vector<Particle>& particles, bool fixed_star) {
    std::vector<int> to_remove_indices;
    // Accretion loop
    if (!particles.empty()) {
        Particle& central_star = particles[0];
        for (size_t i = 1; i < particles.size(); ++i) {
            double dx = central_star.x - particles[i].x;
            double dy = central_star.y - particles[i].y;
            double dz = central_star.z - particles[i].z;
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < ACCRETION_RADIUS_SQ) {
                if (!fixed_star) { // Conserve momentum if star is mobile
                    double combined_mass = central_star.mass + particles[i].mass;
                    central_star.vx = (central_star.vx * central_star.mass + particles[i].vx * particles[i].mass) / combined_mass;
                    central_star.vy = (central_star.vy * central_star.mass + particles[i].vy * particles[i].mass) / combined_mass;
                    central_star.vz = (central_star.vz * central_star.mass + particles[i].vz * particles[i].mass) / combined_mass;
                }
                central_star.mass += particles[i].mass;
                to_remove_indices.push_back(i);
            }
        }
    }
    // Remove accreted particles (in reverse order to not mess up indices)
    std::sort(to_remove_indices.rbegin(), to_remove_indices.rend());
    for (int index : to_remove_indices) {
        particles.erase(particles.begin() + index);
    }
}


void compute_forces_octree(std::vector<Particle>& particles, double theta) {
    if (particles.empty()) return;
    double min_x = particles[0].x, max_x = particles[0].x;
    double min_y = particles[0].y, max_y = particles[0].y;
    double min_z = particles[0].z, max_z = particles[0].z;
    for (size_t i = 1; i < particles.size(); ++i) {
        min_x = std::min(min_x, particles[i].x); max_x = std::max(max_x, particles[i].x);
        min_y = std::min(min_y, particles[i].y); max_y = std::max(max_y, particles[i].y);
        min_z = std::min(min_z, particles[i].z); max_z = std::max(max_z, particles[i].z);
    }
    double domain_side = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    OctreeNode root((max_x+min_x)/2, (max_y+min_y)/2, (max_z+min_z)/2, domain_side * 1.2);
    for (auto& p : particles) root.insert(&p);

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0; particles[i].ay = 0; particles[i].az = 0;
        root.compute_force(&particles[i], particles[i].ax, particles[i].ay, particles[i].az, theta);
    }
}

void leapfrog_step(std::vector<Particle>& particles, double dt, double theta, bool mobile_star) {
    if (particles.empty()) return;
    int start_index = (mobile_star) ? 0 : 1;
    // Half-kick
    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
    // Drift
    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
    // Interactions
    handle_interactions(particles, !mobile_star);
    // Update forces
    compute_forces_octree(particles, theta);
    // Second half-kick
    #pragma omp parallel for
    for (size_t i = start_index; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

double system_energy(const std::vector<Particle>& particles) {
    double ke = 0.0, pe = 0.0;
    #pragma omp parallel for reduction(+:ke, pe)
    for (size_t i = 0; i < particles.size(); ++i) {
        ke += 0.5 * particles[i].mass * (particles[i].vx*particles[i].vx + particles[i].vy*particles[i].vy + particles[i].vz*particles[i].vz);
        for(size_t j = i + 1; j < particles.size(); ++j) {
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + SOFT2;
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions for an Orbiting Disk ---
std::vector<Particle> init_3d_disc_orbiting_central_mass(
    int n_orbiting_particles, double central_mass_val, double min_radius,
    double max_radius, double disc_thickness) {
    std::vector<Particle> particles_vec;
    std::mt19937 rng(std::random_device{}());
    int current_id = 0;

    // Add the central, non-moving star
    if (central_mass_val > 0) {
        particles_vec.emplace_back(current_id++, 0.0, 0.0, 0.0, central_mass_val, 0.0, 0.0, 0.0);
    }

    std::uniform_real_distribution<double> radius_dist(min_radius, max_radius);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> z_offset_dist(-disc_thickness / 2.0, disc_thickness / 2.0);
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);
    std::uniform_real_distribution<double> vel_perturb_dist(-0.05, 0.05);

    for (int i = 0; i < n_orbiting_particles; ++i) {
        double r = radius_dist(rng);
        double angle = angle_dist(rng);
        double particle_mass = mass_dist(rng);
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        double z = z_offset_dist(rng);

        double orbital_speed = std::sqrt(G_CONST * (central_mass_val + particle_mass) / r);
        double vx = -orbital_speed * std::sin(angle) * (1.0 + vel_perturb_dist(rng));
        double vy =  orbital_speed * std::cos(angle) * (1.0 + vel_perturb_dist(rng));
        double vz = vel_perturb_dist(rng) * orbital_speed * 0.1; // Small vertical velocity

        particles_vec.emplace_back(current_id++, x, y, z, particle_mass, vx, vy, vz);
    }
    return particles_vec;
}

// --- Main Simulation ---
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

    // Simple command-line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_orbiting_particles = std::stoi(argv[++i]);
        else if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-dt" && i + 1 < argc) dt = std::stod(argv[++i]);
        else if (arg == "-min_r" && i + 1 < argc) disc_min_radius = std::stod(argv[++i]);
        else if (arg == "-max_r" && i + 1 < argc) disc_max_radius = std::stod(argv[++i]);
    }

    int total_initial_particles = n_orbiting_particles + (central_mass > 0 ? 1 : 0);
    std::string traj_fname = "trajectories_disc_" + std::to_string(total_initial_particles) + ".csv";
    std::string energy_fname = "energy_disc_" + std::to_string(total_initial_particles) + ".csv";

    std::ofstream traj_file(traj_fname);
    traj_file << "step,particle_id,x,y,z,vx,vy,vz,mass\n";
    std::ofstream energy_file(energy_fname);
    energy_file << "step,total_energy,num_particles\n";

    std::vector<Particle> particles = init_3d_disc_orbiting_central_mass(
        n_orbiting_particles, central_mass, disc_min_radius, disc_max_radius, disc_thickness);

    std::cout << "Starting 3D disc simulation. Initial particles: " << particles.size() << std::endl;
    compute_forces_octree(particles, theta);

    for (int step = 0; step < steps; ++step) {
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z
                      << "," << p.vx << "," << p.vy << "," << p.vz << "," << p.mass << "\n";
        }
        energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles)
                    << "," << particles.size() << "\n";

        if (step % 20 == 0) {
            std::cout << "Step " << step << "/" << steps << " | Particles: " << particles.size() << std::endl;
        }

        leapfrog_step(particles, dt, theta, mobile_star);
    }

    std::cout << "Simulation finished. Output files:\n" << traj_fname << "\n" << energy_fname << std::endl;
    traj_file.close();
    energy_file.close();
    return 0;
}