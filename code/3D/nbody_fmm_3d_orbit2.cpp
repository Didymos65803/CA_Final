// nbody_solarsystem_fmm_resonance.cpp
// A solar system simulation for resonance, using the FMM code structure
// but falling back to Barnes-Hut for gravity, and incorporating RK4 and Mars.
// COMPILE WITH:
// g++ nbody_solarsystem_fmm_resonance.cpp -o nbody_solarsystem_fmm_resonance -O3 -std=c++17 -fopenmp -lm

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <complex> // For FMM (though not fully used for force here)
#include <limits>  // For std::numeric_limits

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Physical & Simulation Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.001; // Reduced for better accuracy in resonance
const double SOFT2 = SOFTENING * SOFTENING;
// Accretion radius not used in this specific resonance-focused version,
// but kept from original structure.
const double ACCRETION_RADIUS = 0.05; 
const double ACCRETION_RADIUS_SQ = ACCRETION_RADIUS * ACCRETION_RADIUS;


// --- FMM Parameters (from original code, largely structural here) ---
const int FMM_ORDER = 10; 
const int FMM_TERMS = (FMM_ORDER + 1) * (FMM_ORDER + 1);

// --- Data Structures ---
struct Particle {
    int id;
    std::string name; // Added for identifying bodies
    double x, y, z, mass;
    double vx, vy, vz;
    double ax = 0.0, ay = 0.0, az = 0.0; // Acceleration for RK4

    Particle(int _id, std::string _name, double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
        : id(_id), name(std::move(_name)), x(_x), y(_y), z(_z), mass(_mass),
          vx(_vx), vy(_vy), vz(_vz) {}

    // Function to calculate semi-major axis 'a'
    double semi_major_axis(double central_mass_param) const {
        // For the Sun (particle 0), SMA is not well-defined or needed in this context
        if (id == 0 && name == "Sun") return 0.0; 
        
        double r_val = std::sqrt(x*x + y*y + z*z);
        if (r_val < 1e-6) return 0.0; // Avoid division by zero if at the center

        double v2_val = vx*vx + vy*vy + vz*vz;
        // mu should be G * (M_central + m_particle)
        // Here, central_mass_param is M_sun. We add the particle's own mass for accuracy.
        double mu_val = G_CONST * (central_mass_param + mass); 
        
        // From the vis-viva equation: v^2 = mu * (2/r - 1/a)
        // So, 1/a = 2/r - v^2/mu
        double a_inv_val = (2.0 / r_val) - (v2_val / mu_val);
        
        if (a_inv_val < 1e-9) { // Check for parabolic or hyperbolic (unbound) orbits
            return std::numeric_limits<double>::infinity(); 
        }
        return 1.0 / a_inv_val;
    }
};

// --- FMM Mathematical Utilities (from original code, largely structural here) ---
std::vector<double> precompute_factorials(int n) {
    std::vector<double> fact_vec(n + 1);
    fact_vec[0] = 1.0;
    for (int i = 1; i <= n; ++i) {
        fact_vec[i] = fact_vec[i - 1] * i;
    }
    return fact_vec;
}
const std::vector<double> fact = precompute_factorials(2 * FMM_ORDER);

inline int lm_to_idx(int l, int m) {
    return l * l + l + m;
}

std::complex<double> R_lm(int l, int m, double dx, double dy, double dz) {
    double r_sq = dx*dx + dy*dy + dz*dz;
    if (r_sq < 1e-12) return {0,0}; // Avoid issues at origin
    // This is a placeholder. A full R_lm involves Associated Legendre Polynomials.
    // For a real FMM, this needs to be correctly implemented.
    return std::pow(std::sqrt(r_sq), l) * std::polar(1.0, m * std::atan2(dy, dx));
}

std::complex<double> S_lm(int l, int m, double dx, double dy, double dz) {
    double r_sq = dx*dx + dy*dy + dz*dz;
    if (r_sq < 1e-12) return {std::numeric_limits<double>::infinity(), 0.0}; // Singular
    double r_val = std::sqrt(r_sq);
    // This is a placeholder. A full S_lm involves Associated Legendre Polynomials.
    return std::pow(r_val, -l-1) * std::polar(1.0, -m * std::atan2(dy, dx)); // Note: Y_lm(theta,phi) / r^(l+1)
}


// --- FMM Node Structure (from original code) ---
// This structure would be used by a full FMM implementation.
// In this hybrid code, it's mostly scaffolding as the OctreeNode is used for gravity.
class FMM_Node {
public:
    double cx, cy, cz, size;
    std::vector<std::unique_ptr<FMM_Node>> children;
    std::vector<Particle*> node_particles_fmm; // Renamed to avoid clash
    bool is_leaf = true;
    bool is_empty = true;
    
    std::vector<std::complex<double>> multipole;
    std::vector<std::complex<double>> local;

    FMM_Node(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s) {
        children.resize(8);
        multipole.resize(FMM_TERMS, {0.0, 0.0});
        local.resize(FMM_TERMS, {0.0, 0.0});
    }

    // Subdivide, get_child_index, insert are kept as in the original FMM sketch
    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;
        int child_idx = 0;
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                for (int k = -1; k <= 1; k += 2) {
                    children[child_idx++] = std::make_unique<FMM_Node>(
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
            if (node_particles_fmm.empty()) {
                node_particles_fmm.push_back(p);
            } else { // Max 1 particle per leaf in this simplified FMM sketch for insertion
                Particle* existing_particle = node_particles_fmm[0];
                node_particles_fmm.clear();
                subdivide();
                children[get_child_index(existing_particle)]->insert(existing_particle);
                children[get_child_index(p)]->insert(p);
            }
        } else {
            children[get_child_index(p)]->insert(p);
        }
        is_empty = false;
    }
};

// --- FMM Core Functions (from original code, largely structural placeholders) ---
// These functions outline the FMM steps but are not fully implemented for force calculation.
void p2m(FMM_Node* node) { /* Placeholder from original */ }
void m2m(FMM_Node* node) { /* Placeholder from original */ }
void m2l_interaction(FMM_Node* target, FMM_Node* source) { /* Placeholder from original */ }
void l2l(FMM_Node* node) { /* Placeholder from original */ }
void p2p_l2p(FMM_Node* node, FMM_Node* root) { /* Placeholder from original */ }
void traverse_m2l(FMM_Node* node1, FMM_Node* node2) { /* Placeholder from original */ }

// --- Barnes-Hut Octree (This is used for actual gravity calculation) ---
class OctreeNode {
public:
    double cx, cy, cz, size;
    std::vector<std::unique_ptr<OctreeNode>> children;
    const Particle* p_ptr = nullptr; 
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    bool is_leaf = true;

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s) {
        children.resize(8);
    }
    
    // Helper functions to get child center coordinates based on index (for Barnes-Hut)
    double get_child_cx(int idx) { return cx + ( (idx&1) ? size/4.0 : -size/4.0); }
    double get_child_cy(int idx) { return cy + ( (idx&2) ? size/4.0 : -size/4.0); }
    double get_child_cz(int idx) { return cz + ( (idx&4) ? size/4.0 : -size/4.0); }

    int get_child_index_bh(const Particle* p) const { // Renamed for clarity
        int index = 0;
        if (p->x > cx) index |= 1;
        if (p->y > cy) index |= 2;
        if (p->z > cz) index |= 4;
        return index;
    }

    void insert_bh(const Particle* p_to_insert) { // Renamed for clarity
        if (total_mass == 0.0) { 
            p_ptr = p_to_insert;
            total_mass = p_ptr->mass;
            com_x = p_ptr->x; com_y = p_ptr->y; com_z = p_ptr->z;
            return;
        }

        if (is_leaf) { 
            is_leaf = false;
            int existing_idx = get_child_index_bh(p_ptr);
            if (!children[existing_idx]) children[existing_idx] = std::make_unique<OctreeNode>(get_child_cx(existing_idx), get_child_cy(existing_idx), get_child_cz(existing_idx), size/2.0);
            children[existing_idx]->insert_bh(p_ptr);
            p_ptr = nullptr; 
        }

        int new_idx = get_child_index_bh(p_to_insert);
        if (!children[new_idx]) children[new_idx] = std::make_unique<OctreeNode>(get_child_cx(new_idx), get_child_cy(new_idx), get_child_cz(new_idx), size/2.0);
        children[new_idx]->insert_bh(p_to_insert);
        
        total_mass += p_to_insert->mass;
        com_x = (com_x * (total_mass - p_to_insert->mass) + p_to_insert->x * p_to_insert->mass) / total_mass;
        com_y = (com_y * (total_mass - p_to_insert->mass) + p_to_insert->y * p_to_insert->mass) / total_mass;
        com_z = (com_z * (total_mass - p_to_insert->mass) + p_to_insert->z * p_to_insert->mass) / total_mass;
    }

    void compute_force_bh(Particle& target_p, double theta_sq) const { // Renamed for clarity
        if (total_mass < 1e-12) return;
        double dx, dy, dz, r2;

        if (is_leaf) {
            if (!p_ptr || p_ptr == &target_p) return; 
            dx = p_ptr->x - target_p.x;
            dy = p_ptr->y - target_p.y;
            dz = p_ptr->z - target_p.z;
            r2 = dx * dx + dy * dy + dz * dz;
        } else {
            dx = com_x - target_p.x;
            dy = com_y - target_p.y;
            dz = com_z - target_p.z;
            r2 = dx * dx + dy * dy + dz * dz;
        }
        
        if (!is_leaf && (size * size > theta_sq * r2 && r2 > 1e-6)) { // Added r2 check to prevent division by zero in theta_sq*r2 if r2 is tiny
            for (const auto& child : children) {
                if (child) child->compute_force_bh(target_p, theta_sq);
            }
        } else {
            if (r2 < 1e-12 && p_ptr == &target_p) return; // Avoid self-force if somehow it's a leaf and target is itself
            double inv_r_soft = 1.0 / std::sqrt(r2 + SOFT2);
            double f_factor = G_CONST * total_mass * inv_r_soft * inv_r_soft * inv_r_soft;
            target_p.ax += f_factor * dx;
            target_p.ay += f_factor * dy;
            target_p.az += f_factor * dz;
        }
    }
};

// --- Physics Engine with RK4 Integrator ---
void compute_accelerations_tree(std::vector<Particle>& particles, double theta, double domain_size) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0; particles[i].ay = 0; particles[i].az = 0;
    }

    OctreeNode root(0, 0, 0, domain_size);
    for (auto& p : particles) {
         // Ensure particles are within the domain for Octree insertion
        if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) < domain_size * 0.5) {
             root.insert_bh(&p);
        } else {
            // Handle particles outside domain: simple interaction with Sun or ignore for this step
            if (p.id !=0 && particles[0].name == "Sun") { // If not Sun and Sun exists
                double dx = particles[0].x - p.x;
                double dy = particles[0].y - p.y;
                double dz = particles[0].z - p.z;
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > 1e-6) {
                    double inv_r_soft = 1.0 / std::sqrt(r2 + SOFT2);
                    double f_factor = G_CONST * particles[0].mass * inv_r_soft * inv_r_soft * inv_r_soft;
                    p.ax += f_factor * dx;
                    p.ay += f_factor * dy;
                    p.az += f_factor * dz;
                }
            }
        }
    }

    double theta_sq = theta * theta;
    #pragma omp parallel for
    for (size_t i = 1; i < particles.size(); ++i) { // Start from 1 to keep Sun (p[0]) fixed
        if (std::max({std::abs(particles[i].x), std::abs(particles[i].y), std::abs(particles[i].z)}) < domain_size * 0.5) {
            root.compute_force_bh(particles[i], theta_sq);
        } 
        // If particle was outside domain, its interaction with Sun was already handled (partially)
        // More sophisticated handling for out-of-bounds particles might be needed for very long sims
    }
}

void rk4_step(std::vector<Particle>& particles, double dt, double theta, double domain_size) {
    const size_t n_particles = particles.size();
    if (n_particles == 0) return;
    std::vector<Particle> original_particles = particles;
    std::vector<Particle> k1_v(n_particles, Particle(0,"",0,0,0,0)), k1_x = k1_v;
    std::vector<Particle> k2_v = k1_v, k2_x = k1_v;
    std::vector<Particle> k3_v = k1_v, k3_x = k1_v;
    std::vector<Particle> k4_v = k1_v, k4_x = k1_v;

    // k1
    compute_accelerations_tree(particles, theta, domain_size);
    for(size_t i=0; i<n_particles; ++i) { k1_v[i].ax = particles[i].ax; k1_v[i].ay = particles[i].ay; k1_v[i].az = particles[i].az; k1_x[i].vx = particles[i].vx; k1_x[i].vy = particles[i].vy; k1_x[i].vz = particles[i].vz;}

    // k2
    for (size_t i = 1; i < n_particles; ++i) { // Sun is fixed
        particles[i].x = original_particles[i].x + k1_x[i].vx * 0.5 * dt;
        particles[i].y = original_particles[i].y + k1_x[i].vy * 0.5 * dt;
        particles[i].z = original_particles[i].z + k1_x[i].vz * 0.5 * dt;
        particles[i].vx = original_particles[i].vx + k1_v[i].ax * 0.5 * dt;
        particles[i].vy = original_particles[i].vy + k1_v[i].ay * 0.5 * dt;
        particles[i].vz = original_particles[i].vz + k1_v[i].az * 0.5 * dt;
    }
    compute_accelerations_tree(particles, theta, domain_size);
    for(size_t i=0; i<n_particles; ++i) { k2_v[i].ax = particles[i].ax; k2_v[i].ay = particles[i].ay; k2_v[i].az = particles[i].az; k2_x[i].vx = particles[i].vx; k2_x[i].vy = particles[i].vy; k2_x[i].vz = particles[i].vz;}

    // k3
    for (size_t i = 1; i < n_particles; ++i) {
        particles[i].x = original_particles[i].x + k2_x[i].vx * 0.5 * dt;
        particles[i].y = original_particles[i].y + k2_x[i].vy * 0.5 * dt;
        particles[i].z = original_particles[i].z + k2_x[i].vz * 0.5 * dt;
        particles[i].vx = original_particles[i].vx + k2_v[i].ax * 0.5 * dt;
        particles[i].vy = original_particles[i].vy + k2_v[i].ay * 0.5 * dt;
        particles[i].vz = original_particles[i].vz + k2_v[i].az * 0.5 * dt;
    }
    compute_accelerations_tree(particles, theta, domain_size);
    for(size_t i=0; i<n_particles; ++i) { k3_v[i].ax = particles[i].ax; k3_v[i].ay = particles[i].ay; k3_v[i].az = particles[i].az; k3_x[i].vx = particles[i].vx; k3_x[i].vy = particles[i].vy; k3_x[i].vz = particles[i].vz;}

    // k4
    for (size_t i = 1; i < n_particles; ++i) {
        particles[i].x = original_particles[i].x + k3_x[i].vx * dt;
        particles[i].y = original_particles[i].y + k3_x[i].vy * dt;
        particles[i].z = original_particles[i].z + k3_x[i].vz * dt;
        particles[i].vx = original_particles[i].vx + k3_v[i].ax * dt;
        particles[i].vy = original_particles[i].vy + k3_v[i].ay * dt;
        particles[i].vz = original_particles[i].vz + k3_v[i].az * dt;
    }
    compute_accelerations_tree(particles, theta, domain_size);
    for(size_t i=0; i<n_particles; ++i) { k4_v[i].ax = particles[i].ax; k4_v[i].ay = particles[i].ay; k4_v[i].az = particles[i].az; k4_x[i].vx = particles[i].vx; k4_x[i].vy = particles[i].vy; k4_x[i].vz = particles[i].vz;}
    
    // Combine
    for (size_t i = 1; i < n_particles; ++i) {
        particles[i].x = original_particles[i].x + (dt / 6.0) * (k1_x[i].vx + 2*k2_x[i].vx + 2*k3_x[i].vx + k4_x[i].vx);
        particles[i].y = original_particles[i].y + (dt / 6.0) * (k1_x[i].vy + 2*k2_x[i].vy + 2*k3_x[i].vy + k4_x[i].vy);
        particles[i].z = original_particles[i].z + (dt / 6.0) * (k1_x[i].vz + 2*k2_x[i].vz + 2*k3_x[i].vz + k4_x[i].vz);
        
        particles[i].vx = original_particles[i].vx + (dt / 6.0) * (k1_v[i].ax + 2*k2_v[i].ax + 2*k3_v[i].ax + k4_v[i].ax);
        particles[i].vy = original_particles[i].vy + (dt / 6.0) * (k1_v[i].ay + 2*k2_v[i].ay + 2*k3_v[i].ay + k4_v[i].ay);
        particles[i].vz = original_particles[i].vz + (dt / 6.0) * (k1_v[i].az + 2*k2_v[i].az + 2*k3_v[i].az + k4_v[i].az);
    }
}

// Accretion: simplified for this version, not the primary focus for resonance
void handle_accretion(std::vector<Particle>& particles) {
    if (particles.size() < 2 || particles[0].name != "Sun") return; // Ensure Sun is present
    std::vector<int> to_remove_indices;
    Particle& sun = particles[0];

    for (size_t i = 1; i < particles.size(); ++i) {
        if (particles[i].name == "Jupiter" || particles[i].name == "Mars") continue; // Planets don't get accreted
        
        double dx = sun.x - particles[i].x;
        double dy = sun.y - particles[i].y;
        double dz = sun.z - particles[i].z;
        double r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < ACCRETION_RADIUS_SQ) { // Use predefined accretion radius
            to_remove_indices.push_back(i);
            sun.mass += particles[i].mass; 
        }
    }
    
    std::sort(to_remove_indices.rbegin(), to_remove_indices.rend());
    for (int index : to_remove_indices) {
        particles.erase(particles.begin() + index);
    }
}


// --- Initial Conditions & Main ---
std::vector<Particle> init_solar_system_detailed(
    double central_mass_val, int num_asteroids, double belt_min_r_val,
    double belt_max_r_val, double jupiter_mass_val, double jupiter_r_val, 
    double mars_mass_val, double mars_r_val) {
    std::vector<Particle> particles_vec;
    std::mt19937 rng(std::random_device{}());
    int current_id = 0;

    particles_vec.emplace_back(current_id++, "Sun", 0.0, 0.0, 0.0, central_mass_val);

    double jupiter_speed = std::sqrt(G_CONST * (central_mass_val + jupiter_mass_val) / jupiter_r_val);
    particles_vec.emplace_back(current_id++, "Jupiter", jupiter_r_val, 0.0, 0.0, jupiter_mass_val, 0.0, jupiter_speed, 0.0);
    
    double mars_speed = std::sqrt(G_CONST * (central_mass_val + mars_mass_val) / mars_r_val);
    // Place Mars at a different angle initially to avoid perfect alignment if desired
    double mars_angle = M_PI / 3.0; // Example: 60 degrees ahead
    particles_vec.emplace_back(current_id++, "Mars", 
                               mars_r_val * std::cos(mars_angle), 
                               mars_r_val * std::sin(mars_angle), 0.0, 
                               mars_mass_val, 
                               -mars_speed * std::sin(mars_angle), 
                               mars_speed * std::cos(mars_angle), 0.0);

    std::uniform_real_distribution<double> radius_dist(belt_min_r_val, belt_max_r_val);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> mass_dist_asteroid(1e-10, 1e-9); // Asteroids are very light

    const Particle& sun = particles_vec[0];
    const Particle& jupiter = particles_vec[1];
    const Particle& mars = particles_vec[2];

    for (int i = 0; i < num_asteroids; ++i) {
        double r_a = radius_dist(rng);
        double angle_a = angle_dist(rng);
        double asteroid_mass_val = mass_dist_asteroid(rng);
        double x_a = r_a * std::cos(angle_a);
        double y_a = r_a * std::sin(angle_a);
        
        // Initial velocity calculation trying to account for Sun, Jupiter, Mars for slightly more stable start
        // This is still an approximation. True stable orbits in a 3+ body system are complex.
        double speed_sq_sun = G_CONST * sun.mass / r_a;
        
        double dx_j = x_a - jupiter.x; double dy_j = y_a - jupiter.y; double r_j_sq = dx_j*dx_j + dy_j*dy_j;
        double speed_sq_jupiter_pert = 0;
        if (r_j_sq > 1e-4) speed_sq_jupiter_pert = (G_CONST * jupiter.mass / r_j_sq) * (x_a*dx_j + y_a*dy_j)/r_a;
        
        double dx_m = x_a - mars.x; double dy_m = y_a - mars.y; double r_m_sq = dx_m*dx_m + dy_m*dy_m;
        double speed_sq_mars_pert = 0;
        if (r_m_sq > 1e-4) speed_sq_mars_pert = (G_CONST * mars.mass / r_m_sq) * (x_a*dx_m + y_a*dy_m)/r_a;
        
        double total_speed_sq = speed_sq_sun - speed_sq_jupiter_pert - speed_sq_mars_pert;
        double orbital_speed_a = (total_speed_sq > 0) ? std::sqrt(total_speed_sq) : std::sqrt(speed_sq_sun); // Fallback to Sun only if calculation is off

        double vx_a = -orbital_speed_a * std::sin(angle_a);
        double vy_a =  orbital_speed_a * std::cos(angle_a);
        particles_vec.emplace_back(current_id++, "Asteroid", x_a, y_a, 0.0, asteroid_mass_val, vx_a, vy_a, 0.0);
    }
    std::cout << "Initialized Solar System with " << current_id << " bodies." << std::endl;
    return particles_vec;
}

int main(int argc, char* argv[]) {
    int n_asteroids_val = 5000;      // Number of asteroids
    long long total_steps = 100000; // Total simulation steps for long-term evolution
    double dt_val = 0.005;           // Time step for RK4 (can be larger than for Leapfrog, but still needs care)

    double central_m = 1000.0;       // Mass of the Sun
    double jupiter_m = 1.0;          // Mass of Jupiter (Sun's mass / 1000)
    double jupiter_r_orbit = 5.2;    // Jupiter's orbital radius (AU-like units)
    double mars_m = jupiter_m * 0.00033; // Mars's mass, relative
    double mars_r_orbit = 1.5;       // Mars's orbital radius

    // Asteroid belt range designed to cover important Kirkwood Gaps
    double belt_min_radius = 2.0;    // Inner edge of asteroid belt
    double belt_max_radius = 3.5;    // Outer edge of asteroid belt
    
    double tree_domain_size = jupiter_r_orbit * 2.5; // Domain size for Octree
    double bh_theta = 0.5;                           // Barnes-Hut opening angle for accuracy

    for (int i = 1; i < argc; ++i) {
        std::string arg_str = argv[i];
        if (arg_str == "-n" && i + 1 < argc) n_asteroids_val = std::stoi(argv[++i]);
        else if (arg_str == "-steps" && i + 1 < argc) total_steps = std::stoll(argv[++i]);
        else if (arg_str == "-dt" && i + 1 < argc) dt_val = std::stod(argv[++i]);
        else if (arg_str == "-theta" && i + 1 < argc) bh_theta = std::stod(argv[++i]);
    }

    std::ofstream traj_output_file("trajectories_fmm_resonance.csv");
    traj_output_file << "step,particle_id,name,x,y,z,mass,semi_major_axis\n";

    std::vector<Particle> all_particles = init_solar_system_detailed(
        central_m, n_asteroids_val, belt_min_radius, belt_max_radius, 
        jupiter_m, jupiter_r_orbit, mars_m, mars_r_orbit);

    std::cout << "Starting high-fidelity Solar System simulation (FMM Structure with Octree Gravity, RK4 Integrator)..." << std::endl;
    std::cout << "Target Steps: " << total_steps << ", dt: " << dt_val << ", Asteroids: " << n_asteroids_val << std::endl;
    std::cout << "Jupiter at r=" << jupiter_r_orbit << ", Mars at r=" << mars_r_orbit 
              << ". Asteroid belt from r=" << belt_min_radius << " to " << belt_max_radius << "." << std::endl;
    
    for (long long current_step = 0; current_step < total_steps; ++current_step) {
        // Output data at intervals (e.g., every 5000 steps)
        if (current_step % 500 == 0) { 
            for (const auto& p_obj : all_particles) {
                // Sun (id 0) doesn't need its SMA calculated or output in this context
                double sma = (p_obj.id != 0 && p_obj.mass < central_m * 0.1) ? p_obj.semi_major_axis(central_m) : 0.0;
                traj_output_file << current_step << "," << p_obj.id << "," << p_obj.name << "," 
                                 << p_obj.x << "," << p_obj.y << "," << p_obj.z << "," 
                                 << p_obj.mass << "," << sma << "\n";
            }
            std::cout << "Step " << current_step << "/" << total_steps 
                      << " | Approx. Physical Time: " << current_step * dt_val 
                      << " | Particles: " << all_particles.size() << std::endl;
             if (current_step > 0) traj_output_file.flush(); // Flush occasionally
        }
        
        rk4_step(all_particles, dt_val, bh_theta, tree_domain_size);
        
        // Optional: Handle accretion (removing particles that get too close to the Sun)
        // This is simplified and might not be active if ACCRETION_RADIUS is very small
        if (current_step % 100 == 0) { // Check for accretion less frequently
            handle_accretion(all_particles);
        }
    }
    
    // Final output
    for (const auto& p_obj : all_particles) {
        double sma = (p_obj.id != 0 && p_obj.mass < central_m * 0.1) ? p_obj.semi_major_axis(central_m) : 0.0;
        traj_output_file << total_steps << "," << p_obj.id << "," << p_obj.name << "," 
                         << p_obj.x << "," << p_obj.y << "," << p_obj.z << "," 
                         << p_obj.mass << "," << sma << "\n";
    }

    std::cout << "Simulation finished. Output file: trajectories_fmm_resonance.csv" << std::endl;
    traj_output_file.close();
    return 0;
}