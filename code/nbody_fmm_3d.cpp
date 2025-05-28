// nbody_fmm_3d.cpp
// COMPILE WITH:
// g++ nbody_fmm_3d.cpp -o nbody_fmm_3d -O3 -std=c++17 -fopenmp -lm
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <complex>

// OpenMP include
#ifdef _OPENMP
#include <omp.h>
#endif

// --- Physical Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;
const double ACCRETION_RADIUS_FACTOR = 1.5;
const double ACCRETION_RADIUS = ACCRETION_RADIUS_FACTOR * SOFTENING;
const double ACCRETION_RADIUS_SQ = ACCRETION_RADIUS * ACCRETION_RADIUS;

// --- FMM Parameters ---
const int FMM_TERMS = 10; // Number of terms in multipole/local expansions (p)

// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double mass;
    bool active = true; // To mark for accretion

    Particle(int _id, double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
    : id(_id), x(_x), y(_y), z(_z),
      vx(_vx), vy(_vy), vz(_vz),
      ax(0.0), ay(0.0), az(0.0),
      mass(_mass) {}
};

class OctreeNode {
public:
    double cx, cy, cz, size;
    int level;
    std::vector<std::unique_ptr<OctreeNode>> children;
    std::vector<Particle*> node_particles;

    double total_mass;
    double com_x, com_y, com_z;

    bool is_leaf;
    bool is_empty;

    // FMM expansions
    std::vector<std::complex<double>> multipole;
    std::vector<std::complex<double>> local;

    OctreeNode(double center_x, double center_y, double center_z, double s, int l)
        : cx(center_x), cy(center_y), cz(center_z), size(s), level(l),
          total_mass(0.0), com_x(0.0), com_y(0.0), com_z(0.0),
          is_leaf(true), is_empty(true) {
        // CORRECTED LINE: Default-construct 8 null unique_ptrs.
        children.resize(8);
        multipole.resize(FMM_TERMS, {0.0, 0.0});
        local.resize(FMM_TERMS, {0.0, 0.0});
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
                        cx + k * offset, cy + j * offset, cz + i * offset, child_size, level + 1);
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
                subdivide();
                children[get_child_index(node_particles[0])]->insert(node_particles[0]); // Re-insert existing particle
                node_particles.clear();
                children[get_child_index(p)]->insert(p); // Insert new particle
            }
        } else {
            int child_idx = get_child_index(p);
            if (!children[child_idx]) { // Should not happen with subdivide, but as a safeguard
                 subdivide();
            }
            children[child_idx]->insert(p);
        }
        is_empty = false;
        // Update COM and mass
        double new_total_mass = total_mass + p->mass;
        com_x = (com_x * total_mass + p->x * p->mass) / new_total_mass;
        com_y = (com_y * total_mass + p->y * p->mass) / new_total_mass;
        com_z = (com_z * total_mass + p->z * p->mass) / new_total_mass;
        total_mass = new_total_mass;
    }
};

// --- FMM Core Functions ---

// P2M: Particle to Multipole
void p2m(OctreeNode* node) {
    if (node->is_empty) return;
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) p2m(child.get());
        }
    } else {
        std::fill(node->multipole.begin(), node->multipole.end(), std::complex<double>(0,0));
        for (const auto* p : node->node_particles) {
            // NOTE: This is a simplified 2D FMM logic for the XY plane.
            // A full 3D FMM requires spherical harmonics and is significantly more complex.
            std::complex<double> dz(p->x - node->cx, p->y - node->cy);
            node->multipole[0] += p->mass;
            for (int k = 1; k < FMM_TERMS; ++k) {
                node->multipole[k] += -p->mass * std::pow(dz, k) / (double)k;
            }
        }
    }
}

// M2M: Multipole to Multipole
void m2m(OctreeNode* node) {
    if (node->is_empty || node->is_leaf) return;

    for (const auto& child : node->children) {
        if (!child || child->is_empty) continue;
        m2m(child.get()); // Recurse first

        std::complex<double> d_center(child->cx - node->cx, child->cy - node->cy);
        // Simplified M2M translation
        for (int j = 0; j < FMM_TERMS; ++j) {
            node->multipole[j] += child->multipole[j];
            for (int k = 1; k < j; ++k) {
                node->multipole[j] += child->multipole[k] * std::pow(d_center, j-k);
            }
        }
    }
}


// M2L: Multipole to Local
void m2l(OctreeNode* node, OctreeNode* source_node) {
    if (node == source_node || node->is_empty || source_node->is_empty) return;

    double dx = source_node->cx - node->cx;
    double dy = source_node->cy - node->cy;
    double dist_sq = dx*dx + dy*dy;

    // Well-separated condition (theta criterion)
    if (node->size * node->size < 0.5 * 0.5 * dist_sq) {
        std::complex<double> d_center(dx, dy);
        if (std::abs(d_center) < 1e-9) return; // Avoid division by zero
        // Simplified M2L translation
        for (int j = 0; j < FMM_TERMS; ++j) {
            for (int k = 0; k < FMM_TERMS; ++k) {
                node->local[j] += source_node->multipole[k] / std::pow(d_center, k + j + 1);
            }
        }
        return;
    }

    // If not well-separated, recurse to children
    if (!node->is_leaf && !source_node->is_leaf) {
        for(const auto& child : node->children) {
            for(const auto& source_child : source_node->children) {
                if(child && source_child) {
                    m2l(child.get(), source_child.get());
                }
            }
        }
    }
}


// L2L: Local to Local
void l2l(OctreeNode* node) {
    if (node->is_empty || node->is_leaf) return;

    for (const auto& child : node->children) {
        if (!child || child->is_empty) continue;
        std::complex<double> d_center(child->cx - node->cx, child->cy - node->cy);
        // Simplified L2L translation
        for (int j = 0; j < FMM_TERMS; ++j) {
            for (int k = j; k < FMM_TERMS; ++k) {
                child->local[j] += node->local[k] * std::pow(-d_center, k - j);
            }
        }
        l2l(child.get());
    }
}

// L2P: Local to Particle
void l2p(OctreeNode* node) {
    if (node->is_empty) return;

    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) l2p(child.get());
        }
    } else {
        for (auto* p : node->node_particles) {
            std::complex<double> potential_deriv(0, 0);
            std::complex<double> dz(p->x - node->cx, p->y - node->cy);
            for (int k = 1; k < FMM_TERMS; ++k) {
                potential_deriv += (double)k * node->local[k-1] * std::pow(dz, k-1);
            }
            // Force is derivative of potential
            // This force is for the simplified 2D FMM logic. A 3D force is added in the near-field calc.
            p->ax += G_CONST * potential_deriv.real();
            p->ay += G_CONST * potential_deriv.imag();
        }
    }
}

void compute_forces_direct(Particle& p1, Particle& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dz = p2.z - p1.z;
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < SOFT2) r2 = SOFT2;
    double r = std::sqrt(r2);
    double f_over_r = G_CONST * p2.mass / (r2 * r);
    p1.ax += f_over_r * dx;
    p1.ay += f_over_r * dy;
    p1.az += f_over_r * dz;
}

// Computes direct forces for adjacent nodes
void compute_near_field(OctreeNode* node1, OctreeNode* node2) {
    if (!node1 || !node2 || node1->is_empty || node2->is_empty) return;

    double dx = node2->cx - node1->cx;
    double dy = node2->cy - node1->cy;
    double dz = node2->cz - node1->cz;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    // Only compute for adjacent nodes (simplified check)
    if (dist > (node1->size + node2->size) * 0.75) return;

    if (node1->is_leaf && node2->is_leaf) {
        for (auto* p1 : node1->node_particles) {
            for (auto* p2 : node2->node_particles) {
                if (p1 != p2) compute_forces_direct(*p1, *p2);
            }
        }
    } else if (node1->is_leaf) {
        for (const auto& child2 : node2->children) {
            compute_near_field(node1, child2.get());
        }
    } else if (node2->is_leaf) {
        for (const auto& child1 : node1->children) {
            compute_near_field(child1.get(), node2);
        }
    } else {
        for (const auto& child1 : node1->children) {
            for (const auto& child2 : node2->children) {
                compute_near_field(child1.get(), child2.get());
            }
        }
    }
}

void traverse_fmm(std::vector<Particle>& particles, OctreeNode& root) {
    // 1. Upward Pass: P2M (from particles to multipoles at leaf nodes)
    p2m(&root);

    // 2. Upward Pass: M2M (from child multipoles to parent multipoles)
    m2m(&root);

    // 3. Downward Pass: M2L & L2L
    // This is a simplified interaction loop. A full FMM would have a more complex
    // way to determine the interaction list for M2L.
    m2l(&root, &root);
    l2l(&root);

    // 4. Final Force Evaluation
    l2p(&root); // L2P for far-field forces
    compute_near_field(&root, &root); // Direct summation for near-field forces
}

void compute_forces_fmm(std::vector<Particle>& particles, double domain_size) {
    if (particles.empty()) return;
    
    // Reset accelerations
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0; particles[i].ay = 0; particles[i].az = 0;
    }

    // 1. Build Tree
    OctreeNode root(0,0,0, domain_size * 2, 0);
    for (auto& p : particles) {
        if (p.active) {
            // Ensure particle is within the root bounds
            if (std::abs(p.x) < domain_size && std::abs(p.y) < domain_size && std::abs(p.z) < domain_size) {
                root.insert(&p);
            }
        }
    }
    
    // 2. Traverse tree to calculate forces
    traverse_fmm(particles, root);
}

// --- Collision, Accretion, and Integration ---
void handle_interactions(std::vector<Particle>& particles) {
    std::vector<int> to_remove;
    #pragma omp parallel
    {
        std::vector<int> local_to_remove;
        #pragma omp for
        for (size_t i = 0; i < particles.size(); ++i) {
            if (!particles[i].active) continue;
            for (size_t j = i + 1; j < particles.size(); ++j) {
                if (!particles[j].active) continue;

                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                double r2 = dx*dx + dy*dy + dz*dz;

                // Accretion check
                if (r2 < ACCRETION_RADIUS_SQ && r2 > 1e-9) {
                    Particle& p1 = particles[i];
                    Particle& p2 = particles[j];
                    
                    Particle& accretor = (p1.mass > p2.mass) ? p1 : p2;
                    Particle& accreted = (p1.mass > p2.mass) ? p2 : p1;
                    
                    // Mark the smaller one for removal
                    #pragma omp critical
                    if (accreted.active) { // Check again inside critical section
                        accreted.active = false;
                        local_to_remove.push_back(accreted.id);

                        // Conserve momentum
                        double combined_mass = accretor.mass + accreted.mass;
                        accretor.vx = (accretor.mass * accretor.vx + accreted.mass * accreted.vx) / combined_mass;
                        accretor.vy = (accretor.mass * accretor.vy + accreted.mass * accreted.vy) / combined_mass;
                        accretor.vz = (accretor.mass * accretor.vz + accreted.mass * accreted.vz) / combined_mass;
                        accretor.mass = combined_mass;
                    }
                }
            }
        }
    }

    // Now remove the inactive particles from the main list
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [](const Particle& p) { return !p.active; }), particles.end());
}


void leapfrog_step(std::vector<Particle>& particles, double dt, double domain_size) {
    // 1. Half-kick
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }

    // 2. Drift
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }

    // 3. Handle physical interactions (accretion)
    handle_interactions(particles);

    // 4. Update forces using FMM
    compute_forces_fmm(particles, domain_size);

    // 5. Second half-kick
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].vx += 0.5 * particles[i].ax * dt;
        particles[i].vy += 0.5 * particles[i].ay * dt;
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}


// --- Setup and Main Loop ---
std::vector<Particle> init_particles(int n, double max_radius) {
    std::vector<Particle> p_list;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> radius_dist(0, max_radius);
    std::uniform_real_distribution<double> mass_dist(1.0, 5.0);
    
    // Central massive object
    p_list.emplace_back(0, 0, 0, 0, 1000.0, 0, 0, 0);
    
    for (int i = 1; i < n; ++i) {
        double r = radius_dist(rng);
        double theta = std::uniform_real_distribution<double>(0, 2 * M_PI)(rng);
        double phi = std::acos(2 * std::uniform_real_distribution<double>()(rng) - 1);
        double x = r * std::sin(phi) * std::cos(theta);
        double y = r * std::sin(phi) * std::sin(theta);
        double z = r * std::cos(phi);
        p_list.emplace_back(i, x, y, z, mass_dist(rng), 0,0,0);
    }
    return p_list;
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

int main(int argc, char* argv[]) {
    int n_particles = 1000;
    int steps = 500;
    double dt = 0.01;
    double domain_size = 50.0;

    if (argc > 1) n_particles = std::stoi(argv[1]);
    if (argc > 2) steps = std::stoi(argv[2]);

    std::ofstream traj_file("trajectories_fmm.csv");
    traj_file << "step,particle_id,x,y,z,mass\n";

    std::ofstream energy_file("energy_fmm.csv");
    energy_file << "step,total_energy,num_particles\n";

    std::vector<Particle> particles = init_particles(n_particles, domain_size);
    
    // Initial force calculation
    compute_forces_fmm(particles, domain_size);

    std::cout << "Starting FMM simulation with N=" << n_particles << " for " << steps << " steps." << std::endl;

    for (int step = 0; step < steps; ++step) {
        // Write data
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.x << "," << p.y << "," << p.z << "," << p.mass << "\n";
        }
        energy_file << step << "," << std::fixed << std::setprecision(8) << system_energy(particles) << "," << particles.size() << "\n";
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << "/" << steps << " | Particles: " << particles.size() << std::endl;
        }

        leapfrog_step(particles, dt, domain_size);
    }

    std::cout << "Simulation finished." << std::endl;
    traj_file.close();
    energy_file.close();

    return 0;
}