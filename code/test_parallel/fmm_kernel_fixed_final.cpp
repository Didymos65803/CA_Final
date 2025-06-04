// fmm_kernel_fixed_final.cpp
// Simplified but correct FMM implementation focusing on parallelization and correctness

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Particle structure
struct Particle {
    double x, y, mass;
    double ax, ay;
    int id;
    
    Particle() : x(0), y(0), mass(0), ax(0), ay(0), id(-1) {}
    Particle(double x_, double y_, double m_, int id_) : x(x_), y(y_), mass(m_), ax(0), ay(0), id(id_) {}
};

// Tree node for spatial decomposition
struct TreeNode {
    double center_x, center_y;
    double size;
    int level;
    bool is_leaf;
    
    // Child nodes (quadtree)
    std::unique_ptr<TreeNode> children[4];
    
    // Particles (for leaf nodes)
    std::vector<int> particle_ids;
    
    // Center of mass approximation
    double total_mass;
    double com_x, com_y;
    
    TreeNode(double cx, double cy, double sz, int lvl) 
        : center_x(cx), center_y(cy), size(sz), level(lvl), is_leaf(true),
          total_mass(0), com_x(0), com_y(0) {
    }
};

class SimplifiedFMM {
private:
    std::vector<Particle> particles;
    std::unique_ptr<TreeNode> root;
    double domain_size;
    double theta;
    int max_particles_per_leaf;
    double eps_squared;
    double G_constant;
    int max_level;
    
    // Build the tree recursively
    void build_tree(TreeNode* node, const std::vector<int>& particle_ids, int current_level) {
        if (particle_ids.size() <= static_cast<size_t>(max_particles_per_leaf) || current_level >= max_level) {
            node->is_leaf = true;
            node->particle_ids = particle_ids;
            compute_center_of_mass(node);
            return;
        }
        
        node->is_leaf = false;
        
        // Create four children
        const double half_size = node->size * 0.5;
        const double quarter_size = half_size * 0.5;
        
        node->children[0] = std::make_unique<TreeNode>(
            node->center_x - quarter_size, node->center_y - quarter_size, half_size, current_level + 1);
        node->children[1] = std::make_unique<TreeNode>(
            node->center_x + quarter_size, node->center_y - quarter_size, half_size, current_level + 1);
        node->children[2] = std::make_unique<TreeNode>(
            node->center_x - quarter_size, node->center_y + quarter_size, half_size, current_level + 1);
        node->children[3] = std::make_unique<TreeNode>(
            node->center_x + quarter_size, node->center_y + quarter_size, half_size, current_level + 1);
        
        // Distribute particles to children
        std::vector<std::vector<int>> child_particles(4);
        
        for (int pid : particle_ids) {
            const Particle& p = particles[pid];
            int child_idx = 0;
            if (p.x > node->center_x) child_idx += 1;
            if (p.y > node->center_y) child_idx += 2;
            child_particles[child_idx].push_back(pid);
        }
        
        // Recursively build children
        for (int i = 0; i < 4; ++i) {
            if (!child_particles[i].empty()) {
                build_tree(node->children[i].get(), child_particles[i], current_level + 1);
            }
        }
        
        // Compute center of mass for this node
        compute_center_of_mass(node);
    }
    
    // Compute center of mass for a node
    void compute_center_of_mass(TreeNode* node) {
        node->total_mass = 0.0;
        node->com_x = 0.0;
        node->com_y = 0.0;
        
        if (node->is_leaf) {
            // Compute from particles
            for (int pid : node->particle_ids) {
                const Particle& p = particles[pid];
                node->total_mass += p.mass;
                node->com_x += p.mass * p.x;
                node->com_y += p.mass * p.y;
            }
        } else {
            // Compute from children
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    node->total_mass += node->children[i]->total_mass;
                    node->com_x += node->children[i]->total_mass * node->children[i]->com_x;
                    node->com_y += node->children[i]->total_mass * node->children[i]->com_y;
                }
            }
        }
        
        if (node->total_mass > 0.0) {
            node->com_x /= node->total_mass;
            node->com_y /= node->total_mass;
        }
    }
    
    // Compute force on a particle using the tree
    void compute_particle_force(int particle_id, TreeNode* node) {
        if (!node || node->total_mass <= 0.0) return;
        
        const Particle& p = particles[particle_id];
        const double dx = p.x - node->com_x;
        const double dy = p.y - node->com_y;
        const double r2 = dx*dx + dy*dy + eps_squared;
        const double distance = std::sqrt(r2);
        
        // Barnes-Hut opening criterion
        if (node->is_leaf || (node->size / distance) < theta) {
            // Use this node as a single mass
            if (r2 > eps_squared) {
                const double inv_r = 1.0 / distance;
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double force_mag = G_constant * node->total_mass * inv_r3;
                
                particles[particle_id].ax -= force_mag * dx;
                particles[particle_id].ay -= force_mag * dy;
            }
        } else {
            // Recurse to children
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    compute_particle_force(particle_id, node->children[i].get());
                }
            }
        }
    }
    
    // Handle particle-particle interactions for particles in the same leaf
    void compute_direct_interactions(TreeNode* node) {
        if (!node || !node->is_leaf) return;
        
        const auto& pids = node->particle_ids;
        const size_t n = pids.size();
        
        // Direct interactions within the leaf
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const int pid_i = pids[i];
                const int pid_j = pids[j];
                
                const double dx = particles[pid_i].x - particles[pid_j].x;
                const double dy = particles[pid_i].y - particles[pid_j].y;
                const double r2 = dx*dx + dy*dy + eps_squared;
                
                if (r2 > eps_squared) {
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    
                    const double force_mag_i = G_constant * particles[pid_j].mass * inv_r3;
                    const double force_mag_j = G_constant * particles[pid_i].mass * inv_r3;
                    
                    // Apply Newton's third law
                    particles[pid_i].ax -= force_mag_i * dx;
                    particles[pid_i].ay -= force_mag_i * dy;
                    particles[pid_j].ax += force_mag_j * dx;
                    particles[pid_j].ay += force_mag_j * dy;
                }
            }
        }
    }
    
    // Traverse tree and handle direct interactions
    void handle_direct_interactions(TreeNode* node) {
        if (!node) return;
        
        if (node->is_leaf) {
            compute_direct_interactions(node);
        } else {
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    handle_direct_interactions(node->children[i].get());
                }
            }
        }
    }
    
public:
    void solve_forces(const double* x, const double* y, const double* mass, int n,
                      double domain, double theta_param, int max_particles_per_leaf_param,
                      double epsilon, double G,
                      double* ax, double* ay) {
        
        // Initialize parameters
        domain_size = domain;
        theta = theta_param;
        max_particles_per_leaf = std::max(4, max_particles_per_leaf_param);
        eps_squared = epsilon * epsilon;
        G_constant = G;
        max_level = std::max(3, static_cast<int>(std::log2(n / max_particles_per_leaf)) + 3);
        
        // Copy input data
        particles.resize(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            particles[i] = Particle(x[i], y[i], mass[i], i);
        }
        
        // For very small problems, use direct computation
        if (n < 50) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                double fx = 0.0, fy = 0.0;
                
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        const double dx = particles[i].x - particles[j].x;
                        const double dy = particles[i].y - particles[j].y;
                        const double r2 = dx*dx + dy*dy + eps_squared;
                        
                        if (r2 > eps_squared) {
                            const double inv_r = 1.0 / std::sqrt(r2);
                            const double inv_r3 = inv_r * inv_r * inv_r;
                            const double force_mag = G_constant * particles[j].mass * inv_r3;
                            
                            fx -= force_mag * dx;
                            fy -= force_mag * dy;
                        }
                    }
                }
                
                particles[i].ax = fx;
                particles[i].ay = fy;
            }
        } else {
            // Use simplified FMM for larger problems
            
            // Create particle list
            std::vector<int> all_particles(n);
            std::iota(all_particles.begin(), all_particles.end(), 0);
            
            // Build tree
            root = std::make_unique<TreeNode>(0.0, 0.0, domain_size, 0);
            build_tree(root.get(), all_particles, 0);
            
            // Initialize forces
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                particles[i].ax = 0.0;
                particles[i].ay = 0.0;
            }
            
            // Handle direct interactions within leaves (to avoid double-counting)
            handle_direct_interactions(root.get());
            
            // Compute forces using tree walk
            #pragma omp parallel for schedule(dynamic, 32)
            for (int i = 0; i < n; ++i) {
                // Find the leaf containing this particle
                TreeNode* leaf = find_leaf_containing_particle(i, root.get());
                
                // Compute force from tree, excluding the particle's own leaf
                compute_particle_force_excluding_leaf(i, root.get(), leaf);
            }
        }
        
        // Copy results back
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            ax[i] = particles[i].ax;
            ay[i] = particles[i].ay;
        }
    }
    
private:
    // Find the leaf node containing a specific particle
    TreeNode* find_leaf_containing_particle(int particle_id, TreeNode* node) {
        if (!node) return nullptr;
        
        if (node->is_leaf) {
            // Check if this leaf contains the particle
            for (int pid : node->particle_ids) {
                if (pid == particle_id) {
                    return node;
                }
            }
            return nullptr;
        }
        
        // Recurse to children
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                TreeNode* result = find_leaf_containing_particle(particle_id, node->children[i].get());
                if (result) return result;
            }
        }
        
        return nullptr;
    }
    
    // Compute force excluding a specific leaf (to avoid double-counting direct interactions)
    void compute_particle_force_excluding_leaf(int particle_id, TreeNode* node, TreeNode* exclude_leaf) {
        if (!node || node->total_mass <= 0.0 || node == exclude_leaf) return;
        
        const Particle& p = particles[particle_id];
        const double dx = p.x - node->com_x;
        const double dy = p.y - node->com_y;
        const double r2 = dx*dx + dy*dy + eps_squared;
        const double distance = std::sqrt(r2);
        
        // Barnes-Hut opening criterion
        if (node->is_leaf || (node->size / distance) < theta) {
            // Use this node as a single mass
            if (r2 > eps_squared && node != exclude_leaf) {
                const double inv_r = 1.0 / distance;
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double force_mag = G_constant * node->total_mass * inv_r3;
                
                #pragma omp atomic
                particles[particle_id].ax -= force_mag * dx;
                #pragma omp atomic
                particles[particle_id].ay -= force_mag * dy;
            }
        } else {
            // Recurse to children
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    compute_particle_force_excluding_leaf(particle_id, node->children[i].get(), exclude_leaf);
                }
            }
        }
    }
};

// Python interface function
void fmm_force(const py::array_t<double>& x_arr,
               const py::array_t<double>& y_arr,
               const py::array_t<double>& m_arr,
               int N,
               double domain_size,
               double theta,
               int maxLeaf,
               double eps,
               double G,
               py::array_t<double>& ax_arr,
               py::array_t<double>& ay_arr)
{
    // Get array accessors
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    // Validate array sizes
    if (N != x.shape(0) || N != y.shape(0) || N != m.shape(0) || 
        N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in fmm_force");
    }
    
    try {
        SimplifiedFMM fmm_solver;
        
        // Get data pointers
        const double* x_ptr = x.data(0);
        const double* y_ptr = y.data(0);
        const double* m_ptr = m.data(0);
        double* ax_ptr = ax.mutable_data(0);
        double* ay_ptr = ay.mutable_data(0);
        
        // Solve the system
        fmm_solver.solve_forces(x_ptr, y_ptr, m_ptr, N,
                                domain_size, theta, maxLeaf,
                                eps, G, ax_ptr, ay_ptr);
                                
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("FMM computation failed: ") + e.what());
    }
}

// Python module definition
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "Simplified but correct 2D FMM kernel with proper Barnes-Hut algorithm";
    m.def("fmm_force",
          &fmm_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("N"),
          py::arg("domain_size") = 50.0,
          py::arg("theta") = 0.5,
          py::arg("maxLeaf") = 8,
          py::arg("eps") = 0.01,
          py::arg("G") = 1.0,
          py::arg("ax"),
          py::arg("ay"),
          "Compute gravitational forces using simplified Barnes-Hut FMM");
}
