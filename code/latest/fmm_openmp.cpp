// fmm_openmp_fixed.cpp – Fixed Barnes–Hut FMM with proper OpenMP scaling
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// ---------- Optimized data structures --------------------------------------------------
struct Body { 
    double x, y, m;
    double ax, ay;
    
    Body() : x(0), y(0), m(0), ax(0), ay(0) {}
    Body(double x_, double y_, double m_) : x(x_), y(y_), m(m_), ax(0), ay(0) {}
};

struct Node {
    double cx, cy, size;           // center and size
    double mass, cmx, cmy;         // total mass and center of mass
    bool leaf;
    std::vector<int> particle_ids;
    std::unique_ptr<Node> children[4];
    int depth;                     // Track depth for better parallelization
    
    Node() : cx(0), cy(0), size(0), mass(0), cmx(0), cmy(0), leaf(true), depth(0) {
        for(int i = 0; i < 4; ++i) children[i] = nullptr;
    }
    
    Node(double cx_, double cy_, double size_, int depth_ = 0) 
        : cx(cx_), cy(cy_), size(size_), mass(0), cmx(0), cmy(0), leaf(true), depth(depth_) {
        for(int i = 0; i < 4; ++i) children[i] = nullptr;
    }
};

constexpr int MAX_PARTICLES_PER_LEAF = 16;  // Increased for better parallelization
constexpr double MIN_NODE_SIZE = 1e-8;      // Prevent infinite recursion
constexpr int MIN_PARTICLES_FOR_PARALLEL = 1000;  // Threshold for using OpenMP

// ---------- Enhanced tree construction with better parallelization ---------------------
class EnhancedTree {
private:
    std::unique_ptr<Node> root;
    std::vector<Body>& bodies;
    
    void subdivide_node(Node* node) {
        if (node->size < MIN_NODE_SIZE || node->particle_ids.size() <= MAX_PARTICLES_PER_LEAF) {
            return;
        }
        
        const double half_size = node->size * 0.5;
        const double quarter_size = half_size * 0.5;
        
        // Create children with depth tracking
        node->children[0] = std::make_unique<Node>(node->cx - quarter_size, node->cy - quarter_size, half_size, node->depth + 1);  // SW
        node->children[1] = std::make_unique<Node>(node->cx + quarter_size, node->cy - quarter_size, half_size, node->depth + 1);  // SE
        node->children[2] = std::make_unique<Node>(node->cx - quarter_size, node->cy + quarter_size, half_size, node->depth + 1);  // NW
        node->children[3] = std::make_unique<Node>(node->cx + quarter_size, node->cy + quarter_size, half_size, node->depth + 1);  // NE
        
        // Distribute particles to children
        for (int pid : node->particle_ids) {
            const Body& body = bodies[pid];
            int quadrant = 0;
            if (body.x > node->cx) quadrant += 1;
            if (body.y > node->cy) quadrant += 2;
            node->children[quadrant]->particle_ids.push_back(pid);
        }
        
        node->leaf = false;
        node->particle_ids.clear();
        
        // Recursively subdivide children - use OpenMP at shallow depths only
        bool use_parallel = (node->depth < 3) && (bodies.size() > MIN_PARTICLES_FOR_PARALLEL);
        
        if (use_parallel) {
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < 4; ++i) {
                if (node->children[i] && !node->children[i]->particle_ids.empty()) {
                    subdivide_node(node->children[i].get());
                }
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                if (node->children[i] && !node->children[i]->particle_ids.empty()) {
                    subdivide_node(node->children[i].get());
                }
            }
        }
    }
    
    void compute_mass_properties(Node* node) {
        if (!node) return;
        
        node->mass = 0.0;
        node->cmx = 0.0;
        node->cmy = 0.0;
        
        if (node->leaf) {
            for (int pid : node->particle_ids) {
                const Body& body = bodies[pid];
                node->mass += body.m;
                node->cmx += body.m * body.x;
                node->cmy += body.m * body.y;
            }
        } else {
            // Use OpenMP for mass computation at shallow depths
            bool use_parallel = (node->depth < 4) && (bodies.size() > MIN_PARTICLES_FOR_PARALLEL);
            
            if (use_parallel) {
                double child_masses[4] = {0, 0, 0, 0};
                double child_cmx[4] = {0, 0, 0, 0};
                double child_cmy[4] = {0, 0, 0, 0};
                
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < 4; ++i) {
                    if (node->children[i]) {
                        compute_mass_properties(node->children[i].get());
                        child_masses[i] = node->children[i]->mass;
                        child_cmx[i] = node->children[i]->mass * node->children[i]->cmx;
                        child_cmy[i] = node->children[i]->mass * node->children[i]->cmy;
                    }
                }
                
                for (int i = 0; i < 4; ++i) {
                    node->mass += child_masses[i];
                    node->cmx += child_cmx[i];
                    node->cmy += child_cmy[i];
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    if (node->children[i]) {
                        compute_mass_properties(node->children[i].get());
                        node->mass += node->children[i]->mass;
                        node->cmx += node->children[i]->mass * node->children[i]->cmx;
                        node->cmy += node->children[i]->mass * node->children[i]->cmy;
                    }
                }
            }
        }
        
        if (node->mass > 0) {
            node->cmx /= node->mass;
            node->cmy /= node->mass;
        }
    }
    
public:
    EnhancedTree(std::vector<Body>& bodies_, double domain_size) : bodies(bodies_) {
        root = std::make_unique<Node>(0.0, 0.0, domain_size, 0);
        
        root->particle_ids.reserve(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            root->particle_ids.push_back(static_cast<int>(i));
        }
        
        subdivide_node(root.get());
        compute_mass_properties(root.get());
    }
    
    Node* get_root() { return root.get(); }
};

// ---------- Optimized force calculation with better load balancing --------------------
inline bool should_open_node(const Body& particle, const Node* node, double theta_squared) {
    if (!node || node->mass == 0) return false;
    
    const double dx = particle.x - node->cmx;
    const double dy = particle.y - node->cmy;
    const double distance_squared = dx * dx + dy * dy;
    
    if (distance_squared < 1e-12) return true;
    
    const double size_squared = node->size * node->size;
    return size_squared > theta_squared * distance_squared;
}

void compute_force_from_tree(const std::vector<Body>& bodies, const Node* node, 
                           int particle_idx, double eps2, double theta_squared,
                           double& fx, double& fy) {
    if (!node || node->mass == 0) return;
    
    const Body& particle = bodies[particle_idx];
    
    if (node->leaf || !should_open_node(particle, node, theta_squared)) {
        if (node->leaf) {
            // Direct particle-particle interactions with vectorized operations
            for (int other_idx : node->particle_ids) {
                if (other_idx == particle_idx) continue;
                
                const Body& other = bodies[other_idx];
                const double dx = other.x - particle.x;
                const double dy = other.y - particle.y;
                const double r2 = dx * dx + dy * dy + eps2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double force_magnitude = other.m * inv_r3;
                
                fx += force_magnitude * dx;
                fy += force_magnitude * dy;
            }
        } else {
            // Center of mass approximation
            const double dx = node->cmx - particle.x;
            const double dy = node->cmy - particle.y;
            const double r2 = dx * dx + dy * dy + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double force_magnitude = node->mass * inv_r3;
            
            fx += force_magnitude * dx;
            fy += force_magnitude * dy;
        }
    } else {
        // Recursively traverse children
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                compute_force_from_tree(bodies, node->children[i].get(), particle_idx, 
                                      eps2, theta_squared, fx, fy);
            }
        }
    }
}

// ---------- Enhanced kernels with better parallelization strategies -------------------
void fmm_force_enhanced(const py::array_t<double>& x_arr,
                       const py::array_t<double>& y_arr,
                       const py::array_t<double>& m_arr,
                       double eps2, double domain_size, double theta,
                       py::array_t<double>& ax_arr,
                       py::array_t<double>& ay_arr) {
    
    const auto x = x_arr.unchecked<1>();
    const auto y = y_arr.unchecked<1>();
    const auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = x.shape(0);
    
    std::vector<Body> bodies(N);
    for (int i = 0; i < N; ++i) {
        bodies[i] = Body(x(i), y(i), m(i));
    }
    
    // Build tree (mostly serial, but some parallel components)
    EnhancedTree tree(bodies, domain_size);
    Node* root = tree.get_root();
    
    const double theta_squared = theta * theta;
    
    // Parallel force computation with adaptive scheduling
    int chunk_size = std::max(1, N / (4 * omp_get_max_threads()));
    
    #pragma omp parallel for schedule(dynamic, chunk_size) if(N > MIN_PARTICLES_FOR_PARALLEL)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        compute_force_from_tree(bodies, root, i, eps2, theta_squared, fx, fy);
        bodies[i].ax = fx;
        bodies[i].ay = fy;
    }
    
    // Copy results back with parallel copy for large arrays
    #pragma omp parallel for schedule(static) if(N > MIN_PARTICLES_FOR_PARALLEL)
    for (int i = 0; i < N; ++i) {
        ax(i) = bodies[i].ax;
        ay(i) = bodies[i].ay;
    }
}

// ---------- Enhanced direct force kernel with better parallelization -----------------
void direct_force_enhanced(const py::array_t<double>& x_arr,
                          const py::array_t<double>& y_arr,
                          const py::array_t<double>& m_arr,
                          double eps2,
                          py::array_t<double>& ax_arr,
                          py::array_t<double>& ay_arr) {
    
    const auto x = x_arr.unchecked<1>();
    const auto y = y_arr.unchecked<1>();
    const auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = x.shape(0);
    
    // Initialize accelerations
    #pragma omp parallel for schedule(static) if(N > MIN_PARTICLES_FOR_PARALLEL)
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    // Use block-based parallelization to reduce atomic operations
    const int num_threads = omp_get_max_threads();
    const int block_size = std::max(1, N / (2 * num_threads));
    
    #pragma omp parallel if(N > MIN_PARTICLES_FOR_PARALLEL)
    {
        std::vector<double> local_ax(N, 0.0);
        std::vector<double> local_ay(N, 0.0);
        
        #pragma omp for schedule(dynamic, block_size) nowait
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const double dx = x(j) - x(i);
                const double dy = y(j) - y(i);
                const double r2 = dx * dx + dy * dy + eps2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                
                const double fx = m(j) * inv_r3 * dx;
                const double fy = m(j) * inv_r3 * dy;
                
                local_ax[i] += fx;
                local_ay[i] += fy;
                local_ax[j] -= fx;
                local_ay[j] -= fy;
            }
        }
        
        // Reduce local results to global arrays
        #pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                ax(i) += local_ax[i];
                ay(i) += local_ay[i];
            }
        }
    }
}

// ---------- Wrapper functions for backward compatibility ---------------------------
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& m,
                     double eps2, double domain, double theta,
                     py::array_t<double>& ax,
                     py::array_t<double>& ay) {
    fmm_force_enhanced(x, y, m, eps2, domain, theta, ax, ay);
}

void fmm_force(const py::array_t<double>& x,
               const py::array_t<double>& y,
               const py::array_t<double>& m,
               double eps2, double domain,
               py::array_t<double>& ax,
               py::array_t<double>& ay) {
    fmm_force_enhanced(x, y, m, eps2, domain, 0.6, ax, ay);
}

void direct_force(const py::array_t<double>& x,
                  const py::array_t<double>& y,
                  const py::array_t<double>& m,
                  double eps2,
                  py::array_t<double>& ax,
                  py::array_t<double>& ay) {
    direct_force_enhanced(x, y, m, eps2, ax, ay);
}

// ---------- Python bindings -------------------------------------------------------
PYBIND11_MODULE(fmm_openmp, m) {
    m.doc() = "Enhanced Barnes–Hut FMM with proper OpenMP scaling for larger problems";
    
    m.def("direct_force", &direct_force,
          "Enhanced O(N²) direct force calculation with better OpenMP scaling");
    
    m.def("fmm_force", &fmm_force,
          "Barnes–Hut FMM with default theta=0.6");
    
    m.def("fmm_force_theta", &fmm_force_theta,
          "Barnes–Hut FMM with configurable opening angle theta");
          
#ifdef _OPENMP
    m.def("get_max_threads", &omp_get_max_threads,
          "Get maximum number of OpenMP threads");
    
    m.def("get_num_threads", &omp_get_num_threads,
          "Get current number of OpenMP threads");
#else
    m.def("get_max_threads", []() { return 1; },
          "Get maximum number of OpenMP threads (OpenMP not available)");
    
    m.def("get_num_threads", []() { return 1; },
          "Get current number of OpenMP threads (OpenMP not available)");
#endif
}
