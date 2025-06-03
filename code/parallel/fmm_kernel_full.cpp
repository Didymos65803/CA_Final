// fmm_kernel_full.cpp
// Optimized version with better memory management and parallelization

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Cache-aligned structure to avoid false sharing
struct alignas(64) ThreadLocalData {
    double ax;
    double ay;
    char padding[64 - 2*sizeof(double)];
};

struct FMMNode {
    double cx, cy, size;
    std::vector<int> particles;
    double mass, mx, my;
    bool is_leaf;
    std::unique_ptr<FMMNode> children[4];
    
    FMMNode(double _cx, double _cy, double _size) 
        : cx(_cx), cy(_cy), size(_size), mass(0.0), mx(0.0), my(0.0), is_leaf(true) {
        for (int i = 0; i < 4; ++i) {
            children[i] = nullptr;
        }
    }
};

class OptimizedFMM {
private:
    std::unique_ptr<FMMNode> root;
    const double* x_data;
    const double* y_data;
    const double* m_data;
    int N;
    double domain_size;
    double theta;
    int max_leaf;
    double eps;
    double G;
    
    void build_tree(FMMNode* node, int max_particles) {
        if (node->particles.size() <= static_cast<size_t>(max_particles)) {
            return;
        }
        
        node->is_leaf = false;
        double half_size = node->size * 0.5;
        
        // Create children
        node->children[0] = std::make_unique<FMMNode>(node->cx - half_size, node->cy - half_size, half_size);
        node->children[1] = std::make_unique<FMMNode>(node->cx + half_size, node->cy - half_size, half_size);
        node->children[2] = std::make_unique<FMMNode>(node->cx - half_size, node->cy + half_size, half_size);
        node->children[3] = std::make_unique<FMMNode>(node->cx + half_size, node->cy + half_size, half_size);
        
        // Distribute particles
        for (int pi : node->particles) {
            double px = x_data[pi];
            double py = y_data[pi];
            
            int quadrant = 0;
            if (px > node->cx) quadrant += 1;
            if (py > node->cy) quadrant += 2;
            
            node->children[quadrant]->particles.push_back(pi);
        }
        
        node->particles.clear();
        
        // Recursively build children
        for (int i = 0; i < 4; ++i) {
            if (node->children[i] && !node->children[i]->particles.empty()) {
                build_tree(node->children[i].get(), max_particles);
            }
        }
    }
    
    void compute_mass_center(FMMNode* node) {
        if (node->is_leaf) {
            double total_mass = 0.0;
            double mx_sum = 0.0;
            double my_sum = 0.0;
            
            for (int pi : node->particles) {
                double mi = m_data[pi];
                total_mass += mi;
                mx_sum += mi * x_data[pi];
                my_sum += mi * y_data[pi];
            }
            
            node->mass = total_mass;
            if (total_mass > 0.0) {
                node->mx = mx_sum / total_mass;
                node->my = my_sum / total_mass;
            } else {
                node->mx = node->cx;
                node->my = node->cy;
            }
        } else {
            double total_mass = 0.0;
            double mx_sum = 0.0;
            double my_sum = 0.0;
            
            for (int i = 0; i < 4; ++i) {
                if (node->children[i] && !node->children[i]->particles.empty()) {
                    compute_mass_center(node->children[i].get());
                    
                    double child_mass = node->children[i]->mass;
                    total_mass += child_mass;
                    mx_sum += child_mass * node->children[i]->mx;
                    my_sum += child_mass * node->children[i]->my;
                }
            }
            
            node->mass = total_mass;
            if (total_mass > 0.0) {
                node->mx = mx_sum / total_mass;
                node->my = my_sum / total_mass;
            } else {
                node->mx = node->cx;
                node->my = node->cy;
            }
        }
    }
    
    void evaluate_force(FMMNode* node, double tx, double ty, double& ax, double& ay) {
        if (!node || node->mass == 0.0) return;
        
        double dx = node->mx - tx;
        double dy = node->my - ty;
        double r2 = dx*dx + dy*dy + eps*eps;
        double r = std::sqrt(r2);
        
        if (node->is_leaf || (node->size / r) < theta) {
            if (r > 0.0) {
                double inv_r3 = G / (r2 * r);
                ax += node->mass * dx * inv_r3;
                ay += node->mass * dy * inv_r3;
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    evaluate_force(node->children[i].get(), tx, ty, ax, ay);
                }
            }
        }
    }
    
public:
    void compute_forces(const double* x, const double* y, const double* m, int n,
                       double domain, double th, int max_leaf_particles,
                       double epsilon, double gravity,
                       double* ax, double* ay) {
        
        x_data = x;
        y_data = y;
        m_data = m;
        N = n;
        domain_size = domain;
        theta = th;
        max_leaf = max_leaf_particles;
        eps = epsilon;
        G = gravity;
        
        // Create root node
        root = std::make_unique<FMMNode>(0.0, 0.0, domain_size);
        for (int i = 0; i < N; ++i) {
            root->particles.push_back(i);
        }
        
        // Build tree
        build_tree(root.get(), max_leaf);
        compute_mass_center(root.get());
        
        // Compute forces with optimized parallelization
        const int chunk_size = std::max(1, N / (omp_get_max_threads() * 4));
        
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < N; ++i) {
            ax[i] = 0.0;
            ay[i] = 0.0;
            evaluate_force(root.get(), x[i], y[i], ax[i], ay[i]);
        }
    }
};

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
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    if (N != x.shape(0) || N != y.shape(0) || N != m.shape(0) || 
        N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in fmm_force");
    }
    
    if (N <= 0) {
        throw std::runtime_error("Invalid particle count");
    }
    
    try {
        OptimizedFMM fmm;
        
        const double* x_ptr = x.data(0);
        const double* y_ptr = y.data(0);
        const double* m_ptr = m.data(0);
        double* ax_ptr = ax.mutable_data(0);
        double* ay_ptr = ay.mutable_data(0);
        
        fmm.compute_forces(x_ptr, y_ptr, m_ptr, N,
                          domain_size, theta, maxLeaf,
                          eps, G, ax_ptr, ay_ptr);
                          
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("FMM computation failed: ") + e.what());
    }
}

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "2D Fast Multipole Method (FMM) kernel (Optimized OpenMP)";
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
          py::arg("ay"));
}

