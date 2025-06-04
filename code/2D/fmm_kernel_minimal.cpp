/*
fmm_kernel_minimal.cpp
=====================
Minimal FMM implementation that should compile reliably
Falls back to direct method if FMM fails
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <memory>

// Only include OpenMP if available
#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP
#endif

namespace py = pybind11;

// Fallback direct N-body calculation
py::tuple direct_fallback(const std::vector<double>& x, const std::vector<double>& y, 
                         const std::vector<double>& m, double G, double soft) {
    
    const size_t N = x.size();
    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();
    
    const double soft2 = soft * soft;

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            
            const double dx = x[j] - x[i];
            const double dy = y[j] - y[i];
            const double r2 = dx * dx + dy * dy + soft2;
            const double inv_r3 = 1.0 / std::pow(r2, 1.5);
            
            fx += G * m[j] * dx * inv_r3;
            fy += G * m[j] * dy * inv_r3;
        }
        
        pax(i) = fx;
        pay(i) = fy;
    }
    
    return py::make_tuple(ax, ay);
}

// Simple tree node for minimal FMM
struct FMMNode {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    bool is_leaf;
    std::array<std::unique_ptr<FMMNode>, 4> children;
    
    // Simplified multipole/local expansions (just monopole and dipole)
    double monopole;
    double dipole_x, dipole_y;
    double local_monopole;
    double local_dipole_x, local_dipole_y;
    
    FMMNode(double x, double y, double s, int lev = 0) 
        : cx(x), cy(y), size(s), level(lev), is_leaf(true),
          monopole(0), dipole_x(0), dipole_y(0),
          local_monopole(0), local_dipole_x(0), local_dipole_y(0) {}
};

// Build simple quadtree
void fmm_subdivide(FMMNode* node, const std::vector<double>& x, const std::vector<double>& y, 
                   int max_particles = 16, int max_level = 8) {
    
    if ((int)node->particles.size() <= max_particles || node->level >= max_level) {
        return;
    }
    
    node->is_leaf = false;
    const double half_size = node->size * 0.5;
    
    // Create children
    node->children[0] = std::make_unique<FMMNode>(node->cx - half_size, node->cy - half_size, half_size, node->level + 1);
    node->children[1] = std::make_unique<FMMNode>(node->cx + half_size, node->cy - half_size, half_size, node->level + 1);
    node->children[2] = std::make_unique<FMMNode>(node->cx - half_size, node->cy + half_size, half_size, node->level + 1);
    node->children[3] = std::make_unique<FMMNode>(node->cx + half_size, node->cy + half_size, half_size, node->level + 1);
    
    // Distribute particles
    for (int id : node->particles) {
        const int quad = (x[id] > node->cx ? 1 : 0) + (y[id] > node->cy ? 2 : 0);
        node->children[quad]->particles.push_back(id);
    }
    
    node->particles.clear();
    
    // Recursively subdivide children
    for (auto& child : node->children) {
        if (child && !child->particles.empty()) {
            fmm_subdivide(child.get(), x, y, max_particles, max_level);
        }
    }
}

// Compute simplified multipole moments (P2M and M2M)
void fmm_upward(FMMNode* node, const std::vector<double>& x, const std::vector<double>& y, 
                const std::vector<double>& m) {
    
    if (!node) return;
    
    if (node->is_leaf) {
        // P2M: Particles to multipole
        node->monopole = 0;
        node->dipole_x = 0;
        node->dipole_y = 0;
        
        for (int id : node->particles) {
            const double mass = m[id];
            const double dx = x[id] - node->cx;
            const double dy = y[id] - node->cy;
            
            node->monopole += mass;
            node->dipole_x += mass * dx;
            node->dipole_y += mass * dy;
        }
    } else {
        // M2M: Child to parent translation
        node->monopole = 0;
        node->dipole_x = 0;
        node->dipole_y = 0;
        
        for (auto& child : node->children) {
            if (child && !child->particles.empty()) {
                fmm_upward(child.get(), x, y, m);
                
                const double dx = child->cx - node->cx;
                const double dy = child->cy - node->cy;
                
                // Simple M2M translation
                node->monopole += child->monopole;
                node->dipole_x += child->dipole_x + child->monopole * dx;
                node->dipole_y += child->dipole_y + child->monopole * dy;
            }
        }
    }
}

// Simplified M2L translation
void fmm_m2l(FMMNode* target, FMMNode* source) {
    if (!target || !source || target == source) return;
    
    const double dx = source->cx - target->cx;
    const double dy = source->cy - target->cy;
    const double r2 = dx * dx + dy * dy;
    
    if (r2 < 1e-20) return;
    
    const double r = std::sqrt(r2);
    const double inv_r = 1.0 / r;
    const double inv_r3 = inv_r * inv_r * inv_r;
    
    // Simple M2L: monopole to local
    target->local_monopole += source->monopole * inv_r;
    target->local_dipole_x += source->monopole * dx * inv_r3;
    target->local_dipole_y += source->monopole * dy * inv_r3;
    
    // Dipole to local (simplified)
    target->local_monopole += (source->dipole_x * dx + source->dipole_y * dy) * inv_r3;
}

// Simplified interaction phase
void fmm_interact(FMMNode* node, FMMNode* root, double theta) {
    if (!node) return;
    
    // Simple interaction: interact with all well-separated nodes
    std::function<void(FMMNode*, FMMNode*)> traverse = [&](FMMNode* target, FMMNode* source) {
        if (!source || target == source) return;
        
        const double dx = source->cx - target->cx;
        const double dy = source->cy - target->cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        
        if (dist > 2.0 * std::max(target->size, source->size)) {
            // Well separated - use M2L
            fmm_m2l(target, source);
        } else if (!source->is_leaf) {
            // Not well separated and source has children - recurse
            for (auto& child : source->children) {
                if (child) traverse(target, child.get());
            }
        }
    };
    
    traverse(node, root);
    
    // Recurse to children
    for (auto& child : node->children) {
        if (child) fmm_interact(child.get(), root, theta);
    }
}

// Simplified L2L and force evaluation
void fmm_evaluate(FMMNode* node, const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<double>& m, std::vector<double>& fx, std::vector<double>& fy,
                  double G, double soft2) {
    
    if (!node) return;
    
    if (!node->is_leaf) {
        // L2L: pass local expansion to children
        for (auto& child : node->children) {
            if (child) {
                const double dx = child->cx - node->cx;
                const double dy = child->cy - node->cy;
                
                // Simple L2L translation
                child->local_monopole += node->local_monopole;
                child->local_dipole_x += node->local_dipole_x;
                child->local_dipole_y += node->local_dipole_y;
                
                fmm_evaluate(child.get(), x, y, m, fx, fy, G, soft2);
            }
        }
        return;
    }
    
    // Leaf node: evaluate forces
    for (int i : node->particles) {
        double force_x = 0.0, force_y = 0.0;
        
        // Direct interactions within same leaf
        for (int j : node->particles) {
            if (i != j) {
                const double dx = x[j] - x[i];
                const double dy = y[j] - y[i];
                const double r2 = dx * dx + dy * dy + soft2;
                const double inv_r3 = 1.0 / std::pow(r2, 1.5);
                force_x += G * m[j] * dx * inv_r3;
                force_y += G * m[j] * dy * inv_r3;
            }
        }
        
        // Local expansion contribution (simplified)
        const double dx = x[i] - node->cx;
        const double dy = y[i] - node->cy;
        
        force_x += G * node->local_dipole_x;
        force_y += G * node->local_dipole_y;
        
        fx[i] += force_x;
        fy[i] += force_y;
    }
}

// Main FMM function
py::tuple fmm_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                  double domain, double theta = 0.5, double G = 1.0, double soft = 0.05) {
    
    const size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }
    
    // Copy input data to vectors
    std::vector<double> vx(x.data(), x.data() + N);
    std::vector<double> vy(y.data(), y.data() + N);
    std::vector<double> vm(m.data(), m.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    
    try {
        // Build tree
        auto root = std::make_unique<FMMNode>(0.0, 0.0, domain);
        root->particles.resize(N);
        for (size_t i = 0; i < N; ++i) {
            root->particles[i] = i;
        }
        
        // FMM algorithm
        fmm_subdivide(root.get(), vx, vy, 16, 8);
        fmm_upward(root.get(), vx, vy, vm);
        fmm_interact(root.get(), root.get(), theta);
        fmm_evaluate(root.get(), vx, vy, vm, fx, fy, G, soft * soft);
        
    } catch (...) {
        // If FMM fails, fall back to direct method
        return direct_fallback(vx, vy, vm, G, soft);
    }
    
    // Copy results to NumPy arrays
    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }
    
    return py::make_tuple(ax_out, ay_out);
}

// Python module definition
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "Minimal FMM kernel with direct fallback";
    
    m.def("fmm_omp", &fmm_omp,
          "Fast Multipole Method force calculation",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);

#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
