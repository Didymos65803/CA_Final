/*
force_kernel_minimal.cpp
========================
Minimal, robust version of force kernels that should compile on any system
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

// Simple direct N-body force calculation
py::tuple direct_omp(py::array_t<double> x,
                     py::array_t<double> y,
                     py::array_t<double> m,
                     double G = 1.0,
                     double soft = 0.05) {
    
    const ssize_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }
    
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();
    
    const double soft2 = soft * soft;

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (ssize_t i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        
        for (ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            
            const double dx = px(j) - px(i);
            const double dy = py_(j) - py_(i);
            const double r2 = dx * dx + dy * dy + soft2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            
            fx += G * pm(j) * dx * inv_r3;
            fy += G * pm(j) * dy * inv_r3;
        }
        
        pax(i) = fx;
        pay(i) = fy;
    }
    
    return py::make_tuple(ax, ay);
}

// Simple Barnes-Hut tree node
struct BHNode {
    double cx, cy, size;
    double mass, com_x, com_y;
    bool is_leaf;
    std::vector<int> particles;
    std::array<std::unique_ptr<BHNode>, 4> children;
    
    BHNode(double x, double y, double s) 
        : cx(x), cy(y), size(s), mass(0), com_x(0), com_y(0), is_leaf(true) {}
};

// Insert particle into Barnes-Hut tree
void bh_insert(BHNode* node, int particle_id, 
               const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& m) {
    
    if (node->is_leaf && node->particles.empty()) {
        // Empty leaf - just add particle
        node->particles.push_back(particle_id);
        node->mass = m[particle_id];
        node->com_x = x[particle_id];
        node->com_y = y[particle_id];
        return;
    }
    
    if (node->is_leaf && node->particles.size() < 8) {
        // Leaf with room - add particle and update center of mass
        const double old_mass = node->mass;
        const double new_mass = old_mass + m[particle_id];
        
        if (new_mass > 0) {
            node->com_x = (node->com_x * old_mass + x[particle_id] * m[particle_id]) / new_mass;
            node->com_y = (node->com_y * old_mass + y[particle_id] * m[particle_id]) / new_mass;
        }
        
        node->mass = new_mass;
        node->particles.push_back(particle_id);
        return;
    }
    
    // Need to subdivide
    if (node->is_leaf) {
        node->is_leaf = false;
        const double half_size = node->size * 0.5;
        
        // Create children
        node->children[0] = std::make_unique<BHNode>(node->cx - half_size, node->cy - half_size, half_size);
        node->children[1] = std::make_unique<BHNode>(node->cx + half_size, node->cy - half_size, half_size);
        node->children[2] = std::make_unique<BHNode>(node->cx - half_size, node->cy + half_size, half_size);
        node->children[3] = std::make_unique<BHNode>(node->cx + half_size, node->cy + half_size, half_size);
        
        // Redistribute existing particles
        std::vector<int> existing = node->particles;
        node->particles.clear();
        
        for (int id : existing) {
            const int quad = (x[id] > node->cx ? 1 : 0) + (y[id] > node->cy ? 2 : 0);
            bh_insert(node->children[quad].get(), id, x, y, m);
        }
    }
    
    // Insert new particle
    const int quad = (x[particle_id] > node->cx ? 1 : 0) + (y[particle_id] > node->cy ? 2 : 0);
    bh_insert(node->children[quad].get(), particle_id, x, y, m);
    
    // Update this node's center of mass
    const double old_mass = node->mass;
    const double new_mass = old_mass + m[particle_id];
    
    if (new_mass > 0) {
        node->com_x = (node->com_x * old_mass + x[particle_id] * m[particle_id]) / new_mass;
        node->com_y = (node->com_y * old_mass + y[particle_id] * m[particle_id]) / new_mass;
    }
    
    node->mass = new_mass;
}

// Compute force using Barnes-Hut approximation
void bh_force(const BHNode* node, double px, double py, double theta, double G, double soft2,
              double& fx, double& fy) {
    
    if (!node || node->mass == 0) return;
    
    const double dx = node->com_x - px;
    const double dy = node->com_y - py;
    const double r2 = dx * dx + dy * dy + soft2;
    
    if (r2 < 1e-20) return; // Avoid self-interaction
    
    const double r = std::sqrt(r2);
    
    if (node->is_leaf || (node->size / r < theta)) {
        // Use this node's approximation
        const double inv_r3 = 1.0 / (r2 * r);
        const double force = G * node->mass * inv_r3;
        fx += force * dx;
        fy += force * dy;
    } else {
        // Recurse to children
        for (const auto& child : node->children) {
            if (child) {
                bh_force(child.get(), px, py, theta, G, soft2, fx, fy);
            }
        }
    }
}

// Barnes-Hut method
py::tuple bh_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                 double domain, double theta = 0.5, double G = 1.0, double soft = 0.05) {
    
    const size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }
    
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    // Copy data to vectors for tree building
    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = px(i);
        vy[i] = py_(i);
        vm[i] = pm(i);
    }

    // Build Barnes-Hut tree
    auto root = std::make_unique<BHNode>(0.0, 0.0, domain);
    
    try {
        for (size_t i = 0; i < N; ++i) {
            bh_insert(root.get(), i, vx, vy, vm);
        }
    } catch (...) {
        // If tree building fails, fall back to direct method
        return direct_omp(x, y, m, G, soft);
    }

    // Compute forces
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
        
        try {
            bh_force(root.get(), vx[i], vy[i], theta, G, soft2, fx, fy);
        } catch (...) {
            // If BH force computation fails, use direct calculation for this particle
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                const double dx = vx[j] - vx[i];
                const double dy = vy[j] - vy[i];
                const double r2 = dx * dx + dy * dy + soft2;
                const double inv_r3 = 1.0 / std::pow(r2, 1.5);
                fx += G * vm[j] * dx * inv_r3;
                fy += G * vm[j] * dy * inv_r3;
            }
        }
        
        pax(i) = fx;
        pay(i) = fy;
    }

    return py::make_tuple(ax, ay);
}

// Python module definition
PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "Minimal N-body force kernels";
    
    m.def("direct_omp", &direct_omp,
          "Direct N-body force calculation",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("G") = 1.0, py::arg("soft") = 0.05);
    
    m.def("bh_omp", &bh_omp,
          "Barnes-Hut N-body force calculation",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);

#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
