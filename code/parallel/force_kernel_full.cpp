#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP
#endif

namespace py = pybind11;

// Direct N-body calculation (reference implementation)
py::tuple direct_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                     double G = 1.0, double soft = 0.05) {
    
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

// High-precision Barnes-Hut implementation
struct BHNode {
    double cx, cy, size;
    double total_mass;
    double center_of_mass_x, center_of_mass_y;
    bool is_leaf;
    std::vector<int> particle_indices;
    std::array<std::unique_ptr<BHNode>, 4> children;
    
    BHNode(double x, double y, double s) 
        : cx(x), cy(y), size(s), total_mass(0.0), 
          center_of_mass_x(0.0), center_of_mass_y(0.0), is_leaf(true) {}
    
    void clear() {
        particle_indices.clear();
        total_mass = 0.0;
        center_of_mass_x = 0.0;
        center_of_mass_y = 0.0;
        is_leaf = true;
        for (auto& child : children) {
            child.reset();
        }
    }
};

// Build Barnes-Hut tree with proper center of mass calculation
void bh_insert_particle(BHNode* node, int particle_id, 
                        const std::vector<double>& x, const std::vector<double>& y, 
                        const std::vector<double>& m, int max_depth = 20) {
    
    if (max_depth <= 0) return; // Prevent infinite recursion
    
    const double px = x[particle_id];
    const double py = y[particle_id];
    const double mass = m[particle_id];
    
    // Check if particle is within node bounds (with tolerance)
    const double tolerance = 1e-10;
    if (std::abs(px - node->cx) > node->size + tolerance || 
        std::abs(py - node->cy) > node->size + tolerance) {
        return; // Particle outside this node
    }
    
    // Update center of mass for this node
    const double old_mass = node->total_mass;
    const double new_mass = old_mass + mass;
    
    if (new_mass > 0) {
        node->center_of_mass_x = (node->center_of_mass_x * old_mass + px * mass) / new_mass;
        node->center_of_mass_y = (node->center_of_mass_y * old_mass + py * mass) / new_mass;
    }
    node->total_mass = new_mass;
    
    if (node->is_leaf) {
        // If this is a leaf and empty, just add the particle
        if (node->particle_indices.empty()) {
            node->particle_indices.push_back(particle_id);
            return;
        }
        
        // If this is a leaf with particles, we need to subdivide
        // But only if we have more than one particle and sufficient depth
        if (node->particle_indices.size() >= 1 && max_depth > 1) {
            // Store existing particles
            std::vector<int> existing_particles = node->particle_indices;
            node->particle_indices.clear();
            node->is_leaf = false;
            
            // Create children
            const double half_size = node->size * 0.5;
            node->children[0] = std::make_unique<BHNode>(node->cx - half_size, node->cy - half_size, half_size); // SW
            node->children[1] = std::make_unique<BHNode>(node->cx + half_size, node->cy - half_size, half_size); // SE
            node->children[2] = std::make_unique<BHNode>(node->cx - half_size, node->cy + half_size, half_size); // NW
            node->children[3] = std::make_unique<BHNode>(node->cx + half_size, node->cy + half_size, half_size); // NE
            
            // Re-insert existing particles
            for (int existing_id : existing_particles) {
                const double ex = x[existing_id];
                const double ey = y[existing_id];
                const int child_index = (ex > node->cx ? 1 : 0) + (ey > node->cy ? 2 : 0);
                bh_insert_particle(node->children[child_index].get(), existing_id, x, y, m, max_depth - 1);
            }
        } else {
            // Just add to this leaf (don't subdivide further)
            node->particle_indices.push_back(particle_id);
            return;
        }
    }
    
    // Insert the new particle into appropriate child
    if (!node->is_leaf) {
        const int child_index = (px > node->cx ? 1 : 0) + (py > node->cy ? 2 : 0);
        bh_insert_particle(node->children[child_index].get(), particle_id, x, y, m, max_depth - 1);
    }
}

// Compute force using Barnes-Hut with proper theta criterion
void bh_compute_force(const BHNode* node, double px, double py, double theta, double G, double soft2,
                      double& fx, double& fy) {
    
    if (!node || node->total_mass == 0.0) return;
    
    const double dx = node->center_of_mass_x - px;
    const double dy = node->center_of_mass_y - py;
    const double r2 = dx * dx + dy * dy + soft2;
    
    if (r2 < 1e-20) return; // Avoid self-interaction
    
    const double r = std::sqrt(r2);
    
    // Barnes-Hut approximation criterion: s/d < theta
    const bool use_approximation = node->is_leaf || (node->size / r < theta);
    
    if (use_approximation) {
        // Use this node's center of mass
        const double inv_r = 1.0 / r;
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double force = G * node->total_mass * inv_r3;
        
        fx += force * dx;
        fy += force * dy;
    } else {
        // Recurse to children
        for (const auto& child : node->children) {
            if (child) {
                bh_compute_force(child.get(), px, py, theta, G, soft2, fx, fy);
            }
        }
    }
}

// Main Barnes-Hut function
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
    auto root = std::make_unique<BHNode>(0.0, 0.0, domain * 0.5);
    
    // Insert all particles
    for (size_t i = 0; i < N; ++i) {
        bh_insert_particle(root.get(), i, vx, vy, vm);
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
            bh_compute_force(root.get(), vx[i], vy[i], theta, G, soft2, fx, fy);
        } catch (...) {
            // Fallback to direct calculation for this particle
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

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "High-precision N-body force kernels";
    
    m.def("direct_omp", &direct_omp,
          "Direct N-body force calculation (reference)",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("G") = 1.0, py::arg("soft") = 0.05);
    
    m.def("bh_omp", &bh_omp,
          "High-precision Barnes-Hut force calculation",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);

#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
