#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Direct N-body calculation (reference implementation)
py::tuple direct_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                     double G = 1.0, double soft = 0.05) {
    
    const ssize_t N = x.size();
    if (N == 0) return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();
    
    const double soft2 = soft * soft;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (ssize_t i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        
        for (ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            
            const double dx = px(j) - px(i);
            const double dy = py_(j) - py_(i);
            const double r2 = dx * dx + dy * dy + soft2;
            const double inv_r3 = 1.0 / std::pow(r2, 1.5);
            
            fx += G * pm(j) * dx * inv_r3;
            fy += G * pm(j) * dy * inv_r3;
        }
        
        pax(i) = fx;
        pay(i) = fy;
    }
    
    return py::make_tuple(ax, ay);
}

// High-precision Barnes-Hut tree node
struct BHNode {
    double cx, cy, size;
    double total_mass, com_x, com_y;
    bool is_leaf;
    std::vector<int> particles;
    std::array<std::unique_ptr<BHNode>, 4> children;
    
    BHNode(double x, double y, double s) 
        : cx(x), cy(y), size(s), total_mass(0.0), com_x(0.0), com_y(0.0), is_leaf(true) {}
};

// Insert particle with proper center of mass calculation
void bh_insert(BHNode* node, int pid, const std::vector<double>& x, 
               const std::vector<double>& y, const std::vector<double>& m, int depth = 0) {
    
    if (depth > 30) return; // Prevent infinite recursion
    
    const double px = x[pid], py = y[pid], mass = m[pid];
    
    // Update center of mass
    const double old_mass = node->total_mass;
    const double new_mass = old_mass + mass;
    
    if (new_mass > 0) {
        node->com_x = (node->com_x * old_mass + px * mass) / new_mass;
        node->com_y = (node->com_y * old_mass + py * mass) / new_mass;
    }
    node->total_mass = new_mass;
    
    if (node->is_leaf) {
        if (node->particles.empty()) {
            node->particles.push_back(pid);
            return;
        }
        
        // Subdivide if we have particles and sufficient size
        if (node->size > 1e-10 && node->particles.size() < 8) {
            node->particles.push_back(pid);
            return;
        }
        
        // Create children
        node->is_leaf = false;
        const double hs = node->size * 0.5;
        node->children[0] = std::make_unique<BHNode>(node->cx - hs, node->cy - hs, hs);
        node->children[1] = std::make_unique<BHNode>(node->cx + hs, node->cy - hs, hs);
        node->children[2] = std::make_unique<BHNode>(node->cx - hs, node->cy + hs, hs);
        node->children[3] = std::make_unique<BHNode>(node->cx + hs, node->cy + hs, hs);
        
        // Redistribute existing particles
        for (int existing_pid : node->particles) {
            const int quad = (x[existing_pid] > node->cx) + 2 * (y[existing_pid] > node->cy);
            bh_insert(node->children[quad].get(), existing_pid, x, y, m, depth + 1);
        }
        node->particles.clear();
    }
    
    // Insert new particle
    const int quad = (px > node->cx) + 2 * (py > node->cy);
    bh_insert(node->children[quad].get(), pid, x, y, m, depth + 1);
}

// Compute force with high precision
void bh_force(const BHNode* node, double px, double py, double theta, double G, double soft2,
              double& fx, double& fy) {
    
    if (!node || node->total_mass == 0.0) return;
    
    const double dx = node->com_x - px;
    const double dy = node->com_y - py;
    const double r2 = dx * dx + dy * dy + soft2;
    
    if (r2 < 1e-20) return;
    
    const double r = std::sqrt(r2);
    
    // Barnes-Hut criterion with safety check
    if (node->is_leaf || (node->size > 0 && node->size / r < theta)) {
        const double inv_r3 = 1.0 / (r2 * r);
        const double force = G * node->total_mass * inv_r3;
        fx += force * dx;
        fy += force * dy;
    } else {
        for (const auto& child : node->children) {
            if (child) bh_force(child.get(), px, py, theta, G, soft2, fx, fy);
        }
    }
}

// Main Barnes-Hut function
py::tuple bh_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                 double domain, double theta = 0.5, double G = 1.0, double soft = 0.05) {
    
    const size_t N = x.size();
    if (N == 0) return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = px(i); vy[i] = py_(i); vm[i] = pm(i);
    }

    auto root = std::make_unique<BHNode>(0.0, 0.0, domain);
    for (size_t i = 0; i < N; ++i) {
        bh_insert(root.get(), i, vx, vy, vm);
    }

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();
    
    const double soft2 = soft * soft;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        bh_force(root.get(), vx[i], vy[i], theta, G, soft2, fx, fy);
        pax(i) = fx;
        pay(i) = fy;
    }

    return py::make_tuple(ax, ay);
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "High-precision N-body force kernels";
    m.def("direct_omp", &direct_omp, py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("G") = 1.0, py::arg("soft") = 0.05);
    m.def("bh_omp", &bh_omp, py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);
}
