#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

// Full multipole order matching fmm_scaling_test.py
constexpr int P = 8;

// Precomputed factorials for efficiency
static std::array<double, P + 1> factorial_table = []() {
    std::array<double, P + 1> table;
    table[0] = 1.0;
    for (int i = 1; i <= P; ++i) {
        table[i] = table[i-1] * i;
    }
    return table;
}();

// Binomial coefficient with high precision
static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k; // Use symmetry
    
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// FMM tree node with full multipole expansion
struct FMMCell {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    std::array<cplx, P + 1> multipole{};  // Multipole moments
    std::array<cplx, P + 1> local{};      // Local expansion
    std::array<std::unique_ptr<FMMCell>, 4> children;
    FMMCell* parent;
    bool is_leaf;
    
    FMMCell(double x, double y, double s, int lev = 0, FMMCell* p = nullptr)
        : cx(x), cy(y), size(s), level(lev), parent(p), is_leaf(true) {
        std::fill(multipole.begin(), multipole.end(), cplx(0.0, 0.0));
        std::fill(local.begin(), local.end(), cplx(0.0, 0.0));
    }
};

// Build FMM tree with proper subdivision
void fmm_subdivide(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y, 
                   int max_particles = 16, int max_level = 10) {
    
    if ((int)cell->particles.size() <= max_particles || cell->level >= max_level) {
        return;
    }
    
    cell->is_leaf = false;
    const double half_size = cell->size * 0.5;
    
    // Create children
    cell->children[0] = std::make_unique<FMMCell>(cell->cx - half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[1] = std::make_unique<FMMCell>(cell->cx + half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[2] = std::make_unique<FMMCell>(cell->cx - half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    cell->children[3] = std::make_unique<FMMCell>(cell->cx + half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    
    // Distribute particles to children
    for (int particle_id : cell->particles) {
        const int quadrant = (x[particle_id] > cell->cx ? 1 : 0) + 
                            (y[particle_id] > cell->cy ? 2 : 0);
        cell->children[quadrant]->particles.push_back(particle_id);
    }
    
    cell->particles.clear();
    
    // Recursively subdivide children
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            fmm_subdivide(child.get(), x, y, max_particles, max_level);
        }
    }
}

// P2M and M2M (upward pass) with full multipole expansion
void fmm_upward_pass(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y, 
                     const std::vector<double>& m) {
    
    if (!cell) return;
    
    std::fill(cell->multipole.begin(), cell->multipole.end(), cplx(0.0, 0.0));
    
    if (cell->is_leaf) {
        // P2M: Particles to multipole
        for (int particle_id : cell->particles) {
            const double mass = m[particle_id];
            const double dx = x[particle_id] - cell->cx;
            const double dy = y[particle_id] - cell->cy;
            const cplx z(dx, dy);
            
            // Compute multipole moments: a_k = q * z^k / k!
            cell->multipole[0] += mass;
            cplx z_power = z;
            for (int k = 1; k <= P; ++k) {
                cell->multipole[k] += mass * z_power / factorial_table[k];
                z_power *= z;
            }
        }
    } else {
        // M2M: Child to parent translation
        for (auto& child : cell->children) {
            if (child && !child->particles.empty()) {
                fmm_upward_pass(child.get(), x, y, m);
                
                const double dx = child->cx - cell->cx;
                const double dy = child->cy - cell->cy;
                const cplx z0(dx, dy);
                
                // M2M translation: a_l = sum_{k=0}^l C(l,k) * a_k^child * z0^(l-k)
                for (int l = 0; l <= P; ++l) {
                    cplx z0_power(1.0, 0.0);
                    for (int k = 0; k <= l; ++k) {
                        const double binom_coeff = binomial(l, k);
                        cell->multipole[l] += child->multipole[k] * binom_coeff * z0_power;
                        z0_power *= z0;
                    }
                }
            }
        }
    }
}

// M2L translation with full expansion
void fmm_m2l_translation(FMMCell* target, FMMCell* source) {
    if (!target || !source || target == source) return;
    
    const double dx = source->cx - target->cx;
    const double dy = source->cy - target->cy;
    const double r2 = dx * dx + dy * dy;
    
    if (r2 < 1e-20) return; // Avoid singularity
    
    const cplx z0(dx, dy);
    const double r = std::sqrt(r2);
    
    // Ensure well-separated condition
    if (r < 2.0 * std::max(target->size, source->size)) return;
    
    // M2L translation for full multipole expansion
    // b_j = sum_{k=0}^P (-1)^k * C(j+k,k) * a_k / z0^(j+k+1)
    for (int j = 0; j <= P; ++j) {
        cplx contribution(0.0, 0.0);
        
        for (int k = 0; k <= P; ++k) {
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            const double binom_coeff = binomial(j + k, k);
            const cplx z0_power = std::pow(z0, j + k + 1);
            
            if (std::abs(z0_power) > 1e-15) {
                contribution += sign * binom_coeff * source->multipole[k] / z0_power;
            }
        }
        
        target->local[j] += contribution;
    }
}

// Build interaction lists and perform M2L
void fmm_interaction_pass(FMMCell* cell, FMMCell* root, double theta) {
    if (!cell) return;
    
    // Simple interaction traversal
    std::function<void(FMMCell*, FMMCell*)> traverse = [&](FMMCell* target, FMMCell* source) {
        if (!source || target == source) return;
        
        const double dx = source->cx - target->cx;
        const double dy = source->cy - target->cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        
        const double size_sum = target->size + source->size;
        
        if (dist > 2.0 * size_sum && source->multipole[0] != cplx(0.0, 0.0)) {
            // Well-separated: use M2L
            fmm_m2l_translation(target, source);
        } else if (!source->is_leaf) {
            // Not well-separated and source has children
            for (auto& child : source->children) {
                if (child) {
                    traverse(target, child.get());
                }
            }
        }
    };
    
    traverse(cell, root);
    
    // Recurse to children
    for (auto& child : cell->children) {
        if (child) {
            fmm_interaction_pass(child.get(), root, theta);
        }
    }
}

// L2L and force evaluation (downward pass)
void fmm_downward_pass(FMMCell* cell) {
    if (!cell) return;
    
    for (auto& child : cell->children) {
        if (child) {
            // L2L: Local to local translation
            const double dx = child->cx - cell->cx;
            const double dy = cell->cy - child->cy;
            const cplx z0(dx, dy);
            
            // L2L translation: b_j^child = sum_{k=j}^P C(k,j) * b_k^parent * z0^(k-j)
            for (int j = 0; j <= P; ++j) {
                cplx z0_power(1.0, 0.0);
                for (int k = j; k <= P; ++k) {
                    const double binom_coeff = binomial(k, j);
                    child->local[j] += cell->local[k] * binom_coeff * z0_power;
                    z0_power *= z0;
                }
            }
            
            fmm_downward_pass(child.get());
        }
    }
}

// Force evaluation with local expansion
void fmm_evaluate_forces(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y,
                         const std::vector<double>& m, std::vector<double>& fx, std::vector<double>& fy,
                         double G, double soft2) {
    
    if (!cell) return;
    
    if (!cell->is_leaf) {
        for (auto& child : cell->children) {
            if (child) {
                fmm_evaluate_forces(child.get(), x, y, m, fx, fy, G, soft2);
            }
        }
        return;
    }
    
    // Leaf node: evaluate local expansion + direct interactions
    for (int i : cell->particles) {
        double force_x = 0.0, force_y = 0.0;
        
        // Direct interactions within same leaf
        for (int j : cell->particles) {
            if (i != j) {
                const double dx = x[j] - x[i];
                const double dy = y[j] - y[i];
                const double r2 = dx * dx + dy * dy + soft2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                
                force_x += G * m[j] * dx * inv_r3;
                force_y += G * m[j] * dy * inv_r3;
            }
        }
        
        // Local expansion contribution
        const double dx = x[i] - cell->cx;
        const double dy = y[i] - cell->cy;
        const cplx z(dx, dy);
        
        // Evaluate -∇φ from local expansion
        cplx force_complex(0.0, 0.0);
        cplx z_power(1.0, 0.0);
        
        for (int k = 1; k <= P; ++k) {
            force_complex += double(k) * cell->local[k] * z_power / factorial_table[k];
            z_power *= z;
        }
        
        // Convert to Cartesian force components
        force_x += G * (-force_complex.real());
        force_y += G * (-force_complex.imag());
        
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
    
    // Copy input data
    std::vector<double> vx(x.data(), x.data() + N);
    std::vector<double> vy(y.data(), y.data() + N);
    std::vector<double> vm(m.data(), m.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    
    try {
        // Build FMM tree
        auto root = std::make_unique<FMMCell>(0.0, 0.0, domain * 0.5);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);
        
        // Execute FMM algorithm
        fmm_subdivide(root.get(), vx, vy, 16, 10);
        fmm_upward_pass(root.get(), vx, vy, vm);
        fmm_interaction_pass(root.get(), root.get(), theta);
        fmm_downward_pass(root.get());
        fmm_evaluate_forces(root.get(), vx, vy, vm, fx, fy, G, soft * soft);
        
    } catch (const std::exception& e) {
        // Fallback to direct calculation
        const double soft2 = soft * soft;
        
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                
                const double dx = vx[j] - vx[i];
                const double dy = vy[j] - vy[i];
                const double r2 = dx * dx + dy * dy + soft2;
                const double inv_r3 = 1.0 / std::pow(r2, 1.5);
                
                fx[i] += G * vm[j] * dx * inv_r3;
                fy[i] += G * vm[j] * dy * inv_r3;
            }
        }
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

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "High-precision Fast Multipole Method";
    
    m.def("fmm_omp", &fmm_omp,
          "High-precision FMM force calculation with full P=8 expansion",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);

#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
