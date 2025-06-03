#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <cmath>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using cplx = std::complex<double>;
constexpr int P = 8; // Full multipole order

// Precomputed factorials
static const std::array<double, P + 1> factorial = []() {
    std::array<double, P + 1> f;
    f[0] = 1.0;
    for (int i = 1; i <= P; ++i) f[i] = f[i-1] * i;
    return f;
}();

// Binomial coefficient
static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// FMM tree cell
struct FMMCell {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    std::array<cplx, P + 1> multipole{};
    std::array<cplx, P + 1> local{};
    std::array<std::unique_ptr<FMMCell>, 4> children;
    bool is_leaf = true;
    
    FMMCell(double x, double y, double s, int lev = 0)
        : cx(x), cy(y), size(s), level(lev) {}
};

// Build FMM tree
void fmm_subdivide(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y,
                   int max_particles = 16, int max_level = 10) {
    
    if ((int)cell->particles.size() <= max_particles || cell->level >= max_level) return;
    
    cell->is_leaf = false;
    const double hs = cell->size * 0.5;
    
    cell->children[0] = std::make_unique<FMMCell>(cell->cx - hs, cell->cy - hs, hs, cell->level + 1);
    cell->children[1] = std::make_unique<FMMCell>(cell->cx + hs, cell->cy - hs, hs, cell->level + 1);
    cell->children[2] = std::make_unique<FMMCell>(cell->cx - hs, cell->cy + hs, hs, cell->level + 1);
    cell->children[3] = std::make_unique<FMMCell>(cell->cx + hs, cell->cy + hs, hs, cell->level + 1);
    
    for (int pid : cell->particles) {
        const int quad = (x[pid] > cell->cx) + 2 * (y[pid] > cell->cy);
        cell->children[quad]->particles.push_back(pid);
    }
    cell->particles.clear();
    
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            fmm_subdivide(child.get(), x, y, max_particles, max_level);
        }
    }
}

// P2M and M2M (upward pass)
void fmm_upward(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y,
                const std::vector<double>& m) {
    
    if (!cell) return;
    
    std::fill(cell->multipole.begin(), cell->multipole.end(), cplx(0.0, 0.0));
    
    if (cell->is_leaf) {
        // P2M: particles to multipole
        for (int pid : cell->particles) {
            const double mass = m[pid];
            const cplx z(x[pid] - cell->cx, y[pid] - cell->cy);
            
            cell->multipole[0] += mass;
            cplx z_power = z;
            for (int k = 1; k <= P; ++k) {
                cell->multipole[k] += mass * z_power / factorial[k];
                z_power *= z;
            }
        }
    } else {
        // M2M: child to parent
        for (auto& child : cell->children) {
            if (child && !child->particles.empty()) {
                fmm_upward(child.get(), x, y, m);
                
                const cplx z0(child->cx - cell->cx, child->cy - cell->cy);
                cplx z0_power(1.0, 0.0);
                
                for (int l = 0; l <= P; ++l) {
                    for (int k = 0; k <= l; ++k) {
                        cell->multipole[l] += child->multipole[k] * binomial(l, k) * z0_power;
                        if (k < l) z0_power *= z0;
                    }
                    z0_power = z0;
                }
            }
        }
    }
}

// M2L translation
void fmm_m2l(FMMCell* target, FMMCell* source) {
    if (!target || !source || target == source) return;
    
    const cplx z0(source->cx - target->cx, source->cy - target->cy);
    const double r = std::abs(z0);
    
    if (r < 2.0 * std::max(target->size, source->size)) return;
    
    for (int j = 0; j <= P; ++j) {
        for (int k = 0; k <= P; ++k) {
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            const double binom_coeff = binomial(j + k, k);
            const cplx z_power = std::pow(z0, j + k + 1);
            
            if (std::abs(z_power) > 1e-15) {
                target->local[j] += sign * binom_coeff * source->multipole[k] / z_power;
            }
        }
    }
}

// Interaction phase
void fmm_interact(FMMCell* cell, FMMCell* root) {
    if (!cell) return;
    
    std::function<void(FMMCell*, FMMCell*)> traverse = [&](FMMCell* target, FMMCell* source) {
        if (!source || target == source) return;
        
        const double dx = source->cx - target->cx;
        const double dy = source->cy - target->cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double size_sum = target->size + source->size;
        
        if (dist > 2.0 * size_sum) {
            fmm_m2l(target, source);
        } else if (!source->is_leaf) {
            for (auto& child : source->children) {
                if (child) traverse(target, child.get());
            }
        }
    };
    
    traverse(cell, root);
    
    for (auto& child : cell->children) {
        if (child) fmm_interact(child.get(), root);
    }
}

// L2L and force evaluation
void fmm_evaluate(FMMCell* cell, const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<double>& m, std::vector<double>& fx, std::vector<double>& fy,
                  double G, double soft2) {
    
    if (!cell) return;
    
    if (!cell->is_leaf) {
        for (auto& child : cell->children) {
            if (child) {
                // L2L translation
                const cplx z0(child->cx - cell->cx, child->cy - cell->cy);
                cplx z0_power(1.0, 0.0);
                
                for (int j = 0; j <= P; ++j) {
                    for (int k = j; k <= P; ++k) {
                        child->local[j] += cell->local[k] * binomial(k, j) * z0_power;
                        if (k > j) z0_power *= z0;
                    }
                    z0_power = z0;
                }
                
                fmm_evaluate(child.get(), x, y, m, fx, fy, G, soft2);
            }
        }
        return;
    }
    
    // Leaf: direct + local expansion
    for (int i : cell->particles) {
        double force_x = 0.0, force_y = 0.0;
        
        // Direct interactions within leaf
        for (int j : cell->particles) {
            if (i != j) {
                const double dx = x[j] - x[i];
                const double dy = y[j] - y[i];
                const double r2 = dx * dx + dy * dy + soft2;
                const double inv_r3 = 1.0 / std::pow(r2, 1.5);
                force_x += G * m[j] * dx * inv_r3;
                force_y += G * m[j] * dy * inv_r3;
            }
        }
        
        // Local expansion
        const cplx z(x[i] - cell->cx, y[i] - cell->cy);
        cplx force_complex(0.0, 0.0);
        cplx z_power(1.0, 0.0);
        
        for (int k = 1; k <= P; ++k) {
            force_complex += double(k) * cell->local[k] * z_power / factorial[k];
            z_power *= z;
        }
        
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
    if (N == 0) return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    
    std::vector<double> vx(x.data(), x.data() + N);
    std::vector<double> vy(y.data(), y.data() + N);
    std::vector<double> vm(m.data(), m.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    
    try {
        auto root = std::make_unique<FMMCell>(0.0, 0.0, domain);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);
        
        fmm_subdivide(root.get(), vx, vy, 16, 10);
        fmm_upward(root.get(), vx, vy, vm);
        fmm_interact(root.get(), root.get());
        fmm_evaluate(root.get(), vx, vy, vm, fx, fy, G, soft * soft);
        
    } catch (...) {
        // Fallback to direct
        const double soft2 = soft * soft;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j) {
                    const double dx = vx[j] - vx[i];
                    const double dy = vy[j] - vy[i];
                    const double r2 = dx * dx + dy * dy + soft2;
                    const double inv_r3 = 1.0 / std::pow(r2, 1.5);
                    fx[i] += G * vm[j] * dx * inv_r3;
                    fy[i] += G * vm[j] * dy * inv_r3;
                }
            }
        }
    }
    
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
    m.def("fmm_omp", &fmm_omp, py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = 0.5, py::arg("G") = 1.0, py::arg("soft") = 0.05);
}
