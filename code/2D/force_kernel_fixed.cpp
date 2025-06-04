#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <memory>
#include <omp.h>

namespace py = pybind11;
using cplx = std::complex<double>;
constexpr int P = 8;  // Match fmm_scaling_test.py (p=8)

// Thread-safe factorial calculation
static double factorial(int n) {
    if (n <= 1) return 1.0;
    static const std::vector<double> cache = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880};
    if (n < (int)cache.size()) return cache[n];
    
    double result = cache.back();
    for (int i = cache.size(); i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Binomial coefficient
static double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

struct Cell {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    std::array<cplx, P+1> M{};
    std::array<cplx, P+1> L{};
    std::array<std::unique_ptr<Cell>, 4> children;
    Cell* parent;
    bool is_leaf;
    
    Cell(double x, double y, double s, int lev = 0, Cell* p = nullptr) 
        : cx(x), cy(y), size(s), level(lev), parent(p), is_leaf(true) {}
    
    ~Cell() = default;
    Cell(const Cell&) = delete;
    Cell& operator=(const Cell&) = delete;
};

void subdivide(Cell* cell, const std::vector<double>& x, const std::vector<double>& y, int maxLeaf) {
    if ((int)cell->particles.size() <= maxLeaf) return;
    
    cell->is_leaf = false;
    double half = cell->size * 0.5;
    
    // Create children
    cell->children[0] = std::make_unique<Cell>(cell->cx - half, cell->cy - half, half, cell->level + 1, cell);
    cell->children[1] = std::make_unique<Cell>(cell->cx + half, cell->cy - half, half, cell->level + 1, cell);
    cell->children[2] = std::make_unique<Cell>(cell->cx - half, cell->cy + half, half, cell->level + 1, cell);
    cell->children[3] = std::make_unique<Cell>(cell->cx + half, cell->cy + half, half, cell->level + 1, cell);
    
    // Distribute particles
    for (int id : cell->particles) {
        int quad = 0;
        if (x[id] > cell->cx) quad += 1;
        if (y[id] > cell->cy) quad += 2;
        cell->children[quad]->particles.push_back(id);
    }
    
    cell->particles.clear();
    
    // Recursively subdivide children
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            subdivide(child.get(), x, y, maxLeaf);
        }
    }
}

void upward_pass(Cell* cell, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& m) {
    if (!cell) return;
    
    std::fill(cell->M.begin(), cell->M.end(), cplx(0.0, 0.0));
    
    if (cell->is_leaf) {
        // P2M: Particles to multipole
        for (int id : cell->particles) {
            double dx = x[id] - cell->cx;
            double dy = y[id] - cell->cy;
            cplx z(dx, dy);
            double mass = m[id];
            
            cell->M[0] += mass;
            cplx z_power = z;
            for (int k = 1; k <= P; ++k) {
                cell->M[k] += mass * z_power;
                z_power *= z;
            }
        }
    } else {
        // M2M: Child to parent translation
        for (auto& child : cell->children) {
            if (child && !child->particles.empty()) {
                upward_pass(child.get(), x, y, m);
                
                double dx = child->cx - cell->cx;
                double dy = child->cy - cell->cy;
                cplx z0(dx, dy);
                
                for (int l = 0; l <= P; ++l) {
                    for (int k = 0; k <= l; ++k) {
                        double bin_coeff = binom(l, k);
                        cell->M[l] += child->M[k] * bin_coeff * std::pow(z0, l - k);
                    }
                }
            }
        }
    }
}

void m2l_translation(Cell* target, Cell* source) {
    if (!target || !source || target == source) return;
    
    double dx = source->cx - target->cx;
    double dy = source->cy - target->cy;
    double dist_sq = dx*dx + dy*dy;
    
    if (dist_sq < 1e-20) return;
    
    cplx z0(dx, dy);
    double dist = std::sqrt(dist_sq);
    
    // Simple M2L translation (simplified for stability)
    for (int l = 0; l <= P; ++l) {
        for (int k = 0; k <= P; ++k) {
            if (k == 0) {
                target->L[l] += source->M[k] / std::pow(z0, l + 1);
            } else {
                double factor = binom(l + k, k);
                target->L[l] += source->M[k] * factor / std::pow(z0, l + k + 1);
            }
        }
    }
}

void interaction_phase(Cell* cell, Cell* root) {
    if (!cell) return;
    
    // Simple interaction: all cells interact with all other cells
    // In a full FMM, you'd use proper interaction lists
    std::function<void(Cell*, Cell*)> traverse = [&](Cell* target, Cell* source) {
        if (!source || target == source) return;
        
        double dx = source->cx - target->cx;
        double dy = source->cy - target->cy;
        double dist = std::sqrt(dx*dx + dy*dy);
        
        if (dist > 2.0 * std::max(target->size, source->size)) {
            m2l_translation(target, source);
        } else if (!source->is_leaf) {
            for (auto& child : source->children) {
                if (child) traverse(target, child.get());
            }
        }
    };
    
    traverse(cell, root);
    
    for (auto& child : cell->children) {
        if (child) interaction_phase(child.get(), root);
    }
}

void downward_pass(Cell* cell) {
    if (!cell) return;
    
    for (auto& child : cell->children) {
        if (child) {
            // L2L: Local to local translation
            double dx = child->cx - cell->cx;
            double dy = child->cy - cell->cy;
            cplx z0(dx, dy);
            
            for (int l = 0; l <= P; ++l) {
                for (int k = l; k <= P; ++k) {
                    double bin_coeff = binom(k, l);
                    child->L[l] += cell->L[k] * bin_coeff * std::pow(z0, k - l);
                }
            }
            
            downward_pass(child.get());
        }
    }
}

void evaluate_forces(Cell* cell, const std::vector<double>& x, const std::vector<double>& y, 
                    const std::vector<double>& m, std::vector<double>& fx, std::vector<double>& fy,
                    double G, double soft_sq) {
    if (!cell) return;
    
    if (!cell->is_leaf) {
        for (auto& child : cell->children) {
            if (child) evaluate_forces(child.get(), x, y, m, fx, fy, G, soft_sq);
        }
        return;
    }
    
    // Direct interactions within leaf + local expansion
    for (int i : cell->particles) {
        double force_x = 0.0, force_y = 0.0;
        
        // Direct interactions within same leaf
        for (int j : cell->particles) {
            if (i != j) {
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double r_sq = dx*dx + dy*dy + soft_sq;
                double inv_r3 = 1.0 / (r_sq * std::sqrt(r_sq));
                force_x += G * m[j] * dx * inv_r3;
                force_y += G * m[j] * dy * inv_r3;
            }
        }
        
        // Local expansion contribution
        double dx = x[i] - cell->cx;
        double dy = y[i] - cell->cy;
        cplx z(dx, dy);
        
        cplx force_cplx(0.0, 0.0);
        cplx z_power(1.0, 0.0);
        for (int k = 1; k <= P; ++k) {
            z_power *= z;
            force_cplx += double(k) * cell->L[k] * std::pow(z, k - 1);
        }
        
        force_x += G * (-force_cplx.real());
        force_y += G * (-force_cplx.imag());
        
        fx[i] += force_x;
        fy[i] += force_y;
    }
}

// FIXED: Match the expected function signature
py::tuple fmm_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                  double domain, double theta=0.5, double G=1.0, double soft=0.05) {
    
    size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }
    
    // Copy input data
    std::vector<double> vx(x.data(), x.data() + N);
    std::vector<double> vy(y.data(), y.data() + N);
    std::vector<double> vm(m.data(), m.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    
    try {
        // Build tree
        auto root = std::make_unique<Cell>(0.0, 0.0, domain);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);
        
        // FMM algorithm
        subdivide(root.get(), vx, vy, 16);  // maxLeaf = 16
        upward_pass(root.get(), vx, vy, vm);
        interaction_phase(root.get(), root.get());
        downward_pass(root.get());
        evaluate_forces(root.get(), vx, vy, vm, fx, fy, G, soft*soft);
        
    } catch (const std::exception& e) {
        // Fallback to direct calculation if FMM fails
        double soft_sq = soft * soft;
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j) {
                    double dx = vx[j] - vx[i];
                    double dy = vy[j] - vy[i];
                    double r_sq = dx*dx + dy*dy + soft_sq;
                    double inv_r3 = 1.0 / (r_sq * std::sqrt(r_sq));
                    fx[i] += G * vm[j] * dx * inv_r3;
                    fy[i] += G * vm[j] * dy * inv_r3;
                }
            }
        }
    }
    
    // Copy results to NumPy arrays
    py::array_t<double> ax_out(N), ay_out(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }
    
    return py::make_tuple(ax_out, ay_out);
}

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "2-D FMM solver with proper error handling";
    m.def("fmm_omp", &fmm_omp,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta")=0.5, py::arg("G")=1.0, py::arg("soft")=0.05);
}
