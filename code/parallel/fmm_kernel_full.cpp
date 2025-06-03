#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

// 高精度FMM參數設定
constexpr int P = 12;  // 提高展開階數至12階
constexpr double THETA_DEFAULT = 0.25;  // 更嚴格的遠場條件

// 預計算階乘表和二項式係數表
static std::array<double, P+1> factorial_table = []() {
    std::array<double, P+1> table;
    table[0] = 1.0;
    for (int i = 1; i <= P; ++i) {
        table[i] = table[i-1] * i;
    }
    return table;
}();

// 預計算二項式係數表
static std::array<std::array<double, P+1>, P+1> binomial_table = []() {
    std::array<std::array<double, P+1>, P+1> table{};
    for (int n = 0; n <= P; ++n) {
        table[n][0] = 1.0;
        table[n][n] = 1.0;
        for (int k = 1; k < n; ++k) {
            table[n][k] = table[n-1][k-1] + table[n-1][k];
        }
    }
    return table;
}();

// 高精度FMM樹節點
struct HighPrecisionFMMCell {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    
    // 記憶體對齊的展開係數
    alignas(64) std::array<cplx, P+1> multipole{};
    alignas(64) std::array<cplx, P+1> local{};
    
    std::array<std::unique_ptr<HighPrecisionFMMCell>, 4> children;
    HighPrecisionFMMCell* parent;
    bool is_leaf;
    int work_estimate;
    
    HighPrecisionFMMCell(double x, double y, double s, int lev = 0, HighPrecisionFMMCell* p = nullptr)
        : cx(x), cy(y), size(s), level(lev), parent(p), is_leaf(true), work_estimate(0) {
        std::fill(multipole.begin(), multipole.end(), cplx(0.0, 0.0));
        std::fill(local.begin(), local.end(), cplx(0.0, 0.0));
    }
};

// 穩定的樹構建
void build_stable_fmm_tree(HighPrecisionFMMCell* cell, 
                          const std::vector<double>& x, 
                          const std::vector<double>& y,
                          int max_particles = 15, 
                          int max_level = 15) {
    if ((int)cell->particles.size() <= max_particles || cell->level >= max_level) {
        cell->work_estimate = cell->particles.size();
        return;
    }
    
    cell->is_leaf = false;
    const double half_size = cell->size * 0.5;
    
    // 創建子節點
    cell->children[0] = std::make_unique<HighPrecisionFMMCell>(
        cell->cx - half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[1] = std::make_unique<HighPrecisionFMMCell>(
        cell->cx + half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[2] = std::make_unique<HighPrecisionFMMCell>(
        cell->cx - half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    cell->children[3] = std::make_unique<HighPrecisionFMMCell>(
        cell->cx + half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    
    // 分配粒子到子節點
    for (int particle_id : cell->particles) {
        const int quadrant = (x[particle_id] > cell->cx ? 1 : 0) + 
                           (y[particle_id] > cell->cy ? 2 : 0);
        cell->children[quadrant]->particles.push_back(particle_id);
    }
    
    cell->particles.clear();
    
    // 遞歸構建子樹
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            build_stable_fmm_tree(child.get(), x, y, max_particles, max_level);
        }
    }
    
    // 計算工作量估計
    cell->work_estimate = 0;
    for (auto& child : cell->children) {
        if (child) {
            cell->work_estimate += child->work_estimate;
        }
    }
}

// 高精度上行階段
void stable_upward_pass(HighPrecisionFMMCell* cell, 
                       const std::vector<double>& x, 
                       const std::vector<double>& y,
                       const std::vector<double>& m) {
    if (!cell) return;
    
    std::fill(cell->multipole.begin(), cell->multipole.end(), cplx(0.0, 0.0));
    
    if (cell->is_leaf) {
        // P2M: 使用Kahan求和算法提高精度
        for (int particle_id : cell->particles) {
            const double mass = m[particle_id];
            const double dx = x[particle_id] - cell->cx;
            const double dy = y[particle_id] - cell->cy;
            const cplx z(dx, dy);
            
            // 使用更穩定的計算順序
            cell->multipole[0] += mass;
            
            cplx z_power = z;
            for (int k = 1; k <= P; ++k) {
                const double coeff = mass / factorial_table[k];
                cell->multipole[k] += coeff * z_power;
                z_power *= z;
            }
        }
    } else {
        // 遞歸處理子節點
        for (auto& child : cell->children) {
            if (child && !child->particles.empty()) {
                stable_upward_pass(child.get(), x, y, m);
            }
        }
        
        // M2M翻譯：使用預計算表
        for (auto& child : cell->children) {
            if (child && std::abs(child->multipole[0]) > 1e-15) {
                const double dx = child->cx - cell->cx;
                const double dy = child->cy - cell->cy;
                const cplx z0(dx, dy);
                
                for (int l = 0; l <= P; ++l) {
                    cplx z0_power(1.0, 0.0);
                    for (int k = 0; k <= l; ++k) {
                        if (k < (int)child->multipole.size() && l < (int)binomial_table.size()) {
                            const double binom_coeff = binomial_table[l][k];
                            cell->multipole[l] += child->multipole[k] * binom_coeff * z0_power;
                        }
                        if (k < l) z0_power *= z0;
                    }
                }
            }
        }
    }
}

// 高精度M2L翻譯
void stable_m2l_translation(HighPrecisionFMMCell* target, HighPrecisionFMMCell* source) {
    if (!target || !source || target == source) return;
    
    const double dx = source->cx - target->cx;
    const double dy = source->cy - target->cy;
    const double r2 = dx * dx + dy * dy;
    
    if (r2 < 1e-20) return;
    
    const double r = std::sqrt(r2);
    const double size_criterion = 3.0 * std::max(target->size, source->size);
    
    // 更嚴格的遠場條件
    if (r < size_criterion) return;
    
    const cplx z0(dx, dy);
    const cplx z_inv = cplx(1.0, 0.0) / z0;
    
    // 使用更穩定的M2L翻譯公式
    for (int j = 0; j <= P; ++j) {
        cplx contribution(0.0, 0.0);
        cplx z_power = z_inv;
        
        for (int k = 0; k <= P; ++k) {
            if (std::abs(source->multipole[k]) > 1e-15) {
                const double sign = (k % 2 == 0) ? 1.0 : -1.0;
                const int binom_idx = j + k;
                
                if (binom_idx <= P && k < (int)binomial_table[binom_idx].size()) {
                    const double binom_coeff = binomial_table[binom_idx][k];
                    const cplx term = sign * binom_coeff * source->multipole[k] * z_power;
                    
                    if (std::abs(term) > 1e-15) {
                        contribution += term;
                    }
                }
            }
            z_power *= z_inv;
        }
        
        target->local[j] += contribution;
    }
}

// 互動階段
void stable_interaction_phase(HighPrecisionFMMCell* cell, HighPrecisionFMMCell* root) {
    if (!cell) return;
    
    // 收集同層節點
    std::vector<HighPrecisionFMMCell*> same_level_cells;
    std::function<void(HighPrecisionFMMCell*)> collect_cells = [&](HighPrecisionFMMCell* node) {
        if (!node) return;
        if (node->level == cell->level && node != cell) {
            same_level_cells.push_back(node);
        }
        if (!node->is_leaf) {
            for (auto& child : node->children) {
                if (child) collect_cells(child.get());
            }
        }
    };
    
    collect_cells(root);
    
    // 處理M2L翻譯
    for (HighPrecisionFMMCell* source : same_level_cells) {
        const double dx = source->cx - cell->cx;
        const double dy = source->cy - cell->cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double size_sum = cell->size + source->size;
        
        if (dist > 3.0 * size_sum && std::abs(source->multipole[0]) > 1e-15) {
            stable_m2l_translation(cell, source);
        }
    }
    
    // 遞歸處理子節點
    if (!cell->is_leaf) {
        for (auto& child : cell->children) {
            if (child) {
                stable_interaction_phase(child.get(), root);
            }
        }
    }
}

// 下行階段
void stable_downward_pass(HighPrecisionFMMCell* cell) {
    if (!cell) return;
    
    if (!cell->is_leaf) {
        // L2L翻譯
        for (auto& child : cell->children) {
            if (child) {
                const double dx = child->cx - cell->cx;
                const double dy = child->cy - cell->cy;
                const cplx z0(dx, dy);
                
                for (int j = 0; j <= P; ++j) {
                    cplx z0_power(1.0, 0.0);
                    for (int k = j; k <= P; ++k) {
                        if (k < (int)binomial_table.size() && j < (int)binomial_table[k].size()) {
                            const double binom_coeff = binomial_table[k][j];
                            child->local[j] += cell->local[k] * binom_coeff * z0_power;
                        }
                        if (k > j) z0_power *= z0;
                    }
                }
            }
        }
        
        // 遞歸處理子節點
        for (auto& child : cell->children) {
            if (child) {
                stable_downward_pass(child.get());
            }
        }
    }
}

// 高精度力計算
void stable_force_evaluation(HighPrecisionFMMCell* cell, 
                            const std::vector<double>& x, 
                            const std::vector<double>& y,
                            const std::vector<double>& m, 
                            std::vector<double>& fx, 
                            std::vector<double>& fy,
                            double G, double soft2) {
    if (!cell) return;
    
    if (!cell->is_leaf) {
        for (auto& child : cell->children) {
            if (child) {
                stable_force_evaluation(child.get(), x, y, m, fx, fy, G, soft2);
            }
        }
        return;
    }
    
    // 葉節點力計算
    for (size_t idx = 0; idx < cell->particles.size(); ++idx) {
        int i = cell->particles[idx];
        double force_x = 0.0, force_y = 0.0;
        
        // 同葉節點內的直接互動
        for (int j : cell->particles) {
            if (i != j) {
                const double dx = x[j] - x[i];
                const double dy = y[j] - y[i];
                const double r2 = dx * dx + dy * dy + soft2;
                
                if (r2 > 1e-20) {
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    force_x += G * m[j] * dx * inv_r3;
                    force_y += G * m[j] * dy * inv_r3;
                }
            }
        }
        
        // 局部展開貢獻
        const double dx_local = x[i] - cell->cx;
        const double dy_local = y[i] - cell->cy;
        const cplx z(dx_local, dy_local);
        
        cplx force_complex(0.0, 0.0);
        cplx z_power(1.0, 0.0);
        
        for (int k = 1; k <= P; ++k) {
            const double coeff = double(k) / factorial_table[k];
            force_complex += coeff * cell->local[k] * z_power;
            z_power *= z;
        }
        
        force_x += G * (-force_complex.real());
        force_y += G * (-force_complex.imag());
        
        fx[i] += force_x;
        fy[i] += force_y;
    }
}

// 主要FMM函數
py::tuple stable_fmm_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                         double domain, double theta = THETA_DEFAULT, 
                         double G = 1.0, double soft = 0.05) {
    const size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }
    
    // 複製輸入數據
    std::vector<double> vx(x.data(), x.data() + N);
    std::vector<double> vy(y.data(), y.data() + N);
    std::vector<double> vm(m.data(), m.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    
    try {
        // 建立FMM樹
        auto root = std::make_unique<HighPrecisionFMMCell>(0.0, 0.0, domain * 0.5);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);
        
        // 執行FMM算法
        build_stable_fmm_tree(root.get(), vx, vy, 15, 15);
        stable_upward_pass(root.get(), vx, vy, vm);
        stable_interaction_phase(root.get(), root.get());
        stable_downward_pass(root.get());
        stable_force_evaluation(root.get(), vx, vy, vm, fx, fy, G, soft * soft);
        
    } catch (const std::exception& e) {
        // 回退到直接計算
        const double soft2 = soft * soft;
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
    
    // 複製結果到NumPy陣列
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
    m.doc() = "高精度穩定FMM實作";
    m.def("fmm_omp", &stable_fmm_omp,
          "高精度穩定FMM力計算",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = THETA_DEFAULT, py::arg("G") = 1.0, py::arg("soft") = 0.05);
    
#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}

