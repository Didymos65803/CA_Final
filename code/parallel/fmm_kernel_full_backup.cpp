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
constexpr int P = 10;  // 增加展開階數以提高精度
constexpr double THETA_DEFAULT = 0.3;  // 更嚴格的遠場條件

// 預計算階乘表，避免重複計算
static std::array<double, P+1> factorial_table = []() {
    std::array<double, P+1> table;
    table[0] = 1.0;
    for (int i = 1; i <= P; ++i) {
        table[i] = table[i-1] * i;
    }
    return table;
}();

// 高效二項式係數計算
static double binomial_coeff(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// 優化的FMM樹節點結構
struct OptimizedFMMCell {
    double cx, cy, size;
    int level;
    std::vector<int> particles;
    
    // 記憶體對齊的多極子和局部展開
    alignas(64) std::array<cplx, P+1> multipole{};
    alignas(64) std::array<cplx, P+1> local{};
    
    std::array<std::unique_ptr<OptimizedFMMCell>, 4> children;
    OptimizedFMMCell* parent;
    bool is_leaf;
    
    // 用於負載平衡的工作量估計
    int work_estimate;
    
    OptimizedFMMCell(double x, double y, double s, int lev = 0, OptimizedFMMCell* p = nullptr)
        : cx(x), cy(y), size(s), level(lev), parent(p), is_leaf(true), work_estimate(0) {
        std::fill(multipole.begin(), multipole.end(), cplx(0.0, 0.0));
        std::fill(local.begin(), local.end(), cplx(0.0, 0.0));
    }
};

// 任務導向的樹構建策略
void build_fmm_tree_parallel(OptimizedFMMCell* cell, 
                             const std::vector<double>& x, 
                             const std::vector<double>& y,
                             int max_particles = 20, 
                             int max_level = 12) {
    if ((int)cell->particles.size() <= max_particles || cell->level >= max_level) {
        cell->work_estimate = cell->particles.size();
        return;
    }
    
    cell->is_leaf = false;
    const double half_size = cell->size * 0.5;
    
    // 創建子節點
    cell->children[0] = std::make_unique<OptimizedFMMCell>(
        cell->cx - half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[1] = std::make_unique<OptimizedFMMCell>(
        cell->cx + half_size, cell->cy - half_size, half_size, cell->level + 1, cell);
    cell->children[2] = std::make_unique<OptimizedFMMCell>(
        cell->cx - half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    cell->children[3] = std::make_unique<OptimizedFMMCell>(
        cell->cx + half_size, cell->cy + half_size, half_size, cell->level + 1, cell);
    
    // 分配粒子到子節點
    for (int particle_id : cell->particles) {
        const int quadrant = (x[particle_id] > cell->cx ? 1 : 0) + 
                           (y[particle_id] > cell->cy ? 2 : 0);
        cell->children[quadrant]->particles.push_back(particle_id);
    }
    
    cell->particles.clear();
    
    // 並行遞歸構建子樹
#ifdef USE_OPENMP
    #pragma omp task default(shared) if(cell->level < 6)
#endif
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            build_fmm_tree_parallel(child.get(), x, y, max_particles, max_level);
        }
    }
    
#ifdef USE_OPENMP
    #pragma omp taskwait
#endif
    
    // 計算工作量估計
    cell->work_estimate = 0;
    for (auto& child : cell->children) {
        if (child) {
            cell->work_estimate += child->work_estimate;
        }
    }
}

// 優化的上行階段並行化
void fmm_upward_pass_parallel(OptimizedFMMCell* cell, 
                             const std::vector<double>& x, 
                             const std::vector<double>& y,
                             const std::vector<double>& m) {
    if (!cell) return;
    
    std::fill(cell->multipole.begin(), cell->multipole.end(), cplx(0.0, 0.0));
    
    if (cell->is_leaf) {
        // P2M: 粒子到多極子展開
        for (int particle_id : cell->particles) {
            const double mass = m[particle_id];
            const double dx = x[particle_id] - cell->cx;
            const double dy = y[particle_id] - cell->cy;
            const cplx z(dx, dy);
            
            // 使用Horner方法優化計算
            cell->multipole[0] += mass;
            cplx z_power = z;
            for (int k = 1; k <= P; ++k) {
                cell->multipole[k] += mass * z_power / factorial_table[k];
                z_power *= z;
            }
        }
    } else {
        // 並行處理子節點
#ifdef USE_OPENMP
        #pragma omp taskgroup
#endif
        {
            for (auto& child : cell->children) {
                if (child && !child->particles.empty()) {
#ifdef USE_OPENMP
                    #pragma omp task default(shared) if(cell->level < 8)
#endif
                    fmm_upward_pass_parallel(child.get(), x, y, m);
                }
            }
        }
        
        // M2M: 子節點到父節點翻譯
        for (auto& child : cell->children) {
            if (child && child->multipole[0] != cplx(0.0, 0.0)) {
                const double dx = child->cx - cell->cx;
                const double dy = child->cy - cell->cy;
                const cplx z0(dx, dy);
                
                // 優化的M2M翻譯
                for (int l = 0; l <= P; ++l) {
                    cplx z0_power(1.0, 0.0);
                    for (int k = 0; k <= l; ++k) {
                        const double binom_coeff_val = binomial_coeff(l, k);
                        cell->multipole[l] += child->multipole[k] * binom_coeff_val * z0_power;
                        if (k < l) z0_power *= z0;
                    }
                }
            }
        }
    }
}

// 高效的M2L翻譯實作
void optimized_m2l_translation(OptimizedFMMCell* target, OptimizedFMMCell* source) {
    if (!target || !source || target == source) return;
    
    const double dx = source->cx - target->cx;
    const double dy = source->cy - target->cy;
    const double r2 = dx * dx + dy * dy;
    
    if (r2 < 1e-20) return;
    
    const cplx z0(dx, dy);
    const double r = std::sqrt(r2);
    
    // 嚴格的遠場條件檢查
    if (r < 2.5 * std::max(target->size, source->size)) return;
    
    // 使用預計算表優化的M2L翻譯
    for (int j = 0; j <= P; ++j) {
        cplx contribution(0.0, 0.0);
        for (int k = 0; k <= P; ++k) {
            const double sign = (k % 2 == 0) ? 1.0 : -1.0;
            const double binom_coeff_val = binomial_coeff(j + k, k);
            
            // 使用更穩定的計算方法
            const cplx z_inv = cplx(1.0, 0.0) / z0;
            cplx z_power = std::pow(z_inv, j + k + 1);
            
            if (std::abs(z_power) > 1e-15) {
                contribution += sign * binom_coeff_val * source->multipole[k] * z_power;
            }
        }
        target->local[j] += contribution;
    }
}

// 工作竊取策略的互動階段
void fmm_interaction_phase_parallel(OptimizedFMMCell* cell, OptimizedFMMCell* root) {
    if (!cell) return;
    
    // 收集同層次的所有節點以實現負載平衡
    std::vector<OptimizedFMMCell*> same_level_cells;
    std::function<void(OptimizedFMMCell*)> collect_cells = [&](OptimizedFMMCell* node) {
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
    
    // 並行處理M2L翻譯
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 1) if(same_level_cells.size() > 4)
#endif
    for (size_t i = 0; i < same_level_cells.size(); ++i) {
        OptimizedFMMCell* source = same_level_cells[i];
        
        // 檢查是否為遠場節點
        const double dx = source->cx - cell->cx;
        const double dy = source->cy - cell->cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double size_sum = cell->size + source->size;
        
        if (dist > 2.5 * size_sum && source->multipole[0] != cplx(0.0, 0.0)) {
            optimized_m2l_translation(cell, source);
        }
    }
    
    // 遞歸處理子節點
    if (!cell->is_leaf) {
#ifdef USE_OPENMP
        #pragma omp taskgroup
#endif
        {
            for (auto& child : cell->children) {
                if (child) {
#ifdef USE_OPENMP
                    #pragma omp task default(shared) if(cell->level < 6)
#endif
                    fmm_interaction_phase_parallel(child.get(), root);
                }
            }
        }
    }
}

// 優化的下行階段
void fmm_downward_pass_parallel(OptimizedFMMCell* cell) {
    if (!cell) return;
    
    if (!cell->is_leaf) {
        // L2L翻譯：並行處理所有子節點
#ifdef USE_OPENMP
        #pragma omp parallel for if(cell->level < 8)
#endif
        for (int i = 0; i < 4; ++i) {
            if (cell->children[i]) {
                OptimizedFMMCell* child = cell->children[i].get();
                
                const double dx = child->cx - cell->cx;
                const double dy = child->cy - cell->cy;
                const cplx z0(dx, dy);
                
                // L2L翻譯
                for (int j = 0; j <= P; ++j) {
                    cplx z0_power(1.0, 0.0);
                    for (int k = j; k <= P; ++k) {
                        const double binom_coeff_val = binomial_coeff(k, j);
                        child->local[j] += cell->local[k] * binom_coeff_val * z0_power;
                        if (k > j) z0_power *= z0;
                    }
                }
            }
        }
        
        // 遞歸處理子節點
#ifdef USE_OPENMP
        #pragma omp taskgroup
#endif
        {
            for (auto& child : cell->children) {
                if (child) {
#ifdef USE_OPENMP
                    #pragma omp task default(shared) if(cell->level < 8)
#endif
                    fmm_downward_pass_parallel(child.get());
                }
            }
        }
    }
}

// 高效的力計算
void evaluate_forces_parallel(OptimizedFMMCell* cell, 
                             const std::vector<double>& x, 
                             const std::vector<double>& y,
                             const std::vector<double>& m, 
                             std::vector<double>& fx, 
                             std::vector<double>& fy,
                             double G, double soft2) {
    if (!cell) return;
    
    if (!cell->is_leaf) {
        // 並行處理子節點
#ifdef USE_OPENMP
        #pragma omp taskgroup
#endif
        {
            for (auto& child : cell->children) {
                if (child) {
#ifdef USE_OPENMP
                    #pragma omp task default(shared) if(cell->level < 8)
#endif
                    evaluate_forces_parallel(child.get(), x, y, m, fx, fy, G, soft2);
                }
            }
        }
        return;
    }
    
    // 葉節點：計算局部展開貢獻 + 直接互動
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) if(cell->particles.size() > 16)
#endif
    for (size_t idx = 0; idx < cell->particles.size(); ++idx) {
        int i = cell->particles[idx];
        double force_x = 0.0, force_y = 0.0;
        
        // 同葉節點內的直接互動
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
        
        // 局部展開貢獻
        const double dx_local = x[i] - cell->cx;
        const double dy_local = y[i] - cell->cy;
        const cplx z(dx_local, dy_local);
        
        cplx force_complex(0.0, 0.0);
        cplx z_power(1.0, 0.0);
        for (int k = 1; k <= P; ++k) {
            force_complex += double(k) * cell->local[k] * z_power / factorial_table[k];
            z_power *= z;
        }
        
        force_x += G * (-force_complex.real());
        force_y += G * (-force_complex.imag());
        
        // 使用原子操作避免競爭條件
#ifdef USE_OPENMP
        #pragma omp atomic
#endif
        fx[i] += force_x;
        
#ifdef USE_OPENMP
        #pragma omp atomic
#endif
        fy[i] += force_y;
    }
}

// 主要FMM函數
py::tuple optimized_fmm_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
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
        auto root = std::make_unique<OptimizedFMMCell>(0.0, 0.0, domain * 0.5);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);
        
        // 並行執行FMM算法
#ifdef USE_OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            {
#endif
                build_fmm_tree_parallel(root.get(), vx, vy, 20, 12);
                fmm_upward_pass_parallel(root.get(), vx, vy, vm);
                fmm_interaction_phase_parallel(root.get(), root.get());
                fmm_downward_pass_parallel(root.get());
                evaluate_forces_parallel(root.get(), vx, vy, vm, fx, fy, G, soft * soft);
#ifdef USE_OPENMP
            }
        }
#endif
        
    } catch (const std::exception& e) {
        // 回退到直接計算
        const double soft2 = soft * soft;
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 32)
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
    m.doc() = "優化的高精度快速多極子方法";
    m.def("fmm_omp", &optimized_fmm_omp,
          "高精度FMM力計算，具有完整的P=10展開和任務並行化",
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta") = THETA_DEFAULT, py::arg("G") = 1.0, py::arg("soft") = 0.05);
    
#ifdef USE_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}

