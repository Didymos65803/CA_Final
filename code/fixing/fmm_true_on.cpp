// ---------------------------------------------------------------------------
//  fmm_true_on.cpp
//
//  回到基本：簡單但正確的並行策略
//  不要過度優化，先確保並行有效
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
//  策略1：純並行 O(N²) - 但要做對
// ---------------------------------------------------------------------------
void compute_forces_direct_parallel(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::vector<double>& m,
                                   std::vector<double>& fx,
                                   std::vector<double>& fy,
                                   double eps2) {
    const int N = x.size();
    
    // 最簡單的並行：每個線程負責一些粒子
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double fx_local = 0.0;
        double fy_local = 0.0;
        
        // 計算粒子i受到的所有力
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r2 = dx*dx + dy*dy + eps2;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double f = m[j] * inv_r3;
            
            fx_local += f * dx;
            fy_local += f * dy;
        }
        
        fx[i] = fx_local;
        fy[i] = fy_local;
    }
}

// ---------------------------------------------------------------------------
//  策略2：分塊並行 O(N²) - 更好的cache利用
// ---------------------------------------------------------------------------
void compute_forces_block_parallel(const std::vector<double>& x,
                                  const std::vector<double>& y,
                                  const std::vector<double>& m,
                                  std::vector<double>& fx,
                                  std::vector<double>& fy,
                                  double eps2) {
    const int N = x.size();
    const int BLOCK_SIZE = 256;  // 適合cache的塊大小
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int block_i = 0; block_i < N; block_i += BLOCK_SIZE) {
        int end_i = std::min(block_i + BLOCK_SIZE, N);
        
        // 每個線程處理一個塊
        for (int i = block_i; i < end_i; ++i) {
            double fx_local = 0.0;
            double fy_local = 0.0;
            
            // 分塊處理j方向
            for (int block_j = 0; block_j < N; block_j += BLOCK_SIZE) {
                int end_j = std::min(block_j + BLOCK_SIZE, N);
                
                for (int j = block_j; j < end_j; ++j) {
                    if (i == j) continue;
                    
                    double dx = x[j] - x[i];
                    double dy = y[j] - y[i];
                    double r2 = dx*dx + dy*dy + eps2;
                    double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                    double f = m[j] * inv_r3;
                    
                    fx_local += f * dx;
                    fy_local += f * dy;
                }
            }
            
            fx[i] = fx_local;
            fy[i] = fy_local;
        }
    }
}

// ---------------------------------------------------------------------------
//  策略3：簡化的Barnes-Hut (真正的近似O(N log N))
// ---------------------------------------------------------------------------
struct SimpleCell {
    double xmin, xmax, ymin, ymax;
    double cx, cy, mass;
    bool is_leaf;
    std::vector<int> particles;
    std::vector<SimpleCell> children;
    
    SimpleCell(double x1, double x2, double y1, double y2) 
        : xmin(x1), xmax(x2), ymin(y1), ymax(y2), 
          cx(0.5*(x1+x2)), cy(0.5*(y1+y2)), mass(0), is_leaf(true) {}
    
    double width() const { return xmax - xmin; }
};

void build_simple_tree(SimpleCell& cell, 
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& m,
                      const std::vector<int>& particles,
                      int max_particles = 32,
                      int max_depth = 8) {
    
    // 計算質心
    double total_mass = 0, cx_sum = 0, cy_sum = 0;
    for (int p : particles) {
        total_mass += m[p];
        cx_sum += m[p] * x[p];
        cy_sum += m[p] * y[p];
    }
    
    cell.mass = total_mass;
    if (total_mass > 0) {
        cell.cx = cx_sum / total_mass;
        cell.cy = cy_sum / total_mass;
    }
    
    // 停止條件
    if ((int)particles.size() <= max_particles || max_depth <= 0) {
        cell.is_leaf = true;
        cell.particles = particles;
        return;
    }
    
    // 分割
    cell.is_leaf = false;
    double xmid = 0.5 * (cell.xmin + cell.xmax);
    double ymid = 0.5 * (cell.ymin + cell.ymax);
    
    std::vector<std::vector<int>> child_particles(4);
    for (int p : particles) {
        int quad = (x[p] > xmid) + 2 * (y[p] > ymid);
        child_particles[quad].push_back(p);
    }
    
    // 創建子節點
    cell.children.reserve(4);
    for (int i = 0; i < 4; ++i) {
        if (!child_particles[i].empty()) {
            double x1 = (i & 1) ? xmid : cell.xmin;
            double x2 = (i & 1) ? cell.xmax : xmid;
            double y1 = (i & 2) ? ymid : cell.ymin;
            double y2 = (i & 2) ? cell.ymax : ymid;
            
            cell.children.emplace_back(x1, x2, y1, y2);
            build_simple_tree(cell.children.back(), x, y, m, child_particles[i], 
                            max_particles, max_depth - 1);
        }
    }
}

void compute_cell_force(const SimpleCell& cell,
                       const std::vector<double>& x,
                       const std::vector<double>& y,
                       const std::vector<double>& m,
                       int target, double eps2, double theta2,
                       double& fx, double& fy) {
    
    if (cell.mass == 0) return;
    
    double dx = cell.cx - x[target];
    double dy = cell.cy - y[target];
    double r2 = dx*dx + dy*dy + eps2;
    double s = cell.width();
    
    // Barnes-Hut 開角條件
    if (cell.is_leaf || (s*s < theta2 * r2)) {
        if (cell.is_leaf) {
            // 直接計算
            for (int p : cell.particles) {
                if (p == target) continue;
                
                double dx2 = x[p] - x[target];
                double dy2 = y[p] - y[target];
                double r2_direct = dx2*dx2 + dy2*dy2 + eps2;
                double inv_r3 = 1.0 / (r2_direct * std::sqrt(r2_direct));
                double f = m[p] * inv_r3;
                
                fx += f * dx2;
                fy += f * dy2;
            }
        } else {
            // Monopole 近似
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double f = cell.mass * inv_r3;
            fx += f * dx;
            fy += f * dy;
        }
    } else {
        // 遞歸
        for (const auto& child : cell.children) {
            compute_cell_force(child, x, y, m, target, eps2, theta2, fx, fy);
        }
    }
}

void compute_forces_barnes_hut_parallel(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       const std::vector<double>& m,
                                       std::vector<double>& fx,
                                       std::vector<double>& fy,
                                       double eps2, double domain_size) {
    const int N = x.size();
    
    // 建樹（串行）
    std::vector<int> all_particles(N);
    for (int i = 0; i < N; ++i) all_particles[i] = i;
    
    SimpleCell root(-domain_size/2, domain_size/2, -domain_size/2, domain_size/2);
    build_simple_tree(root, x, y, m, all_particles);
    
    // 並行計算力
    double theta2 = 0.36;  // theta = 0.6
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i) {
        double fx_local = 0.0, fy_local = 0.0;
        compute_cell_force(root, x, y, m, i, eps2, theta2, fx_local, fy_local);
        fx[i] = fx_local;
        fy[i] = fy_local;
    }
}

// ---------------------------------------------------------------------------
//  主函數：智能選擇策略
// ---------------------------------------------------------------------------
void fmm_force_on(py::array_t<double> x_arr,
                  py::array_t<double> y_arr,
                  py::array_t<double> m_arr,
                  double eps2,
                  py::array_t<double> domain_arr,
                  double theta,
                  py::array_t<double> ax_arr,
                  py::array_t<double> ay_arr)
{
    auto x_view = x_arr.unchecked<1>();
    auto y_view = y_arr.unchecked<1>();
    auto m_view = m_arr.unchecked<1>();
    auto domain = domain_arr.unchecked<1>();
    auto axw = ax_arr.mutable_unchecked<1>();
    auto ayw = ay_arr.mutable_unchecked<1>();
    const int N = (int)x_arr.shape(0);
    
    // 複製到連續記憶體
    std::vector<double> x(N), y(N), m(N), fx(N), fy(N);
    for (int i = 0; i < N; ++i) {
        x[i] = x_view(i);
        y[i] = y_view(i);
        m[i] = m_view(i);
    }
    
    double domain_size = domain(1) - domain(0);
    
    // 策略選擇
    if (N < 2000) {
        // 小問題：直接並行
        compute_forces_direct_parallel(x, y, m, fx, fy, eps2);
    } else if (N < 10000) {
        // 中等問題：分塊並行  
        compute_forces_block_parallel(x, y, m, fx, fy, eps2);
    } else {
        // 大問題：Barnes-Hut
        compute_forces_barnes_hut_parallel(x, y, m, fx, fy, eps2, domain_size);
    }
    
    // 複製結果
    for (int i = 0; i < N; ++i) {
        axw(i) = fx[i];
        ayw(i) = fy[i];
    }
}

// ---------------------------------------------------------------------------
//  PyBind11 模組
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fmm_true_on, m) {
    m.doc() = "Simple but effective parallel force calculation";
    m.def("fmm_force_on", &fmm_force_on,
          "fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)");
}
