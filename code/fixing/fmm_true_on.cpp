// ---------------------------------------------------------------------------
//  fmm_true_on.cpp
//
//  專門優化大N並行性能：
//  1. 改善記憶體存取模式
//  2. 減少cache miss
//  3. 更好的負載平衡
//  4. 避免false sharing
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <omp.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
//  優化的資料結構：記憶體對齊，避免false sharing
// ---------------------------------------------------------------------------
struct alignas(64) Particle {  // Cache line對齊
    double x, y, m;
    double fx, fy;
    char padding[24];  // 填充到64位元組
};

struct SimpleCell {
    double xmin, xmax, ymin, ymax;
    double cx, cy, mass;
    bool is_leaf;
    std::vector<int> particles;
    std::vector<std::unique_ptr<SimpleCell>> children;
    
    SimpleCell(double x1, double x2, double y1, double y2) 
        : xmin(x1), xmax(x2), ymin(y1), ymax(y2), 
          cx(0.5*(x1+x2)), cy(0.5*(y1+y2)), mass(0), is_leaf(true) {
        children.reserve(4);
    }
    
    double width() const { return xmax - xmin; }
};

// ---------------------------------------------------------------------------
//  策略1：超並行 O(N²) - 針對小N優化
// ---------------------------------------------------------------------------
void compute_forces_ultra_parallel(std::vector<Particle>& particles, double eps2) {
    const int N = particles.size();
    
    // 極細粒度並行：每個粒子一個工作單元
    #pragma omp parallel for schedule(static, 1) num_threads(omp_get_max_threads())
    for (int i = 0; i < N; ++i) {
        double fx_local = 0.0;
        double fy_local = 0.0;
        
        const Particle& pi = particles[i];
        
        // 向量化友好的內迴圈
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            
            const Particle& pj = particles[j];
            
            double dx = pj.x - pi.x;
            double dy = pj.y - pi.y;
            double r2 = dx*dx + dy*dy + eps2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double f = pj.m * inv_r3;
            
            fx_local += f * dx;
            fy_local += f * dy;
        }
        
        particles[i].fx = fx_local;
        particles[i].fy = fy_local;
    }
}

// ---------------------------------------------------------------------------
//  策略2：NUMA友好的分塊並行
// ---------------------------------------------------------------------------
void compute_forces_numa_parallel(std::vector<Particle>& particles, double eps2) {
    const int N = particles.size();
    const int num_threads = omp_get_max_threads();
    
    // 大塊分割，每個線程處理連續的記憶體區域
    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int chunk_size = (N + num_threads - 1) / num_threads;
        const int start = tid * chunk_size;
        const int end = std::min(start + chunk_size, N);
        
        // 每個線程有自己的工作區域
        for (int i = start; i < end; ++i) {
            double fx_local = 0.0;
            double fy_local = 0.0;
            
            const Particle& pi = particles[i];
            
            // 分塊處理j，提高cache命中率
            const int BLOCK_SIZE = 1024;
            for (int jblock = 0; jblock < N; jblock += BLOCK_SIZE) {
                const int jend = std::min(jblock + BLOCK_SIZE, N);
                
                for (int j = jblock; j < jend; ++j) {
                    if (i == j) continue;
                    
                    const Particle& pj = particles[j];
                    
                    double dx = pj.x - pi.x;
                    double dy = pj.y - pi.y;
                    double r2 = dx*dx + dy*dy + eps2;
                    double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                    double f = pj.m * inv_r3;
                    
                    fx_local += f * dx;
                    fy_local += f * dy;
                }
            }
            
            particles[i].fx = fx_local;
            particles[i].fy = fy_local;
        }
    }
}

// ---------------------------------------------------------------------------
//  策略3：高度優化的Barnes-Hut
// ---------------------------------------------------------------------------
void build_optimized_tree(SimpleCell& cell, 
                         const std::vector<Particle>& particles,
                         const std::vector<int>& indices,
                         int max_particles = 16,  // 減小葉節點，增加並行度
                         int max_depth = 10) {
    
    // 計算質心（向量化）
    double total_mass = 0, cx_sum = 0, cy_sum = 0;
    for (int idx : indices) {
        total_mass += particles[idx].m;
        cx_sum += particles[idx].m * particles[idx].x;
        cy_sum += particles[idx].m * particles[idx].y;
    }
    
    cell.mass = total_mass;
    if (total_mass > 0) {
        cell.cx = cx_sum / total_mass;
        cell.cy = cy_sum / total_mass;
    }
    
    // 停止條件
    if ((int)indices.size() <= max_particles || max_depth <= 0) {
        cell.is_leaf = true;
        cell.particles = indices;
        return;
    }
    
    // 快速分割
    cell.is_leaf = false;
    const double xmid = 0.5 * (cell.xmin + cell.xmax);
    const double ymid = 0.5 * (cell.ymin + cell.ymax);
    
    std::vector<std::vector<int>> child_indices(4);
    child_indices[0].reserve(indices.size()/4);
    child_indices[1].reserve(indices.size()/4);
    child_indices[2].reserve(indices.size()/4);
    child_indices[3].reserve(indices.size()/4);
    
    for (int idx : indices) {
        int quad = (particles[idx].x > xmid) + 2 * (particles[idx].y > ymid);
        child_indices[quad].push_back(idx);
    }
    
    // 創建非空子節點
    for (int i = 0; i < 4; ++i) {
        if (!child_indices[i].empty()) {
            double x1 = (i & 1) ? xmid : cell.xmin;
            double x2 = (i & 1) ? cell.xmax : xmid;
            double y1 = (i & 2) ? ymid : cell.ymin;
            double y2 = (i & 2) ? cell.ymax : ymid;
            
            cell.children.push_back(std::make_unique<SimpleCell>(x1, x2, y1, y2));
            build_optimized_tree(*cell.children.back(), particles, child_indices[i], 
                               max_particles, max_depth - 1);
        }
    }
}

// 內聯的力計算函數
inline void compute_cell_force_optimized(const SimpleCell& cell,
                                        const std::vector<Particle>& particles,
                                        const Particle& target,
                                        double eps2, double theta2,
                                        double& fx, double& fy) {
    
    if (cell.mass == 0) return;
    
    const double dx = cell.cx - target.x;
    const double dy = cell.cy - target.y;
    const double r2 = dx*dx + dy*dy + eps2;
    const double s = cell.width();
    
    // Barnes-Hut 開角條件
    if (cell.is_leaf || (s*s < theta2 * r2)) {
        if (cell.is_leaf) {
            // 直接計算（向量化友好）
            for (int idx : cell.particles) {
                const Particle& p = particles[idx];
                if (&p == &target) continue;
                
                const double dx2 = p.x - target.x;
                const double dy2 = p.y - target.y;
                const double r2_direct = dx2*dx2 + dy2*dy2 + eps2;
                const double inv_r3 = 1.0 / (r2_direct * std::sqrt(r2_direct));
                const double f = p.m * inv_r3;
                
                fx += f * dx2;
                fy += f * dy2;
            }
        } else {
            // Monopole 近似
            const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            const double f = cell.mass * inv_r3;
            fx += f * dx;
            fy += f * dy;
        }
    } else {
        // 遞歸
        for (const auto& child : cell.children) {
            compute_cell_force_optimized(*child, particles, target, eps2, theta2, fx, fy);
        }
    }
}

void compute_forces_barnes_hut_optimized(std::vector<Particle>& particles,
                                       double eps2, double domain_size) {
    const int N = particles.size();
    
    // 建樹
    std::vector<int> all_indices(N);
    for (int i = 0; i < N; ++i) all_indices[i] = i;
    
    SimpleCell root(-domain_size/2, domain_size/2, -domain_size/2, domain_size/2);
    build_optimized_tree(root, particles, all_indices);
    
    // 高度並行的力計算
    const double theta2 = 0.25;  // theta = 0.5, 更積極的近似
    
    #pragma omp parallel
    {
        // 動態調度，小chunk提高負載平衡
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < N; ++i) {
            double fx_local = 0.0, fy_local = 0.0;
            compute_cell_force_optimized(root, particles, particles[i], 
                                       eps2, theta2, fx_local, fy_local);
            particles[i].fx = fx_local;
            particles[i].fy = fy_local;
        }
    }
}

// ---------------------------------------------------------------------------
//  主函數：智能選擇最優策略
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
    
    // 使用對齊的粒子陣列
    std::vector<Particle> particles(N);
    
    // 並行初始化
    #pragma omp parallel for if(N > 1000)
    for (int i = 0; i < N; ++i) {
        particles[i].x = x_view(i);
        particles[i].y = y_view(i);
        particles[i].m = m_view(i);
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
    }
    
    const double domain_size = domain(1) - domain(0);
    const int num_threads = omp_get_max_threads();
    
    // 智能策略選擇
    if (N < 500) {
        // 極小問題：超並行O(N²)
        compute_forces_ultra_parallel(particles, eps2);
    } else if (N < 3000) {
        // 小到中等問題：NUMA友好並行O(N²)
        compute_forces_numa_parallel(particles, eps2);
    } else {
        // 大問題：優化的Barnes-Hut
        compute_forces_barnes_hut_optimized(particles, eps2, domain_size);
    }
    
    // 並行複製結果
    #pragma omp parallel for if(N > 1000)
    for (int i = 0; i < N; ++i) {
        axw(i) = particles[i].fx;
        ayw(i) = particles[i].fy;
    }
}

// ---------------------------------------------------------------------------
//  PyBind11 模組
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fmm_true_on, m) {
    m.doc() = "Highly optimized parallel force calculation for all N ranges";
    m.def("fmm_force_on", &fmm_force_on,
          "fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)");
}
