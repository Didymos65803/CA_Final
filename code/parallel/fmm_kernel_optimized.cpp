// fmm_kernel_optimized.cpp
// 完全重寫的高效FMM實現，解決並行化和記憶體存取問題

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// 修正1: 使用對齊的數據結構避免false sharing
struct alignas(64) ParticleData {
    double x, y, m;
    double ax, ay;
    int cell_id;
    char padding[26]; // 確保64字節對齊
};

// 修正2: 優化網格結構，使用連續記憶體布局
struct alignas(64) GridCell {
    std::vector<int> particle_indices;
    double total_mass;
    double center_x, center_y;
    double moments[4]; // 多極矩：monopole, dipole_x, dipole_y, quadrupole
    bool is_computed;
    char padding[31]; // 避免false sharing
    
    GridCell() : total_mass(0.0), center_x(0.0), center_y(0.0), 
                 is_computed(false) {
        std::fill(moments, moments + 4, 0.0);
    }
};

class OptimizedFMM {
private:
    // 修正3: 使用連續記憶體布局的粒子數據
    std::vector<ParticleData> particles;
    std::vector<GridCell> grid;
    int grid_size;
    double cell_size;
    double domain_size;
    double eps;
    double G;
    int num_threads;
    
    // 修正5: 使用SIMD優化的距離計算
    inline void compute_force_simd(double xi, double yi, double mi,
                                   double xj, double yj, double mj,
                                   double& ax, double& ay) const {
        const double dx = xi - xj;
        const double dy = yi - yj;
        const double r2 = dx*dx + dy*dy + eps*eps;
        
        if (r2 > eps*eps) {
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double force_factor = -G * mj * inv_r3;
            
            ax += force_factor * dx;
            ay += force_factor * dy;
        }
    }
    
    // 修正6: 向量化的多粒子力計算
    void compute_force_vectorized(int start_i, int end_i, 
                                  const std::vector<int>& target_indices) {
        for (int idx_i = start_i; idx_i < end_i; ++idx_i) {
            const int i = target_indices[idx_i];
            double ax_local = 0.0;
            double ay_local = 0.0;
            
            const double xi = particles[i].x;
            const double yi = particles[i].y;
            const double mi = particles[i].m;
            
            // 內層循環使用SIMD
            #pragma omp simd reduction(+:ax_local,ay_local)
            for (size_t idx_j = 0; idx_j < target_indices.size(); ++idx_j) {
                const int j = target_indices[idx_j];
                if (i != j) {
                    double ax_temp = 0.0, ay_temp = 0.0;
                    compute_force_simd(xi, yi, mi, 
                                       particles[j].x, particles[j].y, particles[j].m,
                                       ax_temp, ay_temp);
                    ax_local += ax_temp;
                    ay_local += ay_temp;
                }
            }
            
            particles[i].ax += ax_local;
            particles[i].ay += ay_local;
        }
    }
    
    // 修正7: 改進的網格分配策略
    void assign_particles_to_grid() {
        // 重置網格
        for (auto& cell : grid) {
            cell.particle_indices.clear();
            cell.total_mass = 0.0;
            cell.is_computed = false;
            std::fill(cell.moments, cell.moments + 4, 0.0);
        }
        
        // 順序分配避免競爭條件
        const size_t N = particles.size();
        for (size_t i = 0; i < N; ++i) {
            const int grid_x = std::clamp(
                static_cast<int>((particles[i].x + domain_size) / cell_size),
                0, grid_size - 1);
            const int grid_y = std::clamp(
                static_cast<int>((particles[i].y + domain_size) / cell_size),
                0, grid_size - 1);
            
            const int cell_id = grid_y * grid_size + grid_x;
            particles[i].cell_id = cell_id;
            grid[cell_id].particle_indices.push_back(static_cast<int>(i));
        }
    }
    
    // 修正8: 並行化的質心計算
    void compute_cell_properties() {
        std::vector<int> non_empty_cells;
        non_empty_cells.reserve(grid_size * grid_size / 4); // 預估容量
        
        const size_t grid_total = grid.size();
        for (size_t i = 0; i < grid_total; ++i) {
            if (!grid[i].particle_indices.empty()) {
                non_empty_cells.push_back(static_cast<int>(i));
            }
        }
        
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (size_t idx = 0; idx < non_empty_cells.size(); ++idx) {
            const int cell_id = non_empty_cells[idx];
            GridCell& cell = grid[cell_id];
            
            double total_mass = 0.0;
            double mx_sum = 0.0, my_sum = 0.0;
            
            // 使用局部變數提高cache locality
            for (const int pi : cell.particle_indices) {
                const double mi = particles[pi].m;
                const double xi = particles[pi].x;
                const double yi = particles[pi].y;
                
                total_mass += mi;
                mx_sum += mi * xi;
                my_sum += mi * yi;
            }
            
            cell.total_mass = total_mass;
            if (total_mass > 0.0) {
                cell.center_x = mx_sum / total_mass;
                cell.center_y = my_sum / total_mass;
            }
            
            // 計算多極矩
            compute_multipole_moments(cell);
            cell.is_computed = true;
        }
    }
    
    // 修正9: 多極矩計算
    void compute_multipole_moments(GridCell& cell) {
        if (cell.particle_indices.empty()) return;
        
        // monopole moment (total mass)
        cell.moments[0] = cell.total_mass;
        
        // dipole moments
        double dipole_x = 0.0, dipole_y = 0.0;
        for (const int pi : cell.particle_indices) {
            const double mi = particles[pi].m;
            const double dx = particles[pi].x - cell.center_x;
            const double dy = particles[pi].y - cell.center_y;
            
            dipole_x += mi * dx;
            dipole_y += mi * dy;
        }
        cell.moments[1] = dipole_x;
        cell.moments[2] = dipole_y;
        
        // quadrupole moment (simplified)
        double quadrupole = 0.0;
        for (const int pi : cell.particle_indices) {
            const double mi = particles[pi].m;
            const double dx = particles[pi].x - cell.center_x;
            const double dy = particles[pi].y - cell.center_y;
            const double r2 = dx*dx + dy*dy;
            
            quadrupole += mi * r2;
        }
        cell.moments[3] = quadrupole;
    }
    
    // 修正10: 工作竊取式的並行力計算
    void compute_forces_work_stealing() {
        // 初始化加速度
        const size_t N = particles.size();
        #pragma omp parallel for simd num_threads(num_threads)
        for (size_t i = 0; i < N; ++i) {
            particles[i].ax = 0.0;
            particles[i].ay = 0.0;
        }
        
        // 創建工作隊列
        std::vector<std::pair<int, int>> work_items; // (particle_index, interaction_type)
        work_items.reserve(N * 2);
        
        for (size_t i = 0; i < N; ++i) {
            work_items.emplace_back(static_cast<int>(i), 0); // 近場交互作用
            work_items.emplace_back(static_cast<int>(i), 1); // 遠場交互作用
        }
        
        // 並行處理工作項目
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for schedule(guided, 16) nowait
            for (size_t work_idx = 0; work_idx < work_items.size(); ++work_idx) {
                const int particle_idx = work_items[work_idx].first;
                const int interaction_type = work_items[work_idx].second;
                
                if (interaction_type == 0) {
                    compute_near_field_interaction(particle_idx);
                } else {
                    compute_far_field_interaction(particle_idx);
                }
            }
        }
    }
    
    // 修正11: 優化的近場交互作用
    void compute_near_field_interaction(int particle_idx) {
        const int cell_id = particles[particle_idx].cell_id;
        const int grid_x = cell_id % grid_size;
        const int grid_y = cell_id / grid_size;
        
        double ax_local = 0.0;
        double ay_local = 0.0;
        
        const double xi = particles[particle_idx].x;
        const double yi = particles[particle_idx].y;
        
        // 遍歷相鄰網格（包括自身）
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = grid_x + dx;
                const int ny = grid_y + dy;
                
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                    const int neighbor_cell_id = ny * grid_size + nx;
                    const GridCell& neighbor_cell = grid[neighbor_cell_id];
                    
                    // 內層循環向量化
                    for (const int j : neighbor_cell.particle_indices) {
                        if (particle_idx != j) {
                            compute_force_simd(xi, yi, particles[particle_idx].m,
                                               particles[j].x, particles[j].y, particles[j].m,
                                               ax_local, ay_local);
                        }
                    }
                }
            }
        }
        
        // 原子操作更新結果
        #pragma omp atomic
        particles[particle_idx].ax += ax_local;
        #pragma omp atomic
        particles[particle_idx].ay += ay_local;
    }
    
    // 修正12: 多極展開的遠場交互作用
    void compute_far_field_interaction(int particle_idx) {
        const int cell_id = particles[particle_idx].cell_id;
        const int grid_x = cell_id % grid_size;
        const int grid_y = cell_id / grid_size;
        
        double ax_local = 0.0;
        double ay_local = 0.0;
        
        const double xi = particles[particle_idx].x;
        const double yi = particles[particle_idx].y;
        
        // 遍歷遠場網格
        for (int cy = 0; cy < grid_size; ++cy) {
            for (int cx = 0; cx < grid_size; ++cx) {
                // 跳過近場網格
                if (std::abs(cx - grid_x) <= 1 && std::abs(cy - grid_y) <= 1) {
                    continue;
                }
                
                const int remote_cell_id = cy * grid_size + cx;
                const GridCell& remote_cell = grid[remote_cell_id];
                
                if (!remote_cell.is_computed || remote_cell.total_mass == 0.0) {
                    continue;
                }
                
                // 計算到質心的距離
                const double dx = xi - remote_cell.center_x;
                const double dy = yi - remote_cell.center_y;
                const double r2 = dx*dx + dy*dy + eps*eps;
                
                if (r2 > eps*eps) {
                    // 使用多極展開（這裡簡化為monopole）
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    const double force_factor = -G * remote_cell.total_mass * inv_r3;
                    
                    ax_local += force_factor * dx;
                    ay_local += force_factor * dy;
                }
            }
        }
        
        // 原子操作更新結果
        #pragma omp atomic
        particles[particle_idx].ax += ax_local;
        #pragma omp atomic
        particles[particle_idx].ay += ay_local;
    }
    
public:
    void compute_forces(const double* x, const double* y, const double* m, int n,
                       double domain, double theta, int max_leaf_particles,
                       double epsilon, double gravity,
                       double* ax, double* ay) {
        
        // 初始化參數
        domain_size = domain;
        eps = epsilon;
        G = gravity;
        num_threads = omp_get_max_threads();
        
        // 動態決定網格大小
        if (n < 1000) {
            grid_size = 8;
        } else if (n < 5000) {
            grid_size = 16;
        } else if (n < 20000) {
            grid_size = 32;
        } else {
            grid_size = 64;
        }
        
        cell_size = (2.0 * domain_size) / grid_size;
        
        // 重新調整容器大小
        particles.resize(n);
        grid.resize(grid_size * grid_size);
        
        // 複製數據到優化的結構
        const size_t N_size = static_cast<size_t>(n);
        #pragma omp parallel for simd num_threads(num_threads)
        for (size_t i = 0; i < N_size; ++i) {
            particles[i].x = x[i];
            particles[i].y = y[i];
            particles[i].m = m[i];
            particles[i].ax = 0.0;
            particles[i].ay = 0.0;
        }
        
        // 小問題直接計算
        if (n < 500) {
            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int i = 0; i < n; ++i) {
                double ax_local = 0.0;
                double ay_local = 0.0;
                
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        compute_force_simd(particles[i].x, particles[i].y, particles[i].m,
                                           particles[j].x, particles[j].y, particles[j].m,
                                           ax_local, ay_local);
                    }
                }
                
                particles[i].ax = ax_local;
                particles[i].ay = ay_local;
            }
        } else {
            // 大問題使用FMM
            assign_particles_to_grid();
            compute_cell_properties();
            compute_forces_work_stealing();
        }
        
        // 複製結果
        #pragma omp parallel for simd num_threads(num_threads)
        for (size_t i = 0; i < N_size; ++i) {
            ax[i] = particles[i].ax;
            ay[i] = particles[i].ay;
        }
    }
};

void fmm_force(const py::array_t<double>& x_arr,
               const py::array_t<double>& y_arr,
               const py::array_t<double>& m_arr,
               int N,
               double domain_size,
               double theta,
               int maxLeaf,
               double eps,
               double G,
               py::array_t<double>& ax_arr,
               py::array_t<double>& ay_arr)
{
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    if (N != x.shape(0) || N != y.shape(0) || N != m.shape(0) || 
        N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in fmm_force");
    }
    
    try {
        OptimizedFMM fmm;
        
        const double* x_ptr = x.data(0);
        const double* y_ptr = y.data(0);
        const double* m_ptr = m.data(0);
        double* ax_ptr = ax.mutable_data(0);
        double* ay_ptr = ay.mutable_data(0);
        
        fmm.compute_forces(x_ptr, y_ptr, m_ptr, N,
                          domain_size, theta, maxLeaf,
                          eps, G, ax_ptr, ay_ptr);
                          
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("FMM computation failed: ") + e.what());
    }
}

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "Highly Optimized 2D FMM kernel with proper parallelization";
    m.def("fmm_force",
          &fmm_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("N"),
          py::arg("domain_size") = 50.0,
          py::arg("theta") = 0.5,
          py::arg("maxLeaf") = 8,
          py::arg("eps") = 0.01,
          py::arg("G") = 1.0,
          py::arg("ax"),
          py::arg("ay"));
}
