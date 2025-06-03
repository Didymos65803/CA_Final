// fmm_kernel_full.cpp
// Redesigned with better parallelization strategy

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

const int CACHE_LINE_SIZE = 64;

// Thread-safe spatial grid implementation
class OptimizedSpatialFMM {
private:
    struct alignas(CACHE_LINE_SIZE) GridCell {
        std::vector<int> particles;
        double total_mass;
        double center_x, center_y;
        char padding[CACHE_LINE_SIZE - sizeof(std::vector<int>) - 3*sizeof(double)];
        
        GridCell() : total_mass(0.0), center_x(0.0), center_y(0.0) {}
    };
    
    const double* x_data;
    const double* y_data;
    const double* m_data;
    int N;
    double domain_size;
    double eps;
    double G;
    
    void compute_forces_optimized(double* ax, double* ay) {
        const int grid_size = std::max(4, static_cast<int>(std::sqrt(N / 100.0)));
        const double cell_size = domain_size * 2.0 / grid_size;
        
        // Create thread-safe spatial grid
        std::vector<GridCell> grid(grid_size * grid_size);
        
        // Assign particles to grid cells (sequential to avoid race conditions)
        for (int i = 0; i < N; ++i) {
            int grid_x = std::max(0, std::min(grid_size - 1,
                         static_cast<int>((x_data[i] + domain_size) / cell_size)));
            int grid_y = std::max(0, std::min(grid_size - 1,
                         static_cast<int>((y_data[i] + domain_size) / cell_size)));
            
            int cell_idx = grid_y * grid_size + grid_x;
            grid[cell_idx].particles.push_back(i);
        }
        
        // Compute cell mass centers (parallel)
        #pragma omp parallel for schedule(static)
        for (int cell_idx = 0; cell_idx < grid_size * grid_size; ++cell_idx) {
            GridCell& cell = grid[cell_idx];
            if (cell.particles.empty()) continue;
            
            double total_mass = 0.0;
            double mx_sum = 0.0;
            double my_sum = 0.0;
            
            for (int pi : cell.particles) {
                double mi = m_data[pi];
                total_mass += mi;
                mx_sum += mi * x_data[pi];
                my_sum += mi * y_data[pi];
            }
            
            cell.total_mass = total_mass;
            if (total_mass > 0.0) {
                cell.center_x = mx_sum / total_mass;
                cell.center_y = my_sum / total_mass;
            }
        }
        
        // Compute forces with better parallelization
        const int max_threads = omp_get_max_threads();
        const int chunk_size = std::max(1, N / (max_threads * 8));
        
        #pragma omp parallel for schedule(guided, chunk_size)
        for (int i = 0; i < N; ++i) {
            ax[i] = 0.0;
            ay[i] = 0.0;
            
            const double xi = x_data[i];
            const double yi = y_data[i];
            
            // Find particle's grid cell
            int grid_x = std::max(0, std::min(grid_size - 1,
                         static_cast<int>((xi + domain_size) / cell_size)));
            int grid_y = std::max(0, std::min(grid_size - 1,
                         static_cast<int>((yi + domain_size) / cell_size)));
            
            // Direct interaction with nearby cells
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = grid_x + dx;
                    int ny = grid_y + dy;
                    
                    if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                        const GridCell& cell = grid[ny * grid_size + nx];
                        
                        for (int j : cell.particles) {
                            if (i == j) continue;
                            
                            const double dx_val = xi - x_data[j];
                            const double dy_val = yi - y_data[j];
                            const double r2 = dx_val*dx_val + dy_val*dy_val + eps*eps;
                            
                            if (r2 > eps*eps) {
                                const double inv_r3 = G / (r2 * std::sqrt(r2));
                                const double mj = m_data[j];
                                ax[i] += mj * dx_val * inv_r3;
                                ay[i] += mj * dy_val * inv_r3;
                            }
                        }
                    }
                }
            }
            
            // Multipole approximation for distant cells
            for (int cy = 0; cy < grid_size; ++cy) {
                for (int cx = 0; cx < grid_size; ++cx) {
                    // Skip nearby cells
                    if (std::abs(cx - grid_x) <= 1 && std::abs(cy - grid_y) <= 1) {
                        continue;
                    }
                    
                    const GridCell& cell = grid[cy * grid_size + cx];
                    if (cell.total_mass == 0.0) continue;
                    
                    const double dx_val = xi - cell.center_x;
                    const double dy_val = yi - cell.center_y;
                    const double r2 = dx_val*dx_val + dy_val*dy_val + eps*eps;
                    
                    if (r2 > eps*eps) {
                        const double inv_r3 = G / (r2 * std::sqrt(r2));
                        ax[i] += cell.total_mass * dx_val * inv_r3;
                        ay[i] += cell.total_mass * dy_val * inv_r3;
                    }
                }
            }
        }
    }
    
public:
    void compute_forces(const double* x, const double* y, const double* m, int n,
                       double domain, double th, int max_leaf_particles,
                       double epsilon, double gravity,
                       double* ax, double* ay) {
        
        x_data = x;
        y_data = y;
        m_data = m;
        N = n;
        domain_size = domain;
        eps = epsilon;
        G = gravity;
        
        if (N < 200) {
            // For small problems, use direct method
            #pragma omp parallel for schedule(static) if(N > 50)
            for (int i = 0; i < N; ++i) {
                ax[i] = 0.0;
                ay[i] = 0.0;
                
                for (int j = 0; j < N; ++j) {
                    if (i == j) continue;
                    
                    const double dx = x[i] - x[j];
                    const double dy = y[i] - y[j];
                    const double r2 = dx*dx + dy*dy + eps*eps;
                    
                    if (r2 > eps*eps) {
                        const double inv_r3 = G / (r2 * std::sqrt(r2));
                        const double mj = m[j];
                        ax[i] += mj * dx * inv_r3;
                        ay[i] += mj * dy * inv_r3;
                    }
                }
            }
        } else {
            // Use optimized spatial decomposition
            compute_forces_optimized(ax, ay);
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
        OptimizedSpatialFMM fmm;
        
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
    m.doc() = "2D Optimized Spatial FMM kernel";
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

