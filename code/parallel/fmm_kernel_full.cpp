// fmm_kernel_full.cpp
// Redesigned with spatial decomposition approach

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

// Simple spatial decomposition FMM
class SpatialFMM {
private:
    const double* x_data;
    const double* y_data;
    const double* m_data;
    int N;
    double domain_size;
    double theta;
    double eps;
    double G;
    
    // Use spatial decomposition instead of tree traversal
    void compute_forces_spatial(double* ax, double* ay) {
        const int grid_size = 8; // 8x8 spatial grid
        const double cell_size = domain_size * 2.0 / grid_size;
        
        // Create spatial grid
        std::vector<std::vector<int>> grid(grid_size * grid_size);
        
        // Assign particles to grid cells
        for (int i = 0; i < N; ++i) {
            int grid_x = std::max(0, std::min(grid_size - 1, 
                         static_cast<int>((x_data[i] + domain_size) / cell_size)));
            int grid_y = std::max(0, std::min(grid_size - 1, 
                         static_cast<int>((y_data[i] + domain_size) / cell_size)));
            grid[grid_y * grid_size + grid_x].push_back(i);
        }
        
        // Compute forces using spatial decomposition
        #pragma omp parallel for schedule(dynamic, 1) if(N > 500)
        for (int i = 0; i < N; ++i) {
            ax[i] = 0.0;
            ay[i] = 0.0;
            
            const double xi = x_data[i];
            const double yi = y_data[i];
            
            // Determine which grid cell this particle is in
            int grid_x = std::max(0, std::min(grid_size - 1, 
                         static_cast<int>((xi + domain_size) / cell_size)));
            int grid_y = std::max(0, std::min(grid_size - 1, 
                         static_cast<int>((yi + domain_size) / cell_size)));
            
            // Interact with nearby cells (3x3 neighborhood)
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = grid_x + dx;
                    int ny = grid_y + dy;
                    
                    if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                        const auto& cell = grid[ny * grid_size + nx];
                        
                        // Direct interaction with particles in this cell
                        for (int j : cell) {
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
            
            // For distant cells, use multipole approximation
            for (int cy = 0; cy < grid_size; ++cy) {
                for (int cx = 0; cx < grid_size; ++cx) {
                    // Skip nearby cells (already computed above)
                    if (std::abs(cx - grid_x) <= 1 && std::abs(cy - grid_y) <= 1) {
                        continue;
                    }
                    
                    const auto& cell = grid[cy * grid_size + cx];
                    if (cell.empty()) continue;
                    
                    // Compute cell center and total mass
                    double cell_cx = (cx + 0.5) * cell_size - domain_size;
                    double cell_cy = (cy + 0.5) * cell_size - domain_size;
                    double total_mass = 0.0;
                    double mass_cx = 0.0;
                    double mass_cy = 0.0;
                    
                    for (int j : cell) {
                        double mj = m_data[j];
                        total_mass += mj;
                        mass_cx += mj * x_data[j];
                        mass_cy += mj * y_data[j];
                    }
                    
                    if (total_mass > 0.0) {
                        mass_cx /= total_mass;
                        mass_cy /= total_mass;
                        
                        // Use monopole approximation
                        const double dx_val = xi - mass_cx;
                        const double dy_val = yi - mass_cy;
                        const double r2 = dx_val*dx_val + dy_val*dy_val + eps*eps;
                        
                        if (r2 > eps*eps) {
                            const double inv_r3 = G / (r2 * std::sqrt(r2));
                            ax[i] += total_mass * dx_val * inv_r3;
                            ay[i] += total_mass * dy_val * inv_r3;
                        }
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
        theta = th;
        eps = epsilon;
        G = gravity;
        
        if (N < 1000) {
            // For small problems, fall back to direct method
            #pragma omp parallel for schedule(static) if(N > 100)
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
            // Use spatial decomposition for larger problems
            compute_forces_spatial(ax, ay);
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
    
    if (N <= 0) {
        throw std::runtime_error("Invalid particle count");
    }
    
    try {
        SpatialFMM fmm;
        
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
    m.doc() = "2D Spatial Decomposition FMM kernel (Optimized)";
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

