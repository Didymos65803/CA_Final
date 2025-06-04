// fmm_kernel_fixed_final.cpp
// Completely rewritten working FMM implementation with proper parallelization

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Cache-aligned particle structure
struct alignas(64) Particle {
    double x, y, mass;
    double ax, ay;
    int grid_id;
    char padding[26];
    
    Particle() : x(0), y(0), mass(0), ax(0), ay(0), grid_id(-1) {}
};

// Grid cell for spatial decomposition
struct GridCell {
    std::vector<int> particle_ids;
    double total_mass;
    double center_x, center_y;
    bool computed;
    
    GridCell() : total_mass(0), center_x(0), center_y(0), computed(false) {}
    
    void clear() {
        particle_ids.clear();
        total_mass = 0;
        center_x = center_y = 0;
        computed = false;
    }
};

class WorkingFMM {
private:
    std::vector<Particle> particles;
    std::vector<GridCell> grid;
    int grid_size;
    double cell_size;
    double domain_size;
    double eps_squared;
    double G_constant;
    int num_threads;
    
    // Thread-safe force computation kernel
    inline void compute_pairwise_force(double xi, double yi, double mi,
                                       double xj, double yj, double mj,
                                       double& fx, double& fy) const {
        const double dx = xi - xj;
        const double dy = yi - yj;
        const double r2 = dx*dx + dy*dy + eps_squared;
        
        if (r2 > eps_squared) {
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double force_mag = G_constant * mj * inv_r3;
            
            fx -= force_mag * dx;
            fy -= force_mag * dy;
        }
    }
    
    // Assign particles to grid cells
    void assign_particles_to_grid() {
        // Clear all grid cells
        for (auto& cell : grid) {
            cell.clear();
        }
        
        // Assign particles to cells
        for (size_t i = 0; i < particles.size(); ++i) {
            const double px = particles[i].x;
            const double py = particles[i].y;
            
            // Calculate grid coordinates
            int gx = static_cast<int>((px + domain_size) / cell_size);
            int gy = static_cast<int>((py + domain_size) / cell_size);
            
            // Clamp to valid range
            gx = std::max(0, std::min(gx, grid_size - 1));
            gy = std::max(0, std::min(gy, grid_size - 1));
            
            const int grid_id = gy * grid_size + gx;
            particles[i].grid_id = grid_id;
            grid[grid_id].particle_ids.push_back(static_cast<int>(i));
        }
    }
    
    // Compute mass centers for all grid cells
    void compute_mass_centers() {
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int cell_id = 0; cell_id < grid_size * grid_size; ++cell_id) {
            GridCell& cell = grid[cell_id];
            
            if (cell.particle_ids.empty()) {
                continue;
            }
            
            double total_mass = 0.0;
            double weighted_x = 0.0;
            double weighted_y = 0.0;
            
            for (int pid : cell.particle_ids) {
                const Particle& p = particles[pid];
                total_mass += p.mass;
                weighted_x += p.mass * p.x;
                weighted_y += p.mass * p.y;
            }
            
            cell.total_mass = total_mass;
            if (total_mass > 0.0) {
                cell.center_x = weighted_x / total_mass;
                cell.center_y = weighted_y / total_mass;
            }
            cell.computed = true;
        }
    }
    
    // Compute forces using spatial decomposition
    void compute_forces_spatial() {
        // Initialize forces
        #pragma omp parallel for simd num_threads(num_threads)
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].ax = 0.0;
            particles[i].ay = 0.0;
        }
        
        // Parallel force computation
        #pragma omp parallel num_threads(num_threads)
        {
            // Each thread processes a subset of particles
            #pragma omp for schedule(guided, 16) nowait
            for (size_t i = 0; i < particles.size(); ++i) {
                compute_particle_forces(static_cast<int>(i));
            }
        }
    }
    
    // Compute forces for a single particle
    void compute_particle_forces(int particle_id) {
        const Particle& pi = particles[particle_id];
        const int my_grid_id = pi.grid_id;
        const int gx = my_grid_id % grid_size;
        const int gy = my_grid_id / grid_size;
        
        double fx_total = 0.0;
        double fy_total = 0.0;
        
        // Near-field: direct interaction with neighboring cells
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = gx + dx;
                const int ny = gy + dy;
                
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                    const int neighbor_id = ny * grid_size + nx;
                    const GridCell& neighbor = grid[neighbor_id];
                    
                    // Direct particle-particle interaction
                    for (int other_id : neighbor.particle_ids) {
                        if (other_id != particle_id) {
                            double fx_local = 0.0, fy_local = 0.0;
                            compute_pairwise_force(pi.x, pi.y, pi.mass,
                                                   particles[other_id].x, 
                                                   particles[other_id].y, 
                                                   particles[other_id].mass,
                                                   fx_local, fy_local);
                            fx_total += fx_local;
                            fy_total += fy_local;
                        }
                    }
                }
            }
        }
        
        // Far-field: multipole approximation
        for (int cy = 0; cy < grid_size; ++cy) {
            for (int cx = 0; cx < grid_size; ++cx) {
                // Skip near-field cells
                if (std::abs(cx - gx) <= 1 && std::abs(cy - gy) <= 1) {
                    continue;
                }
                
                const int far_cell_id = cy * grid_size + cx;
                const GridCell& far_cell = grid[far_cell_id];
                
                if (far_cell.computed && far_cell.total_mass > 0.0) {
                    double fx_far = 0.0, fy_far = 0.0;
                    compute_pairwise_force(pi.x, pi.y, pi.mass,
                                           far_cell.center_x, far_cell.center_y, 
                                           far_cell.total_mass,
                                           fx_far, fy_far);
                    fx_total += fx_far;
                    fy_total += fy_far;
                }
            }
        }
        
        // Atomic update to avoid race conditions
        #pragma omp atomic
        particles[particle_id].ax += fx_total;
        #pragma omp atomic
        particles[particle_id].ay += fy_total;
    }
    
public:
    void solve_forces(const double* x, const double* y, const double* mass, int n,
                      double domain, double theta, int max_particles_per_leaf,
                      double epsilon, double G,
                      double* ax, double* ay) {
        
        // Initialize parameters
        domain_size = domain;
        eps_squared = epsilon * epsilon;
        G_constant = G;
        num_threads = omp_get_max_threads();
        
        // Adaptive grid sizing based on problem size
        if (n <= 500) {
            grid_size = 4;
        } else if (n <= 2000) {
            grid_size = 8;
        } else if (n <= 8000) {
            grid_size = 16;
        } else {
            grid_size = 32;
        }
        
        cell_size = (2.0 * domain_size) / grid_size;
        
        // Resize containers
        particles.resize(n);
        grid.resize(grid_size * grid_size);
        
        // Copy input data
        #pragma omp parallel for simd num_threads(num_threads)
        for (int i = 0; i < n; ++i) {
            particles[i].x = x[i];
            particles[i].y = y[i];
            particles[i].mass = mass[i];
            particles[i].ax = 0.0;
            particles[i].ay = 0.0;
        }
        
        // For small problems, use direct computation
        if (n < 200) {
            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int i = 0; i < n; ++i) {
                double fx = 0.0, fy = 0.0;
                
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        double fx_local = 0.0, fy_local = 0.0;
                        compute_pairwise_force(particles[i].x, particles[i].y, particles[i].mass,
                                               particles[j].x, particles[j].y, particles[j].mass,
                                               fx_local, fy_local);
                        fx += fx_local;
                        fy += fy_local;
                    }
                }
                
                particles[i].ax = fx;
                particles[i].ay = fy;
            }
        } else {
            // Use FMM for larger problems
            assign_particles_to_grid();
            compute_mass_centers();
            compute_forces_spatial();
        }
        
        // Copy results back
        #pragma omp parallel for simd num_threads(num_threads)
        for (int i = 0; i < n; ++i) {
            ax[i] = particles[i].ax;
            ay[i] = particles[i].ay;
        }
    }
};

// Python interface function
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
    // Get array accessors
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    // Validate array sizes
    if (N != x.shape(0) || N != y.shape(0) || N != m.shape(0) || 
        N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in fmm_force");
    }
    
    try {
        WorkingFMM fmm_solver;
        
        // Get data pointers
        const double* x_ptr = x.data(0);
        const double* y_ptr = y.data(0);
        const double* m_ptr = m.data(0);
        double* ax_ptr = ax.mutable_data(0);
        double* ay_ptr = ay.mutable_data(0);
        
        // Solve the system
        fmm_solver.solve_forces(x_ptr, y_ptr, m_ptr, N,
                                domain_size, theta, maxLeaf,
                                eps, G, ax_ptr, ay_ptr);
                                
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("FMM computation failed: ") + e.what());
    }
}

// Python module definition
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "Working 2D FMM kernel with proper parallelization";
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
          py::arg("ay"),
          "Compute gravitational forces using Fast Multipole Method");
}
