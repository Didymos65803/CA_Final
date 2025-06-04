// force_kernel_fixed_final.cpp
// Optimized direct force computation with proper load balancing

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Block size for cache optimization
constexpr int BLOCK_SIZE = 64;

// Optimized force computation kernel
inline void compute_force_pair(double xi, double yi, double mi,
                              double xj, double yj, double mj,
                              double eps2, double& fx, double& fy) {
    const double dx = xi - xj;
    const double dy = yi - yj;
    const double r2 = dx*dx + dy*dy + eps2;
    
    if (r2 > eps2) {
        const double inv_r = 1.0 / std::sqrt(r2);
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double force_magnitude = mj * inv_r3;
        
        fx -= force_magnitude * dx;
        fy -= force_magnitude * dy;
    }
}

// Block-based computation for better cache locality
void compute_force_block(const double* x, const double* y, const double* m,
                        int N, double eps2,
                        int i_start, int i_end,
                        int j_start, int j_end,
                        std::vector<double>& fx_local,
                        std::vector<double>& fy_local) {
    
    for (int i = i_start; i < i_end; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double mi = m[i];
        
        double fx_sum = 0.0;
        double fy_sum = 0.0;
        
        #pragma omp simd reduction(+:fx_sum,fy_sum)
        for (int j = j_start; j < j_end; ++j) {
            if (i != j) {
                double fx_temp = 0.0, fy_temp = 0.0;
                compute_force_pair(xi, yi, mi, x[j], y[j], m[j], eps2, fx_temp, fy_temp);
                fx_sum += fx_temp;
                fy_sum += fy_temp;
            }
        }
        
        fx_local[i] += fx_sum;
        fy_local[i] += fy_sum;
    }
}

// Main direct force computation function
void direct_force(const py::array_t<double>& x_arr,
                  const py::array_t<double>& y_arr,
                  const py::array_t<double>& m_arr,
                  double eps2,
                  py::array_t<double>& ax_arr,
                  py::array_t<double>& ay_arr)
{
    // Get array accessors
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    
    // Validate array sizes
    if (N != y.shape(0) || N != m.shape(0) || N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in direct_force");
    }
    
    // Initialize output arrays
    #pragma omp parallel for simd
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    const int num_threads = omp_get_max_threads();
    
    if (N <= 50) {
        // Small problems: simple sequential computation
        for (int i = 0; i < N; ++i) {
            double fx = 0.0, fy = 0.0;
            
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    compute_force_pair(x(i), y(i), m(i), x(j), y(j), m(j), eps2, fx, fy);
                }
            }
            
            ax(i) = fx;
            ay(i) = fy;
        }
    } else if (N <= 1000) {
        // Medium problems: use symmetry optimization
        std::vector<std::vector<double>> fx_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> fy_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& fx_local = fx_threads[tid];
            auto& fy_local = fy_threads[tid];
            
            #pragma omp for schedule(guided, 32) nowait
            for (int i = 0; i < N; ++i) {
                const double xi = x(i);
                const double yi = y(i);
                const double mi = m(i);
                
                // Use symmetry: only compute j > i
                for (int j = i + 1; j < N; ++j) {
                    const double dx = xi - x(j);
                    const double dy = yi - y(j);
                    const double r2 = dx*dx + dy*dy + eps2;
                    
                    if (r2 > eps2) {
                        const double inv_r = 1.0 / std::sqrt(r2);
                        const double inv_r3 = inv_r * inv_r * inv_r;
                        const double mj = m(j);
                        
                        const double fx_ij = dx * inv_r3;
                        const double fy_ij = dy * inv_r3;
                        
                        // Apply Newton's third law
                        fx_local[i] -= mj * fx_ij;
                        fy_local[i] -= mj * fy_ij;
                        fx_local[j] += mi * fx_ij;
                        fy_local[j] += mi * fy_ij;
                    }
                }
            }
        }
        
        // Reduce results from all threads
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            double fx_sum = 0.0, fy_sum = 0.0;
            
            for (int tid = 0; tid < num_threads; ++tid) {
                fx_sum += fx_threads[tid][i];
                fy_sum += fy_threads[tid][i];
            }
            
            ax(i) = fx_sum;
            ay(i) = fy_sum;
        }
    } else {
        // Large problems: use block-based algorithm
        const int block_size = std::min(BLOCK_SIZE, N / num_threads + 1);
        const int num_blocks = (N + block_size - 1) / block_size;
        
        std::vector<std::vector<double>> fx_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> fy_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& fx_local = fx_threads[tid];
            auto& fy_local = fy_threads[tid];
            
            // Block-based computation
            #pragma omp for schedule(guided, 1) collapse(2) nowait
            for (int bi = 0; bi < num_blocks; ++bi) {
                for (int bj = 0; bj < num_blocks; ++bj) {
                    const int i_start = bi * block_size;
                    const int i_end = std::min(i_start + block_size, N);
                    const int j_start = bj * block_size;
                    const int j_end = std::min(j_start + block_size, N);
                    
                    if (bi == bj) {
                        // Diagonal block: use symmetry
                        for (int i = i_start; i < i_end; ++i) {
                            for (int j = std::max(j_start, i + 1); j < j_end; ++j) {
                                const double dx = x(i) - x(j);
                                const double dy = y(i) - y(j);
                                const double r2 = dx*dx + dy*dy + eps2;
                                
                                if (r2 > eps2) {
                                    const double inv_r = 1.0 / std::sqrt(r2);
                                    const double inv_r3 = inv_r * inv_r * inv_r;
                                    
                                    const double fx_ij = dx * inv_r3;
                                    const double fy_ij = dy * inv_r3;
                                    
                                    fx_local[i] -= m(j) * fx_ij;
                                    fy_local[i] -= m(j) * fy_ij;
                                    fx_local[j] += m(i) * fx_ij;
                                    fy_local[j] += m(i) * fy_ij;
                                }
                            }
                        }
                    } else if (bi < bj) {
                        // Upper triangular block: compute and apply symmetry
                        compute_force_block(x.data(0), y.data(0), m.data(0), N, eps2,
                                          i_start, i_end, j_start, j_end, fx_local, fy_local);
                    }
                    // Lower triangular blocks are handled by symmetry
                }
            }
        }
        
        // Final reduction
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < N; ++i) {
            double fx_sum = 0.0, fy_sum = 0.0;
            
            #pragma omp simd reduction(+:fx_sum,fy_sum)
            for (int tid = 0; tid < num_threads; ++tid) {
                fx_sum += fx_threads[tid][i];
                fy_sum += fy_threads[tid][i];
            }
            
            ax(i) = fx_sum;
            ay(i) = fy_sum;
        }
    }
}

// Python module definition
PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "Optimized 2D Direct O(N^2) Gravitational Force Kernel";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"),
          "Compute gravitational forces using direct O(N^2) method");
}
