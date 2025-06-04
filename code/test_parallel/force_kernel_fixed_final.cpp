// force_kernel_fixed_final.cpp
// Highly optimized direct force computation with maximum parallelization

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Force computation with aggressive optimizations
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
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    const int num_threads = omp_get_max_threads();
    
    if (N <= 64) {
        // Very small problems: simple sequential computation
        for (int i = 0; i < N; ++i) {
            double fx = 0.0, fy = 0.0;
            
            #pragma omp simd reduction(+:fx,fy)
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    const double dx = x(i) - x(j);
                    const double dy = y(i) - y(j);
                    const double r2 = dx*dx + dy*dy + eps2;
                    
                    if (r2 > eps2) {
                        const double inv_r = 1.0 / std::sqrt(r2);
                        const double inv_r3 = inv_r * inv_r * inv_r;
                        const double force_mag = m(j) * inv_r3;
                        
                        fx -= force_mag * dx;
                        fy -= force_mag * dy;
                    }
                }
            }
            
            ax(i) = fx;
            ay(i) = fy;
        }
    } else if (N <= 2000) {
        // Medium problems: use symmetry with maximum parallelization
        std::vector<std::vector<double>> fx_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> fy_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& fx_local = fx_threads[tid];
            auto& fy_local = fy_threads[tid];
            
            // Distribute work more evenly using a triangular iteration space
            #pragma omp for schedule(guided, 16) nowait
            for (int i = 0; i < N; ++i) {
                const double xi = x(i);
                const double yi = y(i);
                const double mi = m(i);
                
                // Vectorized inner loop
                #pragma omp simd
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
        
        // Parallel reduction
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
    } else {
        // Large problems: chunk-based parallelization with load balancing
        const int chunk_size = std::max(32, N / (num_threads * 4));
        const int num_chunks = (N + chunk_size - 1) / chunk_size;
        
        // Create thread-local storage
        std::vector<std::vector<double>> fx_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> fy_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& fx_local = fx_threads[tid];
            auto& fy_local = fy_threads[tid];
            
            // Dynamic scheduling for load balancing
            #pragma omp for schedule(guided, 1) collapse(2) nowait
            for (int ci = 0; ci < num_chunks; ++ci) {
                for (int cj = 0; cj < num_chunks; ++cj) {
                    const int i_start = ci * chunk_size;
                    const int i_end = std::min(i_start + chunk_size, N);
                    const int j_start = cj * chunk_size;
                    const int j_end = std::min(j_start + chunk_size, N);
                    
                    if (ci == cj) {
                        // Diagonal chunk: use symmetry
                        for (int i = i_start; i < i_end; ++i) {
                            const double xi = x(i);
                            const double yi = y(i);
                            const double mi = m(i);
                            
                            #pragma omp simd
                            for (int j = std::max(j_start, i + 1); j < j_end; ++j) {
                                const double dx = xi - x(j);
                                const double dy = yi - y(j);
                                const double r2 = dx*dx + dy*dy + eps2;
                                
                                if (r2 > eps2) {
                                    const double inv_r = 1.0 / std::sqrt(r2);
                                    const double inv_r3 = inv_r * inv_r * inv_r;
                                    const double mj = m(j);
                                    
                                    const double fx_ij = dx * inv_r3;
                                    const double fy_ij = dy * inv_r3;
                                    
                                    fx_local[i] -= mj * fx_ij;
                                    fy_local[i] -= mj * fy_ij;
                                    fx_local[j] += mi * fx_ij;
                                    fy_local[j] += mi * fy_ij;
                                }
                            }
                        }
                    } else if (ci < cj) {
                        // Upper triangular chunk: compute and apply symmetry
                        for (int i = i_start; i < i_end; ++i) {
                            const double xi = x(i);
                            const double yi = y(i);
                            const double mi = m(i);
                            
                            #pragma omp simd
                            for (int j = j_start; j < j_end; ++j) {
                                const double dx = xi - x(j);
                                const double dy = yi - y(j);
                                const double r2 = dx*dx + dy*dy + eps2;
                                
                                if (r2 > eps2) {
                                    const double inv_r = 1.0 / std::sqrt(r2);
                                    const double inv_r3 = inv_r * inv_r * inv_r;
                                    const double mj = m(j);
                                    
                                    const double fx_ij = dx * inv_r3;
                                    const double fy_ij = dy * inv_r3;
                                    
                                    fx_local[i] -= mj * fx_ij;
                                    fy_local[i] -= mj * fy_ij;
                                    fx_local[j] += mi * fx_ij;
                                    fy_local[j] += mi * fy_ij;
                                }
                            }
                        }
                    }
                    // Lower triangular chunks are handled by symmetry
                }
            }
        }
        
        // Final parallel reduction with optimal scheduling
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

// High-performance vectorized kernel for small chunks
inline void vectorized_force_kernel(const double* __restrict__ x,
                                   const double* __restrict__ y, 
                                   const double* __restrict__ m,
                                   int i_start, int i_end,
                                   int j_start, int j_end,
                                   double eps2,
                                   double* __restrict__ fx,
                                   double* __restrict__ fy) {
    
    for (int i = i_start; i < i_end; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        
        double fx_acc = 0.0;
        double fy_acc = 0.0;
        
        // Vectorized computation
        #pragma omp simd reduction(+:fx_acc,fy_acc)
        for (int j = j_start; j < j_end; ++j) {
            const double dx = xi - x[j];
            const double dy = yi - y[j];
            const double r2 = dx*dx + dy*dy + eps2;
            
            const double mask = (i != j && r2 > eps2) ? 1.0 : 0.0;
            const double inv_r = mask / std::sqrt(r2 + (1.0 - mask));
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double force_mag = m[j] * inv_r3 * mask;
            
            fx_acc -= force_mag * dx;
            fy_acc -= force_mag * dy;
        }
        
        fx[i] += fx_acc;
        fy[i] += fy_acc;
    }
}

// Alternative high-performance implementation for comparison
void direct_force_alternative(const py::array_t<double>& x_arr,
                              const py::array_t<double>& y_arr,
                              const py::array_t<double>& m_arr,
                              double eps2,
                              py::array_t<double>& ax_arr,
                              py::array_t<double>& ay_arr)
{
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    const int num_threads = omp_get_max_threads();
    
    // Initialize
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    // Maximum parallelization approach
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int chunk_size = (N + num_threads - 1) / num_threads;
        const int i_start = tid * chunk_size;
        const int i_end = std::min(i_start + chunk_size, N);
        
        for (int i = i_start; i < i_end; ++i) {
            const double xi = x(i);
            const double yi = y(i);
            
            double fx_sum = 0.0;
            double fy_sum = 0.0;
            
            // Highly optimized inner loop
            #pragma omp simd reduction(+:fx_sum,fy_sum)
            for (int j = 0; j < N; ++j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                
                const double mask = (i != j && r2 > eps2) ? 1.0 : 0.0;
                const double inv_r = mask / std::sqrt(r2 + (1.0 - mask));
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double force_mag = m(j) * inv_r3 * mask;
                
                fx_sum -= force_mag * dx;
                fy_sum -= force_mag * dy;
            }
            
            ax(i) = fx_sum;
            ay(i) = fy_sum;
        }
    }
}

// Python module definition
PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "Highly optimized 2D Direct O(N^2) Gravitational Force Kernel with maximum parallelization";
    
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"),
          "Compute gravitational forces using optimized direct O(N^2) method");
          
    m.def("direct_force_alt",
          &direct_force_alternative,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"),
          "Alternative high-performance direct force computation");
}
