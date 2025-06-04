// memory_optimized_kernel.cpp
// Fixed and simplified memory-optimized kernel

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Simple cache-blocked algorithm
void cache_blocked_force(const py::array_t<double>& x_arr,
                        const py::array_t<double>& y_arr,
                        const py::array_t<double>& m_arr,
                        double eps2,
                        py::array_t<double>& ax_arr,
                        py::array_t<double>& ay_arr) {
    
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    const int block_size = 64; // Cache-friendly block size
    
    // Initialize output
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    const int num_blocks = (N + block_size - 1) / block_size;
    
    // Process in blocks for better cache utilization
    #pragma omp parallel for schedule(guided) collapse(2)
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            const int i_start = bi * block_size;
            const int i_end = std::min(i_start + block_size, N);
            const int j_start = bj * block_size;
            const int j_end = std::min(j_start + block_size, N);
            
            // Process block (bi, bj)
            for (int i = i_start; i < i_end; i++) {
                const double xi = x(i);
                const double yi = y(i);
                
                double fx_local = 0.0;
                double fy_local = 0.0;
                
                // Vectorized inner loop
                #pragma omp simd reduction(+:fx_local,fy_local)
                for (int j = j_start; j < j_end; j++) {
                    if (i != j) {
                        const double dx = xi - x(j);
                        const double dy = yi - y(j);
                        const double r2 = dx*dx + dy*dy + eps2;
                        const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                        const double force_mag = m(j) * inv_r3;
                        
                        fx_local -= force_mag * dx;
                        fy_local -= force_mag * dy;
                    }
                }
                
                // Atomic update to avoid race conditions
                #pragma omp atomic
                ax(i) += fx_local;
                #pragma omp atomic
                ay(i) += fy_local;
            }
        }
    }
}

// Optimized for memory bandwidth limited systems
void bandwidth_optimized_force(const py::array_t<double>& x_arr,
                              const py::array_t<double>& y_arr,
                              const py::array_t<double>& m_arr,
                              double eps2,
                              py::array_t<double>& ax_arr,
                              py::array_t<double>& ay_arr) {
    
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    
    // Use fewer threads for memory-bound problems
    const int optimal_threads = std::min(omp_get_max_threads(), 
                                        std::max(1, N / 600));
    
    #pragma omp parallel for schedule(guided) num_threads(optimal_threads)
    for (int i = 0; i < N; i++) {
        const double xi = x(i);
        const double yi = y(i);
        
        double fx_sum = 0.0;
        double fy_sum = 0.0;
        
        // Sequential inner loop for memory locality
        for (int j = 0; j < N; j++) {
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                const double force_mag = m(j) * inv_r3;
                
                fx_sum -= force_mag * dx;
                fy_sum -= force_mag * dy;
            }
        }
        
        ax(i) = fx_sum;
        ay(i) = fy_sum;
    }
}

// Symmetry-exploiting version with better memory access
void symmetric_force(const py::array_t<double>& x_arr,
                     const py::array_t<double>& y_arr,
                     const py::array_t<double>& m_arr,
                     double eps2,
                     py::array_t<double>& ax_arr,
                     py::array_t<double>& ay_arr) {
    
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    
    // Initialize output
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    // Use symmetry to reduce memory accesses by half
    #pragma omp parallel for schedule(guided, 32)
    for (int i = 0; i < N; i++) {
        const double xi = x(i);
        const double yi = y(i);
        const double mi = m(i);
        
        double fx_i = 0.0;
        double fy_i = 0.0;
        
        // Only compute j > i, use Newton's third law
        for (int j = i + 1; j < N; j++) {
            const double dx = xi - x(j);
            const double dy = yi - y(j);
            const double r2 = dx*dx + dy*dy + eps2;
            const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            
            const double fx_ij = dx * inv_r3;
            const double fy_ij = dy * inv_r3;
            
            // Force on particle i from j
            fx_i -= m(j) * fx_ij;
            fy_i -= m(j) * fy_ij;
            
            // Force on particle j from i (Newton's third law)
            #pragma omp atomic
            ax(j) += mi * fx_ij;
            #pragma omp atomic
            ay(j) += mi * fy_ij;
        }
        
        // Update particle i (no atomic needed since i is unique per thread)
        ax(i) += fx_i;
        ay(i) += fy_i;
    }
}

// Adaptive algorithm that chooses best method based on problem size
void adaptive_force(const py::array_t<double>& x_arr,
                   const py::array_t<double>& y_arr,
                   const py::array_t<double>& m_arr,
                   double eps2,
                   py::array_t<double>& ax_arr,
                   py::array_t<double>& ay_arr) {
    
    const int N = static_cast<int>(x_arr.shape(0));
    
    // Choose algorithm based on problem size
    if (N < 500) {
        // Small problems: use bandwidth optimization
        bandwidth_optimized_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    } else if (N < 1500) {
        // Medium problems: use symmetry
        symmetric_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    } else {
        // Large problems: use cache blocking
        cache_blocked_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    }
}

// Single-threaded optimized version for comparison
void serial_optimized_force(const py::array_t<double>& x_arr,
                           const py::array_t<double>& y_arr,
                           const py::array_t<double>& m_arr,
                           double eps2,
                           py::array_t<double>& ax_arr,
                           py::array_t<double>& ay_arr) {
    
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = static_cast<int>(x.shape(0));
    
    for (int i = 0; i < N; i++) {
        const double xi = x(i);
        const double yi = y(i);
        
        double fx_sum = 0.0;
        double fy_sum = 0.0;
        
        // Cache-friendly loop with prefetching
        for (int j = 0; j < N; j++) {
            // Prefetch next iteration
            if (j + 8 < N) {
                __builtin_prefetch(&x(j + 8), 0, 1);
                __builtin_prefetch(&y(j + 8), 0, 1);
                __builtin_prefetch(&m(j + 8), 0, 1);
            }
            
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                const double force_mag = m(j) * inv_r3;
                
                fx_sum -= force_mag * dx;
                fy_sum -= force_mag * dy;
            }
        }
        
        ax(i) = fx_sum;
        ay(i) = fy_sum;
    }
}

// Get runtime information
int get_optimal_threads() {
    #ifdef _OPENMP
    return std::min(omp_get_max_threads(), 2); // Conservative for memory-bound
    #else
    return 1;
    #endif
}

// Python module definition
PYBIND11_MODULE(memory_optimized_kernel, m) {
    m.doc() = "Memory-optimized force kernel for high-latency systems";
    
    m.def("adaptive_force", &adaptive_force,
          "Adaptive algorithm that chooses best method based on problem size");
    m.def("bandwidth_optimized_force", &bandwidth_optimized_force,
          "Optimized for memory bandwidth limited systems");
    m.def("cache_blocked_force", &cache_blocked_force,
          "Cache-blocked algorithm for better memory locality");
    m.def("symmetric_force", &symmetric_force,
          "Symmetry-exploiting algorithm");
    m.def("serial_optimized_force", &serial_optimized_force,
          "Single-threaded optimized version");
    m.def("get_optimal_threads", &get_optimal_threads,
          "Get optimal thread count for memory-bound problems");
}
