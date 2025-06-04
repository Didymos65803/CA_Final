// final_optimized_kernel.cpp
// Complete final optimized kernel for Intel Xeon Cascadelake systems

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// High-performance cache-blocked algorithm 
void optimized_cache_blocked_force(const py::array_t<double>& x_arr,
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
    
    // Adaptive block size based on problem size
    int block_size;
    if (N < 500) {
        block_size = 32;
    } else if (N < 1500) {
        block_size = 64;
    } else {
        block_size = 128;
    }
    
    // Initialize output
    std::memset(ax_arr.mutable_data(), 0, N * sizeof(double));
    std::memset(ay_arr.mutable_data(), 0, N * sizeof(double));
    
    const int num_blocks = (N + block_size - 1) / block_size;
    
    // Process blocks with careful parallelization
    #pragma omp parallel for schedule(guided) collapse(2) if(N > 800)
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            const int i_start = bi * block_size;
            const int i_end = std::min(i_start + block_size, N);
            const int j_start = bj * block_size;
            const int j_end = std::min(j_start + block_size, N);
            
            // Inner block computation
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

// Single-threaded optimized version
void single_thread_optimized(const py::array_t<double>& x_arr,
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

// Smart adaptive algorithm
void smart_adaptive_force(const py::array_t<double>& x_arr,
                          const py::array_t<double>& y_arr,
                          const py::array_t<double>& m_arr,
                          double eps2,
                          py::array_t<double>& ax_arr,
                          py::array_t<double>& ay_arr) {
    
    const int N = static_cast<int>(x_arr.shape(0));
    
    if (N < 200) {
        single_thread_optimized(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    } else {
        optimized_cache_blocked_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    }
}

// Benchmark and choose the best algorithm
void benchmark_and_choose(const py::array_t<double>& x_arr,
                         const py::array_t<double>& y_arr,
                         const py::array_t<double>& m_arr,
                         double eps2,
                         py::array_t<double>& ax_arr,
                         py::array_t<double>& ay_arr) {
    
    const int N = static_cast<int>(x_arr.shape(0));
    
    if (N < 100) {
        smart_adaptive_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
        return;
    }
    
    // For larger problems, use cache-blocked which performed best
    optimized_cache_blocked_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
}

int get_optimal_threads_for_system() {
    return 2;  // Optimal for Intel Xeon Cascadelake
}

int get_current_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

// Python module definition
PYBIND11_MODULE(final_optimized_kernel, m) {
    m.doc() = "Final optimized kernel for Intel Xeon Cascadelake systems";
    
    m.def("optimized_cache_blocked_force", &optimized_cache_blocked_force,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("eps2"), py::arg("ax"), py::arg("ay"),
          "Highly optimized cache-blocked algorithm");
          
    m.def("single_thread_optimized", &single_thread_optimized,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("eps2"), py::arg("ax"), py::arg("ay"),
          "Single-threaded optimized version");
          
    m.def("smart_adaptive_force", &smart_adaptive_force,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("eps2"), py::arg("ax"), py::arg("ay"),
          "Smart adaptive algorithm");
          
    m.def("benchmark_and_choose", &benchmark_and_choose,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("eps2"), py::arg("ax"), py::arg("ay"),
          "Automatically choose the best algorithm");
          
    m.def("get_optimal_threads_for_system", &get_optimal_threads_for_system,
          "Get optimal thread count for this system");
          
    m.def("get_current_threads", &get_current_threads,
          "Get current thread count");
}