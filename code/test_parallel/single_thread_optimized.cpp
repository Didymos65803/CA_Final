// final_optimized_kernel.cpp
// Final optimized kernel based on successful cache-blocked approach

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// High-performance cache-blocked algorithm with optimized threading
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
    
    // For your system: use limited parallelization to avoid memory contention
    const int max_parallel_blocks = 4;  // Limit concurrent blocks
    
    // Process blocks with careful parallelization
    for (int block_batch = 0; block_batch < num_blocks; block_batch += max_parallel_blocks) {
        const int batch_end = std::min(block_batch + max_parallel_blocks, num_blocks);
        
        #pragma omp parallel for schedule(static) if(N > 800)
        for (int bi = block_batch; bi < batch_end; bi++) {
            const int i_start = bi * block_size;
            const int i_end = std::min(i_start + block_size, N);
            
            // Process this i-block against all j-blocks
            for (int bj = 0; bj < num_blocks; bj++) {
                const int j_start = bj * block_size;
                const int j_end = std::min(j_start + block_size, N);
                
                // Inner block computation - fully optimized
                for (int i = i_start; i < i_end; i++) {
                    const double xi = x(i);
                    const double yi = y(i);
                    
                    double fx_local = 0.0;
                    double fy_local = 0.0;
                    
                    // Vectorized inner loop with prefetching
                    #pragma omp simd aligned(xi,yi:8) reduction(+:fx_local,fy_local)
                    for (int j = j_start; j < j_end; j++) {
                        // Prefetch next cache line
                        if (j + 4 < j_end) {
                            __builtin_prefetch(&x(j + 4), 0, 1);
                            __builtin_prefetch(&y(j + 4), 0, 1);
                            __builtin_prefetch(&m(j + 4), 0, 1);
                        }
                        
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
                    
                    // Accumulate results (minimize atomic operations)
                    ax(i) += fx_local;
                    ay(i) += fy_local;
                }
            }
        }
    }
}

// Symmetry-exploiting version with better threading
void optimized_symmetric_force(const py::array_t<double>& x_arr,
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
    std::memset(ax_arr.mutable_data(), 0, N * sizeof(double));
    std::memset(ay_arr.mutable_data(), 0, N * sizeof(double));
    
    // Thread-local storage to reduce atomic operations
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> fx_threads(num_threads, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> fy_threads(num_threads, std::vector<double>(N, 0.0));
    
    // Parallel computation with work-optimal scheduling
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& fx_local = fx_threads[tid];
        auto& fy_local = fy_threads[tid];
        
        // Dynamic scheduling for load balancing
        #pragma omp for schedule(guided, 16) nowait
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
                fx_local[j] += mi * fx_ij;
                fy_local[j] += mi * fy_ij;
            }
            
            // Store force on particle i
            fx_local[i] += fx_i;
            fy_local[i] += fy_i;
        }
    }
    
    // Reduction phase - combine results from all threads
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double fx_sum = 0.0, fy_sum = 0.0;
        
        for (int tid = 0; tid < num_threads; tid++) {
            fx_sum += fx_threads[tid][i];
            fy_sum += fy_threads[tid][i];
        }
        
        ax(i) = fx_sum;
        ay(i) = fy_sum;
    }
}

// Smart adaptive algorithm based on your system's characteristics
void smart_adaptive_force(const py::array_t<double>& x_arr,
                          const py::array_t<double>& y_arr,
                          const py::array_t<double>& m_arr,
                          double eps2,
                          py::array_t<double>& ax_arr,
                          py::array_t<double>& ay_arr) {
    
    const int N = static_cast<int>(x_arr.shape(0));
    
    // Algorithm selection based on your system's performance characteristics
    if (N < 200) {
        // Small problems: single-threaded is often fastest
        #pragma omp parallel num_threads(1)
        {
            // Use cache-blocked approach even for small problems
            optimized_cache_blocked_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
        }
    } else if (N < 1000) {
        // Medium problems: limited parallelization
        #pragma omp parallel num_threads(2)
        {
            optimized_symmetric_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
        }
    } else {
        // Large problems: use cache-blocked with careful threading
        optimized_cache_blocked_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
    }
}

// Benchmark multiple algorithms and choose the best
void benchmark_and_choose(const py::array_t<double>& x_arr,
                         const py::array_t<double>& y_arr,
                         const py::array_t<double>& m_arr,
                         double eps2,
                         py::array_t<double>& ax_arr,
                         py::array_t<double>& ay_arr) {
    
    const int N = static_cast<int>(x_arr.shape(0));
    
    if (N < 100) {
        // For very small problems, don't waste time benchmarking
        smart_adaptive_force(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
        return;
    }
    
    // Create test arrays
    py::array_t<double> ax_test = py::array_t<double>(N);
    py::array_t<double> ay_test = py::array_t<double>(N);
    
    struct Algorithm {
        std::string name;
        void (*func)(const py::array_t<double>&, const py::array_t<double>&, 
                     const py::array_t<double>&, double, 
                     py::array_t<double>&, py::array_t<double>&);
        double time;
    };
    
    std::vector<Algorithm> algorithms = {
        {"Cache Blocked", optimized_cache_blocked_force, 0.0},
        {"Symmetric", optimized_symmetric_force, 0.0},
        {"Smart Adaptive", smart_adaptive_force, 0.0}
    };
    
    // Quick benchmark
    for (auto& alg : algorithms) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            alg.func(x_arr, y_arr, m_arr, eps2, ax_test, ay_test);
            auto end = std::chrono::high_resolution_clock::now();
            alg.time = std::chrono::duration<double>(end - start).count();
        } catch (...) {
            alg.time = 1e9;  // Very high time for failed algorithms
        }
    }
    
    // Find the fastest algorithm
    auto best = std::min_element(algorithms.begin(), algorithms.end(),
                                [](const Algorithm& a, const Algorithm& b) {
                                    return a.time < b.time;
                                });
    
    // Use the fastest algorithm for the actual computation
    best->func(x_arr, y_arr, m_arr, eps2, ax_arr, ay_arr);
}

// Get system-specific optimal settings
int get_optimal_threads_for_system() {
    #ifdef _OPENMP
    // For your Intel Xeon Cascadelake with high memory latency
    return 2;  // Empirically determined from your tests
    #else
    return 1;
    #endif
}

// Python module definition
PYBIND11_MODULE(final_optimized_kernel, m) {
    m.doc() = "Final optimized kernel for Intel Xeon Cascadelake systems";
    
    m.def("optimized_cache_blocked_force", &optimized_cache_blocked_force,
          "Highly optimized cache-blocked algorithm");
    m.def("optimized_symmetric_force", &optimized_symmetric_force,
          "Optimized symmetry-exploiting algorithm");
    m.def("smart_adaptive_force", &smart_adaptive_force,
          "Smart adaptive algorithm based on system characteristics");
    m.def("benchmark_and_choose", &benchmark_and_choose,
          "Benchmark and automatically choose the best algorithm");
    m.def("get_optimal_threads_for_system", &get_optimal_threads_for_system,
          "Get optimal thread count for this system");
}
