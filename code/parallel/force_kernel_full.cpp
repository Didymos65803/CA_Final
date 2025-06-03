// force_kernel_full.cpp
// Fixed version with proper cache blocking and false sharing avoidance

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Cache-friendly parameters
const int CACHE_LINE_SIZE = 64;
const int CACHE_BLOCK_SIZE = 64;  // Optimized for L1 cache

// Aligned structure to avoid false sharing
struct alignas(CACHE_LINE_SIZE) ThreadLocalData {
    double ax;
    double ay;
    char padding[CACHE_LINE_SIZE - 2*sizeof(double)];
    
    ThreadLocalData() : ax(0.0), ay(0.0) {}
};

void direct_force(const py::array_t<double>& x_arr,
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
    
    const ssize_t N = x.shape(0);
    
    if (N != y.shape(0) || N != m.shape(0) || N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in direct_force");
    }
    
    // Initialize output arrays
    #pragma omp parallel for schedule(static) if(N > 50)
    for (ssize_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    if (N <= 50) {
        // Sequential for very small problems
        for (ssize_t i = 0; i < N; ++i) {
            const double xi = x(i);
            const double yi = y(i);
            double axi = 0.0;
            double ayi = 0.0;
            
            for (ssize_t j = 0; j < N; ++j) {
                if (i == j) continue;
                
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                
                if (r2 > eps2) {
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    const double mj = m(j);
                    axi -= mj * dx * inv_r3;
                    ayi -= mj * dy * inv_r3;
                }
            }
            
            ax(i) = axi;
            ay(i) = ayi;
        }
    } else {
        // Cache-blocked parallel computation
        const int max_threads = omp_get_max_threads();
        
        // Use static scheduling with cache-friendly chunk size
        #pragma omp parallel num_threads(max_threads)
        {
            const int thread_id = omp_get_thread_num();
            const int num_threads = omp_get_num_threads();
            
            // Calculate work distribution to minimize false sharing
            const ssize_t chunk_size = std::max(ssize_t(1), N / num_threads);
            const ssize_t start = thread_id * chunk_size;
            const ssize_t end = (thread_id == num_threads - 1) ? N : start + chunk_size;
            
            // Each thread works on its assigned range
            for (ssize_t i = start; i < end; ++i) {
                const double xi = x(i);
                const double yi = y(i);
                double axi = 0.0;
                double ayi = 0.0;
                
                // Cache-blocked inner loop
                for (ssize_t j_block = 0; j_block < N; j_block += CACHE_BLOCK_SIZE) {
                    const ssize_t j_end = std::min(j_block + CACHE_BLOCK_SIZE, N);
                    
                    // Vectorizable inner loop with good cache locality
                    for (ssize_t j = j_block; j < j_end; ++j) {
                        if (i == j) continue;
                        
                        const double dx = xi - x(j);
                        const double dy = yi - y(j);
                        const double r2 = dx*dx + dy*dy + eps2;
                        
                        if (r2 > eps2) {
                            const double inv_r = 1.0 / std::sqrt(r2);
                            const double inv_r3 = inv_r * inv_r * inv_r;
                            const double mj = m(j);
                            axi -= mj * dx * inv_r3;
                            ayi -= mj * dy * inv_r3;
                        }
                    }
                }
                
                ax(i) = axi;
                ay(i) = ayi;
            }
        }
    }
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "2D direct O(N^2) gravitational kernel (Cache-optimized)";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"));
}

