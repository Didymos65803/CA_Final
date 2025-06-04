// minimal_force_kernel.cpp
// Absolutely minimal implementation focused purely on parallel scaling

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Minimal direct force computation - no optimizations, just pure parallelism
void minimal_direct_force(const py::array_t<double>& x_arr,
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
    
    // Simple parallel loop - no fancy optimizations
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0;
        double fy = 0.0;
        
        const double xi = x(i);
        const double yi = y(i);
        
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double force_mag = m(j) * inv_r3;
                
                fx -= force_mag * dx;
                fy -= force_mag * dy;
            }
        }
        
        ax(i) = fx;
        ay(i) = fy;
    }
}

// Test different scheduling strategies
void test_schedule_static(const py::array_t<double>& x_arr,
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
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        const double xi = x(i), yi = y(i);
        
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                fx -= m(j) * inv_r3 * dx;
                fy -= m(j) * inv_r3 * dy;
            }
        }
        ax(i) = fx;
        ay(i) = fy;
    }
}

void test_schedule_dynamic(const py::array_t<double>& x_arr,
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
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        const double xi = x(i), yi = y(i);
        
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                fx -= m(j) * inv_r3 * dx;
                fy -= m(j) * inv_r3 * dy;
            }
        }
        ax(i) = fx;
        ay(i) = fy;
    }
}

void test_schedule_guided(const py::array_t<double>& x_arr,
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
    
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        const double xi = x(i), yi = y(i);
        
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                fx -= m(j) * inv_r3 * dx;
                fy -= m(j) * inv_r3 * dy;
            }
        }
        ax(i) = fx;
        ay(i) = fy;
    }
}

// Manually chunked version to test overhead
void test_manual_chunking(const py::array_t<double>& x_arr,
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
    
    #pragma omp parallel
    {
        const int num_threads = omp_get_num_threads();
        const int thread_id = omp_get_thread_num();
        
        const int chunk_size = (N + num_threads - 1) / num_threads;
        const int start = thread_id * chunk_size;
        const int end = std::min(start + chunk_size, N);
        
        for (int i = start; i < end; ++i) {
            double fx = 0.0, fy = 0.0;
            const double xi = x(i), yi = y(i);
            
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    const double dx = xi - x(j);
                    const double dy = yi - y(j);
                    const double r2 = dx*dx + dy*dy + eps2;
                    const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
                    fx -= m(j) * inv_r3 * dx;
                    fy -= m(j) * inv_r3 * dy;
                }
            }
            ax(i) = fx;
            ay(i) = fy;
        }
    }
}

// Check if OpenMP is actually working
int get_max_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

int get_current_threads() {
    #ifdef _OPENMP
    int threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        threads = omp_get_num_threads();
    }
    return threads;
    #else
    return 1;
    #endif
}

// Python module definition
PYBIND11_MODULE(minimal_force_kernel, m) {
    m.doc() = "Minimal force kernel for parallel testing";
    
    m.def("minimal_direct_force", &minimal_direct_force,
          "Basic parallel direct force computation");
    m.def("test_schedule_static", &test_schedule_static,
          "Test static scheduling");
    m.def("test_schedule_dynamic", &test_schedule_dynamic,
          "Test dynamic scheduling");
    m.def("test_schedule_guided", &test_schedule_guided,
          "Test guided scheduling");
    m.def("test_manual_chunking", &test_manual_chunking,
          "Test manual chunking");
    
    m.def("get_max_threads", &get_max_threads,
          "Get maximum OpenMP threads");
    m.def("get_current_threads", &get_current_threads,
          "Get current OpenMP threads");
}
