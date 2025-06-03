// force_kernel_full.cpp
// Optimized version with better parallelization

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

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
    #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    // Optimized O(N^2) loop with better cache usage and load balancing
    #pragma omp parallel for schedule(dynamic, 16) collapse(1)
    for (ssize_t i = 0; i < N; ++i) {
        double xi = x(i);
        double yi = y(i);
        double axi = 0.0;
        double ayi = 0.0;
        
        // Vectorizable inner loop
        #pragma omp simd reduction(+:axi,ayi)
        for (ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            
            double dx = xi - x(j);
            double dy = yi - y(j);
            double r2 = dx*dx + dy*dy + eps2;
            
            if (r2 > 0.0) {
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double mj = m(j);
                axi -= mj * dx * inv_r3;
                ayi -= mj * dy * inv_r3;
            }
        }
        
        ax(i) = axi;
        ay(i) = ayi;
    }
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "2D direct O(N^2) gravitational kernel (Optimized OpenMP)";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"));
}

