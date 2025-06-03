// force_kernel_full.cpp
// --------------------------------------------------
// PyBind11 + OpenMP implementation of a 2D direct
// O(N^2) gravitational‐force kernel with Plummer
// softening, accepting NumPy arrays.
//
// Exposes a single function:
//
//    direct_force(x, y, m, eps2, ax, ay)
//
//   where x, y, m, ax, ay are NumPy arrays of dtype float64.
//
// Build flags (setup.py passes):
//   -std=c++17 -O3 -DNDEBUG -march=native -ffast-math -fopenmp
// --------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

/*
 * void direct_force(
 *     const py::array_t<double>& x_arr,
 *     const py::array_t<double>& y_arr,
 *     const py::array_t<double>& m_arr,
 *     double eps2,
 *     py::array_t<double>& ax_arr,
 *     py::array_t<double>& ay_arr
 * )
 *
 * Computes 2D gravitational accelerations by brute‐force O(N^2) summation:
 *   a_i = - Σ_{j≠i} m_j (r_i - r_j) / ( (|r_i - r_j|^2 + eps2)^(3/2) )
 *
 * Input:
 *   x_arr  : shape (N,), dtype=float64
 *   y_arr  : shape (N,), dtype=float64
 *   m_arr  : shape (N,), dtype=float64
 *   eps2   : double (softening parameter squared)
 * Output (in-place):
 *   ax_arr : shape (N,), dtype=float64
 *   ay_arr : shape (N,), dtype=float64
 */
void direct_force(const py::array_t<double>& x_arr,
                  const py::array_t<double>& y_arr,
                  const py::array_t<double>& m_arr,
                  double eps2,
                  py::array_t<double>& ax_arr,
                  py::array_t<double>& ay_arr)
{
    // Request buffers (unchecked is fastest; no bounds‐checking)
    auto x = x_arr.unchecked<1>();    // read‐only
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();  // writeable
    auto ay = ay_arr.mutable_unchecked<1>();

    const ssize_t N = x.shape(0);
    // Zero out output arrays
    #pragma omp parallel for schedule(static)
    for(ssize_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    // Main O(N^2) loop, parallelized over i
    #pragma omp parallel for schedule(static)
    for(ssize_t i = 0; i < N; ++i) {
        double xi = x(i);
        double yi = y(i);
        double axi = 0.0;
        double ayi = 0.0;

        for(ssize_t j = 0; j < N; ++j) {
            if(i == j) continue;
            double dx = xi - x(j);
            double dy = yi - y(j);
            double r2 = dx*dx + dy*dy + eps2;
            double inv_r3 = 1.0 / (r2 * sqrt(r2));
            double mj = m(j);
            axi -= mj * dx * inv_r3;
            ayi -= mj * dy * inv_r3;
        }
        ax(i) = axi;
        ay(i) = ayi;
    }
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "2D direct O(N^2) gravitational kernel (OpenMP, NumPy arrays)";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"));
}

