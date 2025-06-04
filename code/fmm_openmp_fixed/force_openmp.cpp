// force_openmp.cpp – parallel direct O(N²) reference solver
// =========================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// ------------------------------------------------------------------
// symmetry-aware direct kernel (outer i parallel, inner j>i)
// ------------------------------------------------------------------
void direct_symm(const py::array_t<double>& x_arr,
                 const py::array_t<double>& y_arr,
                 const py::array_t<double>& m_arr,
                 double eps2,
                 py::array_t<double>& ax_arr,
                 py::array_t<double>& ay_arr)
{
    const auto  x = x_arr.unchecked<1>();
    const auto  y = y_arr.unchecked<1>();
    const auto  m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    const int N = x.shape(0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) { ax(i) = 0.0; ay(i) = 0.0; }

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double dx = x(j) - x(i);
            const double dy = y(j) - y(i);
            const double r2 = dx * dx + dy * dy + eps2;
            const double invR  = 1.0 / std::sqrt(r2);
            const double invR3 = invR * invR * invR;
            const double f = m(j) * invR3;

            const double fx = f * dx;
            const double fy = f * dy;

            ax(i) += fx;  ay(i) += fy;
            #pragma omp atomic
            ax(j) -= fx;
            #pragma omp atomic
            ay(j) -= fy;
        }
    }
}

// ------------------------------------------------------------------
// pybind11 wrapper
// ------------------------------------------------------------------
PYBIND11_MODULE(force_openmp, m)
{
    m.doc() = "Symmetry-aware direct N-body solver (OpenMP)";
    m.def("direct_force",
          &direct_symm,
          "O(N²) reference kernel with OpenMP parallelism");
}

