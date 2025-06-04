// force_openmp.cpp
// =================
//
// A symmetry‐aware direct O(N^2) N‐body solver with OpenMP.
// Each thread accumulates partial forces in its own buffer
// and then reduces them in a single critical section to avoid atomic contention.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// direct_symm: computes pairwise forces in O(N^2) fashion.
//   x_arr, y_arr, m_arr: input arrays of length N
//   eps2: softening parameter squared
//   ax_arr, ay_arr: output arrays (length N) to be filled with accelerations
void direct_symm(
    const py::array_t<double>& x_arr,
    const py::array_t<double>& y_arr,
    const py::array_t<double>& m_arr,
    double eps2,
    py::array_t<double>& ax_arr,
    py::array_t<double>& ay_arr)
{
    // 1) Extract raw pointers / read‐only views
    const auto x = x_arr.unchecked<1>();
    const auto y = y_arr.unchecked<1>();
    const auto m = m_arr.unchecked<1>();
    // 2) Get mutable access for ax, ay
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    const int N = x.shape(0);

    // 3) Zero out ax, ay in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    // 4) Each thread accumulates partial forces into private buffers
    #pragma omp parallel
    {
        // Private buffers of size N for each thread
        std::vector<double> ax_loc(N, 0.0), ay_loc(N, 0.0);

        // Compute pairwise interactions (i < j)
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const double dx = x(j) - x(i);
                const double dy = y(j) - y(i);
                const double r2 = dx*dx + dy*dy + eps2;
                const double invR  = 1.0 / std::sqrt(r2);
                const double invR3 = invR * invR * invR;
                const double f = m(j) * invR3;
                const double fx = f * dx;
                const double fy = f * dy;
                // Accumulate in private buffers
                ax_loc[i] += fx;    ay_loc[i] += fy;
                ax_loc[j] -= fx;    ay_loc[j] -= fy;
            }
        }

        // 5) Reduce partial buffers into the global ax, ay in one critical section
        #pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                ax(i) += ax_loc[i];
                ay(i) += ay_loc[i];
            }
        }
    } // end parallel region
}

// Pybind11 module definition
PYBIND11_MODULE(force_openmp, m) {
    m.doc() = "O(N^2) direct N-body solver with OpenMP";
    m.def(
        "direct_symm",
        &direct_symm,
        "direct_symm(x, y, m, eps2, ax, ay)  # O(N^2) pairwise forces",
        py::arg("x_arr"),
        py::arg("y_arr"),
        py::arg("m_arr"),
        py::arg("eps2"),
        py::arg("ax_arr"),
        py::arg("ay_arr")
    );
}

