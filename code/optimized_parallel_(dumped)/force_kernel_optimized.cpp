#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Block sizes tuned for ~64 KB L1‑data cache reuse
constexpr int BLOCK_I = 64;
constexpr int BLOCK_J = 64;

// -----------------------------------------------------------------------------
// direct_force_opt  –  compute accelerations for N bodies (double precision)
// -----------------------------------------------------------------------------
void direct_force_opt(const py::array_t<double>& x_arr,
                      const py::array_t<double>& y_arr,
                      const py::array_t<double>& m_arr,
                      double eps2,
                      py::array_t<double>& ax_arr,
                      py::array_t<double>& ay_arr)
{
    // --- unchecked accessors (no bound checks) --------------------------------
    const auto  x  = x_arr.unchecked<1>();
    const auto  y  = y_arr.unchecked<1>();
    const auto  m  = m_arr.unchecked<1>();
          auto ax = ax_arr.mutable_unchecked<1>();
          auto ay = ay_arr.mutable_unchecked<1>();

    const int N = static_cast<int>(x.shape(0));

    if (N == 0) return;

    // -------------------------------------------------------------------------
    // 1. Zero‑initialise accelerations (independent per element → parallel)
    // -------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    const int nBI = (N + BLOCK_I - 1) / BLOCK_I;
    const int nBJ = (N + BLOCK_J - 1) / BLOCK_J;

    // -------------------------------------------------------------------------
    // 2. Parallel over **i‑blocks** only ⇒ each particle updated by one thread
    //    ‑‑> no atomics / no false sharing.
    // -------------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int bi = 0; bi < nBI; ++bi) {
        const int i0 = bi * BLOCK_I;
        const int i1 = std::min(i0 + BLOCK_I, N);

        for (int bj = 0; bj < nBJ; ++bj) {
            const int j0 = bj * BLOCK_J;
            const int j1 = std::min(j0 + BLOCK_J, N);

            for (int i = i0; i < i1; ++i) {
                const double xi = x(i);
                const double yi = y(i);

                double fx = 0.0;
                double fy = 0.0;

                // ------------------------------------------------------------
                // SIMD inner loop over current j‑block (auto‑vectorised)
                // ------------------------------------------------------------
                #pragma omp simd reduction(+:fx,fy)
                for (int j = j0; j < j1; ++j) {
                    if (i == j) continue;          // skip self‑interaction

                    const double dx = xi - x(j);
                    const double dy = yi - y(j);
                    const double r2 = dx*dx + dy*dy + eps2;
                    const double inv_r  = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;

                    const double f = m(j) * inv_r3;
                    fx -= f * dx;
                    fy -= f * dy;
                }

                ax(i) += fx;
                ay(i) += fy;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// PYBIND11 module wrapper
// -----------------------------------------------------------------------------
PYBIND11_MODULE(force_kernel_opt, m)
{
    m.doc() = "Optimised direct O(N²) N‑body kernel (lock‑free, cache‑blocked)";
    m.def("direct_force", &direct_force_opt,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("eps2"),
          py::arg("ax"), py::arg("ay"),
          R"pbdoc(
        Compute pairwise gravitational accelerations in 2‑D.

        Parameters
        ----------
        x, y, m : numpy.ndarray[float64] (1‑D, length N)
            Particle positions and masses.
        eps2 : float
            Softening length **squared**.
        ax, ay : numpy.ndarray[float64] (writable)
            Output arrays; will be overwritten with accelerations.
    )pbdoc");
}

