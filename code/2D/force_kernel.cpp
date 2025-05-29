#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Direct N-body with OpenMP
void direct_omp(py::array_t<double> x,
                py::array_t<double> y,
                py::array_t<double> m,
                py::array_t<double> ax,
                py::array_t<double> ay,
                double G, double soft2)
{
    auto X = x.unchecked<1>();
    auto Y = y.unchecked<1>();
    auto M = m.unchecked<1>();
    auto AX = ax.mutable_unchecked<1>();
    auto AY = ay.mutable_unchecked<1>();

    ssize_t n = X.shape(0);

    #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < n; ++i) {
        double xi = X(i);
        double yi = Y(i);
        double axi = 0.0;
        double ayi = 0.0;
        for (ssize_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double dx = X(j) - xi;
            double dy = Y(j) - yi;
            double r2 = dx*dx + dy*dy;
            if (r2 < soft2) r2 = soft2;
            double inv_r = 1.0 / std::sqrt(r2);
            double f = G * M(j) * inv_r * inv_r;
            axi += f * dx * inv_r;
            ayi += f * dy * inv_r;
        }
        AX(i) = axi;
        AY(i) = ayi;
    }
}

PYBIND11_MODULE(force_kernel, m) {
    m.def("direct_omp", &direct_omp, "Direct N-body (OpenMP)");
}
