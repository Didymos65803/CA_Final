#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

void direct_omp(const double* x, const double* y, const double* m,
                double* ax, double* ay,
                size_t N, double G, double soft){
    double soft2 = soft * soft;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i){
        double xi = x[i], yi = y[i], axi = 0.0, ayi = 0.0;
        for (size_t j = 0; j < N; ++j){
            if (i == j) continue;
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double r2 = dx*dx + dy*dy + soft2;
            double invr3 = 1.0 / std::pow(r2, 1.5);
            double f = G * m[j] * invr3;
            axi += f * dx;
            ayi += f * dy;
        }
        ax[i] = axi;
        ay[i] = ayi;
    }
}

PYBIND11_MODULE(force_kernel, m){
    m.doc() = "Force kernels: direct_omp";

    m.def("direct_omp", [](py::array_t<double> x,
                           py::array_t<double> y,
                           py::array_t<double> m,
                           double G, double soft) {
        size_t N = x.size();
        auto ax = py::array_t<double>(N);
        auto ay = py::array_t<double>(N);

        direct_omp(x.data(), y.data(), m.data(),
                   ax.mutable_data(), ay.mutable_data(),
                   N, G, soft);
        return py::make_tuple(ax, ay);
    }, py::arg("x"), py::arg("y"), py::arg("m"),
       py::arg("G")=1.0, py::arg("soft")=0.05,
       "Direct gravity computation (O(N^2), OpenMP)");
}

