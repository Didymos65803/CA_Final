// fmm_kernel.cpp – toy 2D FMM-like interface with pybind11 for benchmarking
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

py::tuple fmm_omp(py::array_t<double> x,
                  py::array_t<double> y,
                  py::array_t<double> m,
                  double domain,
                  double G=1.0,
                  double soft=0.05,
                  int order=4){
    const size_t N = x.size();
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();

    double soft2 = soft * soft;

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        double axi = 0.0, ayi = 0.0;
        for (long long j = 0; j < static_cast<long long>(N); ++j) {
            if (i == j) continue;
            double dx = px(j) - px(i);
            double dy = py_(j) - py_(i);
            double r2 = dx*dx + dy*dy + soft2;
            double invr3 = 1.0 / std::pow(r2, 1.5);
            double f = G * pm(j) * invr3;
            axi += f * dx;
            ayi += f * dy;
        }
        pax(i) = axi;
        pay(i) = ayi;
    }
    return py::make_tuple(ax, ay);
}

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "Toy FMM-like gravity solver (placeholder for benchmark)";
    m.def("fmm_omp", &fmm_omp,
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("G")=1.0, py::arg("soft")=0.05, py::arg("order")=4);
}
