#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

py::tuple direct_omp(py::array_t<double> x,
                     py::array_t<double> y,
                     py::array_t<double> m,
                     double G=1.0,
                     double soft=0.05) {
    const ssize_t N = x.size();
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();

    double soft2 = soft * soft;

    #pragma omp parallel for schedule(dynamic)
    for (ssize_t i = 0; i < N; ++i) {
        double axi = 0.0, ayi = 0.0;
        for (ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            double dx = px(j) - px(i);
            double dy = py_(j) - py_(i);
            double r2 = dx * dx + dy * dy + soft2;
            double invr3 = 1.0 / std::pow(r2, 1.5);
            axi += G * pm(j) * dx * invr3;
            ayi += G * pm(j) * dy * invr3;
        }
        pax(i) = axi;
        pay(i) = ayi;
    }
    return py::make_tuple(ax, ay);
}

py::tuple bh_omp(py::array_t<double> x,
                 py::array_t<double> y,
                 py::array_t<double> m,
                 double domain,
                 double theta=0.5,
                 double G=1.0,
                 double soft=0.05) {
    // For now, call the direct_omp – can later plug in real Barnes–Hut tree
    return direct_omp(x, y, m, G, soft);
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "Gravity solvers: BH and Direct with OpenMP";
    m.def("direct_omp", &direct_omp,
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("G")=1.0, py::arg("soft")=0.05);
    m.def("bh_omp", &bh_omp,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta")=0.5, py::arg("G")=1.0, py::arg("soft")=0.05);
}
