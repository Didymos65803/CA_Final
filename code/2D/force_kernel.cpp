#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <omp.h>

namespace py = pybind11;

struct Body { double x, y, m; };

struct QuadNode {
    double cx, cy, size; // center and half-width
    double mass = 0.0, comx = 0.0, comy = 0.0;
    bool leaf = true;
    Body* body = nullptr;
    QuadNode* children[4] = {nullptr};

    QuadNode(double x, double y, double s) : cx(x), cy(y), size(s) {}
    ~QuadNode() { for (auto c : children) delete c; }
};

void insert(QuadNode* node, Body* b) {
    if (node->leaf && node->body == nullptr) {
        node->body = b;
        node->mass = b->m;
        node->comx = b->x;
        node->comy = b->y;
        return;
    }
    if (node->leaf) {
        Body* existing = node->body;
        node->body = nullptr;
        node->leaf = false;
        double s = node->size / 2;
        for (int i = 0; i < 4; ++i) {
            double dx = (i & 1) ? s : -s;
            double dy = (i & 2) ? s : -s;
            node->children[i] = new QuadNode(node->cx + dx, node->cy + dy, s);
        }
        insert(node, existing);
    }
    int idx = (b->x > node->cx) + 2 * (b->y > node->cy);
    insert(node->children[idx], b);

    double M = node->mass + b->m;
    node->comx = (node->comx * node->mass + b->x * b->m) / M;
    node->comy = (node->comy * node->mass + b->y * b->m) / M;
    node->mass = M;
}

void compute_force(const QuadNode* node, const Body& b, double theta, double G, double soft2, double& ax, double& ay) {
    if (!node || (node->leaf && node->body == &b)) return;

    double dx = node->comx - b.x;
    double dy = node->comy - b.y;
    double r2 = dx * dx + dy * dy + soft2;
    double r = std::sqrt(r2);

    if (node->leaf || (node->size / r < theta)) {
        double f = G * node->mass / (r2 * r);
        ax += f * dx;
        ay += f * dy;
    } else {
        for (int i = 0; i < 4; ++i)
            compute_force(node->children[i], b, theta, G, soft2, ax, ay);
    }
}

py::tuple bh_omp(py::array_t<double> x, py::array_t<double> y, py::array_t<double> m,
                 double domain, double theta=0.5, double G=1.0, double soft=0.05) {
    size_t N = x.size();
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    std::vector<Body> bodies(N);
    for (size_t i = 0; i < N; ++i)
        bodies[i] = {px(i), py_(i), pm(i)};

    QuadNode* root = new QuadNode(0.0, 0.0, domain / 2);
    for (auto& b : bodies)
        insert(root, &b);

    auto ax = py::array_t<double>(N);
    auto ay = py::array_t<double>(N);
    auto pax = ax.mutable_unchecked<1>();
    auto pay = ay.mutable_unchecked<1>();
    double soft2 = soft * soft;

    #pragma omp parallel for schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        double fx = 0.0, fy = 0.0;
        compute_force(root, bodies[i], theta, G, soft2, fx, fy);
        pax(i) = fx;
        pay(i) = fy;
    }

    delete root;
    return py::make_tuple(ax, ay);
}

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
        double fx = 0.0, fy = 0.0;
        for (ssize_t j = 0; j < N; ++j) {
            if (i == j) continue;
            double dx = px(j) - px(i);
            double dy = py_(j) - py_(i);
            double r2 = dx * dx + dy * dy + soft2;
            double invr3 = 1.0 / std::pow(r2, 1.5);
            fx += G * pm(j) * dx * invr3;
            fy += G * pm(j) * dy * invr3;
        }
        pax(i) = fx;
        pay(i) = fy;
    }
    return py::make_tuple(ax, ay);
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
