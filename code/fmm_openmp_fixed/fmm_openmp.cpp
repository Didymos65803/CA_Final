// fmm_openmp.cpp
// ================================================================
//  Pybind11 module that exposes two routines:
//     • direct_force()   –  O(N²) reference (multithreaded)
//     • fmm_force()      –  Barnes–Hut Fast Multipole (O(N log N))
//                          with OpenMP‑parallel tree build + traversal.
//
//  Compile on Linux / macOS:
//     g++ -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp -shared -fPIC \
//         `python -m pybind11 --includes` fmm_openmp.cpp \
//         -o fmm_openmp$(python3-config --extension-suffix)
//
//  The resulting shared object can be imported from Python as:
//         import fmm_openmp
//
//  Example driver shown at the end of this file (comment) – copy to a
//  separate Python script to benchmark speed‑up with different
//  OMP_NUM_THREADS.
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <queue>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ------------------------------------------------------------
// Utility struct for particles
// ------------------------------------------------------------
struct Body {
    double x, y, m;
    double ax = 0.0, ay = 0.0;
};

// ------------------------------------------------------------
// 1.  Multithreaded direct solver (reference)
// ------------------------------------------------------------
void direct_force(const py::array_t<double>& x_arr,
                  const py::array_t<double>& y_arr,
                  const py::array_t<double>& m_arr,
                  double soft2,
                  py::array_t<double>& ax_arr,
                  py::array_t<double>& ay_arr)
{
    const auto  x  = x_arr.unchecked<1>();
    const auto  y  = y_arr.unchecked<1>();
    const auto  m  = m_arr.unchecked<1>();
          auto ax = ax_arr.mutable_unchecked<1>();
          auto ay = ay_arr.mutable_unchecked<1>();
    const int N = static_cast<int>(x.shape(0));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        const double xi = x(i), yi = y(i);
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const double dx = x(j) - xi;
            const double dy = y(j) - yi;
            double r2 = dx*dx + dy*dy + soft2;
            double invR = 1.0 / std::sqrt(r2);
            double invR3 = invR * invR * invR;
            double f = m(j) * invR3;
            fx += f * dx;
            fy += f * dy;
        }
        ax(i) = fx;
        ay(i) = fy;
    }
}

// ------------------------------------------------------------
// 2.  Barnes–Hut Fast Multipole (2‑D quadtree)   O(N log N)
// ------------------------------------------------------------
struct Node {
    double cx, cy, size;           // centre + half‑width
    double mass = 0.0, cmx = 0.0, cmy = 0.0;
    bool   leaf = true;
    std::vector<int> ids;          // particle indices (leaf)
    std::unique_ptr<Node> ch[4];   // children
};

constexpr int    MAX_LEAF = 16;
constexpr double THETA2   = 0.36;   // (0.6)^2

static void subdivide(Node* n, const std::vector<Body>& B)
{
    const double h = n->size * 0.5;
    const double off[4][2] = {{-h,-h},{ h,-h},{-h, h},{ h, h}};
    std::vector<int> bucket[4];
    for (int id : n->ids) {
        int q = (B[id].x > n->cx) + 2 * (B[id].y > n->cy);
        bucket[q].push_back(id);
    }
    n->leaf = false;
    n->ids.clear();
    for (int q = 0; q < 4; ++q) if (!bucket[q].empty()) {
        n->ch[q] = std::make_unique<Node>();
        n->ch[q]->cx = n->cx + off[q][0];
        n->ch[q]->cy = n->cy + off[q][1];
        n->ch[q]->size = h;
        n->ch[q]->ids.swap(bucket[q]);
    }
}

static void build_tree(Node* root, const std::vector<Body>& B)
{
#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif
    std::queue<Node*> Q;
    Q.push(root);
    while (!Q.empty()) {
        const std::size_t lvl = Q.size();
        #pragma omp parallel for schedule(dynamic,4)
        for (std::size_t i = 0; i < lvl; ++i) {
            Node* node;
            // pop safely
            #pragma omp critical(pop)
            { node = Q.front(); Q.pop(); }

            if (node->ids.size() > MAX_LEAF)
                subdivide(node, B);

            // compute COM
            node->mass = node->cmx = node->cmy = 0.0;
            if (node->leaf) {
                for (int id : node->ids) {
                    node->mass += B[id].m;
                    node->cmx  += B[id].m * B[id].x;
                    node->cmy  += B[id].m * B[id].y;
                }
            } else {
                for (auto& c : node->ch) if (c) {
                    node->mass += c->mass;
                    node->cmx  += c->mass * c->cmx;
                    node->cmy  += c->mass * c->cmy;
                }
            }
            if (node->mass) {
                node->cmx /= node->mass;
                node->cmy /= node->mass;
            }

            // enqueue children
            if (!node->leaf) {
                for (auto& c : node->ch) if (c) {
                    #pragma omp critical(push)
                    Q.push(c.get());
                }
            }
        }
    }
}

inline bool far_enough(const Body& p, const Node* n)
{
    const double dx = p.x - n->cmx;
    const double dy = p.y - n->cmy;
    return (n->size * n->size) / (dx*dx + dy*dy) < THETA2;
}

static void traverse(const std::vector<Body>& B, Body& p, const Node* n,
                     double soft2, double& fx, double& fy)
{
    if (!n || n->mass == 0.0) return;
    if (n->leaf || far_enough(p, n)) {
        const double dx = n->cmx - p.x;
        const double dy = n->cmy - p.y;
        double r2 = dx*dx + dy*dy + soft2;
        double invR = 1.0 / std::sqrt(r2);
        double invR3 = invR * invR * invR;
        double f = n->mass * invR3;
        fx += f * dx;
        fy += f * dy;
    } else {
        for (const auto& c : n->ch) if (c)
            traverse(B, p, c.get(), soft2, fx, fy);
    }
}

void fmm_force(const py::array_t<double>& x_arr, const py::array_t<double>& y_arr,
               const py::array_t<double>& m_arr, double soft2, double domain,
               py::array_t<double>& ax_arr, py::array_t<double>& ay_arr)
{
    const int N = static_cast<int>(x_arr.shape(0));
    std::vector<Body> B(N);
    for (int i=0;i<N;++i) B[i] = { x_arr.at(i), y_arr.at(i), m_arr.at(i) };

    Node root; root.cx = 0.0; root.cy = 0.0; root.size = domain * 0.5;
    root.ids.resize(N); for (int i=0;i<N;++i) root.ids[i] = i;
    build_tree(&root, B);

    // parallel traversal – one thread per particle
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();

    #pragma omp parallel for schedule(dynamic,64)
    for (int i=0;i<N;++i) {
        double fx = 0.0, fy = 0.0;
        traverse(B, B[i], &root, soft2, fx, fy);
        ax(i) = fx;
        ay(i) = fy;
    }
}

PYBIND11_MODULE(fmm_openmp, m)
{
    m.doc() = "Direct + Barnes–Hut FMM kernels with OpenMP parallelisation";

    m.def("direct_force", &direct_force, py::arg("x"),py::arg("y"),py::arg("m"),
          py::arg("soft2"),py::arg("ax"),py::arg("ay"),
          R"pbdoc(Multithreaded direct O(N²) solver)pbdoc");

    m.def("fmm_force", &fmm_force, py::arg("x"),py::arg("y"),py::arg("m"),
          py::arg("soft2"), py::arg("domain"), py::arg("ax"), py::arg("ay"),
          R"pbdoc(OpenMP Barnes–Hut FMM solver (O(N log N)))pbdoc");
}

/*
=====================  Python benchmark snippet  =====================

import os, time, numpy as np, fmm_openmp as fm

N = 10000
np.random.seed(0)
x = np.random.uniform(-50, 50, N).astype(np.float64)
y = np.random.uniform(-50, 50, N).astype(np.float64)
m = np.ones(N, dtype=np.float64)
ax = np.zeros(N, dtype=np.float64); ay = np.zeros(N, dtype=np.float64)
soft2 = 0.01**2

for threads in (1, 2, 4, 8):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    time.sleep(0.1)
    t0 = time.time()
    fm.fmm_force(x, y, m, soft2, 100.0, ax, ay)
    print(f"{threads} threads : {time.time()-t0:.4f} s")
*/

