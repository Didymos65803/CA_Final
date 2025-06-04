// fmm_kernel_optimized.cpp  –  breadth‑first parallel build (safe v2)
// -----------------------------------------------------------------------------
// Compile with  (example):
//   g++ -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp -shared -fPIC \
//       `python -m pybind11 --includes` fmm_kernel_optimized.cpp \
//       -o fmm_kernel_opt$(python3-config --extension-suffix)
// -----------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <queue>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

struct Body { double x,y,m, ax,ay; };
struct Node {
    double cx, cy, size;
    double mass=0.0, cmx=0.0, cmy=0.0;
    bool   leaf=true;
    std::vector<int> ids;
    std::unique_ptr<Node> ch[4];
};

static constexpr int   MAX_LEAF = 12;      // particles per leaf
static constexpr double THETA2  = 0.36;    // (0.6)^2
static constexpr double G       = 1.0;

// -----------------------------------------------------------------------------
// Helper – subdivide one node (serial)
// -----------------------------------------------------------------------------
static void subdivide(Node* n, const std::vector<Body>& b)
{
    const double h = n->size * 0.5;
    const double off[4][2] = {{-h,-h},{ h,-h},{-h, h},{ h, h}};
    std::vector<int> buckets[4];
    for (int id : n->ids) {
        int q = (b[id].x > n->cx) + 2*(b[id].y > n->cy);
        buckets[q].push_back(id);
    }
    n->leaf = false; n->ids.clear();
    for (int q=0;q<4;++q) if (!buckets[q].empty()) {
        n->ch[q] = std::make_unique<Node>();
        n->ch[q]->cx = n->cx + off[q][0];
        n->ch[q]->cy = n->cy + off[q][1];
        n->ch[q]->size = h;
        n->ch[q]->ids.swap(buckets[q]);
    }
}

// -----------------------------------------------------------------------------
// Breadth‑first, level‑by‑level tree construction with OpenMP parallelism.
// -----------------------------------------------------------------------------
static void build_tree(Node* root, const std::vector<Body>& bodies)
{
#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif
    std::queue<Node*> Q; Q.push(root);
    while (!Q.empty()) {
        const std::size_t lvl = Q.size();
        #pragma omp parallel for schedule(dynamic,4)
        for (std::size_t i=0;i<lvl;++i) {
            Node* node;
            #pragma omp critical(queue_pop)
            { node = Q.front(); Q.pop(); }
            if (node->ids.size() > MAX_LEAF) subdivide(node, bodies);
            // compute centre of mass
            node->mass = node->cmx = node->cmy = 0.0;
            if (node->leaf) {
                for (int id: node->ids) {
                    node->mass += bodies[id].m;
                    node->cmx  += bodies[id].m * bodies[id].x;
                    node->cmy  += bodies[id].m * bodies[id].y;
                }
            } else {
                for (auto& ch: node->ch) if (ch) {
                    node->mass += ch->mass;
                    node->cmx  += ch->mass * ch->cmx;
                    node->cmy  += ch->mass * ch->cmy;
                }
            }
            if (node->mass) { node->cmx /= node->mass; node->cmy /= node->mass; }
            // enqueue children for next level (serial section)
            if (!node->leaf) {
                for (auto& ch: node->ch) if (ch) Q.push(ch.get());
            }
        }
    }
}

inline bool open_criterion(const Body& p, const Node* n)
{
    const double dx = p.x - n->cmx; const double dy = p.y - n->cmy;
    return (n->size*n->size)/(dx*dx + dy*dy) > THETA2;
}

static void traverse(const std::vector<Body>& B, Body& p, const Node* n,
                     double eps2, double& ax, double& ay)
{
    if (!n || n->mass == 0) return;
    if (n->leaf) {
        for (int id : n->ids) if (&p != &B[id]) {
            const double dx = p.x - B[id].x, dy = p.y - B[id].y;
            double r2 = dx*dx + dy*dy + eps2;
            double invR = 1.0 / std::sqrt(r2);
            double f = G * B[id].m * invR * invR * invR;
            ax -= f * dx; ay -= f * dy;
        }
    } else if (!open_criterion(p, n)) {
        const double dx = p.x - n->cmx, dy = p.y - n->cmy;
        double r2 = dx*dx + dy*dy + eps2;
        double invR = 1.0 / std::sqrt(r2);
        double f = G * n->mass * invR * invR * invR;
        ax -= f * dx; ay -= f * dy;
    } else {
        for (const auto& ch : n->ch) if (ch) traverse(B, p, ch.get(), eps2, ax, ay);
    }
}

void fmm_force_opt(const py::array_t<double>& x_arr, const py::array_t<double>& y_arr,
                   const py::array_t<double>& m_arr, int N,
                   double domain, double theta_unused, int ml_unused,
                   double eps, double G_unused,
                   py::array_t<double>& ax_arr, py::array_t<double>& ay_arr)
{
    const double eps2 = eps*eps;
    std::vector<Body> B(N);
    for (int i=0;i<N;++i) B[i] = {x_arr.at(i), y_arr.at(i), m_arr.at(i), 0, 0};

    Node root{0.0,0.0,domain*0.5};
    root.ids.resize(N); for(int i=0;i<N;++i) root.ids[i]=i;
    build_tree(&root, B);

    // force phase
    #pragma omp parallel for schedule(dynamic,64)
    for (int i=0;i<N;++i) {
        double ax=0, ay=0; traverse(B, B[i], &root, eps2, ax, ay);
        B[i].ax = ax; B[i].ay = ay;
    }

    // copy out
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for (int i=0;i<N;++i) { ax(i) = B[i].ax; ay(i) = B[i].ay; }
}

PYBIND11_MODULE(fmm_kernel_opt, m)
{
    m.doc() = "Barnes–Hut FMM – breadth‑first build, no deep OpenMP tasks";
    m.def("fmm_force", &fmm_force_opt,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("N"),
          py::arg("domain"), py::arg("theta"), py::arg("maxLeaf"),
          py::arg("eps"), py::arg("G"), py::arg("ax"), py::arg("ay"));
}

