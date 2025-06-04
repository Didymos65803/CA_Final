// fmm_openmp.cpp  ──  Barnes–Hut FMM (sequential build, OpenMP traversal)
// ----------------------------------------------------------------------------
// • Guard against infinite subdivision: stop if node.h < 1e-6  OR depth > 30.
// • MAX_LEAF = 64  → accurate & fast for 10^2 – 10^5 particles.
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

/* ---------------- data --------------------------------------------------- */
struct Body { double x, y, m; };

struct Node {
    double cx{}, cy{}, h{};                       // centre & half-width
    double mass{}, cmx{}, cmy{};                  // multipole
    bool   leaf{true};
    std::vector<int> ids;                         // bodies (if leaf)
    std::array<std::unique_ptr<Node>,4> ch;       // children SW,SE,NW,NE
};

static constexpr int MAX_LEAF   = 64;
static constexpr int MAX_DEPTH  = 30;   // safety against stack blow-up
static constexpr double H_MIN   = 1e-6; // min half-width before we stop

/* ---------------- helpers ------------------------------------------------ */
inline int quadrant(const Body& b, const Node* n) {
    return (b.x > n->cx) + 2*(b.y > n->cy);
}

/* ---------------- recursive build (with guard) -------------------------- */
void build(Node* n, const std::vector<Body>& B, int depth = 0)
{
    if ((int)n->ids.size() > MAX_LEAF &&
        n->h > H_MIN && depth < MAX_DEPTH)
    {
        n->leaf = false;
        double h2 = 0.5 * n->h;
        for (int q=0;q<4;++q) {
            n->ch[q] = std::make_unique<Node>();
            n->ch[q]->h    = h2;
            n->ch[q]->leaf = true;
            n->ch[q]->cx   = n->cx + (q&1 ? 0.5 : -0.5) * h2;
            n->ch[q]->cy   = n->cy + (q&2 ? 0.5 : -0.5) * h2;
        }
        for (int id : n->ids)
            n->ch[ quadrant(B[id], n) ]->ids.push_back(id);
        n->ids.clear();

        for (auto& c : n->ch) build(c.get(), B, depth+1);
    }

    /* multipole */
    n->mass = n->cmx = n->cmy = 0.0;
    if (n->leaf) {
        for (int id : n->ids) {
            n->mass += B[id].m;
            n->cmx  += B[id].m * B[id].x;
            n->cmy  += B[id].m * B[id].y;
        }
    } else {
        for (auto& c : n->ch) {
            n->mass += c->mass;
            n->cmx  += c->mass * c->cmx;
            n->cmy  += c->mass * c->cmy;
        }
    }
    if (n->mass) { n->cmx /= n->mass;  n->cmy /= n->mass; }
}

/* ---------------- traversal -------------------------------------------- */
inline bool far(const Body& p, const Node* n, double th2)
{
    double dx = p.x - n->cmx,  dy = p.y - n->cmy;
    return (n->h*n->h)/(dx*dx + dy*dy) < th2;
}

void walk(const std::vector<Body>& B, const Node* n, const Body& p,
          double eps2, double th2, double& fx, double& fy)
{
    if (!n || n->mass==0.0) return;
    if (n->leaf || far(p,n,th2)) {
        double dx=n->cmx-p.x, dy=n->cmy-p.y;
        double r2=dx*dx+dy*dy+eps2, invR=1/std::sqrt(r2), invR3=invR*invR*invR;
        double f=n->mass*invR3; fx+=f*dx; fy+=f*dy;
    } else {
        for (auto& c : n->ch) walk(B,c.get(),p,eps2,th2,fx,fy);
    }
}

/* ---------------- main kernel ------------------------------------------ */
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& m,
                     double eps2,double domain,double theta,
                     py::array_t<double>& ax, py::array_t<double>& ay)
{
    int N = x.shape(0);
    std::vector<Body> B(N);
    for (int i=0;i<N;++i) B[i] = { x.at(i), y.at(i), m.at(i) };

    Node root;                         // root encloses  [−domain, +domain]
    root.cx = root.cy = 0.0;
    root.h  = domain;                  // half-width = domain  (200 wide box)
    root.ids.resize(N); std::iota(root.ids.begin(), root.ids.end(), 0);

    std::cout << "[FMM] build_tree N=" << N << '\n'; std::cout.flush();
    build(&root, B);

    auto axw=ax.mutable_unchecked<1>(), ayw=ay.mutable_unchecked<1>();
    double th2 = theta * theta;

    std::cout << "[FMM] traverse N=" << N << '\n'; std::cout.flush();
    #pragma omp parallel for schedule(dynamic,256)
    for (int i=0;i<N;++i) {
        double fx=0.0, fy=0.0;
        walk(B, &root, B[i], eps2, th2, fx, fy);
        axw(i)=fx; ayw(i)=fy;
    }
    std::cout << "[FMM] done N=" << N << '\n'; std::cout.flush();
}

/* ---------------- Python module ---------------------------------------- */
PYBIND11_MODULE(fmm_openmp, m)
{
    m.doc() = "2-D Barnes–Hut FMM (guarded build, OpenMP traversal)";
    m.def("fmm_force_theta", &fmm_force_theta,
          "x,y,m,eps2,domain,theta,ax,ay  → fill ax, ay with accelerations");
}

