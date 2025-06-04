// fmm_openmp.cpp
// ---------------------------------------------------------------------------
// Serial‐build + parallel traversal 2D Barnes–Hut FMM with OpenMP
//
//  • Phase 1: recursive quadtree build (completely serial)
//  • Phase 2: bottom‐up multipole accumulation (completely serial)
//  • Phase 3: parallel traversal           (OpenMP schedule(static,1024))
//
// Build instructions (run these in your shell, not pasted into the file):
//   cd parallel
//   rm -f fmm_openmp.so fmm_openmp.cpython-39-x86_64-linux-gnu.so
//   rm -rf build
//   python3.9 setup_openmp.py build_ext --inplace
//   mv build/lib*/fmm_openmp*.so fmm_openmp.so
//
// Note: force_openmp.cpp and setup_openmp.py are unchanged.
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <cmath>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Each Body holds (x, y, m).  We will fill ax, ay later.
struct Body {
    double x, y, m;
    double ax, ay;
};

// A quadtree node.  We use 'h' = half‐width of this cell.
struct Node {
    double cx{}, cy{}, h{};                   // cell center (cx,cy) and half‐width h
    double mass{}, cmx{}, cmy{};              // accumulated mass and center‐of‐mass
    bool   leaf{true};                        // if true, this is a leaf node
    std::vector<int> ids;                     // indices of bodies if leaf
    std::array<std::unique_ptr<Node>,4> ch;   // children pointers: {SW, SE, NW, NE}
};

static constexpr int MAX_LEAF = 64;  // maximum bodies per leaf before subdividing

// Return quadrant index [0..3] for body b relative to node n:
//  0 = SW (x <= cx, y <= cy)
//  1 = SE (x >  cx, y <= cy)
//  2 = NW (x <= cx, y >  cy)
//  3 = NE (x >  cx, y >  cy)
inline int quadrant(const Body& b, const Node* n) {
    return (b.x > n->cx) + 2 * (b.y > n->cy);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 1: recursive quadtree build (serial).  If node has > MAX_LEAF bodies,
//  split into four children, distribute bodies, and recurse.  Otherwise remain leaf.
// ────────────────────────────────────────────────────────────────────────────
static void build_tree(Node* node, const std::vector<Body>& B) {
    if ((int)node->ids.size() <= MAX_LEAF) {
        node->leaf = true;
        return;
    }
    node->leaf = false;

    double h2 = 0.5 * node->h;
    for (int q = 0; q < 4; ++q) {
        node->ch[q] = std::make_unique<Node>();
        node->ch[q]->h    = h2;
        node->ch[q]->leaf = true;
        node->ch[q]->cx   = node->cx + (q & 1 ? 0.5 : -0.5) * h2;
        node->ch[q]->cy   = node->cy + (q & 2 ? 0.5 : -0.5) * h2;
    }

    for (int id : node->ids) {
        int q = quadrant(B[id], node);
        node->ch[q]->ids.push_back(id);
    }
    node->ids.clear();

    for (auto& cptr : node->ch) {
        build_tree(cptr.get(), B);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2: bottom‐up multipole accumulation (serial).
//  Post‐order: compute children first, then sum into parent.
// ────────────────────────────────────────────────────────────────────────────
static void compute_multipoles(Node* node, const std::vector<Body>& B) {
    if (node->leaf) {
        node->mass = 0.0;
        node->cmx  = 0.0;
        node->cmy  = 0.0;
        for (int id : node->ids) {
            node->mass += B[id].m;
            node->cmx  += B[id].m * B[id].x;
            node->cmy  += B[id].m * B[id].y;
        }
        if (node->mass > 0.0) {
            node->cmx /= node->mass;
            node->cmy /= node->mass;
        }
    } else {
        for (auto& cptr : node->ch) {
            compute_multipoles(cptr.get(), B);
        }
        node->mass = 0.0;
        node->cmx  = 0.0;
        node->cmy  = 0.0;
        for (auto& cptr : node->ch) {
            node->mass += cptr->mass;
            node->cmx  += cptr->mass * cptr->cmx;
            node->cmy  += cptr->mass * cptr->cmy;
        }
        if (node->mass > 0.0) {
            node->cmx /= node->mass;
            node->cmy /= node->mass;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 3: recursive traversal (parallel over targets).
//  If (h² / r² < θ²), use monopole; else descend into children.
// ────────────────────────────────────────────────────────────────────────────
inline bool far(const Body& p, const Node* n, double th2) {
    double dx = p.x - n->cmx;
    double dy = p.y - n->cmy;
    return (n->h * n->h) / (dx*dx + dy*dy) < th2;
}

static void traverse(const std::vector<Body>& B,
                     const Node* node,
                     const Body& p,
                     double eps2,
                     double th2,
                     double& fx,
                     double& fy)
{
    if (!node || node->mass == 0.0) return;

    if (node->leaf || far(p, node, th2)) {
        double dx = node->cmx - p.x;
        double dy = node->cmy - p.y;
        double r2 = dx*dx + dy*dy + eps2;
        double invR  = 1.0 / std::sqrt(r2);
        double invR3 = invR * invR * invR;
        double f    = node->mass * invR3;
        fx += f * dx;
        fy += f * dy;
    } else {
        for (auto& cptr : node->ch) {
            traverse(B, cptr.get(), p, eps2, th2, fx, fy);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Python‐visible entry point:
//   x, y, m : NumPy arrays (length N, dtype=float64)
//   eps2    : softening²
//   domain  : half‐width of root box
//   theta   : opening angle
//   ax, ay  : output NumPy arrays (length N) to fill with accelerations
// ────────────────────────────────────────────────────────────────────────────
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& m,
                     double eps2,
                     double domain,
                     double theta,
                     py::array_t<double>& ax,
                     py::array_t<double>& ay)
{
    int N = x.shape(0);
    std::vector<Body> B(N);
    for (int i = 0; i < N; ++i) {
        B[i].x  = x.at(i);
        B[i].y  = y.at(i);
        B[i].m  = m.at(i);
        B[i].ax = 0.0;
        B[i].ay = 0.0;
    }

    Node root;
    root.cx = 0.0;
    root.cy = 0.0;
    root.h  = domain;
    root.ids.resize(N);
    for (int i = 0; i < N; ++i) {
        root.ids[i] = i;
    }

    // Phase 1: serial quadtree build
    build_tree(&root, B);

    // Phase 2: serial bottom‐up multipoles
    compute_multipoles(&root, B);

    // Phase 3: parallel traversal over each target body
    auto axw = ax.mutable_unchecked<1>();
    auto ayw = ay.mutable_unchecked<1>();
    double th2 = theta * theta;

    #pragma omp parallel for schedule(static,1024)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        traverse(B, &root, B[i], eps2, th2, fx, fy);
        axw(i) = fx;
        ayw(i) = fy;
    }
}

PYBIND11_MODULE(fmm_openmp, m) {
    m.doc() = "2D Barnes–Hut FMM (serial build + parallel traversal)";
    m.def("fmm_force_theta",
          &fmm_force_theta,
          "fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)");
}

