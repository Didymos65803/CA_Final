// fmm_openmp.cpp
// -------------------------------
// 2D Barnes–Hut / FMM with OpenMP
//
//  • build_tree_rec(): sequential, depth‐bounded recursive quadtree build
//  • traverse()     : parallel #pragma omp for
//
// build:   python3 setup_openmp.py build_ext --inplace
// install: mv build/lib*/fmm_openmp*.so  fmm_openmp.so
// -------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// Each particle is (x, y, m).
struct Body {
    double x, y, m;
};

// Quadtree node
struct Node {
    double cx{}, cy{}, h{};               // center + half‐width
    double mass{}, cmx{}, cmy{};          // monopole data
    bool   leaf{true};
    std::vector<int> ids;                 // indices into Bodies, if leaf
    std::array<std::unique_ptr<Node>,4> ch; // 4 children: SW, SE, NW, NE
};

// When a leaf has more than MAX_LEAF points, we subdivide.
static constexpr int MAX_LEAF   = 64;
static constexpr int MAX_DEPTH  = 32;
static constexpr double H_MIN   = 1e-6;

// Determine which quadrant a body belongs to relative to node center
inline int quadrant(const Body& b, const Node* n) {
    return (b.x > n->cx) + 2*(b.y > n->cy);
}

// Recursively build the quadtree. 
// Stops recursing if leaf size ≤ MAX_LEAF, or half‐width < H_MIN, or 
// depth ≥ MAX_DEPTH.
void build_tree_rec(Node* n,
                    const std::vector<Body>& B,
                    int depth = 0)
{
    if ((int)n->ids.size() > MAX_LEAF &&
        n->h > H_MIN &&
        depth   < MAX_DEPTH)
    {
        // Split this node into 4 children
        n->leaf = false;
        double h2 = 0.5 * n->h;
        for (int q = 0; q < 4; ++q) {
            n->ch[q] = std::make_unique<Node>();
            n->ch[q]->h = h2;
            n->ch[q]->leaf = true;
            n->ch[q]->cx = n->cx + (q & 1 ? 0.5 : -0.5) * h2;
            n->ch[q]->cy = n->cy + (q & 2 ? 0.5 : -0.5) * h2;
        }
        // Distribute this node’s particle indices into the correct child
        for (int id : n->ids) {
            int q = quadrant(B[id], n);
            n->ch[q]->ids.push_back(id);
        }
        n->ids.clear();

        // Recurse
        for (auto& c : n->ch) {
            build_tree_rec(c.get(), B, depth + 1);
        }
    }

    // Compute this node’s mass and centroid
    n->mass = 0.0;
    n->cmx  = 0.0;
    n->cmy  = 0.0;
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
    if (n->mass > 0.0) {
        n->cmx /= n->mass;
        n->cmy /= n->mass;
    }
}

// Return true if node n is “far enough” from particle p to approximate by a single
// monopole at (cmx, cmy).
inline bool far(const Body& p, const Node* n, double th2) {
    double dx = p.x - n->cmx;
    double dy = p.y - n->cmy;
    return (n->h * n->h) / (dx*dx + dy*dy) < th2;
}

// Walk the tree to accumulate force on p.  If the node is a leaf or far enough,
// do the monopole approximation; else descend to children.
void traverse(const std::vector<Body>& B,
              const Node* n,
              const Body& p,
              double eps2,
              double th2,
              double& fx,
              double& fy)
{
    if (!n || n->mass == 0.0) return;

    if (n->leaf || far(p, n, th2)) {
        double dx = n->cmx - p.x;
        double dy = n->cmy - p.y;
        double r2 = dx*dx + dy*dy + eps2;
        double invR  = 1.0 / std::sqrt(r2);
        double invR3 = invR * invR * invR;
        double f = n->mass * invR3;
        fx += f * dx;
        fy += f * dy;
    } else {
        for (auto& c : n->ch) {
            traverse(B, c.get(), p, eps2, th2, fx, fy);
        }
    }
}

// The Python‐visible function.  Inputs:
//   x, y, m : NumPy arrays of length N
//   eps2    : softening^2
//   domain  : half‐width of the root’s box  (so box is [−domain, +domain]²)
//   theta   : opening angle
// Outputs:
//   ax, ay  : NumPy arrays (length N) to fill with accelerations
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
        B[i].x = x.at(i);
        B[i].y = y.at(i);
        B[i].m = m.at(i);
    }

    // Build the root node
    Node root;
    root.cx = 0.0;
    root.cy = 0.0;
    root.h  = domain;  // half‐width = domain
    root.ids.resize(N);
    for (int i = 0; i < N; ++i) root.ids[i] = i;

    // 1) Build quadtree (sequential, but depth-bounded)
    std::cout << "[FMM] build_tree N=" << N << "\n";
    std::cout.flush();
    build_tree_rec(&root, B);
    std::cout << "[FMM] done build_tree\n";
    std::cout.flush();

    // 2) Traverse (parallel for)
    auto axw = ax.mutable_unchecked<1>();
    auto ayw = ay.mutable_unchecked<1>();
    double th2 = theta * theta;

    std::cout << "[FMM] traverse N=" << N << "\n";
    std::cout.flush();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        traverse(B, &root, B[i], eps2, th2, fx, fy);
        axw(i) = fx;
        ayw(i) = fy;
    }

    std::cout << "[FMM] done traverse N=" << N << "\n";
    std::cout.flush();
}

PYBIND11_MODULE(fmm_openmp, m) {
    m.doc() = "2D Barnes–Hut FMM (sequential build, OpenMP traversal)";
    m.def("fmm_force_theta",
          &fmm_force_theta,
          "fmm_force_theta(x,y,m,eps2,domain,theta,ax,ay)  # compute accelerations");
}

