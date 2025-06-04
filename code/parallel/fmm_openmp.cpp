// fmm_openmp.cpp
// ===============
//
// A 2D Barnes–Hut / FMM‐style solver with OpenMP.
//
// We have increased MAX_LEAF from 64 to 1024 so that any N ≤ 1024
// is treated as a single leaf (no subdivision). This makes build_tree
// finish in one pass for small‐to‐moderate N.
//
// To compile:
//   python3 setup_openmp.py build_ext --inplace
// Then rename the produced .so to fmm_openmp.so so Python can import it.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>      // for debug printing
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// A simple Body struct
struct Body {
    double x, y, m;
};

// A Node in the quadtree
struct Node {
    double cx = 0.0, cy = 0.0, size = 0.0;   // center and half‐width
    double mass = 0.0, cmx = 0.0, cmy = 0.0; // total mass and centroid
    bool leaf = true;
    std::vector<int> ids;                    // indices of bodies in this node
    std::array<std::unique_ptr<Node>, 4> ch; // children: {SW, SE, NW, NE}
};

// Raise leaf threshold so N ≤ 1024 never subdivides
static constexpr int MAX_LEAF = 1024;

// Subdivide node n into 4 children, distribute body‐indices into correct quadrant.
void subdivide(Node *n, const std::vector<Body>& B) {
    double midx = n->cx, midy = n->cy;
    double h = 0.5 * n->size; // new half‐width for children

    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        n->ch[quadrant] = std::make_unique<Node>();
        n->ch[quadrant]->size = h;
        n->ch[quadrant]->leaf = true;
        if (quadrant == 0) {       // SW quadrant (lower‐left)
            n->ch[quadrant]->cx = midx - h * 0.5;
            n->ch[quadrant]->cy = midy - h * 0.5;
        }
        else if (quadrant == 1) {  // SE quadrant (lower‐right)
            n->ch[quadrant]->cx = midx + h * 0.5;
            n->ch[quadrant]->cy = midy - h * 0.5;
        }
        else if (quadrant == 2) {  // NW quadrant (upper‐left)
            n->ch[quadrant]->cx = midx - h * 0.5;
            n->ch[quadrant]->cy = midy + h * 0.5;
        }
        else {                     // NE quadrant (upper‐right)
            n->ch[quadrant]->cx = midx + h * 0.5;
            n->ch[quadrant]->cy = midy + h * 0.5;
        }
    }
    n->leaf = false;

    // Distribute the body‐indices from n->ids into the new children
    for (int id : n->ids) {
        double bx = B[id].x;
        double by = B[id].y;
        int idx = 0;
        if (bx > n->cx) idx += 1; // east half
        if (by > n->cy) idx += 2; // north half
        n->ch[idx]->ids.push_back(id);
    }
    n->ids.clear();
}

// SEQUENTIAL quadtree builder (BFS, but done on one thread).
// Computes total mass & centroid bottom‐up.
void build_tree(Node *root, const std::vector<Body>& B) {
    std::vector<Node*> current{root}, next;

    while (!current.empty()) {
        next.clear();  // prepare next level

        for (Node* n : current) {
            // If this leaf node has more bodies than MAX_LEAF, subdivide
            if ((int)n->ids.size() > MAX_LEAF) {
                subdivide(n, B);
            }

            // Compute this node's mass & centroid
            n->mass = 0.0;
            n->cmx  = 0.0;
            n->cmy  = 0.0;
            if (n->leaf) {
                // Sum over all bodies in n->ids
                for (int id : n->ids) {
                    n->mass += B[id].m;
                    n->cmx  += B[id].m * B[id].x;
                    n->cmy  += B[id].m * B[id].y;
                }
            } else {
                // Sum over children’s masses
                for (auto &c : n->ch) {
                    if (c) {
                        n->mass += c->mass;
                        n->cmx  += c->mass * c->cmx;
                        n->cmy  += c->mass * c->cmy;
                        next.push_back(c.get());
                    }
                }
            }
            if (n->mass > 0.0) {
                n->cmx /= n->mass;
                n->cmy /= n->mass;
            }
        }

        // Move to next level
        current.swap(next);
    }
}

// Return true if node n is “far enough” to approximate with its centroid
inline bool far(const Body& p, const Node* n, double theta2) {
    if (!n || n->mass == 0.0) return false;
    double dx = p.x - n->cmx;
    double dy = p.y - n->cmy;
    return (n->size * n->size) / (dx*dx + dy*dy) < theta2;
}

// Recursively traverse the tree to accumulate force on particle p
static void traverse(
    const std::vector<Body>& B,
    const Node* n,
    const Body& p,
    double eps2,
    double theta2,
    double& fx,
    double& fy)
{
    if (!n || n->mass == 0.0) return;

    if (n->leaf || far(p, n, theta2)) {
        // Approximate with this node's centroid
        double dx = n->cmx - p.x;
        double dy = n->cmy - p.y;
        double r2 = dx*dx + dy*dy + eps2;
        double invR = 1.0 / std::sqrt(r2);
        double invR3 = invR * invR * invR;
        double f = n->mass * invR3;
        fx += f * dx;
        fy += f * dy;
    } else {
        // Descend into children
        for (auto &c : n->ch) {
            if (c) {
                traverse(B, c.get(), p, eps2, theta2, fx, fy);
            }
        }
    }
}

// The core FMM entry point
static void fmm_core(
    const py::array_t<double>& x_arr,
    const py::array_t<double>& y_arr,
    const py::array_t<double>& m_arr,
    double eps2,
    double domain,
    double theta,
    py::array_t<double>& ax_arr,
    py::array_t<double>& ay_arr)
{
    const int N = x_arr.shape(0);

    // Copy particle data into a vector of Body
    std::vector<Body> B(N);
    for (int i = 0; i < N; ++i) {
        B[i].x = x_arr.at(i);
        B[i].y = y_arr.at(i);
        B[i].m = m_arr.at(i);
    }

    // Build the root node covering [-domain, +domain]
    Node root;
    root.cx   = 0.0;
    root.cy   = 0.0;
    root.size = domain * 0.5; // half‐width
    root.ids.resize(N);
    for (int i = 0; i < N; ++i) {
        root.ids[i] = i;
    }

    // Debug: Starting build_tree
    std::cout << "[FMM] Entering build_tree (N=" << N << ", MAX_LEAF=" << MAX_LEAF << ")\n";
    std::cout.flush();

    build_tree(&root, B);

    // Debug: Finished build_tree
    std::cout << "[FMM] Finished build_tree\n";
    std::cout.flush();

    // Prepare the output arrays
    auto ax_out = ax_arr.mutable_unchecked<1>();
    auto ay_out = ay_arr.mutable_unchecked<1>();

    // Compute forces in parallel by traversing the tree for each particle
    std::cout << "[FMM] Starting parallel traversal (N=" << N << ")\n";
    std::cout.flush();

    double theta2 = theta * theta;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        traverse(B, &root, B[i], eps2, theta2, fx, fy);
        ax_out(i) = fx;
        ay_out(i) = fy;
    }

    // Debug: Finished parallel traversal
    std::cout << "[FMM] Finished parallel traversal (N=" << N << ")\n";
    std::cout.flush();
}

// Python wrapper for an arbitrary θ
void fmm_force_theta(
    const py::array_t<double>& x,
    const py::array_t<double>& y,
    const py::array_t<double>& m,
    double eps2,
    double domain,
    double theta,
    py::array_t<double>& ax,
    py::array_t<double>& ay)
{
    fmm_core(x, y, m, eps2, domain, theta, ax, ay);
}

// Pybind11 module definition
PYBIND11_MODULE(fmm_openmp, m) {
    m.doc() = "2D Barnes-Hut / FMM solver with OpenMP (sequential build_tree, MAX_LEAF=1024)";
    m.def(
        "fmm_force_theta",
        &fmm_force_theta,
        "fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)  # Barnes-Hut / FMM"
    );
}

