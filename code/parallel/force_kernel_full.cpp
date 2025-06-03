// force_kernel_full.cpp
// =====================
//
// High‐precision Direct (O(N²)) and Barnes‐Hut (O(N log N)) force kernels with OpenMP.
// Exports two functions to Python: direct_omp(...) and bh_omp(...), plus a flag has_openmp.
//
// To build (in the same folder as setup.py):
//     python3.12 setup.py build_ext --inplace
// That produces: force_kernel.cpython-<ver>-<arch>.so.
//
// Author: (Adapted for 2025 coursework)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------------
// Part 1: Direct‐sum kernel (O(N²))
// -----------------------------------------------------------------------------------
//
// Given N particles at positions (x[i], y[i]) with masses m[i], compute acceleration
// on each particle via:
//   a_i = Σ_{j≠i} G * m[j] * ( (pos_j - pos_i) / (|pos_j - pos_i|² + soft²)^(3/2) )
//
// We parallelize over the outer loop (each “i”) with OpenMP.  Softening length = soft.
// Gravity constant = G.
// -----------------------------------------------------------------------------------

/**
 * direct_omp(x, y, m, G=1.0, soft=0.01) → (ax, ay)
 *
 * Inputs:
 *   x, y : numpy arrays of length N (dtype=float64)
 *   m    : numpy array of length N (dtype=float64)
 *   G    : double (gravitational constant, default=1.0)
 *   soft : double (softening length, default=0.01)
 *
 * Returns:
 *   (ax, ay) as two numpy arrays of length N, where each is the net acceleration
 *   on particle i due to all others.
 */
py::tuple direct_omp(py::array_t<double> x_in,
                    py::array_t<double> y_in,
                    py::array_t<double> m_in,
                    double G    = 1.0,
                    double soft = 0.01)
{
    // Unpack input array sizes:
    size_t N = x_in.shape(0);

    // If no particles, return empty arrays:
    if (N == 0) {
        return py::make_tuple(
            py::array_t<double>(0),
            py::array_t<double>(0)
        );
    }

    // Copy data into std::vector<double> for fast indexing:
    std::vector<double> x(N), y(N), m(N);
    auto xx = x_in.unchecked<1>();
    auto yy = y_in.unchecked<1>();
    auto mm = m_in.unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        x[i] = xx(i);
        y[i] = yy(i);
        m[i] = mm(i);
    }

    // Prepare output arrays (ax, ay):
    py::array_t<double> ax_out(N), ay_out(N);
    auto ax = ax_out.mutable_unchecked<1>();
    auto ay = ay_out.mutable_unchecked<1>();

    // Zero them first (parallel):
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    // Compute pairwise accelerations:
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i) {
        double xi = x[i];
        double yi = y[i];
        double axi = 0.0;
        double ayi = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double dist2 = dx*dx + dy*dy + soft*soft;
            double invDist3 = 1.0 / (dist2 * std::sqrt(dist2));
            double factor = G * m[j] * invDist3;
            axi += factor * dx;
            ayi += factor * dy;
        }
        ax(i) = axi;
        ay(i) = ayi;
    }

    return py::make_tuple(ax_out, ay_out);
}


// -----------------------------------------------------------------------------------
// Part 2: Barnes‐Hut treecode (O(N log N))
// -----------------------------------------------------------------------------------
//
// We build a quadtree over the 2D domain [-domain,+domain]×[-domain,+domain].  Each node
// holds either a single particle (if leaf) or 4 children subdividing the square.  We store
// in each node: total mass, center‐of‐mass (com_x, com_y).  Then for each target i, we
// traverse the tree.  If (NodeSize / dist_to_node_com) < theta, we treat the entire node
// as one “pseudo‐particle” at its COM.  Otherwise, if leaf, do direct sum over leaf’s single
// particle (which is just itself except when the leaf holds a different particle), or recurse.
// 
// The user supplies “theta” (opening angle), “domain” (root half‐width), G, and softening.
// We parallelize the per‐target loop with OpenMP.
// -----------------------------------------------------------------------------------

static const int BH_BUCKET = 1;  // max particles per leaf = 1

struct BHNode {
    double cx, cy, size;         // center & half‐width of this square cell
    double mass, com_x, com_y;   // total mass & center‐of‐mass of this cell
    bool   is_leaf;              // leaf if contains ≤ BH_BUCKET particles
    int    particle_index;       // index of the single particle if leaf (otherwise unused)
    BHNode* children[4];         // pointers to SW, SE, NW, NE (or nullptr)

    BHNode(double _cx, double _cy, double _size)
      : cx(_cx), cy(_cy), size(_size),
        mass(0.0), com_x(0.0), com_y(0.0),
        is_leaf(true), particle_index(-1)
    {
        for (int i = 0; i < 4; ++i) children[i] = nullptr;
    }

    ~BHNode() {
        for (int i = 0; i < 4; ++i) {
            if (children[i]) {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }
};

/**
 * Insert a single particle “pi” with position (x[pi], y[pi]) and mass m[pi]
 * into the BH‐tree rooted at “node”.  If node is leaf and empty, store the particle.
 * If node is leaf and already contains one particle, subdivide into 4 children,
 * move that old particle down, then insert the new one.  If node is internal,
 * route to the correct child based on quadrant.
 */
static void bh_insert(BHNode* node,
                      const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& m,
                      size_t pi)
{
    // If empty leaf, just store:
    if (node->is_leaf && node->particle_index < 0) {
        node->particle_index = (int)pi;
        return;
    }

    // If leaf & already has one particle → need to subdivide:
    if (node->is_leaf) {
        // Create 4 children by splitting this square into 4:
        double half = node->size / 2.0;
        double x0 = node->cx, y0 = node->cy;
        node->children[0] = new BHNode(x0 - half/2.0, y0 - half/2.0, half/2.0); // SW
        node->children[1] = new BHNode(x0 + half/2.0, y0 - half/2.0, half/2.0); // SE
        node->children[2] = new BHNode(x0 - half/2.0, y0 + half/2.0, half/2.0); // NW
        node->children[3] = new BHNode(x0 + half/2.0, y0 + half/2.0, half/2.0); // NE
        node->is_leaf = false;

        // Re‐insert the previously stored particle into one child:
        int old_pi = node->particle_index;
        double ox = x[old_pi], oy = y[old_pi];
        int octant_old = (ox > x0) + 2*(oy > y0);
        bh_insert(node->children[octant_old], x, y, m, old_pi);

        // Clear this node’s stored index
        node->particle_index = -1;
    }

    // Now node is definitely internal → insert “pi” into correct child:
    double tx = x[pi], ty = y[pi];
    double x0 = node->cx, y0 = node->cy;
    int oct = (tx > x0) + 2*(ty > y0);
    bh_insert(node->children[oct], x, y, m, pi);
}

/**
 * After having inserted all particles, we run a post‐order traversal to compute
 * mass and center‐of‐mass at each node.  At a leaf with one particle, simply store
 * m and com=(x[pi], y[pi]).  At an internal node, sum children’s mass and COMs.
 */
static void bh_compute_mass_distribution(
    BHNode* node,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& m
) {
    if (!node) return;

    // If leaf and holds a particle:
    if (node->is_leaf) {
        int pi = node->particle_index;
        if (pi < 0) {
            // empty leaf (shouldn’t happen in our code)
            node->mass = 0.0;
            node->com_x = node->cx;
            node->com_y = node->cy;
        } else {
            // exactly one particle here
            node->mass  = m[pi];
            node->com_x = x[pi];
            node->com_y = y[pi];
        }
        return;
    }

    // Internal node: traverse children first
    double Mtot = 0.0;
    double sumX = 0.0, sumY = 0.0;
    for (int c = 0; c < 4; ++c) {
        if (node->children[c]) {
            bh_compute_mass_distribution(node->children[c], x, y, m);
            Mtot += node->children[c]->mass;
            sumX += node->children[c]->mass * node->children[c]->com_x;
            sumY += node->children[c]->mass * node->children[c]->com_y;
        }
    }
    node->mass  = Mtot;
    if (Mtot > 0.0) {
        node->com_x = sumX / Mtot;
        node->com_y = sumY / Mtot;
    } else {
        node->com_x = node->cx;
        node->com_y = node->cy;
    }
}

/**
 * Build a Barnes‐Hut quadtree over domain [-domain, +domain]×[-domain,+domain]:
 *   - Insert each particle i into the tree,
 *   - Then do mass & COM pass.
 * Returns root pointer.
 */
static BHNode* bh_build_tree(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const std::vector<double>& m,
                             double domain)
{
    BHNode* root = new BHNode(0.0, 0.0, domain);
    size_t N = x.size();
    for (size_t i = 0; i < N; ++i) {
        bh_insert(root, x, y, m, i);
    }
    bh_compute_mass_distribution(root, x, y, m);
    return root;
}

/**
 * Evaluate acceleration at target (tx, ty) by traversing the BH‐tree.
 * - If (node->size / dist_to_node_COM) < theta → treat node as single mass at COM.
 * - Else if leaf → direct‐sum with that one particle.
 * - Else → recurse into all children.
 */
static void bh_evaluate_target(
    const BHNode* node,
    double tx, double ty,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& m,
    double G, double soft, double theta,
    double& axi, double& ayi
) {
    if (!node || node->mass == 0.0) return;

    double dx = node->com_x - tx;
    double dy = node->com_y - ty;
    double dist2 = dx*dx + dy*dy + soft*soft;
    double dist = std::sqrt(dist2);

    // Multipole condition:
    if (node->size / dist < theta) {
        // Treat entire node as one mass at COM:
        double invDist3 = 1.0 / (dist2 * dist);
        double factor = G * node->mass * invDist3;
        axi += factor * dx;
        ayi += factor * dy;
        return;
    }

    // If leaf, do direct sum with that one particle (unless that particle is ourselves; skip if same index)
    if (node->is_leaf) {
        int pi = node->particle_index;
        if (pi >= 0) {
            double ddx = x[pi] - tx;
            double ddy = y[pi] - ty;
            double d2 = ddx*ddx + ddy*ddy + soft*soft;
            double invD3 = 1.0 / (d2 * std::sqrt(d2));
            axi += G * m[pi] * invD3 * ddx;
            ayi += G * m[pi] * invD3 * ddy;
        }
        return;
    }

    // Otherwise, internal node not far enough → recurse into children
    for (int c = 0; c < 4; ++c) {
        if (node->children[c]) {
            bh_evaluate_target(node->children[c], tx, ty, x, y, m, G, soft, theta, axi, ayi);
        }
    }
}

/**
 * bh_omp(x, y, m, domain, theta=0.5, G=1.0, soft=0.01) → (ax, ay)
 *
 * Builds a Barnes‐Hut tree over [-domain, +domain]², with opening angle “theta”.
 * Returns accelerations (ax, ay) in two numpy arrays of length N, computed in parallel.
 */
py::tuple bh_omp(py::array_t<double> x_in,
                 py::array_t<double> y_in,
                 py::array_t<double> m_in,
                 double domain,
                 double theta = 0.5,
                 double G     = 1.0,
                 double soft  = 0.01)
{
    size_t N = x_in.shape(0);
    if (N == 0) {
        return py::make_tuple(
            py::array_t<double>(0),
            py::array_t<double>(0)
        );
    }

    // Copy arrays into std::vector:
    std::vector<double> x(N), y(N), m(N);
    auto xx = x_in.unchecked<1>();
    auto yy = y_in.unchecked<1>();
    auto mm = m_in.unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        x[i] = xx(i);
        y[i] = yy(i);
        m[i] = mm(i);
    }

    // Build the tree (serially):
    BHNode* root = bh_build_tree(x, y, m, domain);

    // Prepare outputs and zero them:
    py::array_t<double> ax_out(N), ay_out(N);
    auto ax = ax_out.mutable_unchecked<1>();
    auto ay = ay_out.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    // Evaluate each target i in parallel:
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i) {
        double axi = 0.0, ayi = 0.0;
        bh_evaluate_target(root, x[i], y[i], x, y, m, G, soft, theta, axi, ayi);
        ax(i) = axi;
        ay(i) = ayi;
    }

    // Free the tree:
    delete root;
    return py::make_tuple(ax_out, ay_out);
}


// -----------------------------------------------------------------------------------
// PYBIND11 MODULE DEFINITION
// -----------------------------------------------------------------------------------
PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "2D Direct (O(N^2)) and Barnes-Hut (O(N log N)) force kernels (OpenMP)";

    // export direct_omp and bh_omp to Python:
    m.def("direct_omp",
          &direct_omp,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("G")    = 1.0,
          py::arg("soft") = 0.01,
          R"doc(
            Compute accelerations via the direct O(N^2) method (OpenMP parallelized).
            Returns (ax, ay) arrays of length N.
          )doc"
    );

    m.def("bh_omp",
          &bh_omp,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("domain"),
          py::arg("theta") = 0.5,
          py::arg("G")     = 1.0,
          py::arg("soft")  = 0.01,
          R"doc(
            Compute accelerations via Barnes-Hut treecode (OpenMP parallelized).
            domain : half-width of root quadtree cell.
            theta  : opening angle parameter.
            Returns (ax, ay) arrays of length N.
          )doc"
    );

    // Flag to indicate whether OpenMP is enabled:
    #ifdef _OPENMP
      m.attr("has_openmp") = true;
    #else
      m.attr("has_openmp") = false;
    #endif
}

