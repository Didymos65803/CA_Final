// force_kernel_full.cpp
// =====================
//
// High-precision direct‐and‐Barnes‐Hut force kernels with OpenMP.
// This file implements two functions, `direct_omp` and `bh_omp`, both callable from Python.
//
// Build command (in the same folder as setup.py):
//     python setup.py build_ext --inplace
//
// This will generate a shared library `force_kernel*.so`, which can be imported as `import force_kernel`
// and will export: `direct_omp(x,y,m,G,soft)`, `bh_omp(x,y,m,domain,theta,G,soft)`, and
// `force_kernel.has_openmp`.
//
// Author: Your Name (2025-06-XX)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
  #define USE_OPENMP
#endif

namespace py = pybind11;

// -----------------------------------------------------------------------------
// DIRECT N-BODY KERNEL (OMP-parallelized)
// -----------------------------------------------------------------------------
py::tuple direct_omp(
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> m,
    double              G    = 1.0,
    double              soft = 0.05
) {
    size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0),
                              py::array_t<double>(0));
    }

    // Access raw data from NumPy
    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    // Copy into std::vector<double> for contiguous OMP loops
    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = px(i);
        vy[i] = py_(i);
        vm[i] = pm(i);
    }

    // Prepare output force arrays
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    double soft2 = soft * soft;

    // OMP‐parallelized double loop: each i computes force from all j≠i
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < N; ++i) {
        double fx_i = 0.0;
        double fy_i = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            double dx = vx[j] - vx[i];
            double dy = vy[j] - vy[i];
            double r2 = dx*dx + dy*dy + soft2;
            double inv_r3 = 1.0 / std::pow(r2, 1.5);
            fx_i += G * vm[j] * dx * inv_r3;
            fy_i += G * vm[j] * dy * inv_r3;
        }
        fx[i] = fx_i;
        fy[i] = fy_i;
    }

    // Copy back to NumPy arrays
    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }
    return py::make_tuple(ax_out, ay_out);
}

// -----------------------------------------------------------------------------
// BARNES‐HUT TREE NODE DEFINITION
// -----------------------------------------------------------------------------

/**
 * BHNode:
 *   Represents one node in the Barnes‐Hut quadtree.
 *
 *   Fields:
 *     - cx, cy       : center of this box
 *     - size         : half‐width of box
 *     - is_leaf      : true if no children
 *     - total_mass   : sum of masses in this box
 *     - com_x, com_y : center‐of‐mass of all particles in this box
 *     - children[0..3]: four child pointers (NW=0, NE=1, SW=2, SE=3)
 *     - particle_indices: if leaf, list of particles inside
 */
struct BHNode {
    double cx, cy;                // Box center
    double size;                  // Half‐width of box
    bool   is_leaf;               // True if no children

    double total_mass;            // Sum of masses in this node
    double com_x, com_y;          // Center‐of‐mass (over all particles)
    std::vector<size_t> particle_indices;  // Indices if leaf

    std::array<std::unique_ptr<BHNode>, 4> children;  // 0=NW,1=NE,2=SW,3=SE

    BHNode(double _cx, double _cy, double _size)
        : cx(_cx), cy(_cy), size(_size) {
        is_leaf = true;
        total_mass = 0.0;  // will be set when inserting
        com_x = com_y = 0.0;
    }
};

/**
 * bh_insert_particle(node, pid, x, y, m):
 *   Inserts a single particle index pid (with position x[pid], y[pid], mass m[pid])
 *   into the quadtree rooted at ‘node’.  If the node is currently a leaf and already
 *   contains some other particle, it subdivides into 4 children, re‐inserts the existing
 *   particle, and then inserts the new one.  Otherwise, it just propagates downward.
 */
static void bh_insert_particle(
    BHNode*                        node,
    size_t                         pid,
    const std::vector<double>&     x,
    const std::vector<double>&     y,
    const std::vector<double>&     m
) {
    // If this node has no particles yet, just add pid and update mass/COM
    if (node->particle_indices.empty()) {
        node->particle_indices.push_back(pid);
        node->total_mass = m[pid];
        node->com_x = x[pid];
        node->com_y = y[pid];
        return;
    }

    // If this node is a leaf but already has 1 or more particles, we must subdivide
    if (node->is_leaf) {
        // Grab existing list of indices (there can be >1 if we allowed >1 per leaf; adjust as you wish)
        std::vector<size_t> existing = node->particle_indices;
        node->particle_indices.clear();
        node->is_leaf = false;

        double half = node->size / 2.0;

        // Create 4 children
        node->children[0] = std::make_unique<BHNode>(node->cx - half, node->cy - half, half, node);
        node->children[1] = std::make_unique<BHNode>(node->cx + half, node->cy - half, half, node);
        node->children[2] = std::make_unique<BHNode>(node->cx - half, node->cy + half, half, node);
        node->children[3] = std::make_unique<BHNode>(node->cx + half, node->cy + half, half, node);

        // Re-insert all existing particles
        for (size_t old_pid : existing) {
            double ex = x[old_pid], ey = y[old_pid];
            int child_idx = (ex > node->cx ? 1 : 0) + (ey > node->cy ? 2 : 0);
            bh_insert_particle(node->children[child_idx].get(),
                               old_pid, x, y, m);
        }
    }

    // If node now has children, insert pid downward
    if (!node->is_leaf) {
        double px = x[pid];
        double py = y[pid];
        int child_idx = (px > node->cx ? 1 : 0) + (py > node->cy ? 2 : 0);
        bh_insert_particle(node->children[child_idx].get(), pid, x, y, m);
    }
}

/**
 * After building the entire tree (all insertions), we do a post-order traversal
 * to compute total_mass and center‐of‐mass (com_x, com_y) at every internal node.
 */
static void bh_compute_mass_distribution(
    BHNode* node,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& m
) {
    if (!node) return;

    if (node->is_leaf) {
        // Leaf: particle_indices has exactly one element
        size_t pid = node->particle_indices[0];
        node->total_mass = m[pid];
        node->com_x = x[pid];
        node->com_y = y[pid];
    } else {
        // Internal: sum up children
        double sum_m = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (int c = 0; c < 4; ++c) {
            if (node->children[c]) {
                bh_compute_mass_distribution(node->children[c].get(), x, y, m);
                sum_m += node->children[c]->total_mass;
                sum_x += node->children[c]->com_x * node->children[c]->total_mass;
                sum_y += node->children[c]->com_y * node->children[c]->total_mass;
            }
        }
        if (sum_m > 0.0) {
            node->total_mass = sum_m;
            node->com_x = sum_x / sum_m;
            node->com_y = sum_y / sum_m;
        } else {
            node->total_mass = 0.0;
            node->com_x = node->cx;
            node->com_y = node->cy;
        }
    }
}

/**
 * bh_compute_force(node, px, py, pid, x,y,m,theta,G,soft2, &fx, &fy)
 *   Recursively traverse the Barnes‐Hut tree “node” to compute force on a
 *   test particle at (px,py) with ID=pid.  If node is far enough (s/r < theta),
 *   approximate entire node as its center‐of‐mass.  Otherwise, descend into children.
 */
static void bh_compute_force(
    const BHNode*                   node,
    double                           px,
    double                           py,
    size_t                           pid,
    const std::vector<double>&       x,
    const std::vector<double>&       y,
    const std::vector<double>&       m,
    double                           theta,
    double                           G,
    double                           soft2,
    double&                          fx,
    double&                          fy
) {
    if (!node || node->total_mass == 0.0) return;

    if (node->is_leaf) {
        // Direct sum over all particles in this leaf (there may be only 1)
        for (size_t other : node->particle_indices) {
            if (other == pid) continue;
            double dx = x[other] - px;
            double dy = y[other] - py;
            double r2 = dx*dx + dy*dy + soft2;
            if (r2 < 1e-20) continue;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            fx += G * m[other] * dx * inv_r3;
            fy += G * m[other] * dy * inv_r3;
        }
    } else {
        // Internal node: check opening criterion
        double dx = node->com_x - px;
        double dy = node->com_y - py;
        double r2 = dx*dx + dy*dy + soft2;
        double r  = std::sqrt(r2);
        if (node->size / r < theta && r2 > 1e-20) {
            // Approximate entire node by its COM
            double inv_r = 1.0 / r;
            double inv_r3 = inv_r * inv_r * inv_r;
            fx += G * node->total_mass * dx * inv_r3;
            fy += G * node->total_mass * dy * inv_r3;
        } else {
            // Recurse into children
            for (int c = 0; c < 4; ++c) {
                if (node->children[c]) {
                    bh_compute_force(node->children[c].get(), px, py, pid,
                                     x, y, m, theta, G, soft2, fx, fy);
                }
            }
        }
    }
}

/**
 * bh_omp(x, y, m, domain, theta, G, soft)
 *   Build a Barnes‐Hut quadtree, compute mass distribution, then evaluate
 *   force on each particle (i=0..N-1) using bh_compute_force(...).  The outer
 *   loop over i is OpenMP‐parallelized.  Returns (ax, ay) arrays.
 */
py::tuple bh_omp(
    py::array_t<double> x,
    py::array_t<double> y,
    py::array_t<double> m,
    double              domain,
    double              theta = 0.5,
    double              G     = 1.0,
    double              soft  = 0.05
) {
    size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0),
                              py::array_t<double>(0));
    }

    auto px = x.unchecked<1>();
    auto py_ = y.unchecked<1>();
    auto pm = m.unchecked<1>();

    // Copy into vectors for easier indexing
    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = px(i);
        vy[i] = py_(i);
        vm[i] = pm(i);
    }

    // Build root node
    auto root = std::make_unique<BHNode>(0.0, 0.0, domain * 0.5);

    // Insert all particles into the tree
    for (size_t i = 0; i < N; ++i) {
        bh_insert_particle(root.get(), i, vx, vy, vm);
    }

    // Compute mass-distribution (total_mass, COM) in each internal node
    bh_compute_mass_distribution(root.get(), vx, vy, vm);

    // Prepare output arrays
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    double soft2 = soft * soft;

    // Compute force on each particle using BH
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < N; ++i) {
        double fx_i = 0.0;
        double fy_i = 0.0;
        bh_compute_force(root.get(), vx[i], vy[i], i,
                         vx, vy, vm, theta, G, soft2, fx_i, fy_i);
        fx[i] = fx_i;
        fy[i] = fy_i;
    }

    // Copy back to NumPy arrays
    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }

    return py::make_tuple(ax_out, ay_out);
}

// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "High-precision N-body force kernels (Direct & Barnes-Hut) with OpenMP";

    m.def("direct_omp", &direct_omp,
          "Direct N-body (O(N^2)) force calculation (with OpenMP).\n"
          "Args: x, y, m (length N),\n"
          "      G    : gravitational constant (default=1.0),\n"
          "      soft : softening length (default=0.05).\n"
          "Returns: (ax, ay) arrays of length N.",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("G") = 1.0, py::arg("soft") = 0.05);

    m.def("bh_omp", &bh_omp,
          "Barnes-Hut O(N log N) force calculation (with OpenMP).\n"
          "Args: x, y, m (length N),\n"
          "      domain : half-width of bounding box,\n"
          "      theta  : opening angle (default=0.5),\n"
          "      G      : gravitational constant (default=1.0),\n"
          "      soft   : softening length (default=0.05).\n"
          "Returns: (ax, ay) arrays of length N.",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("theta") = 0.5,
          py::arg("G") = 1.0, py::arg("soft") = 0.05);

    #ifdef USE_OPENMP
    m.attr("has_openmp") = true;
    #else
    m.attr("has_openmp") = false;
    #endif
}

