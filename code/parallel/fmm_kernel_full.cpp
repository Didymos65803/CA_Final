// fmm_kernel_full.cpp
// ===================
//
// High-precision Fast Multipole Method (FMM) with OpenMP acceleration.
// This file implements a P=8 complex‐expansion FMM in 2D. The final “evaluate_forces”
// step is parallelized leaf‐by‐leaf using OpenMP.
//
// Build command (in the same folder as setup.py):
//     python setup.py build_ext --inplace
//
// If compiled correctly, you’ll get a shared library `fmm_kernel*.so` which
// can be imported in Python as `import fmm_kernel`.  The symbol `fmm_kernel.has_openmp`
// will tell you whether OpenMP was enabled.
//
// Author: Your Name (2025-06-XX)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <memory>
#include <functional>
#include <iostream>

#ifdef _OPENMP
  #include <omp.h>
  #define USE_OPENMP
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

// ----- PARAMETERS FOR MULTIPOLE EXPANSION -----

// We choose P = 8 total expansion terms (0..8) for a high‐precision 2D FMM.
// If you want to change accuracy, adjust P here (and recompile).
constexpr int P = 8;

// Precompute factorials 0!, 1!, 2!, …, P! for quick lookup.
static std::array<double, P + 1> factorial_table = []() {
    std::array<double, P + 1> table;
    table[0] = 1.0;
    for (int i = 1; i <= P; ++i) {
        table[i] = table[i-1] * i;
    }
    return table;
}();

// Compute “n choose k” via a small loop (exact for n,k ≤ P).
static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// ----- FMM TREE NODE DEFINITION -----

/**
 * FMMCell:
 *   Represents one node in the 2D quadtree. Each node holds:
 *     - center (cx,cy), half‐size “size” (so full box is 2*size × 2*size)
 *     - list of particle indices if leaf
 *     - multipole expansion coefficients multipole[0..P]
 *     - local expansion coefficients local[0..P]
 *     - up to four children (NW, NE, SW, SE)
 *     - parent pointer (nullptr at root)
 *     - is_leaf flag (true if no children)
 */
struct FMMCell {
    double cx, cy;                          // Box center
    double size;                            // Half‐length of box
    int    level;                           // Depth in tree (root = 0)
    bool   is_leaf;                         // True if this is a leaf node

    std::vector<size_t> particles;          // List of particle indices inside this leaf
    std::array<cplx, P + 1> multipole;      // Multipole coefficients a_0..a_P
    std::array<cplx, P + 1> local;          // Local expansion b_0..b_P

    std::array<std::unique_ptr<FMMCell>, 4> children;  // 0=NW,1=NE,2=SW,3=SE
    FMMCell* parent;                        // Parent pointer (nullptr if root)

    // Constructor: initialize center (cx,cy), half‐size “size”, depth level, parent pointer
    FMMCell(double _cx, double _cy, double _size, int _level = 0, FMMCell* _parent = nullptr)
        : cx(_cx), cy(_cy), size(_size), level(_level), parent(_parent) {
        is_leaf = true;
        // Zero‐out expansions
        for (int i = 0; i <= P; ++i) {
            multipole[i] = cplx(0.0, 0.0);
            local[i]     = cplx(0.0, 0.0);
        }
    }
};

// ----- HELPER: COLLECT ALL LEAVES INTO A VECTOR -----

/**
 * collect_leaves(root, leaf_list)
 *   Recursively traverse the quadtree. Whenever a node is a leaf, push its pointer
 *   into leaf_list. This “flattens” all leaves so that we can parallelize leaf‐by‐leaf.
 */
static void collect_leaves(FMMCell* root, std::vector<FMMCell*>& leaf_list) {
    if (!root) return;
    if (root->is_leaf) {
        leaf_list.push_back(root);
    } else {
        for (int qi = 0; qi < 4; ++qi) {
            if (root->children[qi]) {
                collect_leaves(root->children[qi].get(), leaf_list);
            }
        }
    }
}

// ----- FORWARD DECLARATIONS: ORIGINAL SINGLE‐THREADED FMM STEPS -----
//
//   These routines must be copied exactly from your working serial FMM
//   implementation. They remain unmodified so that the FMM algorithm itself
//   is identical. We only change the final "evaluate_forces" to be parallel.
//
//   1) fmm_subdivide: build/quadtree‐refinement until ≤ max_particles or max_level.
//
   void fmm_subdivide(
       FMMCell*                      cell,
       const std::vector<double>&    x,
       const std::vector<double>&    y,
       int                           max_particles,
       int                           max_level
   );
//
//   2) fmm_upward_pass: P2M in leaves, then M2M up the tree to build multipoles.
//
   void fmm_upward_pass(
       FMMCell*                      cell,
       const std::vector<double>&    x,
       const std::vector<double>&    y,
       const std::vector<double>&    m
   );
//
//   3) fmm_m2l_translation: core M2L translation from one source node to one target node.
//
   void fmm_m2l_translation(
       FMMCell*                      target,
       FMMCell*                      source
   );
//
//   4) fmm_interaction_pass: traverse tree to do M2L for each node.
//
   void fmm_interaction_pass(
       FMMCell*                      cell,
       FMMCell*                      root,
       double                        theta
   );
//
//   5) fmm_downward_pass: L2L to propagate local expansions from parent to children.
//
   void fmm_downward_pass(
       FMMCell*                      cell
   );
//
//   6) fmm_evaluate_forces: single‐threaded final force evaluation (direct + local).
//      (We will _replace_ calls to fmm_evaluate_forces with a parallel version.)
//
   void fmm_evaluate_forces(
       FMMCell*                        cell,
       const std::vector<double>&      x,
       const std::vector<double>&      y,
       const std::vector<double>&      m,
       std::vector<double>&            fx,
       std::vector<double>&            fy,
       double                          G,
       double                          soft2
   );

// ============================================================================
// PARALLEL VERSION OF EVALUATE_FORCES (LEAF‐BY‐LEAF using OpenMP)
// ============================================================================

/**
 * parallel_fmm_evaluate_forces(root, x, y, m, fx, fy, G, soft2)
 *
 * Given a fully built FMM tree (with all multipole/local expansions done),
 * this function collects all leaf pointers, then in parallel (OpenMP) loops
 * over leaves.  Each leaf does two things:
 *   1) Near‐field: direct pairwise interactions among particles in that leaf.
 *   2) Far‐field: evaluate the local expansion (L2P) at each particle in that leaf.
 *
 * We use #pragma omp atomic when updating fx[i], fy[i] to avoid race conditions.
 */
static void parallel_fmm_evaluate_forces(
    FMMCell*                         root,
    const std::vector<double>&       x,
    const std::vector<double>&       y,
    const std::vector<double>&       m,
    std::vector<double>&             fx,
    std::vector<double>&             fy,
    double                           G,
    double                           soft2
) {
    // (a) Build a flat list of all leaf cells
    std::vector<FMMCell*> leaf_list;
    leaf_list.reserve(x.size()/4 + 1);  // heuristic
    collect_leaves(root, leaf_list);
    size_t num_leaves = leaf_list.size();

    // (b) Parallel loop over all leaves
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t idx = 0; idx < num_leaves; ++idx) {
        FMMCell* leaf = leaf_list[idx];
        const std::vector<size_t>& plist = leaf->particles;
        size_t leaf_n = plist.size();

        // ---- (1) NEAR‐FIELD DIRECT within this leaf ----
        for (size_t a = 0; a < leaf_n; ++a) {
            size_t i = plist[a];
            for (size_t b = a + 1; b < leaf_n; ++b) {
                size_t j = plist[b];
                if (i == j) continue;
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double r2 = dx*dx + dy*dy + soft2;
                double inv_r3 = 1.0 / std::pow(r2, 1.5);
                double fijx = G * m[j] * dx * inv_r3;
                double fijy = G * m[j] * dy * inv_r3;

                // update forces atomically
                #ifdef USE_OPENMP
                #pragma omp atomic
                #endif
                fx[i] += fijx;
                #ifdef USE_OPENMP
                #pragma omp atomic
                #endif
                fy[i] += fijy;

                #ifdef USE_OPENMP
                #pragma omp atomic
                #endif
                fx[j] -= fijx;
                #ifdef USE_OPENMP
                #pragma omp atomic
                #endif
                fy[j] -= fijy;
            }
        }

        // ---- (2) FAR‐FIELD: evaluate L2P (local expansion) at each particle ----
        double cx = leaf->cx;
        double cy = leaf->cy;
        const std::vector<cplx>& L = leaf->local;
        int Psize = static_cast<int>(L.size());  // should equal (P+1)

        for (size_t a = 0; a < leaf_n; ++a) {
            size_t i = plist[a];
            double rx = x[i] - cx;
            double ry = y[i] - cy;
            cplx z(rx, ry);

            // Evaluate ∇φ from local expansion: sum_{n=1..P} n * L[n] * z^(n-1)
            cplx dz(0.0, 0.0);
            cplx power(1.0, 0.0);   // z^(n-1)

            for (int n = 1; n < Psize; ++n) {
                dz += static_cast<double>(n) * L[n] * power;
                power *= z;   // next power = z^n
            }

            // Convert complex gradient to real force
            //  ∂φ/∂x =  Re(dz),   ∂φ/∂y = −Im(dz)  (convention for 2D FMM)
            double lok_x =  dz.real();
            double lok_y = -dz.imag();

            #ifdef USE_OPENMP
            #pragma omp atomic
            #endif
            fx[i] += lok_x;
            #ifdef USE_OPENMP
            #pragma omp atomic
            #endif
            fy[i] += lok_y;
        }
    }
}

// ============================================================================
// PYBIND11 WRAPPER: fmm_omp()
// ============================================================================

/**
 * fmm_omp(x, y, m, domain, theta, G, soft)
 *
 * Python‐callable entry point.  Steps:
 *   1) Copy input NumPy arrays (x,y,m) into C++ vectors vx, vy, vm.
 *   2) Build root FMMCell with center=(0,0), half-size=(domain/2).
 *   3) Assign all N particles to root->particles, then call fmm_subdivide(…,16,10).
 *   4) fmm_upward_pass(root, vx, vy, vm)  // P2M + M2M
 *   5) fmm_interaction_pass(root, root, theta)  // M2L
 *   6) fmm_downward_pass(root)  // L2L
 *   7) parallel_fmm_evaluate_forces(root, vx, vy, vm, fx, fy, G, soft2)
 *   8) Copy fx,fy into two NumPy arrays and return (ax,ay).
 *   9) Delete the entire tree (avoid memory leak).
 *
 * Returns a tuple (ax_array, ay_array) of length N each.
 */
py::tuple fmm_omp(
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
        // Return two empty arrays if there are no particles
        return py::make_tuple(py::array_t<double>(0),
                              py::array_t<double>(0));
    }

    // Copy NumPy arrays -> std::vector<double>
    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = x.at(i);
        vy[i] = y.at(i);
        vm[i] = m.at(i);
    }

    // Prepare force vectors
    std::vector<double> fx(N, 0.0), fy(N, 0.0);
    double soft2 = soft * soft;

    // (1) Build root cell and insert all particles
    FMMCell* root = new FMMCell(0.0, 0.0, domain * 0.5, 0, nullptr);
    {
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0u);
        // Subdivide until each leaf has ≤ 16 particles or depth ≤ 10
        fmm_subdivide(root, vx, vy, /*max_particles=*/16, /*max_level=*/10);
    }

    // (2) Upward Pass: P2M + M2M
    fmm_upward_pass(root, vx, vy, vm);

    // (3) Interaction Pass: M2L
    fmm_interaction_pass(root, root, theta);

    // (4) Downward Pass: L2L
    fmm_downward_pass(root);

    // (5) Parallel Evaluate: leaf‐by‐leaf near‐field + local expansion
    parallel_fmm_evaluate_forces(root, vx, vy, vm, fx, fy, G, soft2);

    // (6) Copy C++ force arrays back to NumPy arrays
    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }

    // (7) Delete entire quadtree to free memory
    std::function<void(FMMCell*)> delete_tree = [&](FMMCell* c) {
        if (!c) return;
        for (int qi = 0; qi < 4; ++qi) {
            if (c->children[qi]) {
                delete_tree(c->children[qi].release());
            }
        }
        delete c;
    };
    delete_tree(root);

    return py::make_tuple(ax_out, ay_out);
}

// ============================================================================
// MODULE DEFINITION
// ============================================================================

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "High-precision Fast Multipole Method (FMM) with OpenMP acceleration";

    m.def("fmm_omp", &fmm_omp,
          "FMM (OMP-enabled). Args:\n"
          "  x, y, m      : 1D arrays of length N (positions & masses)\n"
          "  domain       : half‐width of bounding box\n"
          "  theta        : opening angle (M2L criterion)\n"
          "  G            : gravitational constant (default=1.0)\n"
          "  soft         : softening length (default=0.05)\n"
          "Returns (ax, ay): two 1D arrays of length N with force components.",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("theta") = 0.5,
          py::arg("G") = 1.0, py::arg("soft") = 0.05);

    #ifdef USE_OPENMP
    m.attr("has_openmp") = true;
    #else
    m.attr("has_openmp") = false;
    #endif
}

