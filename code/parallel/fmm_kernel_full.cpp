// fmm_kernel_full.cpp
// ===================
//
// High-precision Fast Multipole Method (FMM) with OpenMP acceleration.
//
// Compared to the original version, this file:
//   1) Introduces `collect_leaves` to build a flat list of leaf cells.
//   2) Modifies `fmm_evaluate_forces` so that each leaf’s near-field
//      and local-expansion computation runs in an OpenMP parallel loop.
//   3) Retains all original single‐threaded FMM helper functions (subdivide,
//      upward, interaction, downward). You only need to paste your original
//      implementations of those helpers in the marked sections below.
//
// Build command (from the same folder as setup.py):
//   python setup.py build_ext --inplace
//
// If compiled correctly, you’ll get fmm_kernel*.so in this directory.  
// Then Python can do `import fmm_kernel` and see `fmm_kernel.has_openmp == True`.
//
//  Author: (Your Name), Date: 2025-06-XX
// ------------------------------------------------------------------------------------------------

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

/**
 * FMMCell:
 *   - Represents one node (box) in the 2D quadtree.
 *   - Holds multipole expansion (size = P), local expansion (size = P),
 *     list of particle indices, child pointers, etc.
 */
struct FMMCell {
    double cx, cy;                       // Center coordinates of this cell
    double size;                         // Half-size (half the edge length)
    std::array<std::unique_ptr<FMMCell>, 4> children;  // Four quadrants: 0=NW,1=NE,2=SW,3=SE
    std::vector<size_t> particles;       // Particle indices in this cell (only if leaf)
    std::vector<cplx> multipole;         // Multipole expansion coefficients (length = P)
    std::vector<cplx> local;             // Local expansion coefficients (length = P)
    bool is_leaf;                        // True if no children (i.e. a leaf)
    FMMCell* parent;                     // Parent pointer (nullptr for root)

    // Constructor: initialize center (cx,cy), half-size, level, parent
    //   Default P = 15 terms for high precision (you may adjust P as needed)
    FMMCell(double _cx, double _cy, double _size, int level = 0, FMMCell* _parent = nullptr)
        : cx(_cx), cy(_cy), size(_size), is_leaf(true), parent(_parent) {
        const int P = 15;
        multipole.assign(P, cplx(0.0, 0.0));
        local.assign(P, cplx(0.0, 0.0));
    }
};

/**
 * collect_leaves(root, leaf_list)
 *   Recursively traverse the quadtree. Whenever a node has is_leaf == true,
 *   push its pointer into leaf_list. In this way, we flatten the leaves into a
 *   contiguous vector, so that later we can do a single `#pragma omp parallel for`
 *   over all leaves.
 *
 * @param root      Pointer to current FMMCell (node in the tree).
 * @param leaf_list Vector<FMMCell*>& to accumulate pointers of leaf nodes.
 */
static void collect_leaves(FMMCell* root, std::vector<FMMCell*>& leaf_list) {
    if (!root) return;
    if (root->is_leaf) {
        leaf_list.push_back(root);
    } else {
        // Recurse into all non-null children
        for (int qi = 0; qi < 4; ++qi) {
            if (root->children[qi]) {
                collect_leaves(root->children[qi].get(), leaf_list);
            }
        }
    }
}

// Forward declarations of the original single-threaded FMM helper functions.
// You must copy‐paste your original implementations of these *entirely* here,
// without modifying them (so that the algorithm stays exactly as before):
// 
//   - fmm_subdivide: splits a cell into 4 children until ≤ max_particles per leaf
//   - fmm_upward_pass: does P2M on leaves, then M2M combine multipole up the tree
//   - fmm_interaction_pass: for each target cell, do M2L from well-separated sources
//   - fmm_downward_pass: propagate local expansions (L2L) down to children
//   - fmm_evaluate_forces: *ORIGINAL* single-threaded evaluate of near‐field + L2P
//
// In this file, we replace `fmm_evaluate_forces` with a parallel‐enabled version
// further down. But first, declare the signatures so the compiler knows about them.

void fmm_subdivide(FMMCell* cell,
                   const std::vector<double>& x,
                   const std::vector<double>& y,
                   int max_particles,
                   int max_level);

void fmm_upward_pass(FMMCell* cell,
                     const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& m);

void fmm_interaction_pass(FMMCell* cell,
                          FMMCell* root,
                          double theta);

void fmm_downward_pass(FMMCell* cell);

// ------------------------------------------------------------------------------------------------
//  Revised, parallelized version of evaluate_forces. We collect all leaf pointers into `leaf_list`
//  and distribute “leaf‐by‐leaf” to multiple OpenMP threads. Inside each leaf, we do two steps:
//    1) Near‐field direct: Compute pairwise O(n_leaf²) among that leaf’s particles.
//    2) Far‐field local expansion: Evaluate the “local expansion” at each particle in that leaf.
//  We protect every fx[i] += …, fy[i] += … with `#pragma omp atomic` to avoid data races.
//  
//  Note: the *original* single‐threaded implementation of fmm_evaluate_forces can be copied into
//  some separate function if you wish. Here, for clarity, we assume the code below is the ONLY
//  evaluate_forces that will be used (i.e. we no longer call the old serial version).
// ------------------------------------------------------------------------------------------------

/**
 * parallel_fmm_evaluate_forces(root, x, y, m, fx, fy, G, soft2)
 *
 * Given a fully‐built and fully‐translated FMM tree (all multipole/local expansions done),
 * this function:
 *   1) Traverses the tree to collect all leaf cells into `leaf_list`.
 *   2) Splits `leaf_list` among OpenMP threads so that each thread simultaneously
 *      processes several leaves. Inside each leaf:
 *        a) Compute near‐field direct among that leaf’s particles.
 *        b) Add local expansion (L2P) contribution at each particle.
 *
 * @param root   Root pointer of FMM tree.
 * @param x, y, m Vectors of size N (particle positions & masses).
 * @param fx, fy Vectors to accumulate forces (initialized outside to zeros).
 * @param G      Gravitational constant.
 * @param soft2  Softening length squared.
 */
static void parallel_fmm_evaluate_forces(FMMCell* root,
                                         const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         const std::vector<double>& m,
                                         std::vector<double>& fx,
                                         std::vector<double>& fy,
                                         double G,
                                         double soft2)
{
    // (a) Build a flat list of all leaf cells
    std::vector<FMMCell*> leaf_list;
    leaf_list.reserve(x.size() / 4 + 1);  // heuristic: expect ~N/4 leaves
    collect_leaves(root, leaf_list);
    size_t num_leaves = leaf_list.size();

    // (b) Parallel loop over all leaves
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t idx = 0; idx < num_leaves; ++idx) {
        FMMCell* leaf = leaf_list[idx];

        // Each leaf has a small vector of particle indices
        const std::vector<size_t>& plist = leaf->particles;
        size_t leaf_n = plist.size();

        // ----- (1) NEAR-FIELD DIRECT within this leaf -----
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

                // Update forces on i and j (atomic to avoid races)
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

        // ----- (2) FAR-FIELD LOCAL EXPANSION (L2P) from leaf->local -----
        // We assume that `leaf->local[k]` is the k-th coefficient of the local expansion
        // around leaf->(cx,cy). We now evaluate its gradient at each particle in this leaf.
        double cx = leaf->cx;
        double cy = leaf->cy;
        const std::vector<cplx>& L = leaf->local;  // local expansion coefficients
        int P = static_cast<int>(L.size());         // number of expansion terms

        for (size_t a = 0; a < leaf_n; ++a) {
            size_t i = plist[a];
            // Relative coordinate from leaf center
            double rx = x[i] - cx;
            double ry = y[i] - cy;
            cplx z(rx, ry);    // represent as complex z = rx + i ry

            // Evaluate gradient of local expansion: ∑_{n=1..P−1} [ n * L[n] * z^(n-1) ]
            cplx dz(0.0, 0.0); // will accumulate dφ/dx + i·dφ/dy
            cplx power(1.0, 0.0);  // will be z^(n-1) at step n

            for (int n = 1; n < P; ++n) {
                // derivative term for z^n is n * L[n] * z^(n−1)
                dz += static_cast<double>(n) * L[n] * power;
                power *= z;  // update power = z^(n)
            }

            // Convert complex derivative to real force components
            double lok_x =  dz.real();    // ∂φ/∂x
            double lok_y = -dz.imag();    // ∂φ/∂y (sign depends on polynomial convention)

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

/**
 * fmm_omp(x, y, m, domain, theta, G, soft)
 *
 * Pybind11 binding entry point. This is the function that Python calls:
 *   ax, ay = fmm_kernel.fmm_omp(x_array, y_array, m_array, domain, theta, G, soft)
 *
 * Steps:
 *   1) Copy input NumPy arrays to std::vector<double>
 *   2) Build root cell (center=0,0; half-size=domain/2) and put all N particles in it
 *   3) Call fmm_subdivide(...) to create the quadtree (leaf_size ≤ 16 or max_level=10)
 *   4) fmm_upward_pass(...)   – compute multipole expansions
 *   5) fmm_interaction_pass(...) – do M2L between well-separated cells
 *   6) fmm_downward_pass(...)   – propagate local expansions downward (L2L)
 *   7) parallel_fmm_evaluate_forces(...)   – in parallel, each leaf does near-field + L2P
 *   8) Copy fx/fy into NumPy arrays and return as (ax, ay)
 *   9) Delete the tree (avoid memory leak)
 *
 * @param x      (NumPy array[N]) x‐coordinates
 * @param y      (NumPy array[N]) y‐coordinates
 * @param m      (NumPy array[N]) masses
 * @param domain Half‐length of bounding box
 * @param theta  Multipole acceptance criterion
 * @param G      Gravitational constant
 * @param soft   Softening length (we square it inside)
 * @return       Tuple of two NumPy arrays (ax, ay), each length N
 */
py::tuple fmm_omp(py::array_t<double> x,
                  py::array_t<double> y,
                  py::array_t<double> m,
                  double domain,
                  double theta,
                  double G = 1.0,
                  double soft = 0.05) {
    size_t N = x.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0),
                              py::array_t<double>(0));
    }

    // Copy NumPy arrays into C++ vectors
    std::vector<double> vx(N), vy(N), vm(N);
    for (size_t i = 0; i < N; ++i) {
        vx[i] = x.at(i);
        vy[i] = y.at(i);
        vm[i] = m.at(i);
    }

    // Prepare force accumulators
    std::vector<double> fx(N, 0.0), fy(N, 0.0);

    double soft2 = soft * soft;

    // (1) Build root cell and insert all particles
    FMMCell* root = new FMMCell(0.0, 0.0, domain * 0.5, 0, nullptr);
    {
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0u);

        // Subdivide until each leaf ≤ 16 particles or depth ≤ 10
        fmm_subdivide(root, vx, vy, /*max_particles=*/16, /*max_level=*/10);
    }

    // (2) Upward Pass: P2M & M2M build multipole expansions
    fmm_upward_pass(root, vx, vy, vm);

    // (3) Interaction Pass: M2L between well-separated cells
    fmm_interaction_pass(root, root, theta);

    // (4) Downward Pass: L2L propagate local expansions
    fmm_downward_pass(root);

    // (5) Evaluate forces in parallel over leaf nodes
    parallel_fmm_evaluate_forces(root, vx, vy, vm, fx, fy, G, soft2);

    // (6) Copy back to NumPy arrays
    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }

    // (7) Free the quadtree to avoid memory leak
    std::function<void(FMMCell*)> delete_tree = [&](FMMCell* cell) {
        if (!cell) return;
        for (int qi = 0; qi < 4; ++qi) {
            if (cell->children[qi]) {
                delete_tree(cell->children[qi].release());
            }
        }
        delete cell;
    };
    delete_tree(root);

    return py::make_tuple(ax_out, ay_out);
}

// ================================================================================
//  INSERT HERE YOUR ORIGINAL SINGLE-THREADED IMPLEMENTATIONS OF THESE FUNCTIONS:
//    void fmm_subdivide(...);
//    void fmm_upward_pass(...);
//    void fmm_interaction_pass(...);
//    void fmm_downward_pass(...);
//  They must remain exactly as you wrote them before (unmodified), so that the
//  FMM algorithm itself is unchanged—only the final “evaluate_forces” step is parallel.
// ================================================================================

void fmm_subdivide(FMMCell* cell,
                   const std::vector<double>& x,
                   const std::vector<double>& y,
                   int max_particles,
                   int max_level)
{
    // … Copy‐paste your original subdivide code here …
}

void fmm_upward_pass(FMMCell* cell,
                     const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& m)
{
    // … Copy‐paste your original upward pass code here …
}

void fmm_interaction_pass(FMMCell* cell,
                          FMMCell* root,
                          double theta)
{
    // … Copy‐paste your original interaction pass code here …
}

void fmm_downward_pass(FMMCell* cell)
{
    // … Copy‐paste your original downward pass code here …
}

// =================================================================================
//  Finally, the PyBind11 module definition. Remains largely unchanged.
// =================================================================================
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "High-precision Fast Multipole Method (FMM) with OpenMP acceleration";
    m.def("fmm_omp", &fmm_omp,
          "FMM (OMP‐enabled). Args: x,y,m (arrays), domain, theta, G, soft. "
          "Returns: (ax, ay) arrays.",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("theta") = 0.5,
          py::arg("G") = 1.0, py::arg("soft") = 0.05);

    #ifdef USE_OPENMP
    m.attr("has_openmp") = true;
    #else
    m.attr("has_openmp") = false;
    #endif
}

