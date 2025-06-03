// fmm_kernel_full.cpp
// =====================
//
// 2D Fast Multipole Method (FMM) with OpenMP parallelization.
// Exports one function to Python: fmm_omp(...), plus a flag has_openmp.
//
// To build (in the same folder as setup.py):
//     python3.12 setup.py build_ext --inplace
// That produces: fmm_kernel.cpython-<ver>-<arch>.so.
//
// Author: (Corrected for proper cell subdivision to avoid segfault)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------------
// FMM QUADTREE NODE DEFINITION
// -----------------------------------------------------------------------------------
//
// Each node (cell) represents a square region of half‐width `size` centered at (cx, cy).
// That means this cell covers [cx-size, cx+size] × [cy-size, cy+size].
//
// We store:
//   - mass      : total mass inside this cell
//   - mx, my    : dipole moment = Σ( m[i]*x[i], m[i]*y[i] ) for all particles in this cell
//   - is_leaf   : true if this cell has ≤ BUCKET_SIZE particles, false if we subdivided
//   - particles : vector of indices of particles if this cell is a leaf
//   - children  : pointers to four subcells (SW=0, SE=1, NW=2, NE=3) if not a leaf
// -----------------------------------------------------------------------------------

static const int BUCKET_SIZE = 8;  // max # of particles per leaf before subdividing

struct FMMCell {
    double cx, cy;       // center coordinates of this cell
    double size;         // half‐width of this square cell
    double mass;         // total mass in this cell
    double mx, my;       // dipole moment (mass * position sum)
    bool is_leaf;        // whether this cell is a leaf
    std::vector<size_t> particles; // indices of particles if leaf
    FMMCell* children[4];          // pointers to child cells (SW, SE, NW, NE)

    // Constructor: set center (cx, cy), half‐width `size`, zero mass/dipole, leaf=true
    FMMCell(double _cx, double _cy, double _size)
      : cx(_cx), cy(_cy), size(_size),
        mass(0.0), mx(0.0), my(0.0),
        is_leaf(true)
    {
        for (int i = 0; i < 4; ++i) {
            children[i] = nullptr;
        }
    }

    // Destructor: recursively delete children
    ~FMMCell() {
        for (int i = 0; i < 4; ++i) {
            if (children[i]) {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }
};

// -----------------------------------------------------------------------------------
// Insert a single particle `pi` into the quadtree rooted at `cell`.
// If the cell is a leaf and has < BUCKET_SIZE particles, we just push_back(pi).
// If the cell is a leaf and already has BUCKET_SIZE particles, we subdivide it
// into 4 children (each with half the parent’s size), re‐insert all existing
// particles into the appropriate children, then insert `pi` as well.
// If the cell is internal, we route `pi` down to the correct child.
//
// IMPORTANT: We treat `size` as “half‐width.”  When subdividing, each child’s
// `size` becomes exactly parent.size/2.  Child centers are offset by ±child.size.
// -----------------------------------------------------------------------------------
static void fmm_insert_particle(FMMCell* cell,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                size_t pi)
{
    // If this cell is currently a leaf and has space (< BUCKET_SIZE), just store it:
    if (cell->is_leaf && cell->particles.size() < BUCKET_SIZE) {
        cell->particles.push_back(pi);
        return;
    }

    // If this is a leaf but FULL (has exactly BUCKET_SIZE particles), subdivide:
    if (cell->is_leaf) {
        // Compute child size = half of this cell’s size:
        double child_size = cell->size / 2.0;
        double x0 = cell->cx;
        double y0 = cell->cy;

        // Create 4 children: SW, SE, NW, NE
        // SW child: center = (x0 - child_size, y0 - child_size)
        // SE child: center = (x0 + child_size, y0 - child_size)
        // NW child: center = (x0 - child_size, y0 + child_size)
        // NE child: center = (x0 + child_size, y0 + child_size)
        cell->children[0] = new FMMCell(x0 - child_size, y0 - child_size, child_size); // SW
        cell->children[1] = new FMMCell(x0 + child_size, y0 - child_size, child_size); // SE
        cell->children[2] = new FMMCell(x0 - child_size, y0 + child_size, child_size); // NW
        cell->children[3] = new FMMCell(x0 + child_size, y0 + child_size, child_size); // NE

        cell->is_leaf = false;

        // Re‐insert all existing particles (the ones currently in cell->particles)
        for (size_t old_pi : cell->particles) {
            double ox = x[old_pi];
            double oy = y[old_pi];
            // Determine which child holds (ox, oy):
            int oct = (ox > x0 ? 1 : 0) + 2 * (oy > y0 ? 1 : 0);
            fmm_insert_particle(cell->children[oct], x, y, old_pi);
        }

        // Clear the particle list for this node:
        cell->particles.clear();
    }

    // Now cell is definitely not a leaf → determine the correct child and recurse
    double tx = x[pi];
    double ty = y[pi];
    double x0 = cell->cx;
    double y0 = cell->cy;
    int oct = (tx > x0 ? 1 : 0) + 2 * (ty > y0 ? 1 : 0);
    fmm_insert_particle(cell->children[oct], x, y, pi);
}

// -----------------------------------------------------------------------------------
// Build a quadtree over [-domain, +domain] × [-domain, +domain] containing
// N particles at positions (x[i], y[i]).  Returns pointer to the root cell.
// -----------------------------------------------------------------------------------
static FMMCell* fmm_build_tree(const std::vector<double>& x,
                               const std::vector<double>& y,
                               double domain)
{
    // Create root cell centered at (0,0) with half‐width = domain
    FMMCell* root = new FMMCell(0.0, 0.0, domain);
    size_t N = x.size();
    // Insert each particle index into the tree
    for (size_t i = 0; i < N; ++i) {
        fmm_insert_particle(root, x, y, i);
    }
    return root;
}

// -----------------------------------------------------------------------------------
// Upward pass (post‐order) to compute each cell’s total mass and dipole (mx, my):
// If leaf: sum mass & dipole over all its particles.
// If internal: recurse on children, then sum children’s mass & dipole.
// -----------------------------------------------------------------------------------
static void fmm_upward_pass(FMMCell* cell,
                            const std::vector<double>& x,
                            const std::vector<double>& y,
                            const std::vector<double>& m)
{
    if (cell->is_leaf) {
        // Leaf: sum over all contained particles (could be 0..BUCKET_SIZE)
        double M = 0.0, Mx = 0.0, My = 0.0;
        for (size_t pi : cell->particles) {
            M  += m[pi];
            Mx += m[pi] * x[pi];
            My += m[pi] * y[pi];
        }
        cell->mass = M;
        cell->mx   = Mx;
        cell->my   = My;
        return;
    }

    // Internal node: first recurse on children
    double M = 0.0, Mx = 0.0, My = 0.0;
    for (int c = 0; c < 4; ++c) {
        if (cell->children[c]) {
            fmm_upward_pass(cell->children[c], x, y, m);
            M  += cell->children[c]->mass;
            Mx += cell->children[c]->mx;
            My += cell->children[c]->my;
        }
    }
    cell->mass = M;
    cell->mx   = Mx;
    cell->my   = My;
}

// -----------------------------------------------------------------------------------
// Evaluate acceleration at a single target (tx, ty) using FMM “multipole” criterion:
// If (cell->size / r) < theta, treat cell as one point‐mass at its center‐of‐mass.
// If leaf (but not well separated), do direct sum over that leaf’s particles.
// Otherwise, recurse into children.
//
// Adds contribution to (axi, ayi) by reference.
// -----------------------------------------------------------------------------------
static void fmm_evaluate_target(const FMMCell* cell,
                                double tx, double ty,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::vector<double>& m,
                                double G, double soft,
                                double theta,
                                double& axi, double& ayi)
{
    if (!cell || cell->mass == 0.0) return;

    // Distance from target to this cell’s geometric center
    double dx = cell->cx - tx;
    double dy = cell->cy - ty;
    double r2 = dx*dx + dy*dy + 1e-16;  // tiny epsilon to avoid zero
    double r  = std::sqrt(r2);

    // If (cell size / distance) < theta, approximate whole cell by its COM
    if (cell->size / r < theta) {
        // Use center‐of‐mass coordinates:
        double cmx = cell->mx / cell->mass;
        double cmy = cell->my / cell->mass;
        double ddx = cmx - tx;
        double ddy = cmy - ty;
        double dist2 = ddx*ddx + ddy*ddy + soft*soft;
        double invDist3 = 1.0 / (dist2 * std::sqrt(dist2));
        axi += G * cell->mass * invDist3 * ddx;
        ayi += G * cell->mass * invDist3 * ddy;
        return;
    }

    // If leaf (but not well separated), do direct sum over all particles in this leaf:
    if (cell->is_leaf) {
        for (size_t pi : cell->particles) {
            double ddx = x[pi] - tx;
            double ddy = y[pi] - ty;
            double dist2 = ddx*ddx + ddy*ddy + soft*soft;
            double invDist3 = 1.0 / (dist2 * std::sqrt(dist2));
            axi += G * m[pi] * invDist3 * ddx;
            ayi += G * m[pi] * invDist3 * ddy;
        }
        return;
    }

    // Otherwise, not well separated & not a leaf → recurse to children
    for (int c = 0; c < 4; ++c) {
        if (cell->children[c]) {
            fmm_evaluate_target(cell->children[c],
                                tx, ty,
                                x, y, m,
                                G, soft, theta,
                                axi, ayi);
        }
    }
}

// -----------------------------------------------------------------------------------
// Top‐level FMM entry point exposed to Python via pybind11:
//   fmm_omp(x, y, m, domain, theta=0.5, G=1.0, soft=0.01) -> (ax, ay)
//
// Steps:
//  1) Copy numpy arrays into C++ std::vector<double> for faster indexing.
//  2) Build the quadtree over [-domain, +domain]^2 (serial).
//  3) Upward pass (serial) to compute mass & dipole at each cell.
//  4) Allocate output arrays (ax, ay) and zero them (parallel).
//  5) Parallel loop over i=0..N-1: call fmm_evaluate_target on root for each (x[i], y[i]).
//  6) Delete the root (free all cells).  Return (ax, ay).
// -----------------------------------------------------------------------------------
py::tuple fmm_omp(py::array_t<double> x_in,
                  py::array_t<double> y_in,
                  py::array_t<double> m_in,
                  double domain = 10.0,
                  double theta  = 0.5,
                  double G      = 1.0,
                  double soft   = 0.01)
{
    size_t N = x_in.shape(0);
    if (N == 0) {
        // Return empty arrays if no particles
        return py::make_tuple(
            py::array_t<double>(0),
            py::array_t<double>(0)
        );
    }

    // Copy inputs into std::vector<double> for fast indexing
    std::vector<double> x(N), y(N), m(N);
    auto xx = x_in.unchecked<1>();
    auto yy = y_in.unchecked<1>();
    auto mm = m_in.unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        x[i] = xx(i);
        y[i] = yy(i);
        m[i] = mm(i);
    }

    // 1) Build the quadtree (root covers [-domain,+domain]^2)
    FMMCell* root = fmm_build_tree(x, y, domain);

    // 2) Upward pass to compute each cell’s mass & dipole
    fmm_upward_pass(root, x, y, m);

    // 3) Prepare output arrays for accelerations and zero them in parallel
    py::array_t<double> ax_out(N), ay_out(N);
    auto ax = ax_out.mutable_unchecked<1>();
    auto ay = ay_out.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }

    // 4) Evaluate each target in parallel via FMM
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i) {
        double axi = 0.0, ayi = 0.0;
        fmm_evaluate_target(root, x[i], y[i], x, y, m,
                            G, soft, theta, axi, ayi);
        ax(i) = axi;
        ay(i) = ayi;
    }

    // 5) Free the entire quadtree
    delete root;

    return py::make_tuple(ax_out, ay_out);
}

// -----------------------------------------------------------------------------------
// PYBIND11 MODULE DEFINITION
// -----------------------------------------------------------------------------------
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "2D Fast Multipole Method (FMM, OpenMP)";
    m.def("fmm_omp",
          &fmm_omp,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("domain") = 10.0,
          py::arg("theta")  = 0.5,
          py::arg("G")      = 1.0,
          py::arg("soft")   = 0.01,
          R"doc(
            Compute accelerations via 2D Fast Multipole Method (FMM) with OpenMP.
            Returns (ax, ay) arrays of length N.
          )doc"
    );
    m.attr("has_openmp") = true;
}  // <── This closes the PYBIND11_MODULE block, and the file ends here.

