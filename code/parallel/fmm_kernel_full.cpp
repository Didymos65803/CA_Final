// fmm_kernel_full.cpp
// --------------------------------------------------
// PyBind11 + OpenMP implementation of a 2D Fast Multipole
// Method (FMM) gravitational‐force kernel, accepting NumPy arrays.
//
// Exposes a single function:
//
//    fmm_force(x, y, m, N, domain_size, theta, maxLeaf, eps, G, ax, ay)
//
//   where x, y, m, ax, ay are NumPy arrays (dtype=float64).
//
// Build flags (setup.py passes):
//   -std=c++17 -O3 -DNDEBUG -march=native -ffast-math -fopenmp
// --------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

// -------------------------------------------------------------------
// FMMCell: A node in the quadtree
// -------------------------------------------------------------------
struct FMMCell {
    double cx, cy;               // center of cell
    double size;                 // half‐width
    std::vector<int> particles;  // indices inside this cell
    double mass;                 // total mass
    double mx, my;               // center‐of‐mass
    bool isLeaf;                 // leaf? (no children)
    FMMCell* children[4];        // pointers: 0=SW,1=SE,2=NW,3=NE

    FMMCell(double _cx, double _cy, double _size)
      : cx(_cx), cy(_cy), size(_size),
        mass(0.0), mx(0.0), my(0.0), isLeaf(true)
    {
        for(int i=0; i<4; ++i) children[i] = nullptr;
    }

    ~FMMCell() {
        for(int i=0; i<4; ++i) {
            if(children[i]) {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }
};

// -------------------------------------------------------------------
// build_tree: Insert N particles into root quadtree, splitting leaves
// -------------------------------------------------------------------
void build_tree(FMMCell* root,
                const double* x,
                const double* y,
                int N,
                int maxLeaf)
{
    for(int i = 0; i < N; ++i) {
        double px = x[i], py = y[i];
        FMMCell* cell = root;
        while(true) {
            if(cell->isLeaf) {
                cell->particles.push_back(i);
                if((int)cell->particles.size() > maxLeaf) {
                    // Split leaf into 4 children
                    cell->isLeaf = false;
                    for(int q=0; q<4; ++q) {
                        double offsetX = ((q & 1) ? +0.5 : -0.5) * cell->size;
                        double offsetY = ((q & 2) ? +0.5 : -0.5) * cell->size;
                        cell->children[q] = new FMMCell(
                            cell->cx + offsetX,
                            cell->cy + offsetY,
                            cell->size * 0.5
                        );
                    }
                    // Re‐insert existing particles into children
                    for(int idx : cell->particles) {
                        double rx = x[idx], ry = y[idx];
                        int quadrant = ((rx > cell->cx) ? 1 : 0) + ((ry > cell->cy) ? 2 : 0);
                        cell->children[quadrant]->particles.push_back(idx);
                    }
                    cell->particles.clear();
                } else {
                    break;
                }
            } else {
                int quadrant = ((px > cell->cx) ? 1 : 0) + ((py > cell->cy) ? 2 : 0);
                cell = cell->children[quadrant];
            }
        }
    }
}

// -------------------------------------------------------------------
// gather: Collect pointers to all cells, grouped by depth
// -------------------------------------------------------------------
void gather_cells_by_level(FMMCell* root,
                           std::vector<std::vector<FMMCell*>>& levels,
                           int depth)
{
    if(depth >= (int)levels.size()) {
        levels.resize(depth+1);
    }
    levels[depth].push_back(root);
    if(!root->isLeaf) {
        for(int q=0; q<4; ++q) {
            gather_cells_by_level(root->children[q], levels, depth+1);
        }
    }
}

// -------------------------------------------------------------------
// compute_upward_pass: Parallel bottom‐up pass to set mass & center‐of‐mass
// -------------------------------------------------------------------
void compute_upward_pass(FMMCell* root,
                         const double* x,
                         const double* y,
                         const double* m,
                         int N)
{
    // 1) Gather pointers by level (depth)
    std::vector<std::vector<FMMCell*>> levels;
    gather_cells_by_level(root, levels, 0);
    int maxDepth = (int)levels.size() - 1;

    // 2) For each level from bottom → top
    for(int lev = maxDepth; lev >= 0; --lev) {
        auto &cells = levels[lev];
        int nCells = (int)cells.size();
        #pragma omp parallel for schedule(static)
        for(int idx = 0; idx < nCells; ++idx) {
            FMMCell* cell = cells[idx];
            if(cell->isLeaf) {
                double mass_sum = 0.0;
                double mx_sum = 0.0;
                double my_sum = 0.0;
                for(int pi : cell->particles) {
                    double mi = m[pi];
                    mass_sum += mi;
                    mx_sum   += mi * x[pi];
                    my_sum   += mi * y[pi];
                }
                cell->mass = mass_sum;
                if(mass_sum > 0.0) {
                    cell->mx = mx_sum / mass_sum;
                    cell->my = my_sum / mass_sum;
                } else {
                    cell->mx = cell->cx;
                    cell->my = cell->cy;
                }
            } else {
                double mass_sum = 0.0;
                double mx_sum = 0.0;
                double my_sum = 0.0;
                for(int q=0; q<4; ++q) {
                    FMMCell* ch = cell->children[q];
                    mass_sum += ch->mass;
                    mx_sum   += ch->mass * ch->mx;
                    my_sum   += ch->mass * ch->my;
                }
                cell->mass = mass_sum;
                if(mass_sum > 0.0) {
                    cell->mx = mx_sum / mass_sum;
                    cell->my = my_sum / mass_sum;
                } else {
                    cell->mx = cell->cx;
                    cell->my = cell->cy;
                }
            }
        }
    }
}

// -------------------------------------------------------------------
// evaluate_target: Recursively traverse quadtree to accumulate force
// -------------------------------------------------------------------
void evaluate_target(const FMMCell* cell,
                     double tx,
                     double ty,
                     double theta,
                     double eps2,
                     double G,
                     const double* x,
                     const double* y,
                     const double* m,
                     double& axi,
                     double& ayi)
{
    double dx = cell->mx - tx;
    double dy = cell->my - ty;
    double dist2 = dx*dx + dy*dy + eps2;
    double r = sqrt(dist2);

    // If leaf OR (size / r) < theta, do monopole
    if(cell->isLeaf || (cell->size / r) < theta) {
        if(r > 0.0) {
            double inv_r3 = 1.0 / (dist2 * r);
            double mj = cell->mass * G * inv_r3;
            axi += mj * dx;
            ayi += mj * dy;
        }
    } else {
        // Otherwise, open cell
        if(cell->isLeaf) {
            // Direct‐sum over particles in leaf
            for(int pi : cell->particles) {
                double ddx = x[pi] - tx;
                double ddy = y[pi] - ty;
                double d2 = ddx*ddx + ddy*ddy + eps2;
                double rr = sqrt(d2);
                if(rr > 0.0) {
                    double inv_r3 = 1.0 / (d2 * rr);
                    double mj = m[pi] * G * inv_r3;
                    axi += mj * ddx;
                    ayi += mj * ddy;
                }
            }
        } else {
            // Recurse into children
            for(int q=0; q<4; ++q) {
                evaluate_target(cell->children[q],
                                tx, ty,
                                theta, eps2, G,
                                x, y, m,
                                axi, ayi);
            }
        }
    }
}

// -------------------------------------------------------------------
// fmm_force: Build tree → upward pass → parallel downward pass
// -------------------------------------------------------------------
void fmm_force(const py::array_t<double>& x_arr,
               const py::array_t<double>& y_arr,
               const py::array_t<double>& m_arr,
               int N,
               double domain_size,
               double theta,
               int maxLeaf,
               double eps,
               double G,
               py::array_t<double>& ax_arr,
               py::array_t<double>& ay_arr)
{
    // 1) Access raw pointers
    auto x_view = x_arr.unchecked<1>();
    auto y_view = y_arr.unchecked<1>();
    auto m_view = m_arr.unchecked<1>();
    auto ax_view = ax_arr.mutable_unchecked<1>();
    auto ay_view = ay_arr.mutable_unchecked<1>();

    // Copy data pointers into C arrays
    // (NumPy guarantees contiguous double64)
    const double* x = x_view.data(0);
    const double* y = y_view.data(0);
    const double* m = m_view.data(0);
    double* ax = ax_view.mutable_data(0);
    double* ay = ay_view.mutable_data(0);

    // 2) Build quadtree
    FMMCell* root = new FMMCell(0.0, 0.0, domain_size);
    build_tree(root, x, y, N, maxLeaf);

    // 3) Upward pass (compute mass & COM) in parallel
    compute_upward_pass(root, x, y, m, N);

    // 4) Zero out output arrays
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N; ++i) {
        ax[i] = 0.0;
        ay[i] = 0.0;
    }

    // 5) Downward pass: one OpenMP region, each thread handles a block of i
    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif
    int chunk = (N + nthreads - 1) / nthreads;

    #pragma omp parallel firstprivate(nthreads, chunk)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        int istart = std::min(N, tid * chunk);
        int iend   = std::min(N, (tid + 1) * chunk);

        for(int i = istart; i < iend; ++i) {
            double tx = x[i];
            double ty = y[i];
            double axi = 0.0;
            double ayi = 0.0;
            evaluate_target(root, tx, ty, theta, eps*eps, G,
                            x, y, m, axi, ayi);
            ax[i] = axi;
            ay[i] = ayi;
        }
    }

    // 6) Clean up
    delete root;
}

PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "2D Fast Multipole Method (FMM) kernel (OpenMP, NumPy arrays)";
    m.def("fmm_force",
          &fmm_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("N"),
          py::arg("domain_size") = 50.0,
          py::arg("theta")       = 0.5,
          py::arg("maxLeaf")     = 8,
          py::arg("eps")         = 0.01,
          py::arg("G")           = 1.0,
          py::arg("ax"),
          py::arg("ay"));
}

