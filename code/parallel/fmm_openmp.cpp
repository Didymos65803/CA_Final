// fmm_openmp.cpp  ---------------------------------------------------------
// Barnes–Hut / FMM, 2-D
//   * Phase-1: parallel breadth-first splitting (no masses yet)
//   * Phase-2: post-order mass / centroid computation (OpenMP tasks)
//   * Phase-3: parallel traversal (OpenMP for)
//
// Build:   python3 setup_openmp.py build_ext --inplace
// Rename:  mv build/lib*/fmm_openmp*.so  fmm_openmp.so
// -------------------------------------------------------------------------

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

/* ---------------------------------------------------------------- data */
struct Body { double x, y, m; };

struct Node {
    double cx{}, cy{}, h{};
    double mass{}, cmx{}, cmy{};
    bool   leaf{true};
    std::vector<int> ids;
    std::array<std::unique_ptr<Node>,4> ch;
};

static constexpr int    MAX_LEAF  = 64;
static constexpr double H_MIN     = 1e-6;

/* ---------------------------------------------------------------- helpers */
inline int quad(const Body& b, const Node* n)
{ return (b.x > n->cx) + 2*(b.y > n->cy); }

/* ------------------------- phase 1: split ----------------------------- */
void split_level(std::vector<Node*>& level,
                 std::vector<Node*>& next,
                 const std::vector<Body>& B)
{
#pragma omp parallel
    {
        std::vector<Node*> local_next;
#pragma omp for schedule(static)
        for (std::size_t i=0;i<level.size();++i) {
            Node* n = level[i];
            if ((int)n->ids.size() > MAX_LEAF && n->h > H_MIN) {
                n->leaf = false;
                double h2 = 0.5 * n->h;
                for (int q=0;q<4;++q) {
                    n->ch[q] = std::make_unique<Node>();
                    n->ch[q]->h = h2;
                    n->ch[q]->leaf = true;
                    n->ch[q]->cx = n->cx + (q&1 ? 0.5:-0.5)*h2;
                    n->ch[q]->cy = n->cy + (q&2 ? 0.5:-0.5)*h2;
                }
                for (int id : n->ids)
                    n->ch[ quad(B[id],n) ]->ids.push_back(id);
                n->ids.clear();
                for (auto& c : n->ch) local_next.push_back(c.get());
            }
        } /* omp for */
#pragma omp critical
        next.insert(next.end(), local_next.begin(), local_next.end());
    } /* parallel */
}

/* ---------------- phase 2: compute masses (post-order) ---------------- */
void mass_rec(Node* n)
{
    if (!n) return;
    if (!n->leaf){
#pragma omp task shared(n)
        mass_rec(n->ch[0].get());
#pragma omp task shared(n)
        mass_rec(n->ch[1].get());
#pragma omp task shared(n)
        mass_rec(n->ch[2].get());
#pragma omp task shared(n)
        mass_rec(n->ch[3].get());
#pragma omp taskwait
    }

    n->mass = n->cmx = n->cmy = 0.0;
    if (n->leaf){
        for (int id : n->ids){
            n->mass += 1.0;                // mass = 1 for all bodies
            n->cmx  += n->ch.empty() ? 0 : 0; // will be filled in driver
        }
    } else {
        for (auto& c : n->ch){
            n->mass += c->mass;
            n->cmx  += c->mass * c->cmx;
            n->cmy  += c->mass * c->cmy;
        }
    }
    if (n->mass){
        n->cmx /= n->mass;
        n->cmy /= n->mass;
    }
}

/* ---------------- traversal ------------------------------------------ */
inline bool far(const Body& p,const Node* n,double th2){
    double dx=p.x-n->cmx, dy=p.y-n->cmy;
    return (n->h*n->h)/(dx*dx+dy*dy) < th2;
}
void walk(const std::vector<Body>& B,const Node* n,const Body& p,
          double eps2,double th2,double& fx,double& fy)
{
    if(!n||n->mass==0) return;
    if(n->leaf||far(p,n,th2)){
        double dx=n->cmx-p.x, dy=n->cmy-p.y;
        double r2=dx*dx+dy*dy+eps2;
        double invR=1/std::sqrt(r2), invR3=invR*invR*invR;
        double f=n->mass*invR3; fx+=f*dx; fy+=f*dy;
    }else{
        for(auto& c:n->ch) walk(B,c.get(),p,eps2,th2,fx,fy);
    }
}

/* ---------------- main kernel ---------------------------------------- */
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& /*m*/,
                     double eps2,double domain,double theta,
                     py::array_t<double>& ax, py::array_t<double>& ay)
{
    int N=x.shape(0);
    std::vector<Body> B(N);
    for(int i=0;i<N;++i) B[i]={x.at(i),y.at(i),1.0};

    Node root; root.cx=root.cy=0.0; root.h=domain;
    root.ids.resize(N); std::iota(root.ids.begin(),root.ids.end(),0);

    /* -------- phase 1: split */
    std::vector<Node*> level{&root}, next;
    while (!level.empty()){
        next.clear();
        split_level(level,next,B);
        level.swap(next);
    }

    /* -------- phase 2: multipoles */
#pragma omp parallel
#pragma omp single
    mass_rec(&root);

    /* -------- phase 3: traversal */
    auto axw=ax.mutable_unchecked<1>(), ayw=ay.mutable_unchecked<1>();
    double th2=theta*theta;
#pragma omp parallel for schedule(static,1024)
    for(int i=0;i<N;++i){
        double fx=0.0, fy=0.0;
        walk(B,&root,B[i],eps2,th2,fx,fy);
        axw(i)=fx; ayw(i)=fy;
    }
}

/* ---------------- module --------------------------------------------- */
PYBIND11_MODULE(fmm_openmp, m)
{
    m.doc()="Barnes–Hut FMM (parallel build & traverse)";
    m.def("fmm_force_theta",&fmm_force_theta,
          "x,y,mass=1,eps2,domain,theta,ax,ay");
}

