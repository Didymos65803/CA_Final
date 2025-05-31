// bh_omp.cpp – 2‑D Barnes–Hut force kernel (OpenMP) exposed to Python via pybind11
// Build:  g++ -O3 -std=c++17 -fopenmp -shared -fPIC \
//            $(python -m pybind11 --includes) bh_omp.cpp \
//            -o force_kernel$(python3-config --extension-suffix)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

struct Body { double x, y, m; };

struct QuadNode {
    double cx, cy, size;        // square centre & half‑width  (size = half‑length)
    double mass = 0.0, comx = 0.0, comy = 0.0;
    bool   leaf = true;
    Body  *single = nullptr;
    QuadNode* child[4] {nullptr,nullptr,nullptr,nullptr};
    QuadNode(double x0, double y0, double s) : cx(x0), cy(y0), size(s) {}
    ~QuadNode(){ for(auto c: child) delete c; }
};

static inline void insert(QuadNode* n, Body* b){
    if(n->leaf && n->single==nullptr){
        n->single = b;
        n->mass = b->m; n->comx = b->x; n->comy = b->y;
        return;
    }
    if(n->leaf){
        n->leaf = false;
        Body* old = n->single; n->single=nullptr;
        double h = n->size*0.5, q = h*0.5;
        for(int i=0;i<4;i++){
            double dx = (i&1)? q : -q;
            double dy = (i<2)? -q:  q;
            n->child[i] = new QuadNode(n->cx+dx, n->cy+dy, h);
        }
        insert(n, old);
    }
    int idx = (b->x > n->cx) + 2*(b->y > n->cy);
    insert(n->child[idx], b);
    double M = n->mass + b->m;
    n->comx = (n->comx*n->mass + b->x*b->m)/M;
    n->comy = (n->comy*n->mass + b->y*b->m)/M;
    n->mass = M;
}

static inline void force(const QuadNode* n, const Body& b,double theta,double G,double soft2,double &ax,double &ay){
    if(!n) return;
    double dx = n->comx - b.x, dy = n->comy - b.y;
    double r2 = dx*dx + dy*dy + soft2;
    if(n->leaf && n->single==&b) return;             // self
    double r  = std::sqrt(r2);
    if(n->leaf || (n->size/r < theta)){
        double f = G * n->mass / r2;
        ax += f*dx/r; ay += f*dy/r;
    }else{
        for(auto c:n->child) force(c,b,theta,G,soft2,ax,ay);
    }
}

// -----------------  pybind11 wrapper  -----------------
py::tuple bh_omp(py::array_t<double,py::array::c_style|py::array::forcecast> x,
                 py::array_t<double,py::array::c_style|py::array::forcecast> y,
                 py::array_t<double,py::array::c_style|py::array::forcecast> m,
                 double domain, double theta=0.5, double G=1.0, double soft=0.01){
    const size_t N = x.size();
    if(y.size()!=N || m.size()!=N) throw std::runtime_error("x,y,m must have same length");
    auto px = x.data(); auto py_ = y.data(); auto pm = m.data();

    std::vector<Body> bodies(N);
    for(size_t i=0;i<N;++i) bodies[i] = {px[i],py_[i],pm[i]};

    QuadNode* root = new QuadNode(0,0,domain*0.5);   // centred at 0,0
    for(auto &b: bodies) insert(root,&b);

    auto ax_out = py::array_t<double>(N); auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_data(); auto pay = ay_out.mutable_data();

    const double soft2 = soft*soft;
    #pragma omp parallel for schedule(dynamic)
    for(long long i=0;i<static_cast<long long>(N); ++i){
        double ax=0, ay=0; force(root,bodies[i],theta,G,soft2,ax,ay); pax[i]=ax; pay[i]=ay; }

    delete root;
    return py::make_tuple(ax_out,ay_out);
}

PYBIND11_MODULE(force_kernel, m){
    m.doc() = "OpenMP Barnes–Hut 2‑D force kernel";
    m.def("bh_omp", &bh_omp, py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("theta")=0.5, py::arg("G")=1.0, py::arg("soft")=0.01,
          "Compute accelerations with Barnes–Hut + OpenMP");
}
