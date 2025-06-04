// nbody_kernels_full.cpp – direct solver + breadth‑first Barnes‑Hut FMM
// ===========================================================================
// One translation unit -> two Pybind11 modules:
//   force_kernel_opt  (direct O(N²))
//   fmm_kernel_opt    (Barnes–Hut / FMM O(N log N))
// ---------------------------------------------------------------------------
// Build example:
//   g++ -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp -shared -fPIC \
//       `python -m pybind11 --includes` nbody_kernels_full.cpp \
//       -o nbody_kernels_full$(python3-config --extension-suffix)
// ===========================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <queue>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// ---------------------------------------------------------------------------
// 1. Direct kernel (cache‑blocked, lock‑free) – force_kernel_opt
// ---------------------------------------------------------------------------
constexpr int BI = 64;
constexpr int BJ = 64;

void direct_force_opt(const py::array_t<double>& x_arr,
                      const py::array_t<double>& y_arr,
                      const py::array_t<double>& m_arr,
                      double eps2,
                      py::array_t<double>& ax_arr,
                      py::array_t<double>& ay_arr)
{
    const auto  x  = x_arr.unchecked<1>();
    const auto  y  = y_arr.unchecked<1>();
    const auto  m  = m_arr.unchecked<1>();
          auto ax = ax_arr.mutable_unchecked<1>();
          auto ay = ay_arr.mutable_unchecked<1>();
    const int N = static_cast<int>(x.shape(0));

    #pragma omp parallel for schedule(static)
    for (int i=0;i<N;++i){ ax(i)=0.0; ay(i)=0.0; }

    const int nBI=(N+BI-1)/BI, nBJ=(N+ BJ-1)/BJ;

    #pragma omp parallel for schedule(static)
    for (int bi=0; bi<nBI; ++bi){
        const int i0=bi*BI, i1=std::min(i0+BI,N);
        for (int bj=0; bj<nBJ; ++bj){
            const int j0=bj*BJ, j1=std::min(j0+BJ,N);

            for (int i=i0;i<i1;++i){
                const double xi=x(i), yi=y(i);
                double fx=0.0, fy=0.0;
                #pragma omp simd reduction(+:fx,fy)
                for (int j=j0;j<j1;++j){ if(i==j) continue;
                    const double dx=xi-x(j), dy=yi-y(j);
                    const double r2=dx*dx+dy*dy+eps2;
                    const double invR=1.0/std::sqrt(r2);
                    const double f=m(j)*invR*invR*invR;
                    fx-=f*dx; fy-=f*dy; }
                ax(i)+=fx; ay(i)+=fy; }
        }
    }
}

PYBIND11_MODULE(force_kernel_opt, m){
    m.doc()="Cache‑blocked direct N‑body kernel (lock‑free)";
    m.def("direct_force", &direct_force_opt, py::arg("x"),py::arg("y"),py::arg("m"),
          py::arg("eps2"),py::arg("ax"),py::arg("ay"));
}

// ---------------------------------------------------------------------------
// 2. Barnes–Hut FMM kernel – fmm_kernel_opt
// ---------------------------------------------------------------------------
struct Body{ double x,y,m, ax,ay; };
struct Node{
    double cx=0, cy=0, size=0;
    double mass=0, cmx=0, cmy=0;
    bool   leaf=true;
    std::vector<int> ids;
    std::unique_ptr<Node> ch[4];
};

constexpr int    MAX_LEAF = 12;
constexpr double THETA2   = 0.36;  // (0.6)^2
constexpr double GCONST   = 1.0;

static void subdivide(Node* n,const std::vector<Body>& B){
    const double h=n->size*0.5;
    const double off[4][2]={{-h,-h},{h,-h},{-h,h},{h,h}};
    std::vector<int> bucket[4];
    for(int id:n->ids){ int q=(B[id].x>n->cx)+2*(B[id].y>n->cy); bucket[q].push_back(id);}    
    n->leaf=false; n->ids.clear();
    for(int q=0;q<4;++q) if(!bucket[q].empty()){
        n->ch[q]=std::make_unique<Node>();
        n->ch[q]->cx=n->cx+off[q][0]; n->ch[q]->cy=n->cy+off[q][1];
        n->ch[q]->size=h; n->ch[q]->ids.swap(bucket[q]); }
}

static void build_tree(Node* root,const std::vector<Body>& B){
#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif
    std::queue<Node*> Q; Q.push(root);
    while(!Q.empty()){
        const std::size_t lvl=Q.size();
        #pragma omp parallel for schedule(dynamic,4)
        for(std::size_t i=0;i<lvl;++i){
            Node* node; 
            { #pragma omp critical(queue_pop) node=Q.front(); Q.pop(); }
            if(node->ids.size()>MAX_LEAF) subdivide(node,B);
            node->mass=node->cmx=node->cmy=0.0;
            if(node->leaf){ for(int id:node->ids){ node->mass+=B[id].m; node->cmx+=B[id].m*B[id].x; node->cmy+=B[id].m*B[id].y; } }
            else{ for(auto& c:node->ch) if(c){ node->mass+=c->mass; node->cmx+=c->mass*c->cmx; node->cmy+=c->mass*c->cmy; } }
            if(node->mass){ node->cmx/=node->mass; node->cmy/=node->mass; }
            if(!node->leaf){ for(auto& c:node->ch) if(c){ #pragma omp critical(queue_push) Q.push(c.get()); } }
        }
    }
}

inline bool need_open(const Body& p,const Node* n){
    const double dx=p.x-n->cmx, dy=p.y-n->cmy;
    return (n->size*n->size)/(dx*dx+dy*dy) > THETA2;
}

static void traverse(const std::vector<Body>& B,Body& p,const Node* n,double eps2,double& ax,double& ay){
    if(!n||n->mass==0) return;
    if(n->leaf){ for(int id:n->ids) if(&p!=&B[id]){
            const double dx=p.x-B[id].x, dy=p.y-B[id].y;
            const double r2=dx*dx+dy*dy+eps2; const double invR=1.0/std::sqrt(r2);
            const double f=GCONST*B[id].m*invR*invR*invR; ax-=f*dx; ay-=f*dy; } }
    else if(!need_open(p,n)){
        const double dx=p.x-n->cmx, dy=p.y-n->cmy;
        const double r2=dx*dx+dy*dy+eps2; const double invR=1.0/std::sqrt(r2);
        const double f=GCONST*n->mass*invR*invR*invR; ax-=f*dx; ay-=f*dy; }
    else{ for(const auto& c:n->ch) if(c) traverse(B,p,c.get(),eps2,ax,ay); }
}

void fmm_force_opt(const py::array_t<double>& x_arr,const py::array_t<double>& y_arr,
                   const py::array_t<double>& m_arr,int N,double domain,double, int,
                   double eps,double,py::array_t<double>& ax_arr,py::array_t<double>& ay_arr){
    const double eps2=eps*eps;
    std::vector<Body> B(N);
    for(int i=0;i<N;++i) B[i]={x_arr.at(i),y_arr.at(i),m_arr.at(i),0,0};

    Node root; root.cx=0.0; root.cy=0.0; root.size=domain*0.5;
    root.ids.resize(N); for(int i=0;i<N;++i) root.ids[i]=i;
    build_tree(&root,B);

    #pragma omp parallel for schedule(dynamic,64)
    for(int i

