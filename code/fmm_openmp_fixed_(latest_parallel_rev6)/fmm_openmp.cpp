// fmm_openmp.cpp – θ-aware Barnes–Hut / FMM with OpenMP
// =====================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <queue>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// ---------- particle & node --------------------------------------------------
struct Body { double x,y,m, ax,ay; };
struct Node {
    double cx=0, cy=0, size=0;
    double mass=0, cmx=0, cmy=0;
    bool   leaf=true;
    std::vector<int> ids;
    std::unique_ptr<Node> ch[4];
};

constexpr int MAX_LEAF = 16;

// ---------- helpers ----------------------------------------------------------
static void subdivide(Node* n,const std::vector<Body>& B)
{
    const double h=n->size*0.5;
    const double off[4][2]={{-h,-h},{h,-h},{-h,h},{h,h}};
    std::vector<int> bucket[4];
    for(int id:n->ids){
        int q=(B[id].x>n->cx)+2*(B[id].y>n->cy);
        bucket[q].push_back(id);
    }
    n->leaf=false; n->ids.clear();
    for(int q=0;q<4;++q) if(!bucket[q].empty()){
        n->ch[q]=std::make_unique<Node>();
        n->ch[q]->cx=n->cx+off[q][0];
        n->ch[q]->cy=n->cy+off[q][1];
        n->ch[q]->size=h;
        n->ch[q]->ids.swap(bucket[q]);
    }
}

static void build_tree(Node* root,const std::vector<Body>& B)
{
#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif
    std::vector<Node*> current{root}, next;
    while(!current.empty()){
        #pragma omp parallel
        {
            std::vector<Node*> next_local;
            #pragma omp for  schedule(static)
            for(std::size_t i=0;i<current.size();++i){
                Node* n=current[i];
                if(n->ids.size()>MAX_LEAF) subdivide(n,B);
                n->mass=n->cmx=n->cmy=0.0;
                if(n->leaf){
                    for(int id:n->ids){
                        n->mass+=B[id].m;
                        n->cmx +=B[id].m*B[id].x;
                        n->cmy +=B[id].m*B[id].y;
                    }
                } else {
                    for(auto& c:n->ch) if(c){
                        n->mass+=c->mass;
                        n->cmx +=c->mass*c->cmx;
                        n->cmy +=c->mass*c->cmy;
                        next_local.push_back(c.get());
                    }
                }
                if(n->mass){
                    n->cmx/=n->mass; n->cmy/=n->mass;
                }
            }
            #pragma omp critical
            next.insert(next.end(), next_local.begin(), next_local.end());
        }
        current.swap(next); next.clear();
    }
}

inline bool far(const Body& p,const Node* n,double theta2)
{
    const double dx=p.x-n->cmx, dy=p.y-n->cmy;
    return (n->size*n->size)/(dx*dx+dy*dy) < theta2;
}

static void traverse(const std::vector<Body>& B,const Node* n,
                     const Body& p,double eps2,double theta2,
                     double& fx,double& fy)
{
    if(!n||n->mass==0) return;
    if(n->leaf||far(p,n,theta2)){
        double dx=n->cmx-p.x, dy=n->cmy-p.y;
        double r2=dx*dx+dy*dy+eps2;
        double invR=1.0/std::sqrt(r2);
        double f=n->mass*invR*invR*invR;
        fx+=f*dx; fy+=f*dy;
    }else{
        for(const auto& c:n->ch) if(c)
            traverse(B,c.get(),p,eps2,theta2,fx,fy);
    }
}

// ---------- kernels ----------------------------------------------------------
static void fmm_core(const py::array_t<double>& x_arr,
                     const py::array_t<double>& y_arr,
                     const py::array_t<double>& m_arr,
                     double eps2,double domain,double theta,
                     py::array_t<double>& ax_arr,
                     py::array_t<double>& ay_arr)
{
    const int N=x_arr.shape(0);
    std::vector<Body> B(N);
    for(int i=0;i<N;++i)
        B[i]={x_arr.at(i),y_arr.at(i),m_arr.at(i),0,0};

    Node root; root.cx=root.cy=0.0; root.size=domain*0.5;
    root.ids.resize(N); for(int i=0;i<N;++i) root.ids[i]=i;
    build_tree(&root,B);

    const double theta2=theta*theta;
    #pragma omp parallel for schedule(dynamic,64)
    for(int i=0;i<N;++i){
        double fx=0,fy=0;
        traverse(B,&root,B[i],eps2,theta2,fx,fy);
        B[i].ax=fx; B[i].ay=fy;
    }

    auto ax=ax_arr.mutable_unchecked<1>();
    auto ay=ay_arr.mutable_unchecked<1>();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){ ax(i)=B[i].ax; ay(i)=B[i].ay; }
}

// wrapper with explicit θ
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& m,
                     double eps2,double domain,double theta,
                     py::array_t<double>& ax,
                     py::array_t<double>& ay)
{ fmm_core(x,y,m,eps2,domain,theta,ax,ay); }

// wrapper retaining old 7-arg signature (θ = 0.6)
void fmm_force_old(const py::array_t<double>& x,
                   const py::array_t<double>& y,
                   const py::array_t<double>& m,
                   double eps2,double domain,
                   py::array_t<double>& ax,
                   py::array_t<double>& ay)
{ fmm_core(x,y,m,eps2,domain,0.6,ax,ay); }

// ---------- naive direct (unchanged) ----------------------------------------
void direct_symm(const py::array_t<double>& x_arr,
                 const py::array_t<double>& y_arr,
                 const py::array_t<double>& m_arr,
                 double eps2,
                 py::array_t<double>& ax_arr,
                 py::array_t<double>& ay_arr)
{
    const auto  x=x_arr.unchecked<1>();
    const auto  y=y_arr.unchecked<1>();
    const auto  m=m_arr.unchecked<1>();
    auto ax=ax_arr.mutable_unchecked<1>();
    auto ay=ay_arr.mutable_unchecked<1>();
    const int N=x.shape(0);

    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){ ax(i)=0; ay(i)=0; }

    #pragma omp parallel for schedule(dynamic,16)
    for(int i=0;i<N;++i){
        for(int j=i+1;j<N;++j){
            double dx=x(j)-x(i), dy=y(j)-y(i);
            double r2=dx*dx+dy*dy+eps2;
            double invR=1.0/std::sqrt(r2);
            double invR3=invR*invR*invR;
            double f=m(j)*invR3;

            double fx=f*dx, fy=f*dy;
            ax(i)+=fx; ay(i)+=fy;
            #pragma omp atomic
            ax(j)-=fx;
            #pragma omp atomic
            ay(j)-=fy;
        }
    }
}
// -------------- pybind glue --------------------------------------------------
PYBIND11_MODULE(fmm_openmp, m)
{
    m.doc() = "Direct + Barnes–Hut FMM kernels (OpenMP, run-time θ)";

    m.def("direct_force",
          &direct_symm,
          "Symmetry-aware O(N²) reference kernel (OpenMP)");

    // legacy 7-argument signature (θ defaults to 0.6)
    m.def("fmm_force",
          &fmm_force_old,
          "Barnes–Hut FMM kernel (θ = 0.6 – legacy API)");

    // preferred 8-argument version: explicit θ
    m.def("fmm_force_theta",
          &fmm_force_theta,
          "Barnes–Hut FMM kernel with explicit opening angle θ");
}


