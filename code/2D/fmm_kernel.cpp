#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace py = pybind11;
using cplx = std::complex<double>;
constexpr int P = 4;                // multipole order

// ----------------—— binomial coefficient helper (C++17, constexpr) ——----------------
static inline double binom(int n,int k){
    if(k<0||k>n) return 0; if(k==0||k==n) return 1;
    double r=1; for(int i=1;i<=k;++i) r*= (n - (k - i))/(double)i; return r;
}

struct Cell{
    double cx,cy,size;                        // centre & half‑width
    std::vector<int> idx;                     // particle indices
    std::array<cplx,P+1> M{};                 // multipole moments
    std::array<cplx,P+1> L{};                 // local expansion
    Cell* ch[4]{nullptr}; Cell* par=nullptr;
    Cell(double x,double y,double s,Cell* p=nullptr):cx(x),cy(y),size(s),par(p){}
    ~Cell(){ for(auto c:ch) delete c; }
};

static inline cplx to_z(double x,double y){ return {x,y}; }

// ----------------—— build quadtree ——----------------
void subdiv(Cell* n,const std::vector<double>& x,const std::vector<double>& y,int maxLeaf){
    if((int)n->idx.size()<=maxLeaf) return;
    double h=n->size*0.5;
    for(int i=0;i<4;++i){ double dx=(i&1)?h:-h, dy=(i&2)?h:-h;
        n->ch[i]=new Cell(n->cx+dx,n->cy+dy,h,n); }
    for(int id:n->idx){ int q=(x[id]>n->cx)+2*(y[id]>n->cy); n->ch[q]->idx.push_back(id);}    
    n->idx.clear();
    for(auto c:n->ch) if(c) subdiv(c,x,y,maxLeaf);
}

// ----------------—— P2M / M2M ——----------------
void upward(Cell* n,const std::vector<double>& x,const std::vector<double>& y,const std::vector<double>& m){
    if(!n) return;
    if(!n->ch[0]){                                    // leaf: P2M
        for(int id:n->idx){ cplx dz=to_z(x[id]-n->cx,y[id]-n->cy); cplx mm=m[id];
            n->M[0]+=mm; for(int k=1;k<=P;++k) n->M[k]+=mm*std::pow(dz,k); }
    }else{
        for(auto c:n->ch) upward(c,x,y,m);            // recurse first
        for(auto c:n->ch) if(c){                     // M2M combine
            cplx dz=to_z(c->cx-n->cx,c->cy-n->cy);
            for(int k=0;k<=P;++k){ cplx ck=c->M[k];
                for(int j=0;j<=k;++j) n->M[j]+=ck*binom(k,j)*std::pow(dz,k-j); } }
    }
}

// ----------------—— M2L ——----------------
void m2l(Cell* trg,Cell* src,double theta){
    if(!trg||!src||trg==src) return;
    double dx=src->cx-trg->cx, dy=src->cy-trg->cy, d=std::hypot(dx,dy);
    if(!src->ch[0] || src->size/d<theta){             // well‑separated
        cplx z=to_z(dx,dy);
        for(int k=0;k<=P;++k){ cplx Mk=src->M[k];
            for(int j=0;j<=k;++j) trg->L[j]+=Mk*binom(k,j)*std::pow(z,k-j)*std::pow(z,-(k+1)); }
    }else for(auto c:src->ch) m2l(trg,c,theta);
}

void build_M2L(Cell* n,Cell* root,double theta){ if(!n) return; m2l(n,root,theta); for(auto c:n->ch) build_M2L(c,root,theta); }

// ----------------—— L2L ——----------------
void downward(Cell* n){
    if(!n) return; for(auto c:n->ch) if(c){
        cplx dz=to_z(n->cx-c->cx,n->cy-c->cy);
        for(int k=0;k<=P;++k)
            for(int j=k;j<=P;++j) c->L[k]+=n->L[j]*binom(j,k)*std::pow(dz,j-k);
        downward(c);
    }
}

// ----------------—— eval leaf (direct + local) ——----------------
void eval(Cell* n,const std::vector<double>& x,const std::vector<double>& y,const std::vector<double>& m,
          std::vector<double>& ax,std::vector<double>& ay,double G,double soft2){
    if(!n) return; if(n->ch[0]){ for(auto c:n->ch) eval(c,x,y,m,ax,ay,G,soft2); return; }
    for(int i:n->idx){ double fx=0,fy=0;
        for(int j:n->idx) if(i!=j){ double dx=x[j]-x[i],dy=y[j]-y[i]; double r2=dx*dx+dy*dy+soft2; double inv=1/std::pow(r2,1.5);
            fx+=G*m[j]*dx*inv; fy+=G*m[j]*dy*inv; }
        cplx z=to_z(x[i]-n->cx,y[i]-n->cy), dphi=0;
        for(int k=1;k<=P;++k) dphi+=double(k)*n->L[k]*std::pow(z,k-1);
        fx+=G*(-dphi.real()); fy+=G*(-dphi.imag()); ax[i]+=fx; ay[i]+=fy; }
}

// ----------------—— Python wrapper ——----------------
py::tuple fmm_omp(py::array_t<double> x,py::array_t<double> y,py::array_t<double> m,
                  double dom,double theta=0.5,double G=1.0,double soft=0.05,int maxLeaf=16){
    size_t N=x.size();
    std::vector<double> vx(x.data(), x.data()+N);
    std::vector<double> vy(y.data(), y.data()+N);
    std::vector<double> vm(m.data(), m.data()+N);
    std::vector<double> ax(N,0.0), ay(N,0.0);

    Cell* root = new Cell(0,0,dom/2);
    root->idx.resize(N);
    std::iota(root->idx.begin(), root->idx.end(), 0);

    subdiv(root, vx, vy, maxLeaf);
    upward(root, vx, vy, vm);
    build_M2L(root, root, theta);
    downward(root);
    eval(root, vx, vy, vm, ax, ay, G, soft*soft);
    delete root;

    // ---- copy into NumPy memory to avoid dangling pointers ----
    py::array_t<double> ax_out(N), ay_out(N);
    auto pax=ax_out.mutable_unchecked<1>();
    auto pay=ay_out.mutable_unchecked<1>();
    for(size_t i=0;i<N;++i){ pax(i)=ax[i]; pay(i)=ay[i]; }

    return py::make_tuple(ax_out, ay_out);
}

PYBIND11_MODULE(fmm_kernel,m){
    m.doc() = "2-D FMM P=4 (binom‑safe, owns memory)";
    m.def("fmm_omp", &fmm_omp,
          py::arg("x"), py::arg("y"), py::arg("m"), py::arg("domain"),
          py::arg("theta")=0.5, py::arg("G")=1.0, py::arg("soft")=0.05, py::arg("maxLeaf")=16);
}
