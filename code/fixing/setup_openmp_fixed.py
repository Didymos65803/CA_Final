from setuptools import setup, Extension
import pybind11
import os

# 檢查原始碼檔案是否存在
if not os.path.exists("fmm_omp.cpp"):
    print("Error: fmm_omp.cpp not found in current directory")
    print("Available files:", [f for f in os.listdir('.') if f.endswith('.cpp')])
    print("Creating fmm_omp.cpp from template...")
    
    # 創建簡化版的 fmm_omp.cpp
    cpp_content = '''// fmm_omp_simple.cpp
// 簡化版 O(N log N) Barnes-Hut，確保能編譯成功

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

struct Body {
    double x, y, m;
    int idx;
};

struct Node {
    double xmin, xmax, ymin, ymax;
    double mass, cx, cy;
    bool is_leaf;
    std::vector<int> body_ids;
    std::unique_ptr<Node> children[4];

    Node(double x1, double x2, double y1, double y2)
        : xmin(x1), xmax(x2), ymin(y1), ymax(y2),
          mass(0.0), cx(0.0), cy(0.0), is_leaf(true) {}

    double width() const { return xmax - xmin; }
};

const int MAX_BODIES = 32;

void build_tree(Node* node, const std::vector<Body>& bodies, const std::vector<int>& ids) {
    if ((int)ids.size() <= MAX_BODIES) {
        node->is_leaf = true;
        node->body_ids = ids;
        
        double total_mass = 0.0, cx_sum = 0.0, cy_sum = 0.0;
        for (int id : ids) {
            total_mass += bodies[id].m;
            cx_sum += bodies[id].m * bodies[id].x;
            cy_sum += bodies[id].m * bodies[id].y;
        }
        
        node->mass = total_mass;
        if (total_mass > 0.0) {
            node->cx = cx_sum / total_mass;
            node->cy = cy_sum / total_mass;
        }
        return;
    }

    node->is_leaf = false;
    double xmid = 0.5 * (node->xmin + node->xmax);
    double ymid = 0.5 * (node->ymin + node->ymax);

    std::vector<std::vector<int>> child_ids(4);
    for (int id : ids) {
        int quad = 0;
        if (bodies[id].x > xmid) quad += 1;
        if (bodies[id].y > ymid) quad += 2;
        child_ids[quad].push_back(id);
    }

    node->children[0] = std::make_unique<Node>(node->xmin, xmid, ymid, node->ymax);
    node->children[1] = std::make_unique<Node>(xmid, node->xmax, ymid, node->ymax);
    node->children[2] = std::make_unique<Node>(node->xmin, xmid, node->ymin, ymid);
    node->children[3] = std::make_unique<Node>(xmid, node->xmax, node->ymin, ymid);

    for (int i = 0; i < 4; ++i) {
        if (!child_ids[i].empty()) {
            build_tree(node->children[i].get(), bodies, child_ids[i]);
        }
    }

    // 計算內部節點的質量和質心
    double total_mass = 0.0, cx_sum = 0.0, cy_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (node->children[i]) {
            Node* child = node->children[i].get();
            total_mass += child->mass;
            cx_sum += child->mass * child->cx;
            cy_sum += child->mass * child->cy;
        }
    }
    
    node->mass = total_mass;
    if (total_mass > 0.0) {
        node->cx = cx_sum / total_mass;
        node->cy = cy_sum / total_mass;
    }
}

void compute_force(const std::vector<Body>& bodies, const Node* node, int bi,
                  double eps2, double theta2, double& fx, double& fy) {
    if (!node || node->mass == 0.0) return;

    double dx = node->cx - bodies[bi].x;
    double dy = node->cy - bodies[bi].y;
    double r2 = dx*dx + dy*dy + eps2;
    double s = node->width();

    if (node->is_leaf || (s*s / r2 < theta2)) {
        if (node->is_leaf) {
            // 直接計算葉節點內的粒子
            for (int id : node->body_ids) {
                if (id == bi) continue;
                double dx2 = bodies[id].x - bodies[bi].x;
                double dy2 = bodies[id].y - bodies[bi].y;
                double r2_direct = dx2*dx2 + dy2*dy2 + eps2;
                double inv_r3 = 1.0 / (r2_direct * std::sqrt(r2_direct));
                double f = bodies[id].m * inv_r3;
                fx += f * dx2;
                fy += f * dy2;
            }
        } else {
            // 使用質心近似
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            double f = node->mass * inv_r3;
            fx += f * dx;
            fy += f * dy;
        }
    } else {
        // 遞歸到子節點
        for (int i = 0; i < 4; ++i) {
            if (node->children[i]) {
                compute_force(bodies, node->children[i].get(), bi, eps2, theta2, fx, fy);
            }
        }
    }
}

void fmm_force_theta(py::array_t<double> x_arr,
                     py::array_t<double> y_arr,
                     py::array_t<double> m_arr,
                     double eps2,
                     py::array_t<double> domain_arr,
                     double theta,
                     py::array_t<double> ax_arr,
                     py::array_t<double> ay_arr)
{
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto domain = domain_arr.unchecked<1>();
    auto axw = ax_arr.mutable_unchecked<1>();
    auto ayw = ay_arr.mutable_unchecked<1>();
    const int N = (int)x_arr.shape(0);

    // 準備資料
    std::vector<Body> bodies(N);
    std::vector<int> all_ids(N);
    for (int i = 0; i < N; ++i) {
        bodies[i] = {x(i), y(i), m(i), i};
        all_ids[i] = i;
    }

    // 建立樹
    Node root(domain(0), domain(1), domain(2), domain(3));
    build_tree(&root, bodies, all_ids);

    // 計算力
    double theta2 = theta * theta;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        compute_force(bodies, &root, i, eps2, theta2, fx, fy);
        axw(i) = fx;
        ayw(i) = fy;
    }
}

PYBIND11_MODULE(fmm_omp, m) {
    m.doc() = "Simplified O(N log N) Barnes-Hut with OpenMP";
    m.def("fmm_force_theta", &fmm_force_theta,
          "fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)");
}'''
    
    with open("fmm_omp.cpp", "w") as f:
        f.write(cpp_content)
    print("Created fmm_omp.cpp successfully!")

# 獲取 pybind11 的 include 路徑
include_dirs = [
    pybind11.get_include(),
]

fmm_module = Extension(
    name="fmm_omp",
    sources=["fmm_omp.cpp"],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=[
        "-std=c++14",   # 使用 C++14 以支援 make_unique
        "-fopenmp",     # 啟用 OpenMP
        "-O3",          # 優化
        "-march=native" # 本機架構優化
    ],
    extra_link_args=[
        "-fopenmp"
    ],
)

setup(
    name="fmm_omp",
    version="0.1",
    author="(Your Name)",
    description="2D Barnes–Hut FMM (monopole only) with fully parallel OpenMP (O(N log N)).",
    ext_modules=[fmm_module],
    zip_safe=False,
)

print("Compilation completed successfully!")
