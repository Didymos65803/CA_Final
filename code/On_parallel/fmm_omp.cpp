// ---------------------------------------------------------------------------
//  fmm_omp.cpp
//
//  2D Barnes–Hut FMM (monopole-only) with fully parallel tree‐build
//  and leaf‐traversal using OpenMP tasks + parallel for.
//
//  这份代码已调整：
//    1) TASK_THRESHOLD = 50，使得更多子象限 spawn task，增强平行化细粒度。
//    2) 在 build_tree_rec 中，加了一个简单的计数器 spawn_counter，
//       用于演示如何检测实际 spawn 了多少 task。
//    3) 建议在 N 超过 200k 时做测试，以观察真实的并行加速效果。
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <atomic>  // 用于原子计数

namespace py = pybind11;

// ---------------------------------------------------------------------------
//  数据结构：Body 与 Node
// ---------------------------------------------------------------------------
struct Body {
    double x, y;   // 位置
    double m;      // 质量
    int    idx;    // 原始索引 (仅 Debug 用)
};

struct Node {
    double xmin, xmax, ymin, ymax; // 节点覆盖区域 [xmin,xmax]×[ymin,ymax]
    double mass;                   // 此节点内所有粒子的总质量
    double cx, cy;                 // 质心 (center-of-mass)
    bool is_leaf;                  // 是否为叶节点
    std::vector<int> body_ids;     // 若为叶节点，则记录此节点所有粒子索引
    std::unique_ptr<Node> nw, ne, sw, se; // 四个子节点 (NW, NE, SW, SE)

    Node(double _xmin, double _xmax, double _ymin, double _ymax)
      : xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax),
        mass(0.0), cx(0.0), cy(0.0), is_leaf(true)
    {}

    inline double width() const { return xmax - xmin; }
};

// ---------------------------------------------------------------------------
//  可调参数
// ---------------------------------------------------------------------------
// 叶节点最大支持粒子个数
static const int MAX_BODIES_PER_LEAF = 16;
// 子象限内若粒子数 > TASK_THRESHOLD，就 spawn 一个新的 OpenMP task
static const int TASK_THRESHOLD       = 50;

// 原子计数器：统计到底 spawn 了多少个 task (仅用于调试，可以注释掉)
static std::atomic<int> spawn_counter(0);

// ---------------------------------------------------------------------------
//  build_tree_rec()
//    递归构建四叉树，并计算每个节点的质量与质心 (monopole)。
//    如果某子象限内粒子数 > TASK_THRESHOLD，则 spawn OpenMP task。
//
//    总成本 O(N)，平行后每线程约做 O(N/P)。
// ---------------------------------------------------------------------------
void build_tree_rec(Node*                    node,
                    const std::vector<Body>&  B,
                    const std::vector<int>&   ids)
{
    // 1) 如果粒子数 ≤ MAX_BODIES_PER_LEAF，则当成叶节点
    if ((int)ids.size() <= MAX_BODIES_PER_LEAF) {
        node->is_leaf  = true;
        node->body_ids = ids;

        double msum = 0.0, cx = 0.0, cy = 0.0;
        for (int i : ids) {
            msum += B[i].m;
            cx   += B[i].m * B[i].x;
            cy   += B[i].m * B[i].y;
        }
        node->mass = msum;
        if (msum > 0.0) {
            node->cx = cx / msum;
            node->cy = cy / msum;
        } else {
            // 如果没有质量（极端情况），质心就放在节点中央
            node->cx = 0.5*(node->xmin + node->xmax);
            node->cy = 0.5*(node->ymin + node->ymax);
        }
        return;
    }

    // 2) 否则，将粒子分到 4 个子象限
    node->is_leaf = false;
    double xmid = 0.5*(node->xmin + node->xmax);
    double ymid = 0.5*(node->ymin + node->ymax);

    std::vector<int> ids_nw, ids_ne, ids_sw, ids_se;
    ids_nw.reserve(ids.size()/4);
    ids_ne.reserve(ids.size()/4);
    ids_sw.reserve(ids.size()/4);
    ids_se.reserve(ids.size()/4);

    for (int i : ids) {
        double xx = B[i].x;
        double yy = B[i].y;
        if (xx <= xmid && yy >  ymid) ids_nw.push_back(i);
        if (xx >  xmid && yy >  ymid) ids_ne.push_back(i);
        if (xx <= xmid && yy <= ymid) ids_sw.push_back(i);
        if (xx >  xmid && yy <= ymid) ids_se.push_back(i);
    }

    // 3) 手动 new + reset 四个子节点
    node->nw.reset(new Node(node->xmin, xmid,    ymid, node->ymax));
    node->ne.reset(new Node(xmid,    node->xmax, ymid, node->ymax));
    node->sw.reset(new Node(node->xmin, xmid,    node->ymin, ymid));
    node->se.reset(new Node(xmid,    node->xmax, node->ymin, ymid));

    // 4) 如果某个子象限的粒子数 > TASK_THRESHOLD，就 spawn task
    if ((int)ids_nw.size() > TASK_THRESHOLD) {
        spawn_counter.fetch_add(1, std::memory_order_relaxed);
        #pragma omp task
        build_tree_rec(node->nw.get(), B, ids_nw);
    } else {
        build_tree_rec(node->nw.get(), B, ids_nw);
    }

    if ((int)ids_ne.size() > TASK_THRESHOLD) {
        spawn_counter.fetch_add(1, std::memory_order_relaxed);
        #pragma omp task
        build_tree_rec(node->ne.get(), B, ids_ne);
    } else {
        build_tree_rec(node->ne.get(), B, ids_ne);
    }

    if ((int)ids_sw.size() > TASK_THRESHOLD) {
        spawn_counter.fetch_add(1, std::memory_order_relaxed);
        #pragma omp task
        build_tree_rec(node->sw.get(), B, ids_sw);
    } else {
        build_tree_rec(node->sw.get(), B, ids_sw);
    }

    if ((int)ids_se.size() > TASK_THRESHOLD) {
        spawn_counter.fetch_add(1, std::memory_order_relaxed);
        #pragma omp task
        build_tree_rec(node->se.get(), B, ids_se);
    } else {
        build_tree_rec(node->se.get(), B, ids_se);
    }

    #pragma omp taskwait

    // 5) 四个子树都构建完成后，累加各自质点信息
    double msum     = 0.0;
    double cx_total = 0.0;
    double cy_total = 0.0;

    auto accumulate_child = [&](const std::unique_ptr<Node>& cptr) {
        if (cptr) {
            msum       += cptr->mass;
            cx_total   += cptr->mass * cptr->cx;
            cy_total   += cptr->mass * cptr->cy;
        }
    };
    accumulate_child(node->nw);
    accumulate_child(node->ne);
    accumulate_child(node->sw);
    accumulate_child(node->se);

    node->mass = msum;
    if (msum > 0.0) {
        node->cx = cx_total / msum;
        node->cy = cy_total / msum;
    } else {
        node->cx = 0.5*(node->xmin + node->xmax);
        node->cy = 0.5*(node->ymin + node->ymax);
    }
}

// ---------------------------------------------------------------------------
//  traverse()
//    给定目标粒子索引 bi，从 node 开始递归遍历并累加力到 (fx, fy)。
//    使用开角 θ (th2 = θ²) + 软化 eps2。
//    如果 node->is_leaf → 直接做 pairwise；
//    否则如果 s² < th2 * d2 → treat as single monopole；否则递归到子节点。
// ---------------------------------------------------------------------------
void traverse(const std::vector<Body>& B,
              const Node*              node,
              int                      bi,
              double                   eps2,
              double                   th2,
              double&                  fx,
              double&                  fy)
{
    double dx = node->cx - B[bi].x;
    double dy = node->cy - B[bi].y;
    double d2 = dx*dx + dy*dy + eps2;
    double s  = node->width();

    if (node->is_leaf) {
        // 叶节点：直接做 pairwise (跳过自己)
        for (int j : node->body_ids) {
            if (j == bi) continue;
            double dx2 = B[j].x - B[bi].x;
            double dy2 = B[j].y - B[bi].y;
            double r2  = dx2*dx2 + dy2*dy2 + eps2;
            double inv_r3 = 1.0 / (r2 * sqrt(r2));
            double f = B[j].m * inv_r3;
            fx += f * dx2;
            fy += f * dy2;
        }
    }
    else if (s*s < th2 * d2) {
        // 开角条件成立：当作一个质点 (monopole)
        double inv_r3 = 1.0 / (d2 * sqrt(d2));
        double f = node->mass * inv_r3;
        fx += f * dx;
        fy += f * dy;
    }
    else {
        // 递归到各子象限
        if (node->nw) traverse(B, node->nw.get(), bi, eps2, th2, fx, fy);
        if (node->ne) traverse(B, node->ne.get(), bi, eps2, th2, fx, fy);
        if (node->sw) traverse(B, node->sw.get(), bi, eps2, th2, fx, fy);
        if (node->se) traverse(B, node->se.get(), bi, eps2, th2, fx, fy);
    }
}

// ---------------------------------------------------------------------------
//  fmm_force_theta()
//    Python‐exposed 接口：
//      fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)
//    所有重量级计算都在此 C++/OpenMP 层执行，Python 端只负责传参与测试。
// ---------------------------------------------------------------------------
void fmm_force_theta(py::array_t<double> x_arr,
                     py::array_t<double> y_arr,
                     py::array_t<double> m_arr,
                     double eps2,
                     py::array_t<double> domain_arr,
                     double theta,
                     py::array_t<double> ax_arr,
                     py::array_t<double> ay_arr)
{
    auto x      = x_arr.unchecked<1>();
    auto y      = y_arr.unchecked<1>();
    auto m      = m_arr.unchecked<1>();
    auto domain = domain_arr.unchecked<1>(); // [xmin, xmax, ymin, ymax]
    auto axw    = ax_arr.mutable_unchecked<1>();
    auto ayw    = ay_arr.mutable_unchecked<1>();
    const int N = (int)x_arr.shape(0);

    // 1) 复制到 local vector<Body> B
    std::vector<Body> B(N);
    for (int i = 0; i < N; ++i) {
        B[i].x   = x(i);
        B[i].y   = y(i);
        B[i].m   = m(i);
        B[i].idx = i;
    }

    // 2) 创建 root 节点 (cover domain)
    double xmin = domain(0), xmax = domain(1);
    double ymin = domain(2), ymax = domain(3);
    Node root(xmin, xmax, ymin, ymax);

    // 3) 生成 index list [0,1,...,N-1]
    std::vector<int> ids(N);
    for (int i = 0; i < N; ++i) ids[i] = i;

    // 4) OpenMP parallel 区块中调用 build_tree_rec → spawn tasks 构建树
    spawn_counter.store(0, std::memory_order_relaxed);
    #pragma omp parallel
    {
        #pragma omp single
        build_tree_rec(&root, B, ids);
    }
    // 这里可以 print 出 spawn_counter，或将它返回给 Python 以便检测
    // e.g. std::cout << "Spawned tasks: " << spawn_counter.load() << std::endl;

    // 5) 计算 θ²
    double th2 = theta * theta;

    // 6) parallel-for over i=0..N-1，每颗粒子调用 traverse
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        double fx = 0.0, fy = 0.0;
        traverse(B, &root, i, eps2, th2, fx, fy);
        axw(i) = fx;
        ayw(i) = fy;
    }
}

// ---------------------------------------------------------------------------
//  PyBind11 模块注册
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fmm_omp, m) {
    m.doc() = "2D Barnes–Hut FMM (monopole only) with fully parallel OpenMP (O(N)/P).";
    m.def("fmm_force_theta",
          &fmm_force_theta,
          "fmm_force_theta(x, y, m, eps2, domain, theta, ax, ay)");
}

