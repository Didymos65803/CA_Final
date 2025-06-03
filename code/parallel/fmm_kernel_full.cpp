// fmm_kernel_full.cpp
//
// 高精度 FMM 实现 (P = 10)，已加入 OpenMP 并行化 force evaluation
// 编译方式：假设您已经在同一目录下放置 setup.py，执行
//   python setup.py build_ext --inplace
// 就会生成 fmm_kernel*.so（Linux/macOS）或 fmm_kernel*.pyd（Windows）。

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <memory>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP
#endif

namespace py = pybind11;
using cplx = std::complex<double>;

// -----------------------
// Constant definitions
// -----------------------

// Multipole expansion 阶数 P
constexpr int P = 10;

// 预先计算阶乘表 (0! ~ P!)
static std::array<double, P + 1> factorial_table = []() {
    std::array<double, P + 1> table;
    table[0] = 1.0;
    for (int i = 1; i <= P; ++i) {
        table[i] = table[i - 1] * i;
    }
    return table;
}();

// 计算二项式系数 C(n, k)
static double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n - k) k = n - k;
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// -----------------------
// FMMCell 结构定义
// -----------------------
// 代表四叉树节点 (2D)
// 每个节点存储：中心 (cx, cy)、半边长 size、层数 level、
// 是否为叶节点 is_leaf、所含粒子列表 particles、
// multipole 展开系数 multipole[P+1]、local 展开系数 local[P+1]、
// 4 个子节点 children、父节点 parent。

struct FMMCell {
    double cx, cy;                           // 节点中心坐标
    double size;                             // 节点的半边长
    int level;                               // 树的层数
    bool is_leaf;                            // 是否为叶节点
    std::vector<int> particles;              // 此节点内所有粒子的索引
    std::array<cplx, P + 1> multipole{};     // multipole 展开系数 a_k, k = 0..P
    std::array<cplx, P + 1> local{};         // local 展开系数 b_j, j = 0..P
    std::array<std::unique_ptr<FMMCell>, 4> children; // 4 个子节点
    FMMCell* parent;                         // 父节点指针

    // 修改处：将 is_leaf(true) 放到 parent(p) 之前，以符合成员声明顺序
    FMMCell(double x, double y, double s, int lev = 0, FMMCell* p = nullptr)
        : cx(x), cy(y), size(s), level(lev), is_leaf(true), parent(p) {
        std::fill(multipole.begin(), multipole.end(), cplx(0.0, 0.0));
        std::fill(local.begin(), local.end(), cplx(0.0, 0.0));
    }
};

// -----------------------
// 切分空间：fmm_subdivide
// -----------------------
void fmm_subdivide(FMMCell* cell,
                   const std::vector<double>& x,
                   const std::vector<double>& y,
                   int max_particles = 16,
                   int max_level = 10) {
    if ((int)cell->particles.size() <= max_particles || cell->level >= max_level) {
        return;
    }
    cell->is_leaf = false;
    double half = cell->size * 0.5;

    // 创建 4 个子节点 (左下、右下、左上、右上)
    cell->children[0] = std::make_unique<FMMCell>(cell->cx - half, cell->cy - half, half, cell->level + 1, cell);
    cell->children[1] = std::make_unique<FMMCell>(cell->cx + half, cell->cy - half, half, cell->level + 1, cell);
    cell->children[2] = std::make_unique<FMMCell>(cell->cx - half, cell->cy + half, half, cell->level + 1, cell);
    cell->children[3] = std::make_unique<FMMCell>(cell->cx + half, cell->cy + half, half, cell->level + 1, cell);

    // 将当前节点 particles 分配到各子节点
    for (int pid : cell->particles) {
        int quad = 0;
        if (x[pid] > cell->cx) quad += 1;
        if (y[pid] > cell->cy) quad += 2;
        cell->children[quad]->particles.push_back(pid);
    }
    // 清空当前节点粒子列表
    cell->particles.clear();

    // 递归切分
    for (auto& child : cell->children) {
        if (child && !child->particles.empty()) {
            fmm_subdivide(child.get(), x, y, max_particles, max_level);
        }
    }
}

// -----------------------
// Upward Pass: P2M + M2M
// -----------------------
void fmm_upward_pass(FMMCell* cell,
                     const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& m) {
    if (!cell) return;
    // 清空 multipole
    std::fill(cell->multipole.begin(), cell->multipole.end(), cplx(0.0, 0.0));

    if (cell->is_leaf) {
        // P2M: 叶节点把自己的粒子信息累加到 multipole 展开
        for (int pid : cell->particles) {
            double mass = m[pid];
            double dx = x[pid] - cell->cx;
            double dy = y[pid] - cell->cy;
            cplx z(dx, dy);
            // a0 += m
            cell->multipole[0] += mass;
            cplx zpow = z;
            for (int k = 1; k <= P; ++k) {
                cell->multipole[k] += mass * zpow / factorial_table[k];
                zpow *= z;
            }
        }
    } else {
        // M2M: 先对子节点做 recursive upward
        for (auto& child : cell->children) {
            if (child && !child->particles.empty()) {
                fmm_upward_pass(child.get(), x, y, m);
                // 把 child multipole 转到 parent
                double dx = child->cx - cell->cx;
                double dy = child->cy - cell->cy;
                cplx z0(dx, dy);
                for (int l = 0; l <= P; ++l) {
                    cplx z0pow(1.0, 0.0);
                    for (int k = 0; k <= l; ++k) {
                        double bc = binomial(l, k);
                        cell->multipole[l] += child->multipole[k] * bc * z0pow;
                        z0pow *= z0;
                    }
                }
            }
        }
    }
}

// -----------------------
// M2L Translation
// -----------------------
void fmm_m2l_translation(FMMCell* target, FMMCell* source) {
    if (!target || !source || target == source) return;
    double dx = source->cx - target->cx;
    double dy = source->cy - target->cy;
    double r2 = dx * dx + dy * dy;
    double r = std::sqrt(r2);
    double size_sum = target->size + source->size;
    if (r < 2.0 * size_sum) return;

    cplx z0(dx, dy);
    for (int j = 0; j <= P; ++j) {
        cplx contrib(0.0, 0.0);
        for (int k = 0; k <= P; ++k) {
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            double bc = binomial(j + k, k);
            cplx denom = std::pow(z0, j + k + 1);
            if (std::abs(denom) > 1e-15) {
                contrib += sign * bc * source->multipole[k] / denom;
            }
        }
        target->local[j] += contrib;
    }
}

// -----------------------
// Interaction Pass (M2L for all节点)
// -----------------------
void fmm_interaction_pass(FMMCell* cell, FMMCell* root, double theta) {
    if (!cell) return;
    std::function<void(FMMCell*, FMMCell*)> traverse = [&](FMMCell* target, FMMCell* source) {
        if (!source || target == source) return;
        double dx = source->cx - target->cx;
        double dy = source->cy - target->cy;
        double dist = std::sqrt(dx * dx + dy * dy);
        double size_sum = target->size + source->size;
        if (dist > 2.0 * size_sum && source->multipole[0] != cplx(0.0, 0.0)) {
            fmm_m2l_translation(target, source);
        } else if (!source->is_leaf) {
            for (auto& child : source->children) {
                if (child) traverse(target, child.get());
            }
        }
    };
    traverse(cell, root);
    for (auto& child : cell->children) {
        if (child) fmm_interaction_pass(child.get(), root, theta);
    }
}

// -----------------------
// Downward Pass: L2L
// -----------------------
void fmm_downward_pass(FMMCell* cell) {
    if (!cell) return;
    for (auto& child : cell->children) {
        if (child) {
            double dx = child->cx - cell->cx;
            double dy = cell->cy - child->cy; // 注意 y 方向 sign
            cplx z0(dx, dy);
            for (int j = 0; j <= P; ++j) {
                cplx z0pow(1.0, 0.0);
                for (int k = j; k <= P; ++k) {
                    double bc = binomial(k, j);
                    child->local[j] += cell->local[k] * bc * z0pow;
                    z0pow *= z0;
                }
            }
            fmm_downward_pass(child.get());
        }
    }
}

// ====================================================================================
//                                  FMM 主函数 (fmm_omp)
// ====================================================================================
py::tuple fmm_omp(py::array_t<double> x_arr,
                  py::array_t<double> y_arr,
                  py::array_t<double> m_arr,
                  double domain,
                  double theta = 0.5,
                  double G = 1.0,
                  double soft = 0.05) {
    const size_t N = x_arr.size();
    if (N == 0) {
        return py::make_tuple(py::array_t<double>(0), py::array_t<double>(0));
    }

    std::vector<double> vx(x_arr.data(), x_arr.data() + N);
    std::vector<double> vy(y_arr.data(), y_arr.data() + N);
    std::vector<double> vm(m_arr.data(), m_arr.data() + N);
    std::vector<double> fx(N, 0.0), fy(N, 0.0);

    try {
        auto root = std::make_unique<FMMCell>(0.0, 0.0, domain * 0.5);
        root->particles.resize(N);
        std::iota(root->particles.begin(), root->particles.end(), 0);

        fmm_subdivide(root.get(), vx, vy, /*max_particles=*/16, /*max_level=*/10);
        fmm_upward_pass(root.get(), vx, vy, vm);
        fmm_interaction_pass(root.get(), root.get(), theta);
        fmm_downward_pass(root.get());

        // 建立 “粒子 → 叶节点” 映射表
        std::vector<FMMCell*> leaf_of_particle(N, nullptr);
        {
            std::function<void(FMMCell*)> map_leafs = [&](FMMCell* cell) {
                if (!cell) return;
                if (cell->is_leaf) {
                    for (int pid : cell->particles) {
                        leaf_of_particle[pid] = cell;
                    }
                } else {
                    for (auto& child : cell->children) {
                        if (child) map_leafs(child.get());
                    }
                }
            };
            map_leafs(root.get());
        }

        const double soft2 = soft * soft;
        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < N; ++i) {
            double ax_i = 0.0, ay_i = 0.0;
            FMMCell* leaf = leaf_of_particle[i];
            if (!leaf) continue;

            // (a) 同一 leaf 内的 Direct 互相作用
            for (int j : leaf->particles) {
                if ((size_t)j == i) continue;
                double dx = vx[j] - vx[i];
                double dy = vy[j] - vy[i];
                double r2 = dx * dx + dy * dy + soft2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                ax_i += G * vm[j] * dx * inv_r3;
                ay_i += G * vm[j] * dy * inv_r3;
            }

            // (b) Local 展开对 i 的影响: -∇φ
            {
                double dx = vx[i] - leaf->cx;
                double dy = vy[i] - leaf->cy;
                cplx zc(dx, dy);
                cplx zpow(1.0, 0.0), f_c(0.0, 0.0);
                for (int k = 1; k <= P; ++k) {
                    f_c += double(k) * leaf->local[k] * zpow / factorial_table[k];
                    zpow *= zc;
                }
                ax_i += G * (-f_c.real());
                ay_i += G * (-f_c.imag());
            }

            fx[i] = ax_i;
            fy[i] = ay_i;
        }

    } catch (const std::exception& e) {
        // 如果任何步骤抛异常，fallback 到并行的Direct O(N^2)计算
        const double soft2 = soft * soft;
        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < N; ++i) {
            double ax_i = 0.0, ay_i = 0.0;
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                double dx = vx[j] - vx[i];
                double dy = vy[j] - vy[i];
                double r2 = dx * dx + dy * dy + soft2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                ax_i += G * vm[j] * dx * inv_r3;
                ay_i += G * vm[j] * dy * inv_r3;
            }
            fx[i] = ax_i;
            fy[i] = ay_i;
        }
    }

    auto ax_out = py::array_t<double>(N);
    auto ay_out = py::array_t<double>(N);
    auto pax = ax_out.mutable_unchecked<1>();
    auto pay = ay_out.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
        pax(i) = fx[i];
        pay(i) = fy[i];
    }
    return py::make_tuple(ax_out, ay_out);
}

// -----------------------
// PyBind11 注册
// -----------------------
PYBIND11_MODULE(fmm_kernel, m) {
    m.doc() = "High-precision Fast Multipole Method (FMM) with OpenMP parallelization";
    m.def("fmm_omp", &fmm_omp,
          "Compute gravitational forces via FMM (P = 10) with OpenMP parallel evaluation",
          py::arg("x"), py::arg("y"), py::arg("m"),
          py::arg("domain"), py::arg("theta") = 0.5,
          py::arg("G") = 1.0, py::arg("soft") = 0.05);

    #ifdef USE_OPENMP
    m.attr("has_openmp") = true;
    #else
    m.attr("has_openmp") = false;
    #endif
}

