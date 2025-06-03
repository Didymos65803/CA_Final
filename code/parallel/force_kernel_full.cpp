// force_kernel_full.cpp
// HW6-inspired direct method with proper parallelization

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void direct_force(const py::array_t<double>& x_arr,
                  const py::array_t<double>& y_arr,
                  const py::array_t<double>& m_arr,
                  double eps2,
                  py::array_t<double>& ax_arr,
                  py::array_t<double>& ay_arr)
{
    auto x = x_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const ssize_t N = x.shape(0);
    
    if (N != y.shape(0) || N != m.shape(0) || N != ax.shape(0) || N != ay.shape(0)) {
        throw std::runtime_error("Array size mismatch in direct_force");
    }
    
    // 參考 HW6 的初始化模式
    #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    if (N <= 100) {
        // 小問題：順序計算
        for (ssize_t i = 0; i < N; ++i) {
            const double xi = x(i);
            const double yi = y(i);
            double axi = 0.0;
            double ayi = 0.0;
            
            for (ssize_t j = 0; j < N; ++j) {
                if (i == j) continue;
                
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                
                if (r2 > eps2) {
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    const double mj = m(j);
                    axi -= mj * dx * inv_r3;
                    ayi -= mj * dy * inv_r3;
                }
            }
            
            ax(i) = axi;
            ay(i) = ayi;
        }
    } else {
        // 參考 HW6 的成功並行化模式：使用靜態排程
        #pragma omp parallel for schedule(static)
        for (ssize_t i = 0; i < N; ++i) {
            const double xi = x(i);
            const double yi = y(i);
            double axi = 0.0;
            double ayi = 0.0;
            
            // 內層迴圈保持順序，避免快取競爭
            for (ssize_t j = 0; j < N; ++j) {
                if (i == j) continue;
                
                const double dx = xi - x(j);
                const double dy = yi - y(j);
                const double r2 = dx*dx + dy*dy + eps2;
                
                if (r2 > eps2) {
                    const double inv_r = 1.0 / std::sqrt(r2);
                    const double inv_r3 = inv_r * inv_r * inv_r;
                    const double mj = m(j);
                    axi -= mj * dx * inv_r3;
                    ayi -= mj * dy * inv_r3;
                }
            }
            
            ax(i) = axi;
            ay(i) = ayi;
        }
    }
} // 修正：加入這個缺少的大括號

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "2D direct O(N^2) gravitational kernel (HW6-inspired)";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"));
}

