// force_kernel_optimized.cpp
// 優化的直接力計算，解決false sharing和cache locality問題

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>  // AVX2 intrinsics

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// 修正1: 使用塊式算法提高cache locality
constexpr int BLOCK_SIZE = 64;  // 根據cache line大小調整

// 修正2: SIMD優化的力計算核心
inline void compute_force_simd_kernel(double xi, double yi, double mi,
                                      double xj, double yj, double mj,
                                      double eps2, double& ax, double& ay) {
    const double dx = xi - xj;
    const double dy = yi - yj;
    const double r2 = dx*dx + dy*dy + eps2;
    
    if (r2 > eps2) {
        const double inv_r = 1.0 / std::sqrt(r2);
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double mj_inv_r3 = mj * inv_r3;
        
        ax -= mj_inv_r3 * dx;
        ay -= mj_inv_r3 * dy;
    }
}

// 修正3: 向量化的批次計算
void compute_force_block(const double* x, const double* y, const double* m,
                        ssize_t N, double eps2,
                        ssize_t i_start, ssize_t i_end,
                        ssize_t j_start, ssize_t j_end,
                        double* ax, double* ay) {
    
    for (ssize_t i = i_start; i < i_end; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        double ax_local = 0.0;
        double ay_local = 0.0;
        
        // 內層循環向量化
        #pragma omp simd reduction(+:ax_local,ay_local)
        for (ssize_t j = j_start; j < j_end; ++j) {
            if (i != j) {
                compute_force_simd_kernel(xi, yi, m[i], 
                                          x[j], y[j], m[j], 
                                          eps2, ax_local, ay_local);
            }
        }
        
        ax[i] += ax_local;
        ay[i] += ay_local;
    }
}

// 修正4: 對稱性優化的塊計算
void compute_force_block_symmetric(const double* x, const double* y, const double* m,
                                   ssize_t N, double eps2,
                                   ssize_t i_start, ssize_t i_end,
                                   ssize_t j_start, ssize_t j_end,
                                   std::vector<double>& ax_local,
                                   std::vector<double>& ay_local) {
    
    for (ssize_t i = i_start; i < i_end; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double mi = m[i];
        
        for (ssize_t j = std::max(j_start, i + 1); j < j_end; ++j) {
            const double dx = xi - x[j];
            const double dy = yi - y[j];
            const double r2 = dx*dx + dy*dy + eps2;
            
            if (r2 > eps2) {
                const double inv_r = 1.0 / std::sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                
                const double fx = dx * inv_r3;
                const double fy = dy * inv_r3;
                
                // 利用對稱性
                ax_local[i] -= m[j] * fx;
                ay_local[i] -= m[j] * fy;
                ax_local[j] += mi * fx;
                ay_local[j] += mi * fy;
            }
        }
    }
}

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
    
    // 修正5: 優化的記憶體初始化
    #pragma omp parallel for simd
    for (ssize_t i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    const int num_threads = omp_get_max_threads();
    
    if (N <= 100) {
        // 小問題：簡單順序計算
        for (ssize_t i = 0; i < N; ++i) {
            const double xi = x(i);
            const double yi = y(i);
            double axi = 0.0;
            double ayi = 0.0;
            
            #pragma omp simd reduction(+:axi,ayi)
            for (ssize_t j = 0; j < N; ++j) {
                if (i != j) {
                    compute_force_simd_kernel(xi, yi, m(i),
                                              x(j), y(j), m(j),
                                              eps2, axi, ayi);
                }
            }
            
            ax(i) = axi;
            ay(i) = ayi;
        }
    } else if (N <= 2000) {
        // 中等問題：使用對稱性優化
        std::vector<std::vector<double>> ax_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> ay_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& ax_local = ax_threads[tid];
            auto& ay_local = ay_threads[tid];
            
            #pragma omp for schedule(guided, 32) nowait
            for (ssize_t i = 0; i < N; ++i) {
                const double xi = x(i);
                const double yi = y(i);
                const double mi = m(i);
                
                for (ssize_t j = i + 1; j < N; ++j) {
                    const double dx = xi - x(j);
                    const double dy = yi - y(j);
                    const double r2 = dx*dx + dy*dy + eps2;
                    
                    if (r2 > eps2) {
                        const double inv_r = 1.0 / std::sqrt(r2);
                        const double inv_r3 = inv_r * inv_r * inv_r;
                        const double mj = m(j);
                        
                        const double fx = dx * inv_r3;
                        const double fy = dy * inv_r3;
                        
                        ax_local[i] -= mj * fx;
                        ay_local[i] -= mj * fy;
                        ax_local[j] += mi * fx;
                        ay_local[j] += mi * fy;
                    }
                }
            }
        }
        
        // 修正6: 避免false sharing的歸約
        #pragma omp parallel for schedule(static)
        for (ssize_t i = 0; i < N; ++i) {
            double sum_ax = 0.0;
            double sum_ay = 0.0;
            
            for (int tid = 0; tid < num_threads; ++tid) {
                sum_ax += ax_threads[tid][i];
                sum_ay += ay_threads[tid][i];
            }
            
            ax(i) = sum_ax;
            ay(i) = sum_ay;
        }
    } else {
        // 大問題：使用塊式算法
        const ssize_t block_size = std::min(static_cast<ssize_t>(BLOCK_SIZE), N / num_threads + 1);
        const ssize_t num_blocks = (N + block_size - 1) / block_size;
        
        // 創建線程局部緩衝區避免false sharing
        std::vector<std::vector<double>> ax_threads(num_threads, std::vector<double>(N, 0.0));
        std::vector<std::vector<double>> ay_threads(num_threads, std::vector<double>(N, 0.0));
        
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& ax_local = ax_threads[tid];
            auto& ay_local = ay_threads[tid];
            
            // 修正7: 二維塊式並行化提高cache locality
            #pragma omp for schedule(guided, 1) collapse(2) nowait
            for (ssize_t bi = 0; bi < num_blocks; ++bi) {
                for (ssize_t bj = 0; bj < num_blocks; ++bj) {
                    const ssize_t i_start = bi * block_size;
                    const ssize_t i_end = std::min(i_start + block_size, N);
                    const ssize_t j_start = bj * block_size;
                    const ssize_t j_end = std::min(j_start + block_size, N);
                    
                    if (bi == bj) {
                        // 對角塊：使用對稱性
                        compute_force_block_symmetric(x.data(0), y.data(0), m.data(0),
                                                      N, eps2, i_start, i_end, j_start, j_end,
                                                      ax_local, ay_local);
                    } else if (bi < bj) {
                        // 上三角塊：計算並應用對稱性
                        for (ssize_t i = i_start; i < i_end; ++i) {
                            const double xi = x(i);
                            const double yi = y(i);
                            const double mi = m(i);
                            
                            for (ssize_t j = j_start; j < j_end; ++j) {
                                const double dx = xi - x(j);
                                const double dy = yi - y(j);
                                const double r2 = dx*dx + dy*dy + eps2;
                                
                                if (r2 > eps2) {
                                    const double inv_r = 1.0 / std::sqrt(r2);
                                    const double inv_r3 = inv_r * inv_r * inv_r;
                                    const double mj = m(j);
                                    
                                    const double fx = dx * inv_r3;
                                    const double fy = dy * inv_r3;
                                    
                                    ax_local[i] -= mj * fx;
                                    ay_local[i] -= mj * fy;
                                    ax_local[j] += mi * fx;
                                    ay_local[j] += mi * fy;
                                }
                            }
                        }
                    }
                    // 下三角塊由對稱性處理，跳過
                }
            }
        }
        
        // 修正8: 高效的歸約操作
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (ssize_t i = 0; i < N; ++i) {
            double sum_ax = 0.0;
            double sum_ay = 0.0;
            
            // 向量化歸約
            #pragma omp simd reduction(+:sum_ax,sum_ay)
            for (int tid = 0; tid < num_threads; ++tid) {
                sum_ax += ax_threads[tid][i];
                sum_ay += ay_threads[tid][i];
            }
            
            ax(i) = sum_ax;
            ay(i) = sum_ay;
        }
    }
}

PYBIND11_MODULE(force_kernel, m) {
    m.doc() = "Highly Optimized 2D Direct O(N^2) Gravitational Kernel";
    m.def("direct_force",
          &direct_force,
          py::arg("x"),
          py::arg("y"),
          py::arg("m"),
          py::arg("eps2"),
          py::arg("ax"),
          py::arg("ay"));
}
