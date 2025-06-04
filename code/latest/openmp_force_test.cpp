
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    const int N = 10000000;
    double sum = 0.0;
    
    std::cout << "OpenMP available: " << 
#ifdef _OPENMP
    "Yes, version " << _OPENMP << std::endl;
#else
    "No" << std::endl;
#endif
    
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Force parallel execution with more work
    #pragma omp parallel for reduction(+:sum) schedule(static,1000)
    for (int i = 0; i < N; ++i) {
        // Add more computational work to make parallelization worthwhile
        double x = i * 0.0001;
        for (int j = 0; j < 10; ++j) {
            x = sin(x) + cos(x);
        }
        sum += x;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    return 0;
}
