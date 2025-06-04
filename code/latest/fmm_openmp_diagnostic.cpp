// fmm_openmp_diagnostic.cpp – Diagnostic version to identify OpenMP issues
// ========================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <memory>
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

// ---------- OpenMP Diagnostics --------------------------------------------------------
void openmp_diagnostic() {
    std::cout << "=== OpenMP Diagnostic Information ===" << std::endl;
    
#ifdef _OPENMP
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
    
    // Test basic OpenMP functionality
    std::cout << "Testing basic OpenMP parallel region:" << std::endl;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "  Thread " << tid << " of " << nthreads << " threads" << std::endl;
        }
    }
    
    // Test parallel for
    std::cout << "Testing parallel for loop:" << std::endl;
    std::vector<int> thread_used(omp_get_max_threads(), 0);
    
    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        int tid = omp_get_thread_num();
        #pragma omp atomic
        thread_used[tid]++;
    }
    
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        if (thread_used[i] > 0) {
            std::cout << "  Thread " << i << " processed " << thread_used[i] << " iterations" << std::endl;
        }
    }
    
#else
    std::cout << "OpenMP not available!" << std::endl;
#endif
}

// ---------- Simple data structures -----------------------------------------------------
struct Body { 
    double x, y, m;
    double ax, ay;
    
    Body() : x(0), y(0), m(0), ax(0), ay(0) {}
    Body(double x_, double y_, double m_) : x(x_), y(y_), m(m_), ax(0), ay(0) {}
};

// ---------- Direct force implementation with explicit OpenMP testing ------------------
void direct_force_simple(const py::array_t<double>& x_arr,
                         const py::array_t<double>& y_arr,
                         const py::array_t<double>& m_arr,
                         double eps2,
                         py::array_t<double>& ax_arr,
                         py::array_t<double>& ay_arr) {
    
    const auto x = x_arr.unchecked<1>();
    const auto y = y_arr.unchecked<1>();
    const auto m = m_arr.unchecked<1>();
    auto ax = ax_arr.mutable_unchecked<1>();
    auto ay = ay_arr.mutable_unchecked<1>();
    
    const int N = x.shape(0);
    
    std::cout << "Direct force: N=" << N << ", available threads=" << omp_get_max_threads() << std::endl;
    
    // Initialize accelerations
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        ax(i) = 0.0;
        ay(i) = 0.0;
    }
    
    // Check if parallel region is working
    int threads_used = 0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            threads_used = omp_get_num_threads();
        }
    }
    std::cout << "  Threads actually used: " << threads_used << std::endl;
    
    // Simple O(N²) computation with explicit parallelization
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        double local_ax = 0.0, local_ay = 0.0;
        
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            
            const double dx = x(j) - x(i);
            const double dy = y(j) - y(i);
            const double r2 = dx * dx + dy * dy + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double force_magnitude = m(j) * inv_r3;
            
            local_ax += force_magnitude * dx;
            local_ay += force_magnitude * dy;
        }
        
        ax(i) = local_ax;
        ay(i) = local_ay;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Direct computation time: " << duration.count() / 1000.0 << " ms" << std::endl;
}

// ---------- Simplified tree structures ------------------------------------------------
struct SimpleNode {
    double cx, cy, size;
    double mass, cmx, cmy;
    bool leaf;
    std::vector<int> particles;
    std::unique_ptr<SimpleNode> children[4];
    
    SimpleNode(double cx_, double cy_, double size_) 
        : cx(cx_), cy(cy_), size(size_), mass(0), cmx(0), cmy(0), leaf(true) {
        for(int i = 0; i < 4; ++i) children[i] = nullptr;
    }
};

// ---------- Simplified FMM implementation ---------------------------------------------
class SimpleFMM {
private:
    std::vector<Body> bodies;
    std::unique_ptr<SimpleNode> root;
    
    void build_tree(SimpleNode* node, const std::vector<int>& particle_ids) {
        if (particle_ids.size() <= 16 || node->size < 1e-6) {
            node->particles = particle_ids;
            return;
        }
        
        node->leaf = false;
        
        // Create children
        double half_size = node->size * 0.5;
        double quarter_size = half_size * 0.5;
        
        node->children[0] = std::make_unique<SimpleNode>(node->cx - quarter_size, node->cy - quarter_size, half_size);
        node->children[1] = std::make_unique<SimpleNode>(node->cx + quarter_size, node->cy - quarter_size, half_size);
        node->children[2] = std::make_unique<SimpleNode>(node->cx - quarter_size, node->cy + quarter_size, half_size);
        node->children[3] = std::make_unique<SimpleNode>(node->cx + quarter_size, node->cy + quarter_size, half_size);
        
        // Distribute particles
        std::vector<std::vector<int>> child_particles(4);
        for (int pid : particle_ids) {
            int quadrant = 0;
            if (bodies[pid].x > node->cx) quadrant += 1;
            if (bodies[pid].y > node->cy) quadrant += 2;
            child_particles[quadrant].push_back(pid);
        }
        
        // Recursively build children
        for (int i = 0; i < 4; ++i) {
            if (!child_particles[i].empty()) {
                build_tree(node->children[i].get(), child_particles[i]);
            }
        }
    }
    
    void compute_mass_properties(SimpleNode* node) {
        if (!node) return;
        
        node->mass = node->cmx = node->cmy = 0.0;
        
        if (node->leaf) {
            for (int pid : node->particles) {
                node->mass += bodies[pid].m;
                node->cmx += bodies[pid].m * bodies[pid].x;
                node->cmy += bodies[pid].m * bodies[pid].y;
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                if (node->children[i]) {
                    compute_mass_properties(node->children[i].get());
                    node->mass += node->children[i]->mass;
                    node->cmx += node->children[i]->mass * node->children[i]->cmx;
                    node->cmy += node->children[i]->mass * node->children[i]->cmy;
                }
            }
        }
        
        if (node->mass > 0) {
            node->cmx /= node->mass;
            node->cmy /= node->mass;
        }
    }
    
    void compute_force(SimpleNode* node, int particle_idx, double theta2, double eps2, double& fx, double& fy) {
        if (!node || node->mass == 0) return;
        
        const Body& p = bodies[particle_idx];
        
        if (node->leaf) {
            // Direct computation
            for (int other_idx : node->particles) {
                if (other_idx == particle_idx) continue;
                
                const Body& other = bodies[other_idx];
                double dx = other.x - p.x;
                double dy = other.y - p.y;
                double r2 = dx*dx + dy*dy + eps2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double f = other.m * inv_r3;
                
                fx += f * dx;
                fy += f * dy;
            }
        } else {
            // Check opening criterion
            double dx = node->cmx - p.x;
            double dy = node->cmy - p.y;
            double r2 = dx*dx + dy*dy;
            
            if (r2 > 0 && (node->size * node->size) / r2 < theta2) {
                // Use multipole approximation
                double inv_r = 1.0 / std::sqrt(r2 + eps2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double f = node->mass * inv_r3;
                
                fx += f * dx;
                fy += f * dy;
            } else {
                // Recurse to children
                for (int i = 0; i < 4; ++i) {
                    if (node->children[i]) {
                        compute_force(node->children[i].get(), particle_idx, theta2, eps2, fx, fy);
                    }
                }
            }
        }
    }
    
public:
    void setup(const py::array_t<double>& x_arr,
               const py::array_t<double>& y_arr,
               const py::array_t<double>& m_arr,
               double domain_size) {
        
        const auto x = x_arr.unchecked<1>();
        const auto y = y_arr.unchecked<1>();
        const auto m = m_arr.unchecked<1>();
        const int N = x.shape(0);
        
        bodies.resize(N);
        for (int i = 0; i < N; ++i) {
            bodies[i] = Body(x(i), y(i), m(i));
        }
        
        // Build tree
        root = std::make_unique<SimpleNode>(0.0, 0.0, domain_size);
        std::vector<int> all_particles(N);
        for (int i = 0; i < N; ++i) all_particles[i] = i;
        
        build_tree(root.get(), all_particles);
        compute_mass_properties(root.get());
    }
    
    void compute_forces(double theta, double eps2, py::array_t<double>& ax_arr, py::array_t<double>& ay_arr) {
        auto ax = ax_arr.mutable_unchecked<1>();
        auto ay = ay_arr.mutable_unchecked<1>();
        const int N = bodies.size();
        
        std::cout << "FMM force: N=" << N << ", available threads=" << omp_get_max_threads() << std::endl;
        
        int threads_used = 0;
        #pragma omp parallel
        {
            #pragma omp master
            {
                threads_used = omp_get_num_threads();
            }
        }
        std::cout << "  Threads actually used: " << threads_used << std::endl;
        
        double theta2 = theta * theta;
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            double fx = 0.0, fy = 0.0;
            compute_force(root.get(), i, theta2, eps2, fx, fy);
            ax(i) = fx;
            ay(i) = fy;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  FMM computation time: " << duration.count() / 1000.0 << " ms" << std::endl;
    }
};

// ---------- Python interface functions ------------------------------------------------
void fmm_force_theta(const py::array_t<double>& x,
                     const py::array_t<double>& y,
                     const py::array_t<double>& m,
                     double eps2, double domain, double theta,
                     py::array_t<double>& ax,
                     py::array_t<double>& ay) {
    
    SimpleFMM fmm;
    fmm.setup(x, y, m, domain);
    fmm.compute_forces(theta, eps2, ax, ay);
}

void fmm_force(const py::array_t<double>& x,
               const py::array_t<double>& y,
               const py::array_t<double>& m,
               double eps2, double domain,
               py::array_t<double>& ax,
               py::array_t<double>& ay) {
    fmm_force_theta(x, y, m, eps2, domain, 0.6, ax, ay);
}

void direct_force(const py::array_t<double>& x,
                  const py::array_t<double>& y,
                  const py::array_t<double>& m,
                  double eps2,
                  py::array_t<double>& ax,
                  py::array_t<double>& ay) {
    direct_force_simple(x, y, m, eps2, ax, ay);
}

// ---------- Test functions for debugging -----------------------------------------------
void test_openmp_simple() {
    std::cout << "=== Testing Simple OpenMP Operations ===" << std::endl;
    
    const int N = 10000;
    std::vector<double> data(N);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        data[i] = i * 0.001;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Serial version
    double sum_serial = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_serial += std::sin(data[i]) * std::cos(data[i]);
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Parallel version
    double sum_parallel = 0.0;
    #pragma omp parallel for reduction(+:sum_parallel)
    for (int i = 0; i < N; ++i) {
        sum_parallel += std::sin(data[i]) * std::cos(data[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto serial_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Serial result: " << sum_serial << " (time: " << serial_time.count() << " μs)" << std::endl;
    std::cout << "Parallel result: " << sum_parallel << " (time: " << parallel_time.count() << " μs)" << std::endl;
    std::cout << "Speedup: " << (double)serial_time.count() / parallel_time.count() << "×" << std::endl;
    std::cout << "Results match: " << (std::abs(sum_serial - sum_parallel) < 1e-10 ? "Yes" : "No") << std::endl;
}

// ---------- Python bindings -----------------------------------------------------------
PYBIND11_MODULE(fmm_openmp, m) {
    m.doc() = "Diagnostic Barnes–Hut FMM with OpenMP debugging";
    
    m.def("direct_force", &direct_force,
          "Direct O(N²) force calculation with OpenMP diagnostics");
    
    m.def("fmm_force", &fmm_force,
          "Barnes–Hut FMM with default theta=0.6");
    
    m.def("fmm_force_theta", &fmm_force_theta,
          "Barnes–Hut FMM with configurable opening angle theta");
    
    m.def("openmp_diagnostic", &openmp_diagnostic,
          "Run OpenMP diagnostic tests");
    
    m.def("test_openmp_simple", &test_openmp_simple,
          "Test simple OpenMP operations");
          
#ifdef _OPENMP
    m.def("get_max_threads", &omp_get_max_threads,
          "Get maximum number of OpenMP threads");
    
    m.def("get_num_threads", &omp_get_num_threads,
          "Get current number of OpenMP threads");
#else
    m.def("get_max_threads", []() { return 1; },
          "Get maximum number of OpenMP threads (OpenMP not available)");
    
    m.def("get_num_threads", []() { return 1; },
          "Get current number of OpenMP threads (OpenMP not available)");
#endif
}
