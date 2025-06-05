// nbody_comparison.cpp
// A C++ program to benchmark and check the error of Direct, Barnes-Hut, and a structurally improved FMM.
// COMPILE WITH:
// g++ nbody_comparison.cpp -o nbody_comparison -O3 -std=c++17 -fopenmp -lm

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm> // For std::sort, std::unique, std::fill
#include <memory>    // For std::unique_ptr, std::make_unique
#include <complex>
#include <chrono>
#include <map>       // For memoizing factorials
#include <functional> // For std::function REQUIRED for the lambda in perform_fmm_downward_pass

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Constants ---
const double G_CONST = 1.0;
const double SOFTENING = 0.01;
const double SOFT2 = SOFTENING * SOFTENING;

// --- Data Structures ---
struct Particle {
    int id;
    double x, y, z;    // Position
    double mass;       // Mass
    double vx, vy, vz; // Velocity
    double ax, ay, az; // Acceleration

    Particle(int _id, double _x, double _y, double _z, double _m)
        : id(_id), x(_x), y(_y), z(_z), mass(_m),
          vx(0), vy(0), vz(0), ax(0), ay(0), az(0) {}
};

const int FMM_ORDER = 4; 
const int MAX_LEAF_PARTICLES = 32;
const double BH_THETA = 0.5;

class Node {
public:
    double cx, cy, cz;
    double size;
    std::vector<std::unique_ptr<Node>> children;
    std::vector<Particle*> particles;
    Node* parent = nullptr;
    std::vector<std::complex<double>> multipole_coeffs;
    std::vector<std::complex<double>> local_coeffs;
    bool is_leaf = true;
    int level = 0; 
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    std::vector<Node*> list1_U; 
    std::vector<Node*> list2_V;
    std::vector<Node*> list3_W;
    std::vector<Node*> list4_X;

    Node(double center_x, double center_y, double center_z, double s, Node* p, int l)
        : cx(center_x), cy(center_y), cz(center_z), size(s), parent(p), level(l) {
        children.resize(8); 
        multipole_coeffs.resize((FMM_ORDER + 1) * (FMM_ORDER + 1), {0.0, 0.0});
        local_coeffs.resize((FMM_ORDER + 1) * (FMM_ORDER + 1), {0.0, 0.0});
    }

    void insert(Particle* p) {
        if (is_leaf) {
            particles.push_back(p);
            if (particles.size() > MAX_LEAF_PARTICLES && size > 2.0 * SOFTENING && level < 10) {
                subdivide();
            }
        } else {
            children[get_child_index(p)]->insert(p);
        }
    }
    
    bool is_well_separated_from(const Node* other_node, double separation_factor = 2.0) const {
        if (!other_node) return false;
        double dx = cx - other_node->cx;
        double dy = cy - other_node->cy;
        double dz = cz - other_node->cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;
        // A more robust check would involve distance between closest corners vs size, 
        // or use the PDF's |Q| > (c+1)a criterion with c ~= sqrt(3)
        // For simplicity, use distance between centers > factor * (this->size + other_node->size)/2
        // Here, we'll use a simplified version based on own size, assuming comparable cell sizes in IL
        return dist_sq > (separation_factor * separation_factor * size * size);
    }

private:
    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;
        int child_idx = 0;
        for (int i = -1; i <= 1; i += 2) {    
            for (int j = -1; j <= 1; j += 2) { 
                for (int k = -1; k <= 1; k += 2) { 
                    children[child_idx++] = std::make_unique<Node>(
                        cx + k * offset, cy + j * offset, cz + i * offset,
                        child_size, this, level + 1);
                }
            }
        }
        for (auto* p_ptr : particles) {
            children[get_child_index(p_ptr)]->insert(p_ptr);
        }
        particles.clear();
    }

    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1; 
        if (p->y > cy) index |= 2; 
        if (p->z > cz) index |= 4; 
        return index;
    }
};

// --- UTILITY FUNCTIONS ---
void get_all_nodes(Node* node, std::vector<Node*>& nodes_list) {
    if (!node) return;
    nodes_list.push_back(node);
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if(child) get_all_nodes(child.get(), nodes_list);
        }
    }
}

void get_leaf_nodes(Node* node, std::vector<Node*>& leaves_list) {
    if (!node) return;
    if (node->is_leaf) {
        leaves_list.push_back(node);
    } else {
        for (const auto& child : node->children) {
            if(child) get_leaf_nodes(child.get(), leaves_list);
        }
    }
}

void p2p(Particle& p1, Particle& p2) {
    double dx = p2.x - p1.x; double dy = p2.y - p1.y; double dz = p2.z - p1.z;
    double r_sq = dx*dx + dy*dy + dz*dz; double r_sq_soft = r_sq + SOFT2;
    if (r_sq_soft < 1e-12 && r_sq < 1e-12) return;
    double inv_r = 1.0 / std::sqrt(r_sq_soft); double inv_r3 = inv_r*inv_r*inv_r;
    double force_val = G_CONST * p2.mass * inv_r3;
    p1.ax += force_val * dx; p1.ay += force_val * dy; p1.az += force_val * dz;
}

std::map<int, double> factorial_cache;
double factorial(int n) {
    if (n < 0) {
        // std::cerr << "Factorial of negative number requested!" << std::endl;
        return 0.0; // Or throw an exception
    }
    if (n == 0) return 1.0;
    if (factorial_cache.count(n)) return factorial_cache[n];
    double res = 1.0;
    // Limit factorial calculation to avoid overflow if n is too large
    // FMM_ORDER is small so this should be fine.
    if (n > 20) { // For double, 20! is already large. 170! is ~max for double.
         // std::cerr << "Warning: Factorial for n > 20 requested, potential overflow/precision loss. n=" << n << std::endl;
         // For FMM_ORDER usually < 20 this is fine.
    }
    for (int i = 1; i <= n; ++i) res *= i;
    factorial_cache[n] = res;
    return res;
}

// --- ALGORITHM IMPLEMENTATIONS ---

// --- 1. DIRECT O(N^2) ---
void compute_forces_direct(std::vector<Particle>& particles) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            p2p(particles[i], particles[j]);
        }
    }
}

// --- 2. BARNES-HUT O(N log N) ---
void compute_mass_distribution_BH(Node* node) {
    if (!node) return;
    node->total_mass = 0.0; node->com_x = 0.0; node->com_y = 0.0; node->com_z = 0.0;
    if (node->is_leaf) {
        for (const auto* p : node->particles) {
            node->total_mass += p->mass;
            node->com_x += p->x * p->mass; node->com_y += p->y * p->mass; node->com_z += p->z * p->mass;
        }
    } else {
        for (const auto& child : node->children) {
            if (child) {
                compute_mass_distribution_BH(child.get());
                node->total_mass += child->total_mass;
                node->com_x += child->com_x * child->total_mass;
                node->com_y += child->com_y * child->total_mass;
                node->com_z += child->com_z * child->total_mass;
            }
        }
    }
    if (node->total_mass > 1e-12) {
        node->com_x /= node->total_mass; node->com_y /= node->total_mass; node->com_z /= node->total_mass;
    } else {
        node->com_x = node->cx; node->com_y = node->cy; node->com_z = node->cz;
    }
}

void compute_force_on_particle_BH(Particle* target_p, Node* current_node) {
    if (!current_node || current_node->total_mass < 1e-12) return;
    double dx = current_node->com_x - target_p->x; double dy = current_node->com_y - target_p->y; double dz = current_node->com_z - target_p->z;
    double d_sq = dx * dx + dy * dy + dz * dz;
    bool is_self_node = current_node->is_leaf && current_node->particles.size() == 1 && current_node->particles[0] == target_p;
    if (is_self_node) return;
    if (current_node->is_leaf) {
        for (auto* source_p : current_node->particles) {
            if (target_p != source_p) p2p(*target_p, *source_p);
        }
    } else {
        if (current_node->size * current_node->size < d_sq * BH_THETA * BH_THETA) {
            double r_sq_soft = d_sq + SOFT2; if (r_sq_soft < 1e-12) return;
            double inv_r = 1.0 / std::sqrt(r_sq_soft); double inv_r3 = inv_r * inv_r * inv_r;
            double f_scale = G_CONST * current_node->total_mass * inv_r3;
            target_p->ax += f_scale * dx; target_p->ay += f_scale * dy; target_p->az += f_scale * dz;
        } else {
            for (const auto& child : current_node->children) {
                if (child) compute_force_on_particle_BH(target_p, child.get());
            }
        }
    }
}

void compute_forces_BH(std::vector<Particle>& particles, Node& root) {
    compute_mass_distribution_BH(&root);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0;
        compute_force_on_particle_BH(&particles[i], &root);
    }
}

// --- 3. FMM O(N) ---
namespace FMM_Math {
    inline int lm_to_idx(int l, int m) { return l * l + l + m; } 

    void cart_to_sph(double x, double y, double z, double& r, double& theta, double& phi) {
        r = std::sqrt(x*x + y*y + z*z);
        if (r < 1e-12) { theta = 0.0; phi = 0.0; } 
        else { theta = std::acos(z / r); phi = std::atan2(y, x); }
    }
    
    double legendreP(int l, int m_abs, double x) { 
        if (m_abs < 0 || m_abs > l ) return 0.0; 
        if (x > 1.0) x = 1.0; if (x < -1.0) x = -1.0;
        double pmm = 1.0; 
        if (m_abs > 0) {
            double somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x))); // Ensure arg to sqrt is non-negative
            double fact = 1.0; 
            for (int i = 1; i <= m_abs; i++) {
                pmm *= -fact * somx2; 
                fact += 2.0;
            }
        } 
        if (l == m_abs) return pmm;
        double pmmp1 = x * (2.0 * m_abs + 1.0) * pmm; 
        if (l == m_abs + 1) return pmmp1;
        double pll = 0.0; 
        for (int ll_iter = m_abs + 2; ll_iter <= l; ll_iter++) { 
            pll = ((2.0 * ll_iter - 1.0) * x * pmmp1 - (ll_iter + m_abs - 1.0) * pmm) / (ll_iter - m_abs);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        return pll;
    }
    
    std::complex<double> Y_lm_pdf(int n, int m, double theta, double phi) {
        if (n < 0 || std::abs(m) > n) return {0.0, 0.0};
        double m_abs = std::abs(m);
        double norm_factor_sq_num = factorial(n - m_abs);
        double norm_factor_sq_den = factorial(n + m_abs);
        double norm_factor = 0.0;

        if (norm_factor_sq_den > 1e-20) {
             norm_factor = std::sqrt(norm_factor_sq_num / norm_factor_sq_den);
        } else if (norm_factor_sq_num < 1e-20) { 
             norm_factor = 1.0; 
        } // else norm_factor remains 0.0, implies potential issue if num > 0 & den = 0

        double p_val = legendreP(n, m_abs, std::cos(theta)); 
        return norm_factor * p_val * std::exp(std::complex<double>(0.0, m * phi));
    }
}

void p2m_FMM(Node* leaf_node) { 
    if (!leaf_node || !leaf_node->is_leaf) return;
    std::fill(leaf_node->multipole_coeffs.begin(), leaf_node->multipole_coeffs.end(), std::complex<double>(0.0,0.0));
    for (const auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        FMM_Math::cart_to_sph(p->x - leaf_node->cx, p->y - leaf_node->cy, p->z - leaf_node->cz, 
                              r_rel, theta_rel, phi_rel);
        for (int n = 0; n <= FMM_ORDER; ++n) {
            for (int m = -n; m <= n; ++m) {
                leaf_node->multipole_coeffs[FMM_Math::lm_to_idx(n,m)] += 
                    p->mass * std::pow(r_rel, n) * FMM_Math::Y_lm_pdf(n, -m, theta_rel, phi_rel);
            }
        }
    }
}

void M2M_translation(const Node* child_node, Node* parent_node) {
    // Placeholder: Implements Theorem 5.3 from PDF.
    // This is a complex formula involving sums over child's multipole coeffs,
    // displacement vector (child center to parent center) in spherical coords,
    // A_n^m constants, and Y_n^{-m} of displacement.
    // For now, only the 0-th order moment (total mass/charge) is correctly propagated.
    if (!child_node || !parent_node) return;
    if (!parent_node->multipole_coeffs.empty() && !child_node->multipole_coeffs.empty()) {
         parent_node->multipole_coeffs[FMM_Math::lm_to_idx(0,0)] += child_node->multipole_coeffs[FMM_Math::lm_to_idx(0,0)];
    }
    // std::cout << "Warning: M2M_translation is a placeholder for higher-order moments." << std::endl;
}

void perform_m2m_pass(Node* node) {
    if (!node || node->is_leaf) {
        return; 
    }
    std::fill(node->multipole_coeffs.begin(), node->multipole_coeffs.end(), std::complex<double>(0.0,0.0));
    for (const auto& child : node->children) {
        if (child) {
            perform_m2m_pass(child.get()); 
            M2M_translation(child.get(), node); 
        }
    }
}

void M2L_translation(const Node* source_node_well_separated, Node* target_node) {
    // Placeholder: Implements Theorem 5.4 from PDF.
    // Converts source_node's multipole expansion to a local expansion at target_node's center.
    // Involves source multipole coeffs, displacement vector (target center to source center),
    // A_n^m constants, Y_{j+n}^{m-k} of displacement. Extremely complex.
    if(!source_node_well_separated || !target_node) return;
    // std::cout << "Warning: M2L_translation is a placeholder." << std::endl;
}

void L2L_translation(const Node* parent_node, Node* child_node) {
    // Placeholder: Implements Theorem 5.5 from PDF.
    // Translates parent_node's local expansion to child_node's center.
    // Involves parent local coeffs, displacement vector (child center to parent center),
    // A_n^m constants, Y_{n-j}^{m-k} of displacement.
    if(!parent_node || !child_node) return;
    // std::cout << "Warning: L2L_translation is a placeholder." << std::endl;
}

void l2p_FMM(Node* leaf_node) { 
    if (!leaf_node || !leaf_node->is_leaf) return;
    for (auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        FMM_Math::cart_to_sph(p->x - leaf_node->cx, p->y - leaf_node->cy, p->z - leaf_node->cz, 
                              r_rel, theta_rel, phi_rel);
        std::complex<double> pot_grad_x(0,0), pot_grad_y(0,0), pot_grad_z(0,0);
        for (int n = 0; n <= FMM_ORDER; ++n) { 
            for (int m = -n; m <= n; ++m) {
                int idx = FMM_Math::lm_to_idx(n, m);
                if (std::abs(leaf_node->local_coeffs[idx].real()) < 1e-20 && std::abs(leaf_node->local_coeffs[idx].imag()) < 1e-20) continue;
                if (n == 0) continue; 
                double r_pow_nm1 = (r_rel > 1e-9) ? (n * std::pow(r_rel, n - 1)) : 0.0;
                auto Ylm_val = FMM_Math::Y_lm_pdf(n, m, theta_rel, phi_rel);
                pot_grad_x += leaf_node->local_coeffs[idx] * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::cos(phi_rel);
                pot_grad_y += leaf_node->local_coeffs[idx] * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::sin(phi_rel);
                pot_grad_z += leaf_node->local_coeffs[idx] * r_pow_nm1 * Ylm_val * std::cos(theta_rel);
            }
        }
        p->ax -= G_CONST * pot_grad_x.real(); 
        p->ay -= G_CONST * pot_grad_y.real();
        p->az -= G_CONST * pot_grad_z.real();
    }
}

void perform_fmm_downward_pass(Node* node, const std::vector<Node*>& /* all_nodes_for_m2l_potentially */) {
    // The all_nodes_for_m2l_potentially is not used in this simplified interaction list finding
    if (!node) return;

    if (node->parent && node->parent->parent) { 
        Node* p_node = node->parent;
        Node* gp_node = p_node->parent;
        for (const auto& uncle_block_ptr : gp_node->children) {
            Node* uncle_block = uncle_block_ptr.get();
            if (uncle_block == p_node || !uncle_block ) continue; 

            std::vector<Node*> source_candidates_from_uncle_block;
            // Lambda to collect leaf descendants or self if leaf
            std::function<void(Node*)> collect_leaves_recursive = 
                [&](Node* current_scan_node) {
                if (!current_scan_node) return;
                if (current_scan_node->is_leaf) {
                    if(!current_scan_node->particles.empty()) // Only consider non-empty leaves
                        source_candidates_from_uncle_block.push_back(current_scan_node);
                } else {
                    for(const auto& child_of_scan_node : current_scan_node->children) {
                        collect_leaves_recursive(child_of_scan_node.get());
                    }
                }
            };            
            collect_leaves_recursive(uncle_block);

            for (Node* source_node : source_candidates_from_uncle_block) {
                 if (node->is_well_separated_from(source_node)) { 
                    M2L_translation(source_node, node); 
                 }
            }
        }
    }

    if (node->is_leaf) {
        l2p_FMM(node);
        // P2P: Simplified to siblings. A full FMM would use a more robust neighbor list (e.g., 3x3x3 stencil of finest-level cells)
        if (node->parent) { 
            for (const auto& sibling_ptr : node->parent->children) {
                Node* sibling = sibling_ptr.get();
                if (sibling && sibling != node && sibling->is_leaf && !sibling->particles.empty()) { 
                    for (auto* p1 : node->particles) {
                        for (auto* p2 : sibling->particles) p2p(*p1, *p2);
                    }
                }
            }
        }
    } else { 
        for (const auto& child : node->children) {
            if (child) {
                L2L_translation(node, child.get()); 
                perform_fmm_downward_pass(child.get(), /* all_nodes_for_m2l_potentially */ {}); // Pass empty vector or manage properly
            }
        }
    }
}

void compute_forces_FMM_structured(std::vector<Particle>& particles, Node& root) {
    for (auto& p : particles) { p.ax = p.ay = p.az = 0; }
    
    std::vector<Node*> leaves;
    get_leaf_nodes(&root, leaves); 

    #pragma omp parallel for
    for (size_t i = 0; i < leaves.size(); ++i) {
        if(leaves[i]) p2m_FMM(leaves[i]);
    }
    
    perform_m2m_pass(&root); 

    std::vector<Node*> all_nodes; // Not strictly needed if downward pass handles its own traversal
    get_all_nodes(&root, all_nodes); 
    for (Node* n : all_nodes) {
        if(n) std::fill(n->local_coeffs.begin(), n->local_coeffs.end(), std::complex<double>(0.0,0.0));
    }
    
    perform_fmm_downward_pass(&root, all_nodes); 
}

// --- PARTICLE INITIALIZATION ---
std::vector<Particle> init_particles(int num_particles, unsigned int seed, double domain_half_size = 25.0) {
    std::vector<Particle> p_list;
    p_list.reserve(num_particles);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_dist(-domain_half_size, domain_half_size);
    std::uniform_real_distribution<double> mass_dist(0.5, 1.5); 
    for (int i = 0; i < num_particles; ++i) {
        p_list.emplace_back(i, pos_dist(rng), pos_dist(rng), pos_dist(rng), mass_dist(rng));
    }
    return p_list;
}

// --- ERROR CALCULATION ---
double calculate_rms_relative_error(const std::vector<Particle>& approx_particles, const std::vector<Particle>& exact_particles) {
    if (approx_particles.size() != exact_particles.size()) {
        std::cerr << "Error: Particle vector sizes mismatch for error calculation." << std::endl;
        return -1.0; 
    }
    double sum_sq_diff = 0.0;
    double sum_sq_exact = 0.0;
    for (size_t i = 0; i < approx_particles.size(); ++i) {
        double dx = approx_particles[i].ax - exact_particles[i].ax;
        double dy = approx_particles[i].ay - exact_particles[i].ay;
        double dz = approx_particles[i].az - exact_particles[i].az;
        sum_sq_diff += dx*dx + dy*dy + dz*dz;
        sum_sq_exact += exact_particles[i].ax*exact_particles[i].ax + exact_particles[i].ay*exact_particles[i].ay + exact_particles[i].az*exact_particles[i].az;
    }
    if (sum_sq_exact < 1e-24) {
        return (sum_sq_diff < 1e-24) ? 0.0 : 1.0;
    }
    return std::sqrt(sum_sq_diff / sum_sq_exact);
}

// --- BENCHMARKING MAIN ---
int main() {
    std::cout << std::fixed << std::setprecision(5);
    factorial_cache.clear(); // Initialize factorial cache

    std::vector<int> N_values = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}; // Significantly reduced N for FMM dev.
    int max_N_for_direct_sum = 32768; // Limit for direct sum method, can be adjusted based on performance

    std::vector<int> thread_counts_to_test;
    int max_hw_threads = 1;
    #ifdef _OPENMP
    max_hw_threads = omp_get_max_threads();
    #endif
    thread_counts_to_test.push_back(1); 
    for (int tc = 2; tc <= max_hw_threads; tc *= 2) {
        thread_counts_to_test.push_back(tc);
    }
    if (std::find(thread_counts_to_test.begin(), thread_counts_to_test.end(), max_hw_threads) == thread_counts_to_test.end() && max_hw_threads > 1) {
        thread_counts_to_test.push_back(max_hw_threads);
    }
    std::sort(thread_counts_to_test.begin(), thread_counts_to_test.end());
    thread_counts_to_test.erase(std::unique(thread_counts_to_test.begin(), thread_counts_to_test.end()), thread_counts_to_test.end());
    
    #ifndef _OPENMP
    if (thread_counts_to_test.size() > 1 || thread_counts_to_test[0] != 1) {
        // std::cout << "Warning: OpenMP not enabled or not effective. Thread scaling tests will effectively run on 1 thread." << std::endl;
    }
    thread_counts_to_test = {1}; 
    #endif

    std::ofstream results_file("performance_results.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open performance_results.csv for writing." << std::endl;
        return 1;
    }
    results_file << "N,Method,Num_Threads,Time_sec,Relative_Error\n"; 
    unsigned int seed = 42;
    double domain_half_size = 25.0;
    double root_node_size = 2.0 * domain_half_size + 1.0; 

    for (int N : N_values) {
        std::cout << "\n--- Testing N = " << N << " ---" << std::endl;
        
        std::vector<Particle> particles_direct_results;
        bool direct_computed_for_N = false;

        if (N <= max_N_for_direct_sum) {
            auto temp_particles = init_particles(N, seed, domain_half_size);
            #ifdef _OPENMP
            omp_set_num_threads(max_hw_threads); 
            #endif
            compute_forces_direct(temp_particles); 
            particles_direct_results = temp_particles;
            direct_computed_for_N = true;
        }

        std::vector<std::string> methods_to_test = {"Direct", "BH", "FMM"};

        for (const std::string& method_name : methods_to_test) {
            if (method_name == "Direct" && N > max_N_for_direct_sum) { 
                 if (N_values.empty() || N != N_values[0] || N_values[0] <= max_N_for_direct_sum) { 
                    std::cout << "Skipping Direct O(N^2) method for N=" << N << " (N > max_N_for_direct_sum=" << max_N_for_direct_sum << ")" << std::endl;
                    continue;
                 }
            }

            for (int num_threads : thread_counts_to_test) {
                #ifdef _OPENMP
                omp_set_num_threads(num_threads);
                #else
                if (num_threads > 1) continue; 
                #endif

                auto current_particles = init_particles(N, seed, domain_half_size);
                std::chrono::duration<double> time_diff;
                
                std::cout << "Running " << method_name << " (N=" << N << ", Threads=" << num_threads << ")..." << std::flush;

                if (method_name == "Direct") {
                    auto start = std::chrono::high_resolution_clock::now();
                    compute_forces_direct(current_particles);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_diff = end - start;
                } else { 
                    Node root_node(0, 0, 0, root_node_size, nullptr, 0); 
                    for(auto& p : current_particles) root_node.insert(&p);
                    
                    auto start = std::chrono::high_resolution_clock::now();
                    if (method_name == "BH") {
                        compute_forces_BH(current_particles, root_node);
                    } else if (method_name == "FMM") {
                        compute_forces_FMM_structured(current_particles, root_node); 
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    time_diff = end - start;
                }
                
                double error_val = -1.0;
                if (method_name == "Direct") {
                    error_val = 0.0;
                } else if (direct_computed_for_N) {
                    error_val = calculate_rms_relative_error(current_particles, particles_direct_results);
                }
                
                results_file << N << "," << method_name << "," << num_threads << "," << time_diff.count() << "," << error_val << "\n";
                std::cout << " Time: " << time_diff.count() << "s";
                if (error_val != -1.0) {
                     std::cout << ", Error: " << error_val;
                }
                std::cout << std::endl;
            }
        }
    }
    results_file.close();
    std::cout << "\nBenchmark finished. Results saved to performance_results.csv" << std::endl;
    std::cout << "Note: The FMM implementation is structurally improved but core translations (M2M, M2L, L2L) are placeholders." << std::endl;
    return 0;
}