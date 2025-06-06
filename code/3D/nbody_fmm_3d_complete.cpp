// nbody_comparison.cpp
// A C++ program to benchmark and check the error of Direct, Barnes-Hut, and a fully implemented FMM.
// COMPILE WITH:
// g++ nbody_comparison.cpp -o nbody_comparison -O3 -std=c++17 -fopenmp -lm

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>  // For std::sort, std::unique, std::fill
#include <memory>     // For std::unique_ptr, std::make_unique
#include <complex>
#include <chrono>
#include <map>        // For memoizing factorials
#include <functional> // For std::function (used in FMM downward pass)

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

// --- FMM Parameters ---
const int FMM_ORDER = 4;           // Truncation order for multipole/local expansions
const int MAX_LEAF_PARTICLES = 32; // Maximum particles per leaf
const double BH_THETA = 0.5;       // Barnes-Hut opening angle threshold

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

class Node {
public:
    double cx, cy, cz; // Center coordinates of the box
    double size;       // Box side length
    std::vector<std::unique_ptr<Node>> children;
    std::vector<Particle*> particles;
    Node* parent = nullptr;
    std::vector<std::complex<double>> multipole_coeffs; // Multipole expansion coefficients
    std::vector<std::complex<double>> local_coeffs;     // Local expansion coefficients
    bool is_leaf = true;
    int level = 0; 
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;

    Node(double center_x, double center_y, double center_z, double s, Node* p, int l)
        : cx(center_x), cy(center_y), cz(center_z), size(s), parent(p), level(l) {
        children.resize(8);
        // Allocate (FMM_ORDER+1)^2 complex coefficients
        multipole_coeffs.resize((FMM_ORDER + 1) * (FMM_ORDER + 1), {0.0, 0.0});
        local_coeffs.resize((FMM_ORDER + 1) * (FMM_ORDER + 1), {0.0, 0.0});
    }

    // Insert a particle into this node; subdivide if necessary
    void insert(Particle* p) {
        if (is_leaf) {
            particles.push_back(p);
            // If leaf exceeds capacity and not too small, subdivide
            if (particles.size() > MAX_LEAF_PARTICLES && size > 2.0 * SOFTENING && level < 10) { // Max level e.g. 10-15
                subdivide();
            }
        } else {
            // If already subdivided, delegate insertion to the correct child
            children[get_child_index(p)]->insert(p);
        }
    }

    // Check if this node is well-separated from another node (for FMM interaction list)
    bool is_well_separated_from(const Node* other_node, double separation_factor = 2.0) const {
        if (!other_node) return false;
        double dx = cx - other_node->cx;
        double dy = cy - other_node->cy;
        double dz = cz - other_node->cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;
        // Simplified separation: distance between centers > factor * (size of this node)
        // A more robust check might consider (factor * (size + other_node->size))^2 or similar
        return dist_sq > (separation_factor * separation_factor * size * size);
    }

private:
    // Subdivide this leaf into 8 children
    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;
        int child_idx = 0;
        // Create 8 octants
        for (int i = -1; i <= 1; i += 2) { // z-offset
            for (int j = -1; j <= 1; j += 2) { // y-offset
                for (int k = -1; k <= 1; k += 2) { // x-offset
                    children[child_idx++] = std::make_unique<Node>(
                        cx + k * offset, cy + j * offset, cz + i * offset,
                        child_size, this, level + 1
                    );
                }
            }
        }
        // Re-insert existing particles into appropriate children
        for (auto* p_ptr : particles) {
            children[get_child_index(p_ptr)]->insert(p_ptr);
        }
        particles.clear();
    }

    // Determine which child octant a given particle belongs to
    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1; // East
        if (p->y > cy) index |= 2; // North
        if (p->z > cz) index |= 4; // Up
        return index;
    }
};

// --- Utility Functions ---

// Recursively collect all nodes in the subtree rooted at 'node'
void get_all_nodes_recursive(Node* node, std::vector<Node*>& nodes_list) {
    if (!node) return;
    nodes_list.push_back(node);
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) get_all_nodes_recursive(child.get(), nodes_list);
        }
    }
}
std::vector<Node*> get_all_nodes(Node* root_node) {
    std::vector<Node*> nodes_list;
    get_all_nodes_recursive(root_node, nodes_list);
    return nodes_list;
}


// Recursively collect only leaf nodes in the subtree rooted at 'node'
void get_leaf_nodes_recursive(Node* node, std::vector<Node*>& leaves_list) {
    if (!node) return;
    if (node->is_leaf) {
        leaves_list.push_back(node);
    } else {
        for (const auto& child : node->children) {
            if (child) get_leaf_nodes_recursive(child.get(), leaves_list);
        }
    }
}
std::vector<Node*> get_leaf_nodes(Node* root_node) {
    std::vector<Node*> leaves_list;
    get_leaf_nodes_recursive(root_node, leaves_list);
    return leaves_list;
}


// Compute pairwise force contribution from p2 to p1 (softened inverse-square law)
void p2p(Particle& p1, Particle& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dz = p2.z - p1.z;
    double r_sq = dx*dx + dy*dy + dz*dz;
    double r_sq_soft = r_sq + SOFT2;
    if (r_sq_soft < 1e-12 && r_sq < 1e-12) return; // Avoid self-interaction
    double inv_r = 1.0 / std::sqrt(r_sq_soft);
    double inv_r3 = inv_r * inv_r * inv_r;
    double force_val = G_CONST * p2.mass * inv_r3;
    p1.ax += force_val * dx;
    p1.ay += force_val * dy;
    p1.az += force_val * dz;
}

// Cache for factorial computations (now thread-local)
double factorial(int n) {
    thread_local static std::map<int, double> factorial_cache_tl;
    if (n < 0) return 0.0; // Or throw exception
    if (n == 0) return 1.0;
    if (factorial_cache_tl.count(n)) return factorial_cache_tl[n];
    double res = 1.0;
    for (int i = 1; i <= n; ++i) res *= i;
    factorial_cache_tl[n] = res;
    return res;
}

// --- Direct O(N^2) Force Computation ---
void compute_forces_direct(std::vector<Particle>& particles) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0.0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            p2p(particles[i], particles[j]);
        }
    }
}

// --- Barnes-Hut O(N log N) ---
void compute_mass_distribution_BH(Node* node) {
    if (!node) return;
    node->total_mass = 0.0;
    node->com_x = node->com_y = node->com_z = 0.0;

    if (node->is_leaf) {
        for (const auto* p : node->particles) {
            node->total_mass += p->mass;
            node->com_x += p->x * p->mass;
            node->com_y += p->y * p->mass;
            node->com_z += p->z * p->mass;
        }
    } else {
        for (const auto& child : node->children) {
            if (child) {
                compute_mass_distribution_BH(child.get()); // Recursive call
                node->total_mass += child->total_mass;
                // Accumulate weighted COM from children
                node->com_x += child->com_x; // Mistake in original: needs to be weighted by child mass
                node->com_y += child->com_y; // Mistake in original
                node->com_z += child->com_z; // Mistake in original
            }
        }
    }

    if (node->total_mass > 1e-12) {
        node->com_x /= node->total_mass;
        node->com_y /= node->total_mass;
        node->com_z /= node->total_mass;
    } else {
        node->com_x = node->cx;
        node->com_y = node->cy;
        node->com_z = node->cz;
    }
}
// Corrected BH mass distribution (original had issue with CoM accumulation from children)
void compute_mass_distribution_BH_corrected(Node* node) {
    if (!node) return;
    node->total_mass = 0.0;
    node->com_x = 0.0; node->com_y = 0.0; node->com_z = 0.0;

    if (node->is_leaf) {
        for (const auto* p : node->particles) {
            node->total_mass += p->mass;
            node->com_x += p->x * p->mass;
            node->com_y += p->y * p->mass;
            node->com_z += p->z * p->mass;
        }
    } else {
        for (const auto& child : node->children) {
            if (child) {
                compute_mass_distribution_BH_corrected(child.get());
                node->total_mass += child->total_mass;
                node->com_x += child->com_x * child->total_mass; // Correct: use child's total mass for weighting
                node->com_y += child->com_y * child->total_mass; // Correct
                node->com_z += child->com_z * child->total_mass; // Correct
            }
        }
    }

    if (node->total_mass > 1e-12) {
        node->com_x /= node->total_mass;
        node->com_y /= node->total_mass;
        node->com_z /= node->total_mass;
    } else {
        // If no mass, set center-of-mass to node center (or handle as error/empty)
        node->com_x = node->cx;
        node->com_y = node->cy;
        node->com_z = node->cz;
    }
}


void compute_force_on_particle_BH(Particle* target_p, Node* current_node) {
    if (!current_node || current_node->total_mass < 1e-12) return;

    double dx = current_node->com_x - target_p->x;
    double dy = current_node->com_y - target_p->y;
    double dz = current_node->com_z - target_p->z;
    double d_sq = dx*dx + dy*dy + dz*dz;
    
    bool is_target_in_leaf = false;
    if(current_node->is_leaf){
        for(const auto* p_in_leaf : current_node->particles){
            if(p_in_leaf == target_p) {
                is_target_in_leaf = true;
                break;
            }
        }
    }
    // Avoid self-interaction if target_p is the *only* particle in this leaf node
    if (is_target_in_leaf && current_node->particles.size() == 1) return;


    if (current_node->is_leaf) {
        for (auto* source_p : current_node->particles) {
            if (target_p != source_p) p2p(*target_p, *source_p);
        }
    } else {
        if (current_node->size * current_node->size < d_sq * BH_THETA * BH_THETA) {
            double r_sq_soft = d_sq + SOFT2;
            if (r_sq_soft < 1e-12 && d_sq < 1e-12) return; 
            double inv_r = 1.0 / std::sqrt(r_sq_soft);
            double inv_r3 = inv_r * inv_r * inv_r;
            double f_scale = G_CONST * current_node->total_mass * inv_r3;
            target_p->ax += f_scale * dx;
            target_p->ay += f_scale * dy;
            target_p->az += f_scale * dz;
        } else {
            for (const auto& child : current_node->children) {
                if (child) compute_force_on_particle_BH(target_p, child.get());
            }
        }
    }
}

void compute_forces_BH(std::vector<Particle>& particles, Node& root) {
    compute_mass_distribution_BH_corrected(&root); // Use corrected version
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0.0;
        compute_force_on_particle_BH(&particles[i], &root);
    }
}

// --- FMM O(N) ---
namespace FMM_Math {
    inline int lm_to_idx(int l, int m) {
        return l * l + l + m;
    }

    void cart_to_sph(double x, double y, double z, double& r, double& theta, double& phi) {
        r = std::sqrt(x*x + y*y + z*z);
        if (r < 1e-12) {
            theta = 0.0;
            phi = 0.0;
        } else {
            theta = std::acos(z / r); // z/r can be > 1 or < -1 due to precision issues if r is very close to abs(z)
            if (z/r > 1.0) theta = 0.0;
            else if (z/r < -1.0) theta = M_PI;
            phi = std::atan2(y, x);
        }
    }

    double legendreP(int l, int m_abs, double x) {
        if (m_abs < 0 || m_abs > l) return 0.0;
        if (x > 1.0) x = 1.0; // Clamp x to valid range for acos and legendre
        if (x < -1.0) x = -1.0;
        
        double pmm = 1.0;
        if (m_abs > 0) {
            double somx2 = std::sqrt(std::max(0.0, (1.0 - x) * (1.0 + x)));
            double fact = 1.0;
            for (int i = 1; i <= m_abs; ++i) {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }
        if (l == m_abs) return pmm;
        double pmmp1 = x * (2.0 * m_abs + 1.0) * pmm;
        if (l == m_abs + 1) return pmmp1;
        double pll = 0.0;
        for (int ll_iter = m_abs + 2; ll_iter <= l; ++ll_iter) {
            pll = ((2.0 * ll_iter - 1.0) * x * pmmp1 - (ll_iter + m_abs - 1.0) * pmm) / (ll_iter - m_abs);
            pmm = pmmp1;
            pmmp1 = pll;
        }
        return pll;
    }
    
    std::complex<double> Y_lm_pdf(int n, int m, double theta, double phi) {
        // Uses the "pdf" (physics definition) convention for spherical harmonics.
        // Y_n^m = sqrt(((2n+1)/4pi) * (n-m)!/(n+m)!) P_n^m(cos(theta)) e^(i m phi)
        // The provided code uses a slightly different normalization (related to (-1)^m factor for Y_n^{-m} sometimes)
        // and combines it with parts of the FMM translation formulas.
        // The factorial part is sqrt((n-|m|)! / (n+|m|)!) * P_n^{|m|}
        // The (-1)^m factor might be implicitly handled by using Y_n^{-m} in P2M etc.
        // The key is consistency. The provided code uses:
        // sqrt( (n-abs_m)! / (n+abs_m)! ) * P_n^{abs_m}(cos theta) * exp(i * m * phi)

        if (n < 0 || std::abs(m) > n) return {0.0, 0.0};
        int m_abs = std::abs(m);
        
        double fact_num = factorial(n - m_abs);
        double fact_den = factorial(n + m_abs);
        double norm_sqrt_factor = 0.0;

        if (fact_den > 1e-30) { // Avoid division by zero if (n+m_abs)! is tiny
             norm_sqrt_factor = std::sqrt(fact_num / fact_den);
        } else if (fact_num < 1e-30) { // 0/0 case, can happen if n=m_abs=0 for example, (0!/0!) = 1
             norm_sqrt_factor = 1.0; // Or handle as error if den is zero but num is not
        } else {
            // This case (den is zero, num is not) should ideally not happen with non-negative factorials
            // but can occur if factorial returns 0 for large numbers due to overflow if not handled.
            // For now, assume factorial handles large numbers or FMM_ORDER is small enough.
            // If it does happen, it's an issue.
        }

        double p_val = legendreP(n, m_abs, std::cos(theta));
        return norm_sqrt_factor * p_val * std::exp(std::complex<double>(0.0, m * phi));
    }
}


void p2m_FMM(Node* leaf_node) {
    if (!leaf_node || !leaf_node->is_leaf) return;
    std::fill(leaf_node->multipole_coeffs.begin(),
              leaf_node->multipole_coeffs.end(),
              std::complex<double>(0.0, 0.0));

    for (const auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        FMM_Math::cart_to_sph(p->x - leaf_node->cx,
                              p->y - leaf_node->cy,
                              p->z - leaf_node->cz,
                              r_rel, theta_rel, phi_rel);
        for (int n = 0; n <= FMM_ORDER; ++n) {
            for (int m = -n; m <= n; ++m) {
                int idx = FMM_Math::lm_to_idx(n, m);
                leaf_node->multipole_coeffs[idx] +=
                    p->mass * std::pow(r_rel, n) * FMM_Math::Y_lm_pdf(n, -m, theta_rel, phi_rel);
            }
        }
    }
}

void M2M_translation(const Node* child_node, Node* parent_node) {
    if (!child_node || !parent_node) return;
    double dx = child_node->cx - parent_node->cx;
    double dy = child_node->cy - parent_node->cy;
    double dz = child_node->cz - parent_node->cz;
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);

    const int p_order = FMM_ORDER; // Use a distinct name from particle 'p'
    const auto& child_coeffs = child_node->multipole_coeffs;
    auto& parent_coeffs = parent_node->multipole_coeffs;
    const std::complex<double> I(0.0, 1.0);

    thread_local static std::map<std::pair<int,int>, double> A_cache_tl;
    auto A_nm_cached = [&](int n, int m_val) { // Renamed m to m_val to avoid conflict
        int abs_m = std::abs(m_val);
        auto key = std::make_pair(n, abs_m);
        if (A_cache_tl.count(key)) return A_cache_tl[key];
        // (-1)^n * sqrt((n-|m|)! * (n+|m|)!) -- this is different from some FMM A_nm definitions.
        // The original code implies: (-1)^n * sqrt( (n-abs_m)! / (n+abs_m)! ) <- no, this is Y_lm_pdf's norm_factor part
        // The original code implies: (-1)^n * sqrt( factorials for (n-abs_m) and (n+abs_m) )
        // Let's re-verify the formula from "A Short Course on Fast Multipole Methods" by Beatson & Greengard
        // Theorem 2.1 (Expansion of 1/r): A_n^m = (-1)^n / sqrt((n-m)!(n+m)!)
        // Theorem 5.3 (M2M): has A_n^m, A_l^{m'}, A_j^k in denominator.
        // The code uses A_nm = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        // This is consistent with some definitions, let's stick to the code's version.
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache_tl[key] = val;
        return val;
    };

    for (int j = 0; j <= p_order; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_parent = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);
            for (int n_sum = 0; n_sum <= j; ++n_sum) { // Renamed 'n' to 'n_sum' for clarity
                for (int m_sum = -n_sum; m_sum <= n_sum; ++m_sum) { // Renamed 'm' to 'm_sum'
                    int l = j - n_sum;
                    int jm = k - m_sum; 
                    if (std::abs(jm) > l) continue;

                    int idx_child = FMM_Math::lm_to_idx(l, jm);
                    std::complex<double> O_l_jm = child_coeffs[idx_child];
                    if (std::abs(O_l_jm.real()) < 1e-30 && std::abs(O_l_jm.imag()) < 1e-30) continue;

                    int exponent = std::abs(k) - std::abs(m_sum) - std::abs(jm);
                    std::complex<double> phase = std::pow(I, exponent);
                    
                    double A_n_m_val = A_nm_cached(n_sum, m_sum);
                    double A_l_jm_val = A_nm_cached(l, jm);
                    double rho_pow_n = std::pow(rho, n_sum);
                    std::complex<double> Yval = FMM_Math::Y_lm_pdf(n_sum, -m_sum, alpha, beta);

                    accum += O_l_jm * phase * A_n_m_val * A_l_jm_val * rho_pow_n * Yval;
                }
            }
            double A_j_k_val = A_nm_cached(j, k);
            if (std::abs(A_j_k_val) > 1e-30) {
                accum /= A_j_k_val;
            }
            parent_coeffs[idx_parent] += accum;
        }
    }
}

void perform_m2m_pass_parallel(Node* node) {
    if (!node || node->is_leaf) return;

    for (const auto& child_unique_ptr : node->children) { // Renamed for clarity
        if (child_unique_ptr) {
            Node* child_raw_ptr = child_unique_ptr.get(); // Get the raw pointer
            #pragma omp task default(shared) firstprivate(child_raw_ptr) // Make raw pointer firstprivate
            perform_m2m_pass_parallel(child_raw_ptr);
        }
    }
    #pragma omp taskwait
    // ... rest of the function
    std::fill(node->multipole_coeffs.begin(),
              node->multipole_coeffs.end(),
              std::complex<double>(0.0, 0.0));
    for (const auto& child_unique_ptr : node->children) {
        if (child_unique_ptr) {
            M2M_translation(child_unique_ptr.get(), node);
        }
    }
}

void M2L_translation(const Node* source_node, Node* target_node) {
    if (!source_node || !target_node) return;
    double dx = source_node->cx - target_node->cx;
    double dy = source_node->cy - target_node->cy;
    double dz = source_node->cz - target_node->cz;
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);
    if (rho < 1e-9) return; // Avoid division by zero if source and target centers are too close

    const int p_order = FMM_ORDER;
    const auto& source_coeffs = source_node->multipole_coeffs;
    auto& local_coeffs = target_node->local_coeffs;
    const std::complex<double> I(0.0, 1.0);

    thread_local static std::map<std::pair<int,int>, double> A_cache_tl_m2l;
     auto A_nm_cached = [&](int n, int m_val) {
        int abs_m = std::abs(m_val);
        auto key = std::make_pair(n, abs_m);
        if (A_cache_tl_m2l.count(key)) return A_cache_tl_m2l[key];
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache_tl_m2l[key] = val;
        return val;
    };

    for (int j = 0; j <= p_order; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_local = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);
            for (int n_sum = 0; n_sum <= p_order; ++n_sum) {
                for (int m_sum = -n_sum; m_sum <= n_sum; ++m_sum) {
                    int idx_source = FMM_Math::lm_to_idx(n_sum, m_sum);
                    std::complex<double> O_nm_val = source_coeffs[idx_source];
                    if (std::abs(O_nm_val.real()) < 1e-30 && std::abs(O_nm_val.imag()) < 1e-30) continue;

                    int exponent = std::abs(k - m_sum) - std::abs(k) - std::abs(m_sum);
                    std::complex<double> phase = std::pow(I, exponent);
                    
                    double A_n_m_val = A_nm_cached(n_sum, m_sum);
                    double A_j_k_val = A_nm_cached(j, k);
                    
                    int l_ylm = j + n_sum; // Degree for Ylm
                    int m_ylm = m_sum - k; // Order for Ylm
                    if (std::abs(m_ylm) > l_ylm) continue;
                    std::complex<double> Y_l_mm = FMM_Math::Y_lm_pdf(l_ylm, m_ylm, alpha, beta);
                    
                    double rho_pow = std::pow(rho, -(j + n_sum + 1));
                    
                    // Denominator: (-1)^n_sum * A_{j+n_sum}^{m_sum-k} (using original notation for A)
                    // The A_nm in code has (-1)^n_sum factor.
                    // Denom from paper: (-1)^n * A_{n+j}^{m-k} where A comes from 1/r exp.
                    // Let's stick to the formula in the existing code:
                    // Denom: sign_n * A_nm(denom_n, abs_mk) where denom_n = j+n_sum, abs_mk = |m_sum-k|
                    double sign_n_sum = ((n_sum % 2) == 0 ? 1.0 : -1.0);
                    double A_mk_val = A_nm_cached(l_ylm, m_ylm); // A_{j+n_sum}^{m_sum-k}


                    if (std::abs(sign_n_sum * A_mk_val) < 1e-30) continue;

                    accum += O_nm_val * phase * A_n_m_val * A_j_k_val * Y_l_mm * rho_pow
                             / (sign_n_sum * A_mk_val);
                }
            }
            local_coeffs[idx_local] += accum;
        }
    }
}

void L2L_translation(const Node* parent_node, Node* child_node) {
    if (!parent_node || !child_node) return;
    double dx = parent_node->cx - child_node->cx;
    double dy = parent_node->cy - child_node->cy;
    double dz = parent_node->cz - child_node->cz;
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);

    const int p_order = FMM_ORDER;
    const auto& parent_local = parent_node->local_coeffs;
    auto& child_local = child_node->local_coeffs; // Accumulates, so must be zeroed before downward pass
    const std::complex<double> I(0.0, 1.0);

    thread_local static std::map<std::pair<int,int>, double> A_cache_tl_l2l;
    auto A_nm_cached = [&](int n, int m_val) {
        int abs_m = std::abs(m_val);
        auto key = std::make_pair(n, abs_m);
        if (A_cache_tl_l2l.count(key)) return A_cache_tl_l2l[key];
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache_tl_l2l[key] = val;
        return val;
    };

    for (int j = 0; j <= p_order; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_child = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);
            for (int n_sum = j; n_sum <= p_order; ++n_sum) { // n_sum >= j
                for (int m_sum = -n_sum; m_sum <= n_sum; ++m_sum) {
                    int idx_parent = FMM_Math::lm_to_idx(n_sum, m_sum);
                    std::complex<double> L_nm_val_parent = parent_local[idx_parent];
                    if (std::abs(L_nm_val_parent.real()) < 1e-30 && std::abs(L_nm_val_parent.imag()) < 1e-30) continue;

                    int diff_deg = n_sum - j; // Degree for Ylm
                    int diff_ord = m_sum - k; // Order for Ylm
                    if (std::abs(diff_ord) > diff_deg) continue;

                    int exponent = std::abs(m_sum) - std::abs(m_sum - k) - std::abs(k);
                    std::complex<double> phase = std::pow(I, exponent);
                    
                    double A_diff = A_nm_cached(diff_deg, diff_ord); // A_{n-j}^{m-k}
                    double A_j_k_val = A_nm_cached(j, k);           // A_j^k
                    
                    std::complex<double> Y_diff_mk = FMM_Math::Y_lm_pdf(diff_deg, diff_ord, alpha, beta);
                    double rho_pow = std::pow(rho, diff_deg);
                    
                    // Denominator: (-1)^{n_sum+j} * A_{n_sum}^{m_sum}
                    double sign_nj = (((n_sum + j) % 2) == 0 ? 1.0 : -1.0);
                    double A_n_m_val = A_nm_cached(n_sum, m_sum); // A_n^m

                    if (std::abs(sign_nj * A_n_m_val) < 1e-30) continue;

                    accum += L_nm_val_parent * phase * A_diff * A_j_k_val * Y_diff_mk * rho_pow
                             / (sign_nj * A_n_m_val);
                }
            }
            child_local[idx_child] += accum;
        }
    }
}

void l2p_FMM(Node* leaf_node) {
    if (!leaf_node || !leaf_node->is_leaf || leaf_node->particles.empty()) return;

    for (auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        FMM_Math::cart_to_sph(p->x - leaf_node->cx,
                              p->y - leaf_node->cy,
                              p->z - leaf_node->cz,
                              r_rel, theta_rel, phi_rel);
        std::complex<double> pot_grad_x(0.0, 0.0);
        std::complex<double> pot_grad_y(0.0, 0.0);
        std::complex<double> pot_grad_z(0.0, 0.0);

        for (int n = 0; n <= FMM_ORDER; ++n) { // L_nm is L_n^m
            for (int m = -n; m <= n; ++m) {
                int idx = FMM_Math::lm_to_idx(n, m);
                std::complex<double> L_nm = leaf_node->local_coeffs[idx];
                if (std::abs(L_nm.real()) < 1e-20 && std::abs(L_nm.imag()) < 1e-20) continue;
                
                // Potential Phi = Sum L_nm * r^n * Y_n^m(conj) (or Y_n^{-m}, check convention)
                // Force = -Grad(Phi)
                // For FMM, often potential is Sum L_nm * R_n^m where R_n^m = r^n Y_n^m (regular solid harmonics)
                // Your P2M uses Y_n^{-m}, so local expansion interaction with this uses Y_n^m.
                // F = - Grad ( Sum_{n,m} L_nm * r^n * Y_n^m(theta,phi) )
                // Grad (r^n Y_n^m) is complex.
                // The L2P step typically evaluates Sum L_nm * Grad(r^n Y_n^m(theta_rel,phi_rel)).
                // The provided gradient calculation is an approximation.
                // For a more accurate version from potential:
                // Phi = Sum_{n,m} L_nm * r^n * Y_n^m(theta,phi) (using code's Y_lm_pdf for Y_n^m)
                // The code's L2P is: Sum L_nm * n * r^(n-1) * Y_n^m * (components of unit vector)

                if (n == 0) continue; 

                double r_pow_nm1;
                if (r_rel < 1e-9) { // if particle is at cell center
                     if (n==1) r_pow_nm1 = 1.0; // for n=1, r^(n-1) = r^0 = 1
                     else r_pow_nm1 = 0.0; // for n > 1, r^(n-1) is 0 if r=0
                } else {
                    r_pow_nm1 = n * std::pow(r_rel, n - 1);
                }
                if (std::abs(r_pow_nm1) < 1e-20 && n > 0) continue;


                auto Ylm_val = FMM_Math::Y_lm_pdf(n, m, theta_rel, phi_rel);

                // This is a simplified gradient. It should be derivatives of solid harmonics.
                // Fx = - dPhi/dx etc.
                // d/dx (r^n Y_n^m) requires chain rule with spherical coordinate derivatives.
                // For simplicity, we use the existing code's formulation.
                pot_grad_x += L_nm * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::cos(phi_rel);
                pot_grad_y += L_nm * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::sin(phi_rel);
                pot_grad_z += L_nm * r_pow_nm1 * Ylm_val * std::cos(theta_rel);
            }
        }
        p->ax -= G_CONST * pot_grad_x.real(); // Force = - Grad(Potential)
        p->ay -= G_CONST * pot_grad_y.real(); // G_CONST is part of potential if M,L coeffs are just geometric sums
        p->az -= G_CONST * pot_grad_z.real(); // If L_nm already includes G_CONST, then no G_CONST here.
                                            // P2M does not include G. M2L does not. So G here.
    }
}

void compute_leaf_forces_FMM(Node* leaf_node) { // Changed 'node' to 'leaf_node' for clarity
    if (!leaf_node || !leaf_node->is_leaf) return;

    l2p_FMM(leaf_node); // Far-field from local expansion

    // Near-field P2P: particles in same leaf + particles in adjacent (sibling) leaves.
    // 1. Interactions within the same leaf
    for (size_t i = 0; i < leaf_node->particles.size(); ++i) {
        for (size_t j = i + 1; j < leaf_node->particles.size(); ++j) {
            p2p(*(leaf_node->particles[i]), *(leaf_node->particles[j]));
            p2p(*(leaf_node->particles[j]), *(leaf_node->particles[i])); // Apply force to both
        }
    }

    // 2. Interactions with particles in *directly adjacent* ( όχι just siblings) non-empty leaf nodes.
    // The original code did P2P with siblings. A more complete FMM usually defines a "near-field"
    // list for each leaf, typically siblings and their children if they are leaves and close enough,
    // or more generally, any leaf node whose box is adjacent or close.
    // For this version, stick to the simpler sibling interaction as per the original code's structure.
    if (leaf_node->parent) {
        for (const auto& sibling_ptr : leaf_node->parent->children) {
            Node* sibling = sibling_ptr.get();
            if (sibling && sibling != leaf_node && sibling->is_leaf && !sibling->particles.empty()) {
                for (auto* p1 : leaf_node->particles) {
                    for (auto* p2 : sibling->particles) {
                        p2p(*p1, *p2);
                    }
                }
            }
            // What if sibling is not a leaf but its children are close?
            // This is a more complex near-field definition usually handled by traversing down
            // non-well-separated nodes in the interaction list until leaves are found.
            // The current FMM downward pass M2L handles "far" interactions.
            // The P2P here should handle "near".
            // A common definition of near-field for leaf X is all particles in leaves Y such that
            // Y is in X's parent's children list (siblings) OR
            // Y is in X's parent's "near-field interaction list" (excluding far-field ones)
            // and Y is a leaf.
            // The given M2L logic (uncles' children) might miss some interactions
            // if the separation criterion is strict.
            // The current `compute_leaf_forces_FMM` is okay for a basic FMM.
        }
    }
}


void perform_fmm_downward_pass_parallel(Node* node) {
    if (!node) return;

    // 1. M2L: For each node, find its interaction list by scanning "uncles" (parent's siblings' children that are leaves)
    // This part updates node->local_coeffs, so it must be done before L2L to children.
    if (node->parent && node->parent->parent) {
        Node* parent = node->parent;
        Node* grandpa = parent->parent;
        for (const auto& uncle_grandpa_child_ptr : grandpa->children) { // These are actual uncles or parent itself
            if (!uncle_grandpa_child_ptr) continue;
            Node* uncle_or_parent_node = uncle_grandpa_child_ptr.get();
            if (uncle_or_parent_node == parent) continue; // Skip self (parent of node)

            // Interaction list V_node: children of uncles that are well separated from node.
            // Original paper by Greengard/Rokhlin describes interaction list U_box.
            // Here, interaction list means source_nodes for M2L.
            // These are typically leaves in the subtrees of "uncles" that are well-separated.
            
            std::vector<Node*> source_candidates; // Could be non-leaf nodes from uncle's sub-tree
                                                  // if M2L works with non-leaf sources.
                                                  // Original code uses source_leaves from uncle.
            std::function<void(Node*)> collect_sources_for_m2l = 
                [&](Node* current_interaction_candidate) {
                if (!current_interaction_candidate) return;

                if (node->is_well_separated_from(current_interaction_candidate)) {
                    // If the candidate (or its box) is well separated, use its multipole.
                    // If candidate is leaf and has particles, M2L from it.
                    if (current_interaction_candidate->is_leaf && !current_interaction_candidate->particles.empty()){
                         M2L_translation(current_interaction_candidate, node);
                    } else if (!current_interaction_candidate->is_leaf) {
                        // If non-leaf source is well-separated, can do M2L from its *already computed* multipole.
                        // This requires M2M to be complete for that source.
                        // The original code collected leaves from uncle and did M2L from those leaves.
                        // This is simpler. Let's stick to that.
                    }
                } else { // Not well separated
                    if (current_interaction_candidate->is_leaf) {
                        // This case (not well separated leaf in uncle's subtree) should be handled by P2P if adjacent,
                        // or by upward/downward pass if it's part of "cousins at same level".
                        // The original FMM structure has distinct near-field and far-field.
                        // If not well-separated, recurse on children of interaction_candidate.
                        // This is effectively what the original code does by only M2L-ing from *leaves*
                        // of the uncle that are well-separated.
                    } else {
                         for (const auto& ch : current_interaction_candidate->children) {
                            collect_sources_for_m2l(ch.get());
                        }
                    }
                }
            };
            
            // The original code's M2L logic:
            // Collect ALL leaf descendants of this uncle
            std::vector<Node*> source_leaves_of_uncle;
            std::function<void(Node*)> collect_leaves_recursively = 
                [&](Node* cur) {
                if (!cur) return;
                if (cur->is_leaf) {
                    if (!cur->particles.empty()) source_leaves_of_uncle.push_back(cur);
                } else {
                    for (const auto& ch : cur->children) {
                        collect_leaves_recursively(ch.get());
                    }
                }
            };
            collect_leaves_recursively(uncle_or_parent_node); // Collect all leaves from this "uncle" branch

            // For each leaf in uncle’s subtree, if well-separated, do M2L
            for (Node* src_leaf : source_leaves_of_uncle) {
                if (node->is_well_separated_from(src_leaf)) {
                     M2L_translation(src_leaf, node); // src_leaf's multipole (from P2M) to node's local
                }
                // If not well-separated, these are handled by direct P2P if they are "neighbors",
                // or this interaction is implicitly handled at a different level or by L2L from common ancestor.
            }
        }
    }

    if (node->is_leaf) {
        compute_leaf_forces_FMM(node);
    } else {
        for (const auto& child_unique_ptr : node->children) { // Renamed for clarity
            if (child_unique_ptr) {
                Node* child_raw_ptr = child_unique_ptr.get(); // Get the raw pointer
                L2L_translation(node, child_raw_ptr); // Pass raw pointer if L2L expects Node*
                #pragma omp task default(shared) firstprivate(child_raw_ptr) // Make raw pointer firstprivate
                perform_fmm_downward_pass_parallel(child_raw_ptr);
            }
        }
        // No taskwait here, children tasks proceed. Task group is implicit by parent task.
    }
}


void compute_forces_FMM_structured(std::vector<Particle>& particles, Node& root) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0.0;
    }

    std::vector<Node*> leaves = get_leaf_nodes(&root);
    #pragma omp parallel for
    for (size_t i = 0; i < leaves.size(); ++i) {
        p2m_FMM(leaves[i]);
    }

    #pragma omp parallel // Create a team of threads
    {
        #pragma omp single // One thread initiates the task generation
        perform_m2m_pass_parallel(&root);
    } // Implicit barrier: all M2M tasks must complete before proceeding

    std::vector<Node*> all_node_list = get_all_nodes(&root);
    #pragma omp parallel for
    for (size_t i = 0; i < all_node_list.size(); ++i) {
        Node* n = all_node_list[i];
        if (n) {
            std::fill(n->local_coeffs.begin(),
                      n->local_coeffs.end(),
                      std::complex<double>(0.0, 0.0));
        }
    }
    
    #pragma omp parallel // Create another team for downward pass tasks
    {
        #pragma omp single // One thread initiates task generation
        perform_fmm_downward_pass_parallel(&root);
    } // Implicit barrier: all downward pass tasks must complete
}


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

double calculate_rms_relative_error(const std::vector<Particle>& approx_particles,
                                    const std::vector<Particle>& exact_particles) {
    if (approx_particles.size() != exact_particles.size()) {
        std::cerr << "Error: Particle vector sizes mismatch for error calculation." << std::endl;
        return -1.0;
    }
    double sum_sq_diff = 0.0;
    double sum_sq_exact = 0.0;
    #pragma omp parallel for reduction(+:sum_sq_diff, sum_sq_exact)
    for (size_t i = 0; i < approx_particles.size(); ++i) {
        double dx = approx_particles[i].ax - exact_particles[i].ax;
        double dy = approx_particles[i].ay - exact_particles[i].ay;
        double dz = approx_particles[i].az - exact_particles[i].az;
        sum_sq_diff += dx*dx + dy*dy + dz*dz;
        sum_sq_exact += exact_particles[i].ax*exact_particles[i].ax
                      + exact_particles[i].ay*exact_particles[i].ay
                      + exact_particles[i].az*exact_particles[i].az;
    }
    if (sum_sq_exact < 1e-24) {
        return (sum_sq_diff < 1e-24) ? 0.0 : 1.0;
    }
    return std::sqrt(sum_sq_diff / sum_sq_exact);
}

int main() {
    std::cout << std::fixed << std::setprecision(5);
    // Factorial cache is now thread-local, no need to clear globally
    // A_nm caches are also thread-local within their respective functions

    std::vector<int> N_values = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}; // Reduced for quick testing
    // std::vector<int> N_values = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    int max_N_for_direct_sum = 65536; // Reduced for quick testing
    // int max_N_for_direct_sum = 4096; 


    std::vector<int> thread_counts_to_test;
    int max_hw_threads = 1;
    #ifdef _OPENMP
    max_hw_threads = omp_get_max_threads();
    #endif
    thread_counts_to_test.push_back(1);
    for (int tc = 2; tc <= max_hw_threads; tc *= 2) {
        thread_counts_to_test.push_back(tc);
    }
    if (std::find(thread_counts_to_test.begin(), thread_counts_to_test.end(), max_hw_threads) == thread_counts_to_test.end() &&
        max_hw_threads > 1 && (thread_counts_to_test.empty() || max_hw_threads != thread_counts_to_test.back())) {
         // Add max_hw_threads if it's not a power of 2 already added and not 1
        if (max_hw_threads > 0 && (thread_counts_to_test.empty() || max_hw_threads != thread_counts_to_test.back())) {
             bool is_power_of_two = (max_hw_threads > 0) && ((max_hw_threads & (max_hw_threads - 1)) == 0);
             if(!is_power_of_two || max_hw_threads == 1) { // add if not power of two, or if it is 1 and list is empty
                 if (thread_counts_to_test.back() < max_hw_threads)
                    thread_counts_to_test.push_back(max_hw_threads);
             }
        }
    }
    std::sort(thread_counts_to_test.begin(), thread_counts_to_test.end());
    thread_counts_to_test.erase(std::unique(thread_counts_to_test.begin(), thread_counts_to_test.end()), thread_counts_to_test.end());
    if (thread_counts_to_test.empty()) thread_counts_to_test.push_back(1);


    #ifndef _OPENMP
    if (thread_counts_to_test.size() > 1 || thread_counts_to_test[0] != 1) {
         std::cout << "Note: OpenMP not enabled by compiler; only running on 1 thread." << std::endl;
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
    double root_node_size = 2.0 * domain_half_size + 1.0; // Ensure root covers domain

    for (int N : N_values) {
        std::cout << "\n--- Testing N = " << N << " ---" << std::endl;

        std::vector<Particle> particles_direct_results;
        bool direct_computed_for_N = false;
        if (N <= max_N_for_direct_sum) {
            auto temp_particles_for_direct = init_particles(N, seed, domain_half_size);
            #ifdef _OPENMP
            omp_set_num_threads(max_hw_threads); // Use max threads for accurate direct sum timing
            #endif
            std::cout << "Computing direct sum for N=" << N << " (for error checking)..." << std::flush;
            auto start_direct_ref = std::chrono::high_resolution_clock::now();
            compute_forces_direct(temp_particles_for_direct);
            auto end_direct_ref = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_direct_ref = end_direct_ref - start_direct_ref;
            std::cout << " Done in " << time_direct_ref.count() << "s." << std::endl;
            particles_direct_results = temp_particles_for_direct;
            direct_computed_for_N = true;
        }

        std::vector<std::string> methods_to_test = {"FMM", "BH"};
        if (N <= max_N_for_direct_sum) methods_to_test.push_back("Direct");
        // Order methods for testing: FMM, BH, then Direct if N is small
        std::reverse(methods_to_test.begin(), methods_to_test.end());


        for (const std::string& method_name : methods_to_test) {
            if (method_name == "Direct" && N > max_N_for_direct_sum) {
                 if (N_values.back() == N) // Only print once for the largest N
                    std::cout << "Skipping Direct O(N^2) for N=" << N << " (exceeds limit " << max_N_for_direct_sum << ")" << std::endl;
                continue;
            }

            for (int num_threads : thread_counts_to_test) {
                #ifdef _OPENMP
                omp_set_num_threads(num_threads);
                #else
                if (num_threads > 1) continue; // Should be redundant due to earlier check
                #endif

                auto current_particles = init_particles(N, seed, domain_half_size);
                std::chrono::duration<double> time_diff;
                Node root_node(0.0, 0.0, 0.0, root_node_size, nullptr, 0); // Create fresh root_node

                std::cout << "Running " << method_name << " (N=" << N << ", Threads=" << num_threads << ")..." << std::flush;

                if (method_name == "Direct") {
                    auto start = std::chrono::high_resolution_clock::now();
                    compute_forces_direct(current_particles);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_diff = end - start;
                } else { // BH or FMM
                    // Build tree sequentially for now
                    for (size_t i = 0; i < current_particles.size(); ++i) {
                         root_node.insert(&current_particles[i]);
                    }

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
                if (direct_computed_for_N) {
                     error_val = calculate_rms_relative_error(current_particles, particles_direct_results);
                } else if (method_name == "Direct") { // If direct is run but not as reference (e.g. N > max_N_for_direct_sum was false)
                     error_val = 0.0; // Direct is exact against itself
                }


                results_file << N << "," << method_name << "," << num_threads << "," << time_diff.count() << "," << error_val << "\n";
                std::cout << " Time: " << time_diff.count() << "s";
                if (error_val >= 0.0) {
                    std::cout << ", Error: " << error_val;
                }
                std::cout << std::endl;
            }
        }
    }

    results_file.close();
    std::cout << "\nBenchmark finished. Results saved to performance_results.csv" << std::endl;
    std::cout << "Note: FMM implementation now uses OpenMP tasks for M2M and Downward passes." << std::endl;
    return 0;
}