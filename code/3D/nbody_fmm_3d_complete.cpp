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
            if (particles.size() > MAX_LEAF_PARTICLES && size > 2.0 * SOFTENING && level < 10) {
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
        // Simplified separation: distance between centers > factor * (size)
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
        for (int i = -1; i <= 1; i += 2) {
            for (int j = -1; j <= 1; j += 2) {
                for (int k = -1; k <= 1; k += 2) {
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
        if (p->x > cx) index |= 1;
        if (p->y > cy) index |= 2;
        if (p->z > cz) index |= 4;
        return index;
    }
};

// --- Utility Functions ---

// Recursively collect all nodes in the subtree rooted at 'node'
void get_all_nodes(Node* node, std::vector<Node*>& nodes_list) {
    if (!node) return;
    nodes_list.push_back(node);
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) get_all_nodes(child.get(), nodes_list);
        }
    }
}

// Recursively collect only leaf nodes in the subtree rooted at 'node'
void get_leaf_nodes(Node* node, std::vector<Node*>& leaves_list) {
    if (!node) return;
    if (node->is_leaf) {
        leaves_list.push_back(node);
    } else {
        for (const auto& child : node->children) {
            if (child) get_leaf_nodes(child.get(), leaves_list);
        }
    }
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

// Cache for factorial computations
std::map<int, double> factorial_cache;
double factorial(int n) {
    if (n < 0) return 0.0;
    if (n == 0) return 1.0;
    if (factorial_cache.count(n)) return factorial_cache[n];
    double res = 1.0;
    for (int i = 1; i <= n; ++i) res *= i;
    factorial_cache[n] = res;
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

// Compute total mass and center-of-mass for each node (post-order traversal)
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
                compute_mass_distribution_BH(child.get());
                node->total_mass += child->total_mass;
                node->com_x += child->com_x * child->total_mass;
                node->com_y += child->com_y * child->total_mass;
                node->com_z += child->com_z * child->total_mass;
            }
        }
    }

    if (node->total_mass > 1e-12) {
        node->com_x /= node->total_mass;
        node->com_y /= node->total_mass;
        node->com_z /= node->total_mass;
    } else {
        // If no mass, set center-of-mass to node center
        node->com_x = node->cx;
        node->com_y = node->cy;
        node->com_z = node->cz;
    }
}

// Recursively compute force on a single particle using Barnes-Hut criterion
void compute_force_on_particle_BH(Particle* target_p, Node* current_node) {
    if (!current_node || current_node->total_mass < 1e-12) return;

    double dx = current_node->com_x - target_p->x;
    double dy = current_node->com_y - target_p->y;
    double dz = current_node->com_z - target_p->z;
    double d_sq = dx*dx + dy*dy + dz*dz;

    // Check if current_node is exactly the leaf containing target_p
    bool is_self_node = current_node->is_leaf &&
                        current_node->particles.size() == 1 &&
                        current_node->particles[0] == target_p;
    if (is_self_node) return;

    // If leaf, do direct P2P for particles inside this leaf
    if (current_node->is_leaf) {
        for (auto* source_p : current_node->particles) {
            if (target_p != source_p) p2p(*target_p, *source_p);
        }
    } else {
        // If node is sufficiently far (size / distance < theta), approximate
        if (current_node->size * current_node->size < d_sq * BH_THETA * BH_THETA) {
            double r_sq_soft = d_sq + SOFT2;
            if (r_sq_soft < 1e-12) return;
            double inv_r = 1.0 / std::sqrt(r_sq_soft);
            double inv_r3 = inv_r * inv_r * inv_r;
            double f_scale = G_CONST * current_node->total_mass * inv_r3;
            target_p->ax += f_scale * dx;
            target_p->ay += f_scale * dy;
            target_p->az += f_scale * dz;
        } else {
            // Otherwise, recurse into children
            for (const auto& child : current_node->children) {
                if (child) compute_force_on_particle_BH(target_p, child.get());
            }
        }
    }
}

// Compute forces on all particles using Barnes-Hut
void compute_forces_BH(std::vector<Particle>& particles, Node& root) {
    compute_mass_distribution_BH(&root);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = particles[i].ay = particles[i].az = 0.0;
        compute_force_on_particle_BH(&particles[i], &root);
    }
}

// --- FMM O(N) ---

namespace FMM_Math {
    // Map (l,m) to linear index: idx = l^2 + l + m
    inline int lm_to_idx(int l, int m) {
        return l * l + l + m;
    }

    // Convert Cartesian coordinates (x,y,z) to spherical (r,theta,phi)
    void cart_to_sph(double x, double y, double z, double& r, double& theta, double& phi) {
        r = std::sqrt(x*x + y*y + z*z);
        if (r < 1e-12) {
            theta = 0.0;
            phi = 0.0;
        } else {
            theta = std::acos(z / r);
            phi = std::atan2(y, x);
        }
    }

    // Compute associated Legendre polynomial P_l^m_abs(x)
    double legendreP(int l, int m_abs, double x) {
        if (m_abs < 0 || m_abs > l) return 0.0;
        if (x > 1.0) x = 1.0;
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

    // Compute spherical harmonic Y_l^m(theta, phi) in condensed form (unnormalized times phase)
    std::complex<double> Y_lm_pdf(int n, int m, double theta, double phi) {
        if (n < 0 || std::abs(m) > n) return {0.0, 0.0};
        int m_abs = std::abs(m);
        // Compute normalization factor sqrt((n - m)!(n + m)! )
        double num = factorial(n - m_abs);
        double den = factorial(n + m_abs);
        double norm_factor = 0.0;
        if (den > 1e-20) {
            norm_factor = std::sqrt(num / den);
        } else if (num < 1e-20) {
            norm_factor = 1.0;
        }
        double p_val = legendreP(n, m_abs, std::cos(theta));
        // Y_l^m = norm_factor * P_l^m(cos(theta)) * e^{i m phi}
        return norm_factor * p_val * std::exp(std::complex<double>(0.0, m * phi));
    }
}

// ----------------------------------------------------------------------------
// STEP 1: P2M (Particle-to-Multipole): accumulate each leaf’s particles into its
// multipole_coeffs. Only 0-th order (monopole) plus higher orders.
// ----------------------------------------------------------------------------
void p2m_FMM(Node* leaf_node) {
    if (!leaf_node || !leaf_node->is_leaf) return;
    // Zero out multipole coefficients first
    std::fill(leaf_node->multipole_coeffs.begin(),
              leaf_node->multipole_coeffs.end(),
              std::complex<double>(0.0, 0.0));

    // For each particle in the leaf, add its contribution
    for (const auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        // Compute relative coordinates from leaf center to particle
        FMM_Math::cart_to_sph(p->x - leaf_node->cx,
                              p->y - leaf_node->cy,
                              p->z - leaf_node->cz,
                              r_rel, theta_rel, phi_rel);
        // For each multipole order (n,m)
        for (int n = 0; n <= FMM_ORDER; ++n) {
            for (int m = -n; m <= n; ++m) {
                int idx = FMM_Math::lm_to_idx(n, m);
                // Contribution: mass * r_rel^n * Y_n^{-m}(theta_rel, phi_rel)
                leaf_node->multipole_coeffs[idx] +=
                    p->mass * std::pow(r_rel, n) * FMM_Math::Y_lm_pdf(n, -m, theta_rel, phi_rel);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// STEP 2: M2M (Multipole-to-Multipole Translation)
// Merge each child’s multipole expansion into parent’ multipole expansion.
// Implements Theorem 5.3 from “A Short Course on Fast Multipole Methods”.
// ----------------------------------------------------------------------------
void M2M_translation(const Node* child_node, Node* parent_node) {
    if (!child_node || !parent_node) return;

    // Compute relative displacement from parent center to child center
    double dx = child_node->cx - parent_node->cx;
    double dy = child_node->cy - parent_node->cy;
    double dz = child_node->cz - parent_node->cz;

    // Convert displacement to spherical coordinates (rho, alpha, beta)
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);

    const int p = FMM_ORDER;
    const auto& child_coeffs = child_node->multipole_coeffs;
    auto& parent_coeffs = parent_node->multipole_coeffs;

    const std::complex<double> I(0.0, 1.0);

    // Cache A_n^m = (-1)^n * sqrt((n - |m|)! * (n + |m|)!)
    static std::map<std::pair<int,int>, double> A_cache;
    auto A_nm = [&](int n, int m) {
        int abs_m = std::abs(m);
        auto key = std::make_pair(n, abs_m);
        if (A_cache.count(key)) return A_cache[key];
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache[key] = val;
        return val;
    };

    // Loop over parent indices (j,k)
    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_parent = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);

            // Sum contributions from child multipole coefficients O_{l}^{(k-m)},
            // where l = j - n and m runs from -n..n
            for (int n = 0; n <= j; ++n) {
                for (int m = -n; m <= n; ++m) {
                    int l = j - n;
                    int jm = k - m;  // m' = k - m
                    if (jm < -l || jm > l) continue; // out of valid range

                    int idx_child = FMM_Math::lm_to_idx(l, jm);
                    std::complex<double> O_l_jm = child_coeffs[idx_child];
                    if (std::abs(O_l_jm) < 1e-30) continue;

                    // Phase factor i^{|k| - |m| - |jm|}
                    int exponent = std::abs(k) - std::abs(m) - std::abs(jm);
                    std::complex<double> phase = std::pow(I, exponent);

                    // Compute A_n^m and A_{l}^{jm}
                    double A_n_m = A_nm(n, m);
                    double A_l_jm = A_nm(l, jm);

                    // Compute rho^n
                    double rho_pow_n = std::pow(rho, n);

                    // Compute Y_n^{-m}(alpha, beta)
                    std::complex<double> Yval = FMM_Math::Y_lm_pdf(n, -m, alpha, beta);

                    // Accumulate to parent's coefficient
                    accum += O_l_jm * phase * A_n_m * A_l_jm * rho_pow_n * Yval;
                }
            }

            // Divide by A_j^k
            double A_j_k = A_nm(j, k);
            if (std::abs(A_j_k) > 1e-30) {
                accum /= A_j_k;
            }

            parent_coeffs[idx_parent] += accum;
        }
    }
}

// Recursively perform M2M on the subtree: post-order traversal
void perform_m2m_pass(Node* node) {
    if (!node || node->is_leaf) return;
    // First process children
    for (const auto& child : node->children) {
        if (child) {
            perform_m2m_pass(child.get());
        }
    }
    // Zero out this node's multipole_coeffs
    std::fill(node->multipole_coeffs.begin(),
              node->multipole_coeffs.end(),
              std::complex<double>(0.0, 0.0));
    // Accumulate from children using M2M_translation
    for (const auto& child : node->children) {
        if (child) {
            M2M_translation(child.get(), node);
        }
    }
}

// ----------------------------------------------------------------------------
// STEP 3: M2L (Multipole-to-Local Translation)
// Translate a well-separated source multipole expansion into a target’s local expansion.
// Implements Theorem 5.4 from “A Short Course on Fast Multipole Methods”.
// ----------------------------------------------------------------------------
void M2L_translation(const Node* source_node, Node* target_node) {
    if (!source_node || !target_node) return;

    // Displacement from target center to source center
    double dx = source_node->cx - target_node->cx;
    double dy = source_node->cy - target_node->cy;
    double dz = source_node->cz - target_node->cz;

    // Convert displacement to spherical (rho, alpha, beta)
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);

    const int p = FMM_ORDER;
    const auto& source_coeffs = source_node->multipole_coeffs;
    auto& local_coeffs = target_node->local_coeffs;
    const std::complex<double> I(0.0, 1.0);

    // Cache A_n^m = (-1)^n * sqrt((n - |m|)! * (n + |m|)!)
    static std::map<std::pair<int,int>, double> A_cache;
    auto A_nm = [&](int n, int m) {
        int abs_m = std::abs(m);
        auto key = std::make_pair(n, abs_m);
        if (A_cache.count(key)) return A_cache[key];
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache[key] = val;
        return val;
    };

    // Loop over target local indices (j,k)
    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_local = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);

            // Sum over source multipole indices (n,m)
            for (int n = 0; n <= p; ++n) {
                for (int m = -n; m <= n; ++m) {
                    int idx_source = FMM_Math::lm_to_idx(n, m);
                    std::complex<double> O_nm_val = source_coeffs[idx_source];
                    if (std::abs(O_nm_val) < 1e-30) continue;

                    // Phase: i^{|k - m| - |k| - |m|}
                    int exponent = std::abs(k - m) - std::abs(k) - std::abs(m);
                    std::complex<double> phase = std::pow(I, exponent);

                    // A_n^m and A_j^k
                    double A_n_m = A_nm(n, m);
                    double A_j_k = A_nm(j, k);

                    // Compute Y_{j+n}^{m-k}(alpha, beta)
                    int l = j + n;
                    int mm = m - k;
                    if (std::abs(mm) > l) continue;
                    std::complex<double> Y_l_mm = FMM_Math::Y_lm_pdf(l, mm, alpha, beta);

                    // rho^{-(j+n+1)}
                    double rho_pow = std::pow(rho, -(j + n + 1));

                    // Denominator: (-1)^n * A_{|m-k|}^{j+n}
                    int abs_mk = std::abs(m - k);
                    int denom_n = j + n;
                    double A_mk = A_nm(denom_n, abs_mk);
                    double sign_n = ((n % 2) == 0 ? 1.0 : -1.0);

                    accum += O_nm_val * phase * A_n_m * A_j_k * Y_l_mm * rho_pow
                             / (sign_n * A_mk);
                }
            }

            local_coeffs[idx_local] += accum;
        }
    }
}

// ----------------------------------------------------------------------------
// STEP 4: L2L (Local-to-Local Translation)
// Translate a parent’s local expansion to its child node.
// Implements Theorem 5.5 from “A Short Course on Fast Multipole Methods”.
// ----------------------------------------------------------------------------
void L2L_translation(const Node* parent_node, Node* child_node) {
    if (!parent_node || !child_node) return;

    // Displacement from child center to parent center
    double dx = parent_node->cx - child_node->cx;
    double dy = parent_node->cy - child_node->cy;
    double dz = parent_node->cz - child_node->cz;

    // Convert displacement to spherical (rho, alpha, beta)
    double rho, alpha, beta;
    FMM_Math::cart_to_sph(dx, dy, dz, rho, alpha, beta);

    const int p = FMM_ORDER;
    const auto& parent_local = parent_node->local_coeffs;
    auto& child_local = child_node->local_coeffs;
    const std::complex<double> I(0.0, 1.0);

    // Cache A_n^m = (-1)^n * sqrt((n - |m|)! * (n + |m|)!)
    static std::map<std::pair<int,int>, double> A_cache;
    auto A_nm = [&](int n, int m) {
        int abs_m = std::abs(m);
        auto key = std::make_pair(n, abs_m);
        if (A_cache.count(key)) return A_cache[key];
        double val = std::pow(-1.0, n) * std::sqrt(factorial(n - abs_m) * factorial(n + abs_m));
        A_cache[key] = val;
        return val;
    };

    // Loop over child local indices (j,k)
    for (int j = 0; j <= p; ++j) {
        for (int k = -j; k <= j; ++k) {
            int idx_child = FMM_Math::lm_to_idx(j, k);
            std::complex<double> accum(0.0, 0.0);

            // Sum over parent local indices (n,m) where n >= j
            for (int n = j; n <= p; ++n) {
                for (int m = -n; m <= n; ++m) {
                    int idx_parent = FMM_Math::lm_to_idx(n, m);
                    std::complex<double> O_nm_val = parent_local[idx_parent];
                    if (std::abs(O_nm_val) < 1e-30) continue;

                    int diff = n - j;
                    int mk = m - k;
                    if (std::abs(mk) > diff) continue;

                    // Phase: i^{|m| - |m-k| - |k|}
                    int exponent = std::abs(m) - std::abs(m - k) - std::abs(k);
                    std::complex<double> phase = std::pow(I, exponent);

                    // A_{m-k}^{n-j} and A_j^k
                    double A_mk = A_nm(diff, mk);
                    double A_j_k = A_nm(j, k);

                    // Y_{n-j}^{m-k}(alpha, beta)
                    std::complex<double> Y_diff_mk = FMM_Math::Y_lm_pdf(diff, mk, alpha, beta);

                    // rho^{n-j}
                    double rho_pow = std::pow(rho, diff);

                    // Denominator: (-1)^{n+j} * A_n^m
                    double sign_nj = (((n + j) % 2) == 0 ? 1.0 : -1.0);
                    double A_n_m = A_nm(n, m);

                    accum += O_nm_val * phase * A_mk * A_j_k * Y_diff_mk * rho_pow
                             / (sign_nj * A_n_m);
                }
            }

            child_local[idx_child] += accum;
        }
    }
}

// ----------------------------------------------------------------------------
// STEP 5: L2P (Local-to-Particle) + Near-field P2P Direct Interaction
// Evaluate local expansion at each leaf’s particles and compute near-field direct P2P.
// ----------------------------------------------------------------------------
void l2p_FMM(Node* leaf_node) {
    if (!leaf_node || !leaf_node->is_leaf) return;

    // For each particle in this leaf, evaluate local expansion gradient to get acceleration
    for (auto* p : leaf_node->particles) {
        double r_rel, theta_rel, phi_rel;
        // Relative coordinates from leaf center to particle
        FMM_Math::cart_to_sph(p->x - leaf_node->cx,
                              p->y - leaf_node->cy,
                              p->z - leaf_node->cz,
                              r_rel, theta_rel, phi_rel);
        std::complex<double> pot_grad_x(0.0, 0.0);
        std::complex<double> pot_grad_y(0.0, 0.0);
        std::complex<double> pot_grad_z(0.0, 0.0);

        // Sum contributions from each local coefficient (n,m)
        for (int n = 0; n <= FMM_ORDER; ++n) {
            for (int m = -n; m <= n; ++m) {
                int idx = FMM_Math::lm_to_idx(n, m);
                std::complex<double> L_nm = leaf_node->local_coeffs[idx];
                if (std::abs(L_nm.real()) < 1e-20 && std::abs(L_nm.imag()) < 1e-20) continue;
                if (n == 0) continue; // Zeroth order term does not contribute to gradient

                // Compute r_rel^(n-1) * Y_n^m(theta_rel, phi_rel) times directional factors
                double r_pow_nm1 = (r_rel > 1e-9) ? (n * std::pow(r_rel, n - 1)) : 0.0;
                auto Ylm_val = FMM_Math::Y_lm_pdf(n, m, theta_rel, phi_rel);

                // Compute partial derivatives in x,y,z directions (approximate gradient)
                pot_grad_x += L_nm * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::cos(phi_rel);
                pot_grad_y += L_nm * r_pow_nm1 * Ylm_val * std::sin(theta_rel) * std::sin(phi_rel);
                pot_grad_z += L_nm * r_pow_nm1 * Ylm_val * std::cos(theta_rel);
            }
        }

        // Multiply by -G_CONST to convert potential gradient to force (acceleration)
        p->ax -= G_CONST * pot_grad_x.real();
        p->ay -= G_CONST * pot_grad_y.real();
        p->az -= G_CONST * pot_grad_z.real();
    }
}

// ----------------------------------------------------------------------------
// Combine far-field (L2P) and near-field P2P interactions for a leaf node
// ----------------------------------------------------------------------------
void compute_leaf_forces_FMM(Node* node) {
    // Evaluate far-field contributions via local expansion
    l2p_FMM(node);

    // Compute near-field P2P interactions: siblings within same parent
    if (node->parent) {
        for (const auto& sibling_ptr : node->parent->children) {
            Node* sibling = sibling_ptr.get();
            if (sibling && sibling != node && sibling->is_leaf && !sibling->particles.empty()) {
                for (auto* p1 : node->particles) {
                    for (auto* p2 : sibling->particles) {
                        p2p(*p1, *p2);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Downward pass for FMM: perform M2L for well-separated nodes, L2L to children,
// then L2P + near-field P2P at leaves.
// ----------------------------------------------------------------------------
void perform_fmm_downward_pass(Node* node, const std::vector<Node*>& all_nodes) {
    if (!node) return;

    // 1. M2L: For each node, find its interaction list by scanning "uncles" (parent's siblings)
    if (node->parent && node->parent->parent) {
        Node* parent = node->parent;
        Node* grandpa = parent->parent;
        for (const auto& uncle_ptr : grandpa->children) {
            if (!uncle_ptr) continue;
            Node* uncle = uncle_ptr.get();
            if (uncle == parent) continue;

            // Collect all leaf descendants of this uncle
            std::vector<Node*> source_leaves;
            std::function<void(Node*)> collect_leaves = [&](Node* cur) {
                if (!cur) return;
                if (cur->is_leaf) {
                    if (!cur->particles.empty()) source_leaves.push_back(cur);
                } else {
                    for (const auto& ch : cur->children) {
                        collect_leaves(ch.get());
                    }
                }
            };
            collect_leaves(uncle);

            // For each leaf in uncle’s subtree, if well-separated, do M2L
            for (Node* src_leaf : source_leaves) {
                if (node->is_well_separated_from(src_leaf)) {
                    M2L_translation(src_leaf, node);
                }
            }
        }
    }

    // 2. If this node is a leaf, do L2P + near-field P2P
    if (node->is_leaf) {
        compute_leaf_forces_FMM(node);
    } else {
        // 3. If not leaf, perform L2L to each child, then recurse down
        for (const auto& child_ptr : node->children) {
            if (child_ptr) {
                L2L_translation(node, child_ptr.get());
                perform_fmm_downward_pass(child_ptr.get(), all_nodes);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Main FMM force computation: orchestrate P2M, M2M, M2L, L2L, L2P, and near-field P2P.
// ----------------------------------------------------------------------------
void compute_forces_FMM_structured(std::vector<Particle>& particles, Node& root) {
    // 1. Zero out accelerations
    for (auto& p : particles) {
        p.ax = p.ay = p.az = 0.0;
    }

    // 2. Collect all leaf nodes
    std::vector<Node*> leaves;
    get_leaf_nodes(&root, leaves);

    // 3. P2M: For each leaf, accumulate its particles into multipole expansion
    #pragma omp parallel for
    for (size_t i = 0; i < leaves.size(); ++i) {
        p2m_FMM(leaves[i]);
    }

    // 4. M2M: Upward pass to build root’s multipole expansion
    perform_m2m_pass(&root);

    // 5. Zero out all local expansions
    std::vector<Node*> all_nodes;
    get_all_nodes(&root, all_nodes);
    for (Node* n : all_nodes) {
        if (n) {
            std::fill(n->local_coeffs.begin(),
                      n->local_coeffs.end(),
                      std::complex<double>(0.0, 0.0));
        }
    }

    // 6. M2L + L2L + L2P + near-field P2P: Downward pass
    perform_fmm_downward_pass(&root, all_nodes);
}

// --- Particle Initialization ---

// Create N particles with random positions in [-domain_half_size, +domain_half_size]^3 and random masses [0.5, 1.5]
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

// --- Error Calculation ---

// Compute RMS relative error between approx_particles (FH/BH/FMM) and exact_particles (Direct).
double calculate_rms_relative_error(const std::vector<Particle>& approx_particles,
                                    const std::vector<Particle>& exact_particles) {
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
        sum_sq_exact += exact_particles[i].ax*exact_particles[i].ax
                      + exact_particles[i].ay*exact_particles[i].ay
                      + exact_particles[i].az*exact_particles[i].az;
    }
    if (sum_sq_exact < 1e-24) {
        return (sum_sq_diff < 1e-24) ? 0.0 : 1.0;
    }
    return std::sqrt(sum_sq_diff / sum_sq_exact);
}

// --- Benchmarking Main ---

int main() {
    std::cout << std::fixed << std::setprecision(5);
    factorial_cache.clear(); // Initialize factorial cache

    // Define N values to test (powers of two up to 32768)
    std::vector<int> N_values = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int max_N_for_direct_sum = 32768; // Maximum N to run the direct O(N^2) method

    // Determine thread counts to test: 1, 2, 4, ..., up to hardware max
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
        max_hw_threads > 1) {
        thread_counts_to_test.push_back(max_hw_threads);
    }
    std::sort(thread_counts_to_test.begin(), thread_counts_to_test.end());
    thread_counts_to_test.erase(std::unique(thread_counts_to_test.begin(), thread_counts_to_test.end()), thread_counts_to_test.end());

    #ifndef _OPENMP
    // If OpenMP is not enabled, restrict to single thread
    if (thread_counts_to_test.size() > 1 || thread_counts_to_test[0] != 1) {
        // std::cout << "Warning: OpenMP not enabled; only running on 1 thread." << std::endl;
    }
    thread_counts_to_test = {1};
    #endif

    // Open CSV file to record performance results
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

        // 1. Compute direct results once (if N is small enough), to use for error comparison
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
            // Skip direct method for large N to avoid O(N^2) blowup
            if (method_name == "Direct" && N > max_N_for_direct_sum) {
                std::cout << "Skipping Direct O(N^2) for N=" << N << " (exceeds limit " << max_N_for_direct_sum << ")" << std::endl;
                continue;
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
                    // Build the octree: root covers [-domain_half_size..+domain_half_size]^3
                    Node root_node(0.0, 0.0, 0.0, root_node_size, nullptr, 0);
                    // Insert all particles into the tree
                    for (auto& p : current_particles) {
                        root_node.insert(&p);
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
                if (method_name == "Direct") {
                    error_val = 0.0; // Direct has no approximation
                } else if (direct_computed_for_N) {
                    error_val = calculate_rms_relative_error(current_particles, particles_direct_results);
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
    std::cout << "Note: The FMM implementation now includes full M2M, M2L, and L2L translations." << std::endl;
    return 0;
}

