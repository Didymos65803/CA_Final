// ---------------------------------------------------------------------------
//  fmm_true_on.cpp
//
//  基於成功C++實現的關鍵優化：
//  1. 正確的FMM四階段並行化
//  2. 使用C++實現中證明有效的數據結構
//  3. 級別註冊表的並行處理
//  4. 優化的交互列表和近場計算
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <complex>
#include <map>
#include <algorithm>
#include <memory>
#include <omp.h>

namespace py = pybind11;

// 從成功的C++實現中採用的常數
const double G_CONST = 1.0;
const double SOFTENING = 0.001;
const int FMM_P_TERMS = 16;  // 與C++版本相同
const int MAX_LEVEL_DEFAULT = 20;

// ---------------------------------------------------------------------------
//  複製C++實現的粒子結構
// ---------------------------------------------------------------------------
struct Particle {
    double x, y, mass;
    double vx, vy;  // 雖然不用於力計算，但保持兼容性
    double ax, ay;

    Particle() : x(0), y(0), mass(0), vx(0), vy(0), ax(0), ay(0) {}
    Particle(double _x, double _y, double _mass) 
        : x(_x), y(_y), mass(_mass), vx(0), vy(0), ax(0), ay(0) {}
};

// ---------------------------------------------------------------------------
//  基於C++成功實現的QuadTreeNode
// ---------------------------------------------------------------------------
class QuadTreeNode {
public:
    double cx, cy, size;
    int level;
    int max_level;
    QuadTreeNode* children[4];
    QuadTreeNode* parent;

    std::vector<Particle*> particles_in_node;
    double total_mass;
    double com_x, com_y;
    bool is_leaf;
    bool is_empty;

    std::vector<std::complex<double>> multipole;
    std::vector<std::complex<double>> local_expansion;
    int p_terms;

    std::pair<int, int> grid_key;

    // 關鍵：從C++實現複製的靜態註冊表
    static std::map<int, std::vector<QuadTreeNode*>> global_level_registry;
    static std::map<int, std::map<std::pair<int, int>, QuadTreeNode*>> level_hash;

    QuadTreeNode(double _cx, double _cy, double _size, int _level = 0, 
                 int _max_level = MAX_LEVEL_DEFAULT, QuadTreeNode* _parent = nullptr)
        : cx(_cx), cy(_cy), size(_size), level(_level), max_level(_max_level), parent(_parent),
          total_mass(0.0), com_x(0.0), com_y(0.0), is_leaf(true), is_empty(true), p_terms(FMM_P_TERMS) {
        
        for (int i = 0; i < 4; ++i) children[i] = nullptr;
        multipole.resize(p_terms, {0.0, 0.0});
        local_expansion.resize(p_terms, {0.0, 0.0});

        // 從C++實現複製的網格鍵值計算
        grid_key = {
            static_cast<int>((cx + 50.0) / size),  // 假設domain size為100
            static_cast<int>((cy + 50.0) / size)
        };
        level_hash[level][grid_key] = this;
        global_level_registry[level].push_back(this);
    }

    ~QuadTreeNode() {
        for (int i = 0; i < 4; ++i) {
            delete children[i];
            children[i] = nullptr;
        }
    }

    static void clear_static_registries() {
        global_level_registry.clear();
        level_hash.clear();
    }

    void insert(Particle* p) {
        is_empty = false;
        if (is_leaf) {
            if (particles_in_node.empty() || level >= max_level) {
                particles_in_node.push_back(p);
                // 增量更新質心
                double old_total_mass = total_mass;
                total_mass += p->mass;
                if (total_mass > 1e-9) {
                    com_x = (com_x * old_total_mass + p->x * p->mass) / total_mass;
                    com_y = (com_y * old_total_mass + p->y * p->mass) / total_mass;
                } else {
                    com_x = cx;
                    com_y = cy;
                }
                return;
            } else {
                is_leaf = false;
                std::vector<Particle*> old_particles = particles_in_node;
                particles_in_node.clear();

                double half = size / 2.0;
                double quarter = half / 2.0;
                children[0] = new QuadTreeNode(cx - quarter, cy - quarter, half, level + 1, max_level, this);
                children[1] = new QuadTreeNode(cx + quarter, cy - quarter, half, level + 1, max_level, this);
                children[2] = new QuadTreeNode(cx - quarter, cy + quarter, half, level + 1, max_level, this);
                children[3] = new QuadTreeNode(cx + quarter, cy + quarter, half, level + 1, max_level, this);

                for (Particle* old_p : old_particles) {
                    _insert_to_child(old_p);
                }
            }
        }

        _insert_to_child(p);
        double old_total_mass = total_mass;
        total_mass += p->mass;
        if (total_mass > 1e-9) {
            com_x = (com_x * old_total_mass + p->x * p->mass) / total_mass;
            com_y = (com_y * old_total_mass + p->y * p->mass) / total_mass;
        } else {
            com_x = cx; 
            com_y = cy;
        }
    }

    void _insert_to_child(Particle* particle) {
        int index = 0;
        if (particle->x > cx) index += 1;
        if (particle->y > cy) index += 2;
        children[index]->insert(particle);
    }

    // 從C++實現複製的二項式係數計算
    unsigned long long binomial_coefficient(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = n - k;
        unsigned long long res = 1;
        for (int i = 1; i <= k; ++i) {
            res = res * (n - i + 1) / i;
        }
        return res;
    }

    // 從C++實現複製的P2M
    void compute_multipole_expansion_P2M() {
        if (is_empty || !is_leaf) return;
        
        multipole.assign(p_terms, {0.0, 0.0});
        if (particles_in_node.empty()) return;

        double node_total_mass = 0.0;
        for (Particle* p : particles_in_node) {
            node_total_mass += p->mass;
            std::complex<double> z_rel = {p->x - cx, p->y - cy};
            for (int l = 1; l < p_terms; ++l) {
                multipole[l] -= p->mass * std::pow(z_rel, l) / static_cast<double>(l);
            }
        }
        multipole[0] = node_total_mass;
    }
    
    // 從C++實現複製的M2M
    void compute_multipole_expansion_M2M() {
        if (is_empty || is_leaf) return;

        multipole.assign(p_terms, {0.0, 0.0});
        double node_total_mass = 0.0;

        for (int i = 0; i < 4; ++i) {
            QuadTreeNode* child = children[i];
            if (child && !child->is_empty) {
                node_total_mass += child->multipole[0].real();
                std::complex<double> z0_child_to_parent = {child->cx - cx, child->cy - cy};
                
                for (int l = 1; l < p_terms; ++l) {
                    multipole[l] -= child->multipole[0] * std::pow(z0_child_to_parent, l) / static_cast<double>(l);
                }
                
                for (int l = 1; l < p_terms; ++l) {
                    for (int k = 1; k <= l && k < p_terms; ++k) {
                         if (std::abs(child->multipole[k]) > 1e-30) {
                            multipole[l] += child->multipole[k] * static_cast<double>(binomial_coefficient(l - 1, k - 1)) * std::pow(z0_child_to_parent, l - k);
                        }
                    }
                }
            }
        }
        multipole[0] = node_total_mass;
    }

    // 從C++實現複製的鄰居獲取
    std::vector<QuadTreeNode*> get_neighbors() {
        std::vector<QuadTreeNode*> nbrs;
        if (level_hash.find(level) == level_hash.end()) return nbrs;

        auto& current_level_map = level_hash[level];
        int gi = grid_key.first;
        int gj = grid_key.second;

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue;
                std::pair<int, int> neighbor_key = {gi + di, gj + dj};
                if (current_level_map.count(neighbor_key)) {
                    nbrs.push_back(current_level_map[neighbor_key]);
                }
            }
        }
        return nbrs;
    }
    
    bool are_neighbors_or_self(QuadTreeNode* other) {
        if (!other) return false;
        double dist_x = std::abs(cx - other->cx);
        double dist_y = std::abs(cy - other->cy);
        double max_half_size_sum = (size + other->size) / 2.0;
        return dist_x <= max_half_size_sum + SOFTENING && dist_y <= max_half_size_sum + SOFTENING;
    }

    // 從C++實現複製的交互列表
    std::vector<QuadTreeNode*> get_interaction_list() {
        std::vector<QuadTreeNode*> interaction;
        if (!parent) return interaction;

        std::vector<QuadTreeNode*> candidates;
        std::vector<QuadTreeNode*> parent_neighbors = parent->get_neighbors();
        for(QuadTreeNode* p_neighbor : parent_neighbors) {
            if (p_neighbor && !p_neighbor->is_empty && !p_neighbor->is_leaf) {
                for(int i=0; i<4; ++i) {
                    if (p_neighbor->children[i] && !p_neighbor->children[i]->is_empty) {
                        candidates.push_back(p_neighbor->children[i]);
                    }
                }
            } else if (p_neighbor && !p_neighbor->is_empty && p_neighbor->is_leaf) {
                 candidates.push_back(p_neighbor);
            }
        }

        for (QuadTreeNode* node : candidates) {
            if (!are_neighbors_or_self(node)) {
                 interaction.push_back(node);
            }
        }
        return interaction;
    }

    // 從C++實現複製的M2L
    void compute_local_expansion_M2L(const std::vector<QuadTreeNode*>& interaction_list) {
        if (is_empty) return;
        
        for (QuadTreeNode* source_node : interaction_list) {
            if (!source_node || source_node->is_empty || std::abs(source_node->multipole[0]) < 1e-20) continue;

            std::complex<double> z0_source_to_target = {source_node->cx - cx, source_node->cy - cy};
            if (std::abs(z0_source_to_target) < SOFTENING) continue;

            const auto& source_multipoles = source_node->multipole;

            for (int l = 0; l < p_terms; ++l) {
                std::complex<double> term_sum_for_L_l = {0.0, 0.0};
                for (int k = 0; k < p_terms; ++k) {
                    if (std::abs(source_multipoles[k]) < 1e-30) continue;

                    double C_lk = static_cast<double>(binomial_coefficient(l + k, k));
                    std::complex<double> term = std::pow(-1.0, k) * source_multipoles[k] * C_lk / std::pow(z0_source_to_target, l + k + 1);
                    term_sum_for_L_l += term;
                }
                local_expansion[l] += term_sum_for_L_l;
            }
        }
    }
    
    // 從C++實現複製的L2L
    void compute_local_expansion_L2L(QuadTreeNode* source_parent_node) {
        if (is_empty || !source_parent_node) return;

        std::complex<double> z0_child_to_parent = {cx - source_parent_node->cx, cy - source_parent_node->cy};
        const auto& parent_local = source_parent_node->local_expansion;
        
        if (parent_local.empty() || (parent_local.size() > 1 && std::abs(parent_local[0]) < 1e-30 && std::abs(parent_local[1]) < 1e-30)) {
             return; 
        }

        std::vector<std::complex<double>> temp_b = parent_local;
        for (int k = p_terms - 2; k >= 0; --k) {
             if (k + 1 < p_terms) {
                temp_b[k] += z0_child_to_parent * temp_b[k+1];
             }
        }
        for(int k=0; k < p_terms; ++k) {
            local_expansion[k] += temp_b[k];
        }
    }

    // 從C++實現複製的L2P
    void evaluate_local_expansion_L2P(Particle* p) {
        if (is_empty || local_expansion.empty()) return;
        if (std::abs(local_expansion[0]) < 1e-30 && (local_expansion.size() > 1 && std::abs(local_expansion[1]) < 1e-30)) return;

        std::complex<double> z_rel_particle_to_center = {p->x - cx, p->y - cy};
        std::complex<double> force_complex = {0.0, 0.0};
        std::complex<double> z_power = {1.0, 0.0};

        for (int k = 1; k < p_terms; ++k) { 
            if (k > 0 && static_cast<size_t>(k) < local_expansion.size() && std::abs(local_expansion[k]) > 1e-30) {
                 force_complex += local_expansion[k] * static_cast<double>(k) * z_power;
            }
            if (k < p_terms -1) { 
                 z_power *= z_rel_particle_to_center; 
            }
        }
        
        p->ax -= force_complex.real() * G_CONST; 
        p->ay += force_complex.imag() * G_CONST; 
    }

    // 從C++實現複製的P2P
    void compute_direct_force_on_particle_P2P(Particle* p, const std::vector<QuadTreeNode*>& near_field_nodes) {
        double soft2 = SOFTENING * SOFTENING;
        for (QuadTreeNode* source_node : near_field_nodes) {
            if (!source_node || source_node->is_empty) continue;
            
            if (source_node->is_leaf) {
                for (Particle* other_p : source_node->particles_in_node) {
                    if (p == other_p) continue;
                    double dx = other_p->x - p->x;
                    double dy = other_p->y - p->y;
                    double r2 = dx * dx + dy * dy + soft2;
                    if (r2 < 1e-9) r2 = 1e-9; 
                    double inv_r = 1.0 / std::sqrt(r2);
                    double inv_r3 = inv_r * inv_r * inv_r;
                    
                    double force_mag_over_m = G_CONST * other_p->mass * inv_r3;
                    p->ax += force_mag_over_m * dx;
                    p->ay += force_mag_over_m * dy;
                }
            } 
        }
    }

    std::vector<QuadTreeNode*> get_near_field_cells_for_leaf() {
        std::vector<QuadTreeNode*> near_field;
        near_field.push_back(this);
        std::vector<QuadTreeNode*> neighbors = this->get_neighbors();
        for (QuadTreeNode* nbr : neighbors) {
            if (nbr && !nbr->is_empty) {
                collect_leaf_nodes_recursive(nbr, near_field);
            }
        }
        return near_field;
    }

    void collect_leaf_nodes_recursive(QuadTreeNode* node, std::vector<QuadTreeNode*>& leaves) {
        if (!node || node->is_empty) return;
        if (node->is_leaf) {
            leaves.push_back(node);
        } else {
            for (int i=0; i<4; ++i) {
                collect_leaf_nodes_recursive(node->children[i], leaves);
            }
        }
    }
};

// 初始化靜態成員
std::map<int, std::vector<QuadTreeNode*>> QuadTreeNode::global_level_registry;
std::map<int, std::map<std::pair<int, int>, QuadTreeNode*>> QuadTreeNode::level_hash;

// ---------------------------------------------------------------------------
//  從C++實現複製的直接方法
// ---------------------------------------------------------------------------
void compute_forces_direct(std::vector<Particle>& particles) {
    double soft2 = SOFTENING * SOFTENING;
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
        double ax_priv = 0.0;
        double ay_priv = 0.0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double r2 = dx * dx + dy * dy + soft2;
            if (r2 < 1e-9) r2 = 1e-9;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            
            double force_mag_over_m = G_CONST * particles[j].mass * inv_r3; 
            ax_priv += force_mag_over_m * dx;
            ay_priv += force_mag_over_m * dy;
        }
        particles[i].ax = ax_priv;
        particles[i].ay = ay_priv;
    }
}

// ---------------------------------------------------------------------------
//  從C++實現複製的FMM方法
// ---------------------------------------------------------------------------
void compute_forces_fmm(std::vector<Particle>& particles, double domain_size_val, int max_tree_level) {
    if (particles.empty()) return;

    QuadTreeNode::clear_static_registries();

    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0;
        particles[i].ay = 0.0;
    }

    QuadTreeNode* root = new QuadTreeNode(0.0, 0.0, domain_size_val, 0, max_tree_level);
    for (size_t i = 0; i < particles.size(); ++i) {
        root->insert(&particles[i]);
    }

    int max_observed_level = 0;
    for(auto const& pair_level_nodes : QuadTreeNode::global_level_registry) {
        if (pair_level_nodes.first > max_observed_level) {
            max_observed_level = pair_level_nodes.first;
        }
    }
    
    // 上行階段：並行計算多極展開
    for (int l = max_observed_level; l >= 0; --l) {
        if (QuadTreeNode::global_level_registry.count(l)) {
            const auto& nodes_at_level = QuadTreeNode::global_level_registry[l];
            #pragma omp parallel for
            for (size_t i = 0; i < nodes_at_level.size(); ++i) {
                QuadTreeNode* node = nodes_at_level[i];
                if (node->is_leaf) {
                    node->compute_multipole_expansion_P2M();
                } else {
                    node->compute_multipole_expansion_M2M();
                }
            }
        }
    }
    
    // 下行階段：並行計算局部展開
    for (int l = 0; l <= max_observed_level; ++l) { 
        if (QuadTreeNode::global_level_registry.count(l)) {
            const auto& nodes_at_level = QuadTreeNode::global_level_registry[l];
            #pragma omp parallel for
            for (size_t i = 0; i < nodes_at_level.size(); ++i) {
                QuadTreeNode* node = nodes_at_level[i];
                if (node->is_empty) continue;

                node->local_expansion.assign(FMM_P_TERMS, {0.0,0.0}); 

                std::vector<QuadTreeNode*> interaction_list_nodes;
                if (node->parent) { 
                     interaction_list_nodes = node->get_interaction_list(); 
                }
                node->compute_local_expansion_M2L(interaction_list_nodes);

                if (node->parent) { 
                    node->compute_local_expansion_L2L(node->parent);
                }
            }
        }
    }

    // 收集所有葉節點並並行處理
    std::vector<QuadTreeNode*> all_leaf_nodes;
    for (int l = 0; l <= max_observed_level; ++l) {
        if (QuadTreeNode::global_level_registry.count(l)) {
            const auto& nodes_at_level = QuadTreeNode::global_level_registry.at(l);
            for(QuadTreeNode* node : nodes_at_level){
                if(node && node->is_leaf && !node->is_empty){
                    all_leaf_nodes.push_back(node);
                }
            }
        }
    }
        
    #pragma omp parallel for
    for (size_t i=0; i < all_leaf_nodes.size(); ++i) {
        QuadTreeNode* leaf = all_leaf_nodes[i];
        if (!leaf || leaf->particles_in_node.empty()) continue;

        std::vector<QuadTreeNode*> near_field_cells = leaf->get_near_field_cells_for_leaf();

        for (Particle* p : leaf->particles_in_node) {
            leaf->evaluate_local_expansion_L2P(p);
            leaf->compute_direct_force_on_particle_P2P(p, near_field_cells);
        }
    }
    
    delete root; 
    QuadTreeNode::clear_static_registries(); 
}

// ---------------------------------------------------------------------------
//  主函數：使用與C++相同的策略
// ---------------------------------------------------------------------------
void fmm_force_on(py::array_t<double> x_arr,
                  py::array_t<double> y_arr,
                  py::array_t<double> m_arr,
                  double eps2,
                  py::array_t<double> domain_arr,
                  double theta,
                  py::array_t<double> ax_arr,
                  py::array_t<double> ay_arr)
{
    auto x_view = x_arr.unchecked<1>();
    auto y_view = y_arr.unchecked<1>();
    auto m_view = m_arr.unchecked<1>();
    auto domain = domain_arr.unchecked<1>();
    auto axw = ax_arr.mutable_unchecked<1>();
    auto ayw = ay_arr.mutable_unchecked<1>();
    const int N = (int)x_arr.shape(0);
    
    // 創建粒子向量，與C++實現兼容
    std::vector<Particle> particles(N);
    for (int i = 0; i < N; ++i) {
        particles[i] = Particle(x_view(i), y_view(i), m_view(i));
    }
    
    const double domain_size = domain(1) - domain(0);
    
    // 使用與C++實現相同的策略選擇
    if (N <= 32768) {  // 與C++實現相同的閾值
        compute_forces_direct(particles);
    } else {
        compute_forces_fmm(particles, domain_size, MAX_LEVEL_DEFAULT);
    }
    
    // 複製結果
    for (int i = 0; i < N; ++i) {
        axw(i) = particles[i].ax;
        ayw(i) = particles[i].ay;
    }
}

// ---------------------------------------------------------------------------
//  PyBind11 模組
// ---------------------------------------------------------------------------
PYBIND11_MODULE(fmm_true_on, m) {
    m.doc() = "Force calculation based on successful C++ FMM implementation";
    m.def("fmm_force_on", &fmm_force_on,
          "fmm_force_on(x, y, m, eps2, domain, theta, ax, ay)");
}
