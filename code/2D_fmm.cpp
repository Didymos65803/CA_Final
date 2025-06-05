#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <map>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip> // For std::fixed and std::setprecision
#include <omp.h>   // OpenMP
#include <algorithm> // For std::min, std::max

// Constants
const double G_CONST = 1.0;
const double SOFTENING = 0.001;
const double DOMAIN_SIZE = 100.0;
const int FMM_P_TERMS = 16; // Number of terms in multipole/local expansions
const int MAX_LEVEL_DEFAULT = 20; // Default max_level for quadtree

// Forward declaration
class QuadTreeNode;

struct Particle {
    double x, y, mass;
    double vx, vy;
    double ax, ay;

    Particle(double _x, double _y, double _mass, double _vx = 0.0, double _vy = 0.0)
        : x(_x), y(_y), mass(_mass), vx(_vx), vy(_vy), ax(0.0), ay(0.0) {}
};

// Helper for pair hashing in std::map for level_hash (not strictly needed for std::map but good practice for unordered_map)
struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Simple hash combining
        return h1 ^ (h2 << 1);
    }
};

class QuadTreeNode {
public:
    double cx, cy, size;
    int level;
    int max_level;
    QuadTreeNode* children[4]; // NW, NE, SW, SE
    QuadTreeNode* parent;

    std::vector<Particle*> particles_in_node; // Particles directly in this leaf node
    double total_mass;
    double com_x, com_y;
    bool is_leaf;
    bool is_empty;

    std::vector<std::complex<double>> multipole;
    std::vector<std::complex<double>> local_expansion;
    int p_terms;

    std::pair<int, int> grid_key;

    // Static members for FMM registry
    static std::map<int, std::vector<QuadTreeNode*>> global_level_registry;
    static std::map<int, std::map<std::pair<int, int>, QuadTreeNode*>> level_hash;


    QuadTreeNode(double _cx, double _cy, double _size, int _level = 0, int _max_level = MAX_LEVEL_DEFAULT, QuadTreeNode* _parent = nullptr)
        : cx(_cx), cy(_cy), size(_size), level(_level), max_level(_max_level), parent(_parent),
          total_mass(0.0), com_x(0.0), com_y(0.0), is_leaf(true), is_empty(true), p_terms(FMM_P_TERMS) {
        for (int i = 0; i < 4; ++i) children[i] = nullptr;
        multipole.resize(p_terms, {0.0, 0.0});
        local_expansion.resize(p_terms, {0.0, 0.0});

        // Grid key
        grid_key = {
            static_cast<int>((cx + DOMAIN_SIZE / 2.0) / size),
            static_cast<int>((cy + DOMAIN_SIZE / 2.0) / size)
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
                // Update CoM and total_mass incrementally
                double old_total_mass = total_mass;
                total_mass += p->mass;
                if (total_mass > 1e-9) { // Avoid division by zero for massless particles if any
                    com_x = (com_x * old_total_mass + p->x * p->mass) / total_mass;
                    com_y = (com_y * old_total_mass + p->y * p->mass) / total_mass;
                } else {
                    com_x = cx; // Or some other appropriate default
                    com_y = cy;
                }
                return;
            } else {
                is_leaf = false;
                std::vector<Particle*> old_particles = particles_in_node;
                particles_in_node.clear(); // Particles will move to children

                double half = size / 2.0;
                double quarter = half / 2.0;
                children[0] = new QuadTreeNode(cx - quarter, cy - quarter, half, level + 1, max_level, this); // NW
                children[1] = new QuadTreeNode(cx + quarter, cy - quarter, half, level + 1, max_level, this); // NE
                children[2] = new QuadTreeNode(cx - quarter, cy + quarter, half, level + 1, max_level, this); // SW
                children[3] = new QuadTreeNode(cx + quarter, cy + quarter, half, level + 1, max_level, this); // SE

                for (Particle* old_p : old_particles) {
                    _insert_to_child(old_p);
                }
            }
        }

        _insert_to_child(p);
        // Update CoM and total_mass incrementally
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
        if (particle->x > cx) index += 1; // East
        if (particle->y > cy) index += 2; // South
        children[index]->insert(particle);
    }
    
    // --- FMM Specific Methods ---
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

    void compute_multipole_expansion_P2M() { // Particle to Multipole (for leaf)
        if (is_empty || !is_leaf) return;
        
        multipole.assign(p_terms, {0.0, 0.0});
        if (particles_in_node.empty()) return;

        double node_total_mass = 0.0;
        for (Particle* p : particles_in_node) {
            node_total_mass += p->mass;
            std::complex<double> z_rel = {p->x - cx, p->y - cy};
            for (int l = 1; l < p_terms; ++l) { // l from 1 to p-1 for a_l
                multipole[l] -= p->mass * std::pow(z_rel, l) / static_cast<double>(l);
            }
        }
        multipole[0] = node_total_mass;
    }
    
    void compute_multipole_expansion_M2M() { // Multipole to Multipole (for internal node)
        if (is_empty || is_leaf) return;

        multipole.assign(p_terms, {0.0, 0.0});
        double node_total_mass = 0.0;

        for (int i = 0; i < 4; ++i) {
            QuadTreeNode* child = children[i];
            if (child && !child->is_empty) {
                node_total_mass += child->multipole[0].real(); // child's total mass
                std::complex<double> z0_child_to_parent = {child->cx - cx, child->cy - cy};
                
                // Contribution from child's a_0 (total mass)
                for (int l = 1; l < p_terms; ++l) {
                    multipole[l] -= child->multipole[0] * std::pow(z0_child_to_parent, l) / static_cast<double>(l);
                }
                // Contribution from child's a_k (k > 0)
                for (int l = 1; l < p_terms; ++l) {
                    for (int k = 1; k <= l && k < p_terms; ++k) {
                         if (std::abs(child->multipole[k]) > 1e-30) { // if child_expansion[k] is significant
                            multipole[l] += child->multipole[k] * static_cast<double>(binomial_coefficient(l - 1, k - 1)) * std::pow(z0_child_to_parent, l - k);
                        }
                    }
                }
            }
        }
         multipole[0] = node_total_mass; // Parent's a_0 is sum of children's a_0
    }

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
        // Check if bounding boxes touch or overlap
        double dist_x = std::abs(cx - other->cx);
        double dist_y = std::abs(cy - other->cy);
        double max_half_size_sum = (size + other->size) / 2.0;
        return dist_x <= max_half_size_sum + SOFTENING && dist_y <= max_half_size_sum + SOFTENING;
    }


    std::vector<QuadTreeNode*> get_interaction_list() {
        std::vector<QuadTreeNode*> interaction;
        if (!parent) return interaction; // Root has no interaction list this way

        std::vector<QuadTreeNode*> candidates;
        // Parent's neighbors (V-list in some terminologies, or Greengard's U-list related)
        std::vector<QuadTreeNode*> parent_neighbors = parent->get_neighbors();
        for(QuadTreeNode* p_neighbor : parent_neighbors) {
            if (p_neighbor && !p_neighbor->is_empty && !p_neighbor->is_leaf) {
                for(int i=0; i<4; ++i) {
                    if (p_neighbor->children[i] && !p_neighbor->children[i]->is_empty) {
                        candidates.push_back(p_neighbor->children[i]);
                    }
                }
            } else if (p_neighbor && !p_neighbor->is_empty && p_neighbor->is_leaf) {
                 candidates.push_back(p_neighbor); // if parent's neighbor is a leaf
            }
        }
        // Also consider children of parent's parent (if parent is not root) for more distant interactions
        // This simplified version only considers parent's neighbors' children. A full one is more complex.

        for (QuadTreeNode* node : candidates) {
            if (!are_neighbors_or_self(node)) {
                 interaction.push_back(node);
            }
        }
        return interaction;
    }


    void compute_local_expansion_M2L(const std::vector<QuadTreeNode*>& interaction_list) { // Multipole to Local
        if (is_empty) return;
        // local_expansion should be initialized to zero before this function is called for a node in downward pass
        
        for (QuadTreeNode* source_node : interaction_list) {
            if (!source_node || source_node->is_empty || std::abs(source_node->multipole[0]) < 1e-20) continue;

            std::complex<double> z0_source_to_target = {source_node->cx - cx, source_node->cy - cy};
            if (std::abs(z0_source_to_target) < SOFTENING) continue; // Avoid singularity

            const auto& source_multipoles = source_node->multipole;

            for (int l = 0; l < p_terms; ++l) { // l for L_l (target local expansion index)
                std::complex<double> term_sum_for_L_l = {0.0, 0.0};
                for (int k = 0; k < p_terms; ++k) { // k for M_k (source multipole index)
                    if (std::abs(source_multipoles[k]) < 1e-30) continue;

                    double C_lk = static_cast<double>(binomial_coefficient(l + k, k));
                    std::complex<double> term = std::pow(-1.0, k) * source_multipoles[k] * C_lk / std::pow(z0_source_to_target, l + k + 1);
                    term_sum_for_L_l += term;
                }
                local_expansion[l] += term_sum_for_L_l;
            }
        }
    }
    
    void compute_local_expansion_L2L(QuadTreeNode* source_parent_node) { // Local to Local
        if (is_empty || !source_parent_node) return;
        // local_expansion should be initialized (e.g. from M2L pass or empty)
        // This function ADDS the parent's contribution.

        std::complex<double> z0_child_to_parent = {cx - source_parent_node->cx, cy - source_parent_node->cy};
        const auto& parent_local = source_parent_node->local_expansion;
        
        // Optimization: if parent local expansion is effectively zero
        // Check if empty OR (size is sufficient for two elements AND first two elements are small)
        if (parent_local.empty() || (parent_local.size() > 1 && std::abs(parent_local[0]) < 1e-30 && std::abs(parent_local[1]) < 1e-30)) {
            // No significant contribution from parent's local expansion
             return; 
        }

        std::vector<std::complex<double>> temp_b = parent_local;
        for (int k = p_terms - 2; k >= 0; --k) {
             if (k + 1 < p_terms) { // Ensure temp_b[k+1] is in bounds
                temp_b[k] += z0_child_to_parent * temp_b[k+1];
             }
        }
        for(int k=0; k < p_terms; ++k) {
            local_expansion[k] += temp_b[k];
        }
    }

    void evaluate_local_expansion_L2P(Particle* p) { // Local to Particle
        if (is_empty || local_expansion.empty()) return;
        if (std::abs(local_expansion[0]) < 1e-30 && (local_expansion.size() > 1 && std::abs(local_expansion[1]) < 1e-30)) return;


        std::complex<double> z_rel_particle_to_center = {p->x - cx, p->y - cy};
        std::complex<double> force_complex = {0.0, 0.0};
        std::complex<double> z_power = {1.0, 0.0}; // z^0

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
        near_field.push_back(this); // Self
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

// Initialize static members
std::map<int, std::vector<QuadTreeNode*>> QuadTreeNode::global_level_registry;
std::map<int, std::map<std::pair<int, int>, QuadTreeNode*>> QuadTreeNode::level_hash;


// --- Force Computation Algorithms ---
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
    // C++11 compatible loop for map
    for(auto const& pair_level_nodes : QuadTreeNode::global_level_registry) {
        if (pair_level_nodes.first > max_observed_level) {
            max_observed_level = pair_level_nodes.first;
        }
    }
    
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

    std::vector<QuadTreeNode*> all_leaf_nodes;
    for (int l = 0; l <= max_observed_level; ++l) {
        if (QuadTreeNode::global_level_registry.count(l)) {
            // C++11 compatible loop for map iteration
            const auto& nodes_at_level = QuadTreeNode::global_level_registry.at(l); // Use .at() for const access if sure key exists
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

// [PREVIOUS C++ CODE: includes, constants, Particle, QuadTreeNode, force computation functions]
// ... (all the code from the previous response up to main())

int main(int argc, char* argv[]) {
    std::vector<int> n_particles_to_test = {
       2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
       2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288
                                         // Direct method will be skipped for N > 4000
    };

    std::ofstream csv_file("timing_results.csv");
    csv_file << "NumParticles,Algorithm,NumCores,TimeSeconds\n";

    std::vector<int> core_counts = {1, 2, 4, 8};
    int max_threads_system = omp_get_max_threads();

    // Seed for random number generation (consistent across N for particle positions if desired)
    unsigned int base_seed = 0; 

    for (int num_particles : n_particles_to_test) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "Processing for N = " << num_particles << " particles" << std::endl;
        std::cout << "============================================" << std::endl;

        // Particle generation for the current num_particles
        // Using a different seed for each N, or a fixed offset from base_seed,
        // ensures different particle distributions for different N,
        // which is typical for such tests.
        std::mt19937 rng(base_seed + num_particles); 
        std::uniform_real_distribution<double> pos_dist(-DOMAIN_SIZE / 2.0, DOMAIN_SIZE / 2.0);
        std::uniform_real_distribution<double> mass_dist(1.0, 3.0);

        std::vector<Particle> particles_master_copy;
        particles_master_copy.reserve(num_particles);
        for (int i = 0; i < num_particles; ++i) {
            double x = pos_dist(rng);
            double y = pos_dist(rng);
            double mass = mass_dist(rng);
            particles_master_copy.emplace_back(x, y, mass);
        }

        std::vector<Particle> particles_direct;
        std::vector<Particle> particles_fmm;

        // --- Direct N-body Method ---
        std::cout << "\nBenchmarking Direct N-body Method for N = " << num_particles << "..." << std::endl;
        if (num_particles <= 32768) { // Direct method is slow, only run for smaller N
            for (int cores : core_counts) {
                if (cores > max_threads_system) {
                    std::cout << "Skipping Direct (" << cores << " cores) - exceeds max system threads (" << max_threads_system << ")" << std::endl;
                    continue;
                }
                omp_set_num_threads(cores);
                
                particles_direct = particles_master_copy; // Reset particles from master copy

                auto start_time = std::chrono::high_resolution_clock::now();
                compute_forces_direct(particles_direct);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end_time - start_time;
                
                std::cout << "  Direct (N=" << num_particles << ", " << cores << " cores): " << std::fixed << std::setprecision(6) << diff.count() << " s" << std::endl;
                csv_file << num_particles << ",Direct," << cores << "," << diff.count() << "\n";
                csv_file.flush(); // Ensure data is written immediately
            }
        } else {
            std::cout << "  Skipping Direct method for N = " << num_particles << " (too large)." << std::endl;
        }

        // --- FMM Method ---
        std::cout << "\nBenchmarking FMM Method for N = " << num_particles << "..." << std::endl;
        for (int cores : core_counts) {
            if (cores > max_threads_system) {
                 std::cout << "Skipping FMM (" << cores << " cores) - exceeds max system threads (" << max_threads_system << ")" << std::endl;
                continue;
            }
            omp_set_num_threads(cores);

            particles_fmm = particles_master_copy; // Reset particles from master copy

            auto start_time = std::chrono::high_resolution_clock::now();
            compute_forces_fmm(particles_fmm, DOMAIN_SIZE, MAX_LEVEL_DEFAULT);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end_time - start_time;

            std::cout << "  FMM (N=" << num_particles << ", " << cores << " cores): " << std::fixed << std::setprecision(6) << diff.count() << " s" << std::endl;
            csv_file << num_particles << ",FMM," << cores << "," << diff.count() << "\n";
            csv_file.flush(); // Ensure data is written immediately
        }

        // Optional: Print some resulting accelerations to verify for the smallest N and last core count
        if (num_particles == n_particles_to_test[0]) { // Only for the first N in the list
            if (num_particles <= 4000 && !particles_direct.empty() && !particles_fmm.empty()) {
                 std::cout << std::fixed << std::setprecision(5);
                 std::cout << "\n  Particle 0 Accel (Direct, N=" << num_particles << "): ax=" << particles_direct[0].ax << ", ay=" << particles_direct[0].ay << std::endl;
                 std::cout << "  Particle 0 Accel (FMM,    N=" << num_particles << "): ax=" << particles_fmm[0].ax << ", ay=" << particles_fmm[0].ay << std::endl;
            } else if (!particles_fmm.empty()) {
                 std::cout << std::fixed << std::setprecision(5);
                 std::cout << "\n  Particle 0 Accel (FMM,    N=" << num_particles << "): ax=" << particles_fmm[0].ax << ", ay=" << particles_fmm[0].ay << std::endl;
            }
        }
    } // End loop over n_particles_to_test

    csv_file.close();
    std::cout << "\n\nAll tests completed. Results saved to timing_results.csv" << std::endl;

    return 0;
}