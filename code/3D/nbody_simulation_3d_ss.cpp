// nbody_simulation_solarsystem.cpp
// Compile with:
// g++ nbody_simulation_solarsystem.cpp -o nbody_solarsystem -O3 -std=c++17 -fopenmp -Wall
#include <iostream>
#include <vector>
#include <string>
#include <cmath>     // For std::sqrt, std::cos, std::sin, M_PI
#include <random>    // For std::mt19937, std::uniform_real_distribution, std::random_device
#include <fstream>   // For std::ofstream
#include <iomanip>   // For std::fixed, std::setprecision, std::scientific
#include <algorithm> // For std::max, std::min, std::max({})
#include <memory>    // For std::unique_ptr, std::make_unique (for OctreeNode)

// OpenMP include
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Real Physical Constants ---
const double G_CONST = 6.67430e-11; // m^3 kg^-1 s^-2
const double AU_TO_METERS = 1.495978707e11; // meters per AU
const double DAYS_TO_SECONDS = 86400.0;    // seconds per day

const double SOFTENING = 1000.0; // 1 km in meters, for numerical stability at extremely close (unphysical) approaches
const double SOFT2 = SOFTENING * SOFTENING;

// --- Data Structures ---
struct Particle {
    int id;
    std::string name;
    double x, y, z;    // m, position relative to Solar System Barycenter (SSB)
    double vx, vy, vz; // m/s, velocity relative to SSB
    double ax, ay, az; // m/s^2, acceleration
    double mass;       // kg

    Particle(int _id, std::string _name,
             double _x, double _y, double _z, double _mass,
             double _vx = 0.0, double _vy = 0.0, double _vz = 0.0)
        : id(_id), name(std::move(_name)),
          x(_x), y(_y), z(_z),
          vx(_vx), vy(_vy), vz(_vz),
          ax(0.0), ay(0.0), az(0.0),
          mass(_mass) {}
};

// OctreeNode class (Barnes-Hut method, alternative to direct summation)
class OctreeNode {
public:
    double cx, cy, cz, size;
    std::vector<std::unique_ptr<OctreeNode>> children;
    std::vector<Particle*> node_particles;
    double total_mass;
    double com_x, com_y, com_z;
    bool is_leaf;
    bool is_empty;
    static const int MAX_PARTICLES_PER_LEAF = 1;

    OctreeNode(double center_x, double center_y, double center_z, double s)
        : cx(center_x), cy(center_y), cz(center_z), size(s),
          total_mass(0.0), com_x(0.0), com_y(0.0), com_z(0.0),
          is_leaf(true), is_empty(true) {
        children.resize(8);
    }

    void subdivide() {
        is_leaf = false;
        double child_size = size / 2.0;
        double offset = size / 4.0;
        int child_idx_counter = 0;
        for (int i = -1; i <= 1; i += 2) { // z_factor
            for (int j = -1; j <= 1; j += 2) { // y_factor
                for (int k = -1; k <= 1; k += 2) { // x_factor
                    children[child_idx_counter++] = std::make_unique<OctreeNode>(
                        cx + k * offset, cy + j * offset, cz + i * offset, child_size);
                }
            }
        }
    }

    int get_child_index(const Particle* p) const {
        int index = 0;
        if (p->x > cx) index |= 1;
        if (p->y > cy) index |= 2;
        if (p->z > cz) index |= 4;
        return index;
    }
    
    void insert_into_child_and_update_parent(Particle* p) { // Renamed for clarity
        int child_idx = get_child_index(p);
        if (!children[child_idx]) {
             // This should ideally not happen if subdivide is called correctly.
             std::cerr << "Error: Octree child " << child_idx << " not initialized during insert_into_child. Forcing subdivision." << std::endl;
             subdivide(); // Attempt to recover, though this indicates a logic flaw.
             if(!children[child_idx]){ // Check again
                std::cerr << "FATAL Error: Could not create child node " << child_idx << " even after forced subdivision." << std::endl;
                return;
             }
        }
        
        // Store parent's (this node's) properties BEFORE 'p' is added to its subtree via the child
        double old_parent_total_mass = this->total_mass;
        double old_parent_com_x = this->com_x;
        double old_parent_com_y = this->com_y;
        double old_parent_com_z = this->com_z;

        children[child_idx]->insert(p); // Recursive call

        // After recursive call returns, 'this' (parent) node must update its own COM/mass
        // to reflect that particle 'p' has been added to its overall subtree via that child.
        this->total_mass = old_parent_total_mass + p->mass; // Incrementally add p's mass
        if (old_parent_total_mass == 0.0) { // If parent's subtree was effectively empty mass-wise before p
            this->com_x = p->x;
            this->com_y = p->y;
            this->com_z = p->z;
        } else { // Parent already had mass, do a weighted average
            this->com_x = (old_parent_com_x * old_parent_total_mass + p->x * p->mass) / this->total_mass;
            this->com_y = (old_parent_com_y * old_parent_total_mass + p->y * p->mass) / this->total_mass;
            this->com_z = (old_parent_com_z * old_parent_total_mass + p->z * p->mass) / this->total_mass;
        }
    }


    void insert(Particle* p) {
        this->is_empty = false; 

        if (this->is_leaf) {
            if (this->node_particles.empty()) { // Leaf is empty, can take particle p
                this->node_particles.push_back(p);
                this->total_mass = p->mass;
                this->com_x = p->x; this->com_y = p->y; this->com_z = p->z;
                return; // This leaf's properties are now set for p.
            } else { // Leaf is full (has 1 particle), must subdivide.
                Particle* existing_particle = this->node_particles[0];
                this->node_particles.clear(); // Will no longer hold particles directly.
                
                this->subdivide(); // Converts 'this' to an internal node. Children created.

                // Re-insert existing_particle into the new child structure.
                // This will make 'this' node's mass/COM reflect existing_particle being in its subtree.
                this->insert_into_child_and_update_parent(existing_particle); 
                
                // Now insert the new particle 'p' into the new child structure.
                // This will further update 'this' node's mass/COM to reflect 'p' also being in its subtree.
                this->insert_into_child_and_update_parent(p);
                return; // All updates handled by the calls above for this subdivided node.
            }
        } else { // Already an internal node.
            this->insert_into_child_and_update_parent(p);
            return; // Update handled by the call above.
        }
    }


    void compute_force(const Particle* target_p, double& force_x, double& force_y, double& force_z, double theta) const {
        if (is_empty) return;
        if (is_leaf) {
            for (const Particle* p_in_node : node_particles) {
                if (p_in_node == target_p) continue;
                double dx = p_in_node->x - target_p->x; double dy = p_in_node->y - target_p->y; double dz = p_in_node->z - target_p->z;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < SOFT2) r2 = SOFT2;
                double r = std::sqrt(r2);
                if (r == 0) continue; 
                double f_over_r = G_CONST * p_in_node->mass / (r2 * r);
                force_x += f_over_r * dx; force_y += f_over_r * dy; force_z += f_over_r * dz;
            }
            return;
        }
        double dx_com = com_x - target_p->x; double dy_com = com_y - target_p->y; double dz_com = com_z - target_p->z;
        double r2_com = dx_com * dx_com + dy_com * dy_com + dz_com * dz_com;
        if (r2_com < SOFT2) r2_com = SOFT2;
        double r_com = std::sqrt(r2_com);

        if (r_com == 0) { // Target particle is at the COM of this internal node
            for (int i = 0; i < 8; ++i) if (children[i] && !children[i]->is_empty) children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
            return;
        }
        if ((size / r_com) < theta || total_mass == 0) { // If node is far enough, or has no mass to approximate
            if (total_mass > 0) { // only apply force if there's mass
                 double f_over_r_com = G_CONST * total_mass / (r2_com * r_com);
                 force_x += f_over_r_com * dx_com; force_y += f_over_r_com * dy_com; force_z += f_over_r_com * dz_com;
            }
        } else {
            for (int i = 0; i < 8; ++i) if (children[i] && !children[i]->is_empty) children[i]->compute_force(target_p, force_x, force_y, force_z, theta);
        }
    }
};

// --- Force Calculation ---
void compute_forces_direct(std::vector<Particle>& particles) {
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax = 0.0; particles[i].ay = 0.0; particles[i].az = 0.0;
        double acc_x_local = 0.0, acc_y_local = 0.0, acc_z_local = 0.0;
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < SOFT2) {
                r2 = SOFT2;
            }
            if (r2 == 0) continue; 
            double inv_r = 1.0 / std::sqrt(r2);
            double f_factor = G_CONST * particles[j].mass * inv_r * inv_r * inv_r; // G*m_j/r^3
            acc_x_local += f_factor * dx;
            acc_y_local += f_factor * dy;
            acc_z_local += f_factor * dz;
        }
        particles[i].ax = acc_x_local;
        particles[i].ay = acc_y_local;
        particles[i].az = acc_z_local;
    }
}

void compute_forces_octree(std::vector<Particle>& particles, double theta) {
    if (particles.empty()) return;
    double min_x = particles[0].x, max_x = particles[0].x;
    double min_y = particles[0].y, max_y = particles[0].y;
    double min_z = particles[0].z, max_z = particles[0].z;

    for (size_t i = 1; i < particles.size(); ++i) {
        const auto& p = particles[i];
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }
    double dx_domain = max_x - min_x, dy_domain = max_y - min_y, dz_domain = max_z - min_z;
    double domain_size = std::max({dx_domain, dy_domain, dz_domain});
    if (domain_size == 0) domain_size = 2 * AU_TO_METERS; // Handle case where all particles at same point initially, give some size
    domain_size *= 1.2; // Add padding
    domain_size = std::max(domain_size, 1.0e5); // Ensure a minimum practical size (e.g. 100km)

    double center_x = (min_x + max_x) / 2.0;
    double center_y = (min_y + max_y) / 2.0;
    double center_z = (min_z + max_z) / 2.0;
    
    OctreeNode root(center_x, center_y, center_z, domain_size);

    for (auto& p : particles) {
        if (std::abs(p.x - root.cx) <= root.size / 2.0 + 1e-9 && // Add small tolerance for strict float comparison
            std::abs(p.y - root.cy) <= root.size / 2.0 + 1e-9 &&
            std::abs(p.z - root.cz) <= root.size / 2.0 + 1e-9) {
            root.insert(&p);
        } else {
             std::cerr << "Warning: Particle " << p.id << " (" << p.name <<") at " 
                       << p.x << "," << p.y << "," << p.z 
                       << " is outside initial root bounds (Center: " << root.cx << "," << root.cy << "," << root.cz 
                       << ", Size: " << root.size << "). Skipping for Octree insertion." << std::endl;
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].ax=0.0; particles[i].ay=0.0; particles[i].az=0.0;
        double fx=0.0, fy=0.0, fz=0.0; 
        root.compute_force(&particles[i], fx, fy, fz, theta);
        particles[i].ax=fx; particles[i].ay=fy; particles[i].az=fz;
    }
}

// --- Integrator ---
void leapfrog_step(std::vector<Particle>& particles, double dt, const std::string& method, double theta, bool mobile_sun) {
    bool sun_exists_and_is_particle_0 = !particles.empty() && particles[0].id == 0 && particles[0].name == "Sun";

    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && sun_exists_and_is_particle_0 && !mobile_sun) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt; 
        particles[i].vy += 0.5 * particles[i].ay * dt; 
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && sun_exists_and_is_particle_0 && !mobile_sun) continue;
        particles[i].x += particles[i].vx * dt; 
        particles[i].y += particles[i].vy * dt; 
        particles[i].z += particles[i].vz * dt;
    }

    if (method == "direct") compute_forces_direct(particles);
    else if (method == "fmm" || method == "octree") compute_forces_octree(particles, theta);
    else { std::cerr << "Unknown method: " << method << ". Defaulting to direct." << std::endl; compute_forces_direct(particles); }
    
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0 && sun_exists_and_is_particle_0 && !mobile_sun) continue;
        particles[i].vx += 0.5 * particles[i].ax * dt; 
        particles[i].vy += 0.5 * particles[i].ay * dt; 
        particles[i].vz += 0.5 * particles[i].az * dt;
    }
}

// --- Diagnostics ---
double system_energy(const std::vector<Particle>& particles) {
    double ke = 0.0, pe = 0.0;
    for (const auto& p : particles) ke += 0.5 * p.mass * (p.vx*p.vx + p.vy*p.vy + p.vz*p.vz);
    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            double dx = particles[j].x-particles[i].x; double dy = particles[j].y-particles[i].y; double dz = particles[j].z-particles[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + SOFT2; 
            pe -= G_CONST * particles[i].mass * particles[j].mass / std::sqrt(r2);
        }
    }
    return ke + pe;
}

// --- Initial Conditions for Solar System ---
struct PlanetEphemeris {
    std::string name;
    double mass_kg;
    double x_au, y_au, z_au; // Position (AU) from SSB
    double vx_aud, vy_aud, vz_aud; // Velocity (AU/day) relative to SSB
};

std::vector<Particle> init_solar_system() {
    std::vector<Particle> particles_vec;
    int current_id = 0;

    // Data for JD 2451545.0 (2000-Jan-01 12:00:00.0000 TDB)
    // Source: JPL HORIZONS system (queries via web interface or API)
    // Target Body: Solar System Barycenter (SSB) [500@0]
    // Coordinate System: ICRF/J2000.0
    std::vector<PlanetEphemeris> solar_system_data = {
        // Name, Mass (kg), X (AU), Y (AU), Z (AU), VX (AU/day), VY (AU/day), VZ (AU/day)
        {"Sun",     1.988500e+30, -4.703265888996817E-03,  2.020896338344178E-03,  2.300822831930518E-05, -4.311604937990003E-06, -9.042310058510105E-06,  1.411701402003169E-07},
        {"Mercury", 3.301140e+23, -1.575608503748138E-01,  4.440050882520828E-01,  2.004971783143501E-02, -2.211890007853935E-02, -5.742282540117116E-03,  1.843223965904333E-03},
        {"Venus",   4.867470e+24,  2.222349437704366E-01, -6.831685254906887E-01, -3.019053791733102E-02,  1.898664762798181E-02,  6.373948415280699E-03, -9.042925354300879E-04},
        {"Earth",   5.972370e+24, -9.779570901960066E-01,  1.790986791213419E-01, -2.322841222568360E-05, -3.580782880111865E-03, -1.691097382355837E-02,  3.531464441413169E-07}, // Earth-Moon Barycenter often used
        {"Mars",    6.417120e+23, -3.591255778704809E-01,  1.489295350627074E+00,  4.203699623547142E-02, -1.347489808034809E-02, -2.208977841532108E-03,  2.616493967416403E-04},
        {"Jupiter", 1.898187e+27,  3.204240086701498E+00, -4.119949074777258E+00, -7.250064228629112E-02,  5.556764502281973E-03,  4.691074903954076E-03, -1.310435297548823E-04},
        {"Saturn",  5.683420e+26,  9.007258774343130E+00,  3.002478140802385E+00, -3.021544841910142E-01, -1.865043048628939E-03,  4.953285081615811E-03,  6.301797576178536E-05},
        {"Uranus",  8.681270e+25,  1.874462093421343E+01, -7.507853342449902E+00, -3.348298814082154E-01,  1.442087769847788E-03,  3.242031042652039E-03, -4.733323475402258E-05},
        {"Neptune", 1.024126e+26,  2.940975708390846E+01,  5.412605273570493E+00, -5.891819856220199E-01, -5.642493869220977E-04,  2.981130490260478E-03,  6.723978855960388E-05}
    };

    for (const auto& data : solar_system_data) {
        particles_vec.emplace_back(
            current_id++, data.name,
            data.x_au * AU_TO_METERS, data.y_au * AU_TO_METERS, data.z_au * AU_TO_METERS,
            data.mass_kg,
            data.vx_aud * AU_TO_METERS / DAYS_TO_SECONDS,
            data.vy_aud * AU_TO_METERS / DAYS_TO_SECONDS,
            data.vz_aud * AU_TO_METERS / DAYS_TO_SECONDS
        );
    }
    return particles_vec;
}


// --- Main Simulation ---
int main(int argc, char* argv[]) {
    // Default to 365 steps, with dt = 1 day, for a 1-year simulation.
    int steps = 365;
    double dt = DAYS_TO_SECONDS * 1.0; // Time step: 1 day in seconds
    std::string method = "fmm";     // Direct summation is best for N=9
    double theta = 0.5;                // Not used if method is "direct"
    bool mobile_sun = true;            // Sun (particle 0) is mobile, as coordinates are SSB
    
    std::string traj_fname_base = "trajectories_solarsystem";
    std::string energy_fname_base = "energy_drift_solarsystem";

    // Command-line argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "-dt_days" && i + 1 < argc) dt = std::stod(argv[++i]) * DAYS_TO_SECONDS;
        else if (arg == "-method" && i + 1 < argc) method = argv[++i];
        else if (arg == "-theta" && i + 1 < argc) theta = std::stod(argv[++i]);
        else if (arg == "--fixed-sun") mobile_sun = false; 
        else if (arg == "-traj_out" && i + 1 < argc) traj_fname_base = argv[++i];
        else if (arg == "-energy_out" && i + 1 < argc) energy_fname_base = argv[++i];
        else {
            std::cerr << "Usage: " << argv[0] 
                      << " [-steps N_STEPS (default: " << steps << " for " << steps*dt/DAYS_TO_SECONDS/365.25 << " years)]"
                      << " [-dt_days DT_IN_DAYS (default: " << dt/DAYS_TO_SECONDS << " days)]"
                      << " [-method direct|fmm (default: " << method << ")]"
                      << " [--fixed-sun (default: mobile Sun/SSB frame)]"
                      << " [-traj_out TRAJ_FNAME_BASE] [-energy_out ENERGY_FNAME_BASE]"
                      << std::endl;
            return 1;
        }
    }
    
    std::vector<Particle> particles = init_solar_system();
    int total_particles_in_sim = particles.size(); // Should be 9 for Sun + 8 planets

    if (total_particles_in_sim == 0) {
        std::cerr << "Error: No particles initialized. Exiting." << std::endl;
        return 1;
    }

    std::string traj_fname = traj_fname_base + "_N" + std::to_string(total_particles_in_sim) + "_rev1.csv";
    std::string energy_fname = energy_fname_base + "_N" + std::to_string(total_particles_in_sim) + "_rev1.csv";

    std::ofstream traj_file(traj_fname);
    std::ofstream energy_file(energy_fname);

    if (!traj_file.is_open()) { std::cerr << "Error: Could not open trajectory file " << traj_fname << std::endl; return 1;}
    if (!energy_file.is_open()) { std::cerr << "Error: Could not open energy file " << energy_fname << std::endl; return 1;}

    traj_file << "step,particle_id,name,x,y,z,vx,vy,vz\n"; 
    energy_file << "step,total_energy\n";
    
    // Initial force calculation
    if (method == "direct") compute_forces_direct(particles);
    else compute_forces_octree(particles, theta);
    
    std::cout << "Starting Solar System simulation: N=" << total_particles_in_sim
              << ", Steps=" << steps << " (Total " << steps * dt / DAYS_TO_SECONDS / 365.25 << " years)"
              << ", dt=" << dt << " s (" << dt/DAYS_TO_SECONDS << " days per step)"
              << ", Method=" << method
              << (mobile_sun ? ", Mobile Sun (SSB Frame)" : ", Fixed Sun (Warning: Inconsistent with SSB initial data if Sun is not particle 0 or data is not heliocentric)")
              << std::endl;
    std::cout << "Outputting trajectories to: " << traj_fname << std::endl;
    std::cout << "Outputting energy to: " << energy_fname << std::endl;

    for (int step = 0; step < steps; ++step) {
        for (const auto& p : particles) {
            traj_file << step << "," << p.id << "," << p.name << ","
                      << p.x << "," << p.y << "," << p.z << ","
                      << p.vx << "," << p.vy << "," << p.vz << "\n";
        }
        double current_energy = system_energy(particles);
        energy_file << step << "," << std::scientific << std::setprecision(10) << current_energy << "\n";

        if (step == 0 || (step + 1) % std::max(1, steps / 20) == 0 || step == steps - 1) { 
             double years_simulated = (step + 1) * dt / DAYS_TO_SECONDS / 365.25;
             std::cout << "Step " << step + 1 << "/" << steps 
                       << " (Year " << std::fixed << std::setprecision(3) << years_simulated << ")"
                       << " | Energy: " << std::scientific << std::setprecision(10) << current_energy << std::endl;
        }
        leapfrog_step(particles, dt, method, theta, mobile_sun);
    }
    
    std::cout << "Simulation finished." << std::endl;
    traj_file.close();
    energy_file.close();
    return 0;
}