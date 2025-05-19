# plot_simulation_data_3d.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

def plot_static_3d_trajectories(trajectory_df, n_total_particles, method_str, output_filename="trajectories_3d_rev1.png"):
    """Plots the static 3D trajectories of all particles."""
    if trajectory_df.empty:
        print("[Warn] Trajectory data is empty. Skipping static 3D trajectory plot.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Trajectories – N={n_total_particles} ({method_str})")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Check if particle_id 0 (central star) exists
    star_initial_pos = trajectory_df[(trajectory_df['particle_id'] == 0) & (trajectory_df['step'] == 0)]

    for i in range(n_total_particles):
        particle_traj = trajectory_df[trajectory_df['particle_id'] == i]
        if not particle_traj.empty:
            if i == 0 and not star_initial_pos.empty: # Central star
                ax.plot(particle_traj['x'], particle_traj['y'], particle_traj['z'],
                        lw=1.0, alpha=0.8, color='red', label="Star (Particle 0)" if n_total_particles < 10 else None)
                ax.scatter(star_initial_pos['x'].iloc[0], star_initial_pos['y'].iloc[0], star_initial_pos['z'].iloc[0],
                           s=150, c="red", marker='*', edgecolors='black', label="Star Start")
            else: # Other particles
                ax.plot(particle_traj['x'], particle_traj['y'], particle_traj['z'],
                        lw=0.5, alpha=0.5, label=f"Particle {i}" if n_total_particles < 10 else None)
    
    # Determine plot limits based on all particle positions
    all_x = trajectory_df['x']
    all_y = trajectory_df['y']
    all_z = trajectory_df['z']

    if not all_x.empty: # Check if dataframe is not empty
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()

        center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.6 # ensure 0.6 factor for padding

        ax.set_xlim(center_x - max_range, center_x + max_range)
        ax.set_ylim(center_y - max_range, center_y + max_range)
        ax.set_zlim(center_z - max_range, center_z + max_range)
    else: # Fallback limits if no data
        ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_zlim(-40, 40)


    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")
    
    if n_total_particles < 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend if outside
    plt.savefig(output_filename, dpi=300)
    print(f"[Saved] Static 3D trajectory plot: {output_filename}")
    plt.close(fig)

def plot_energy_drift(energy_df, n_total_particles, output_filename="energy_drift_3d_rev1.png"):
    """Plots the total system energy over time (2D plot)."""
    if energy_df.empty:
        print("[Warn] Energy data is empty. Skipping energy plot.")
        return
        
    plt.figure(figsize=(7, 4))
    plt.plot(energy_df['step'], energy_df['total_energy'])
    plt.xlabel("Simulation Step")
    plt.ylabel("$E_{total}$")
    plt.title(f"System Energy Drift (N={n_total_particles})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[Saved] Energy drift plot: {output_filename}")
    plt.close()

def animate_3d_simulation(trajectory_df, n_total_particles, num_steps, output_filename="simulation_3d_rev1.gif"):
    """Creates and saves a 3D GIF animation of the simulation."""
    if trajectory_df.empty:
        print("[Warn] Trajectory data is empty. Skipping 3D animation.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine plot limits from the whole dataset for consistent animation window
    all_x, all_y, all_z = trajectory_df['x'], trajectory_df['y'], trajectory_df['z']
    if not all_x.empty:
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()

        center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.6
        ax.set_xlim(center_x - max_range, center_x + max_range)
        ax.set_ylim(center_y - max_range, center_y + max_range)
        ax.set_zlim(center_z - max_range, center_z + max_range)
    else:
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_zlim(-50, 50)
        
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Initial scatter plot elements
    # Star (particle 0) - assuming it exists
    star_data_initial = trajectory_df[(trajectory_df['particle_id'] == 0) & (trajectory_df['step'] == 0)]
    if not star_data_initial.empty:
        scat_star = ax.scatter(star_data_initial['x'], star_data_initial['y'], star_data_initial['z'], s=100, c="red", marker='*')
    else: # Dummy scatter if no star at step 0
        scat_star = ax.scatter([], [], [], s=100, c="red", marker='*')


    # Other particles (particles 1 to N-1)
    # For performance, group other particles into one scatter plot if many, or plot individually if few.
    # Here, we prepare for individual updating if colors are desired, or one for all.
    # Let's try one scatter for all other particles for simplicity in updating.
    other_particles_initial = trajectory_df[(trajectory_df['particle_id'] != 0) & (trajectory_df['step'] == 0)]
    if not other_particles_initial.empty:
        scat_others = ax.scatter(other_particles_initial['x'], other_particles_initial['y'], other_particles_initial['z'], s=15, c="blue", alpha=0.7)
    else: # Dummy scatter
        scat_others = ax.scatter([], [], [], s=15, c="blue", alpha=0.7)

    title = ax.set_title(f"3D Simulation N={n_total_particles} - Step 0")
    # Optional: Set a fixed view angle
    ax.view_init(elev=20, azim=30)


    def update(frame):
        # Star
        current_star_pos = trajectory_df[(trajectory_df['particle_id'] == 0) & (trajectory_df['step'] == frame)]
        if not current_star_pos.empty:
            # For 3D scatter, need to update _offsets3d (a bit of a hack but common)
            scat_star._offsets3d = (current_star_pos['x'].values, current_star_pos['y'].values, current_star_pos['z'].values)
        
        # Other particles
        current_other_particles_data = trajectory_df[(trajectory_df['particle_id'] != 0) & (trajectory_df['step'] == frame)]
        if not current_other_particles_data.empty:
            scat_others._offsets3d = (current_other_particles_data['x'].values, 
                                      current_other_particles_data['y'].values, 
                                      current_other_particles_data['z'].values)
        else: # Hide if no data
            scat_others._offsets3d = (np.array([]), np.array([]), np.array([]))

        title.set_text(f"3D Simulation N={n_total_particles} - Step {frame}")
        # scat_star and scat_others are part of the returned list
        return [scat_star, scat_others, title]


    actual_num_steps = trajectory_df['step'].max() + 1 if not trajectory_df.empty else num_steps
    print(f"Creating 3D animation for {actual_num_steps} steps...")
    
    ani = FuncAnimation(fig, update, frames=tqdm(range(actual_num_steps), desc="Animating 3D"), blit=True, interval=50) # blit=True might have issues with 3D sometimes
    
    try:
        ani.save(output_filename, writer=PillowWriter(fps=20))
        print(f"[Saved] 3D Animation: {output_filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Try setting blit=False in FuncAnimation if you encounter issues, or ensure Pillow is correctly installed.")
        print("For MP4 output (recommended for 3D), ensure ffmpeg is installed and use writer='ffmpeg' or FFMpegWriter.")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 3D N-body simulation data from CSV files.")
    parser.add_argument("--traj_file", type=str, required=True, help="Path to the 3D trajectory CSV file (cols: step,particle_id,x,y,z,vx,vy,vz).")
    parser.add_argument("--energy_file", type=str, required=True, help="Path to the energy CSV file.")
    parser.add_argument("--n_particles", type=int, required=True, help="Total number of particles in the simulation.")
    parser.add_argument("--steps", type=int, required=True, help="Number of simulation steps (for animation frame count).")
    parser.add_argument("--method", type=str, default="unknown", help="Method used for simulation (for plot titles).")
    parser.add_argument("--output_traj_png", type=str, default="trajectories_3d_plot_rev1.png", help="Output filename for static 3D trajectory plot.")
    parser.add_argument("--output_energy_png", type=str, default="energy_drift_3d_plot_rev1.png", help="Output filename for energy drift plot.")
    parser.add_argument("--output_gif", type=str, default="simulation_3d_anim_rev1.gif", help="Output filename for 3D GIF animation.")
    parser.add_argument("--skip_animation", action="store_true", help="Skip creating the GIF animation.")

    args = parser.parse_args()

    print(f"Reading 3D trajectory data from: {args.traj_file}")
    try:
        trajectory_df = pd.read_csv(args.traj_file)
        # Ensure correct data types if not automatically inferred
        trajectory_df['x'] = pd.to_numeric(trajectory_df['x'], errors='coerce')
        trajectory_df['y'] = pd.to_numeric(trajectory_df['y'], errors='coerce')
        trajectory_df['z'] = pd.to_numeric(trajectory_df['z'], errors='coerce')
        trajectory_df = trajectory_df.dropna(subset=['x', 'y', 'z']) # Drop rows where conversion failed
    except FileNotFoundError:
        print(f"Error: Trajectory file not found: {args.traj_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Trajectory file is empty: {args.traj_file}")
        trajectory_df = pd.DataFrame()

    print(f"Reading energy data from: {args.energy_file}")
    try:
        energy_df = pd.read_csv(args.energy_file)
    except FileNotFoundError:
        print(f"Error: Energy file not found: {args.energy_file}")
        energy_df = pd.DataFrame() 
    except pd.errors.EmptyDataError:
        print(f"Error: Energy file is empty: {args.energy_file}")
        energy_df = pd.DataFrame()

    plot_static_3d_trajectories(trajectory_df, args.n_particles, args.method, args.output_traj_png)
    plot_energy_drift(energy_df, args.n_particles, args.output_energy_png)
        
    if not args.skip_animation:
        animate_3d_simulation(trajectory_df, args.n_particles, args.steps, args.output_gif)
    else:
        print("3D Animation skipped by user request.")
        
    print("3D Plotting complete.")

if __name__ == "__main__":
    main()