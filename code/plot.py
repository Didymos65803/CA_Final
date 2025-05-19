# plot_simulation_data.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm # For progress bar if reading large files or for animation progress

def plot_static_trajectories(trajectory_df, n_particles, method_str, output_filename="trajectories_rev1.png"):
    """Plots the static trajectories of all particles."""
    plt.figure(figsize=(8, 8))
    plt.title(f"Trajectories – n={n_particles} ({method_str})")
    plt.grid(True, linestyle="--", alpha=0.7)

    central_star_pos = trajectory_df[trajectory_df['particle_id'] == 0][['x', 'y']].iloc[0]

    for i in range(1, n_particles): # Skip central star for path plotting
        particle_traj = trajectory_df[trajectory_df['particle_id'] == i]
        if not particle_traj.empty:
            plt.plot(particle_traj['x'], particle_traj['y'], lw=0.6, alpha=0.6, label=f"Particle {i}" if n_particles < 10 else None)

    plt.scatter(central_star_pos['x'], central_star_pos['y'], s=100, c="red", marker='*', label="Star (Initial)")
    
    # Determine plot limits based on all particle positions
    all_x = trajectory_df['x']
    all_y = trajectory_df['y']
    if not all_x.empty and not all_y.empty:
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 0.55
        center_x = (all_x.max() + all_x.min()) / 2
        center_y = (all_y.max() + all_y.min()) / 2
        plt.xlim(center_x - max_range, center_x + max_range)
        plt.ylim(center_y - max_range, center_y + max_range)
    else: # Fallback limits
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.gca().set_aspect('equal', adjustable='box')
    if n_particles < 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[Saved] Static trajectory plot: {output_filename}")
    plt.close()

def plot_energy_drift(energy_df, n_particles, output_filename="energy_drift_rev1.png"):
    """Plots the total system energy over time."""
    if energy_df.empty:
        print("[Warn] Energy data is empty. Skipping energy plot.")
        return
        
    plt.figure(figsize=(7, 4))
    plt.plot(energy_df['step'], energy_df['total_energy'])
    plt.xlabel("Simulation Step")
    plt.ylabel("$E_{total}$")
    plt.title(f"System Energy Drift (N={n_particles})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[Saved] Energy drift plot: {output_filename}")
    plt.close()

def animate_simulation(trajectory_df, n_particles, num_steps, output_filename="simulation_rev1.gif"):
    """Creates and saves a GIF animation of the simulation."""
    if trajectory_df.empty:
        print("[Warn] Trajectory data is empty. Skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Determine plot limits from the whole dataset for consistent animation window
    all_x = trajectory_df['x']
    all_y = trajectory_df['y']
    if not all_x.empty and not all_y.empty:
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    else: # Fallback limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    # Star (particle 0)
    star_df = trajectory_df[trajectory_df['particle_id'] == 0]
    scat_star = ax.scatter([], [], s=100, c="red", marker='*')
    
    # Other particles
    other_particles_df = trajectory_df[trajectory_df['particle_id'] != 0]
    # Use distinct colors for a few particles if N is small
    colors = plt.cm.viridis(np.linspace(0, 1, n_particles -1)) if n_particles > 1 else ['blue']
    
    # Create a list of scatter objects, one for each non-star particle
    scat_others_list = []
    if n_particles > 1:
      for i in range(1, n_particles):
          # Find initial data for particle i to set color, can be improved if colors should be fixed per particle_id
          particle_initial_data = other_particles_df[(other_particles_df['particle_id'] == i) & (other_particles_df['step'] == 0)]
          if not particle_initial_data.empty:
              scat_others_list.append(ax.scatter([], [], s=15, color=colors[i-1 if n_particles > 1 else 0]))
          else: # Add a dummy scatter if particle i doesn't exist at step 0 (should not happen with current init)
              scat_others_list.append(ax.scatter([],[], s=0)) # invisible
    
    title = ax.set_title(f"Simulation N={n_particles} - Step 0")

    def update(frame):
        # Star
        current_star_pos = star_df[star_df['step'] == frame]
        if not current_star_pos.empty:
            scat_star.set_offsets(current_star_pos[['x', 'y']].values)
        
        # Other particles
        current_other_particles_data = other_particles_df[other_particles_df['step'] == frame]
        for i, scat_obj in enumerate(scat_others_list):
            particle_id_to_plot = i + 1 # particle_id = 1, 2, ...
            pos_data = current_other_particles_data[current_other_particles_data['particle_id'] == particle_id_to_plot]
            if not pos_data.empty:
                scat_obj.set_offsets(pos_data[['x', 'y']].values)
            else: # Hide if no data for this particle at this frame
                 scat_obj.set_offsets(np.empty((0,2)))


        title.set_text(f"Simulation N={n_particles} - Step {frame}")
        return [scat_star] + scat_others_list + [title]

    # Determine the number of unique steps from the data
    actual_num_steps = trajectory_df['step'].max() + 1 if not trajectory_df.empty else num_steps

    print(f"Creating animation for {actual_num_steps} steps...")
    # Wrap frames with tqdm for a progress bar
    ani = FuncAnimation(fig, update, frames=tqdm(range(actual_num_steps), desc="Animating"), blit=True, interval=30)
    
    ani.save(output_filename, writer=PillowWriter(fps=30))
    print(f"[Saved] Animation: {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot N-body simulation data from CSV files.")
    parser.add_argument("--traj_file", type=str, required=True, help="Path to the trajectory CSV file.")
    parser.add_argument("--energy_file", type=str, required=True, help="Path to the energy CSV file.")
    parser.add_argument("--n_particles", type=int, required=True, help="Number of particles in the simulation (used for plotting).")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps in the simulation (used for animation).")
    parser.add_argument("--method", type=str, default="unknown", help="Method used for simulation (for plot titles).")
    parser.add_argument("--output_traj_png", type=str, default="trajectories_plot_rev1.png", help="Output filename for static trajectory plot.")
    parser.add_argument("--output_energy_png", type=str, default="energy_drift_plot_rev1.png", help="Output filename for energy drift plot.")
    parser.add_argument("--output_gif", type=str, default="simulation_anim_rev1.gif", help="Output filename for GIF animation.")
    parser.add_argument("--skip_animation", action="store_true", help="Skip creating the GIF animation.")


    args = parser.parse_args()

    print(f"Reading trajectory data from: {args.traj_file}")
    try:
        trajectory_df = pd.read_csv(args.traj_file)
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
        energy_df = pd.DataFrame() # Create empty dataframe to avoid downstream errors
    except pd.errors.EmptyDataError:
        print(f"Error: Energy file is empty: {args.energy_file}")
        energy_df = pd.DataFrame()


    if not trajectory_df.empty:
        plot_static_trajectories(trajectory_df, args.n_particles, args.method, args.output_traj_png)
    else:
        print("Skipping static trajectory plot due to empty trajectory data.")

    if not energy_df.empty:
        plot_energy_drift(energy_df, args.n_particles, args.output_energy_png)
    else:
        print("Skipping energy drift plot due to empty energy data.")
        
    if not args.skip_animation:
        if not trajectory_df.empty:
            animate_simulation(trajectory_df, args.n_particles, args.steps, args.output_gif)
        else:
            print("Skipping animation due to empty trajectory data.")
    else:
        print("Animation skipped by user request.")
        
    print("Plotting complete.")

if __name__ == "__main__":
    main()