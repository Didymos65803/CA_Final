#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define a sensible maximum extent for plot axes to prevent overflow issues
# Adjust this value based on the expected scale of your simulation
# If particles are expected to go further, increase this. If they shouldn't,
# this will clip the view, but prevent crashes.
SENSIBLE_MAX_PLOT_EXTENT = 1000.0 # Example: plot axes won't go beyond +/-1000

def create_animation(traj_file, bins, output_mp4):
    """
    Creates a side-by-side animation of particle positions and
    radial distribution from a trajectory CSV file, with robust axis limit handling.
    """
    print("[Loading] Reading trajectory data...")
    try:
        df = pd.read_csv(traj_file)
    except FileNotFoundError:
        print(f"[Error] Trajectory file not found: {traj_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"[Error] Trajectory file is empty: {traj_file}")
        return
    except Exception as e:
        print(f"[Error] Could not read trajectory file: {e}")
        return

    if df.empty:
        print("[Error] DataFrame is empty after loading. Cannot create animation.")
        return

    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['x', 'y'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"[Info] Dropped {dropped_rows} rows containing NaN/Inf in 'x' or 'y' columns.")

    if df.empty:
        print("[Error] DataFrame is empty after dropping NaN/Inf rows. Cannot create animation.")
        return

    steps = sorted(df['step'].unique())
    if not steps:
        print("[Error] No unique steps found in the filtered trajectory data.")
        return
    num_frames = len(steps)

    print(f"[Info] Using {df['particle_id'].nunique()} particles over {num_frames} steps for animation.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Pre-calculate robust axis limits ---
    # Position plot limits
    if df['x'].empty or df['y'].empty:
        print("[Warning] No valid x or y data after filtering. Using default position axis limits.")
        min_coord_x, max_coord_x = -10.0, 10.0
        min_coord_y, max_coord_y = -10.0, 10.0
    else:
        min_coord_x = df['x'].min()
        max_coord_x = df['x'].max()
        min_coord_y = df['y'].min()
        max_coord_y = df['y'].max()

    # Clip to sensible extents to prevent overflow
    min_coord_x = np.clip(min_coord_x, -SENSIBLE_MAX_PLOT_EXTENT, SENSIBLE_MAX_PLOT_EXTENT)
    max_coord_x = np.clip(max_coord_x, -SENSIBLE_MAX_PLOT_EXTENT, SENSIBLE_MAX_PLOT_EXTENT)
    min_coord_y = np.clip(min_coord_y, -SENSIBLE_MAX_PLOT_EXTENT, SENSIBLE_MAX_PLOT_EXTENT)
    max_coord_y = np.clip(max_coord_y, -SENSIBLE_MAX_PLOT_EXTENT, SENSIBLE_MAX_PLOT_EXTENT)
    
    # Ensure min is less than max after clipping
    if min_coord_x >= max_coord_x: max_coord_x = min_coord_x + 1.0
    if min_coord_y >= max_coord_y: max_coord_y = min_coord_y + 1.0


    max_abs_coord = max(abs(min_coord_x), abs(max_coord_x), abs(min_coord_y), abs(max_coord_y), 1.0)
    # The final display limit for scatter plot, ensuring it's symmetrical and scaled
    # This `max_coord` will be used for set_xlim and set_ylim
    max_coord_display = max_abs_coord * 1.1 
    # Further clip this display limit if it's still too large
    max_coord_display = np.clip(max_coord_display, 1.0, SENSIBLE_MAX_PLOT_EXTENT * 1.1)


    # Histogram plot limits
    df['radius'] = np.sqrt(df['x']**2 + df['y']**2) # Calculate radius on cleaned df
    
    if df['radius'].empty or df['radius'].isnull().all():
        print("[Warning] No valid radius data after filtering. Using default histogram limits.")
        hist_max_radius_data = 10.0
    else:
        hist_max_radius_data = df['radius'].max()

    hist_max_radius_display = np.clip(hist_max_radius_data, 0.1, SENSIBLE_MAX_PLOT_EXTENT)
    if not np.isfinite(hist_max_radius_display) or hist_max_radius_display <= 0:
        print(f"[Warning] hist_max_radius_display is invalid ({hist_max_radius_display}). Setting to default 10.0.")
        hist_max_radius_display = 10.0
        
    max_hist_y = 0
    for step_val in steps:
        step_df = df[df['step'] == step_val]
        asteroids_step_df = step_df[step_df['particle_id'] >= 2]
        if not asteroids_step_df.empty:
            radii_in_step = asteroids_step_df['radius']
            if not radii_in_step.empty and radii_in_step.notnull().any():
                # Ensure range for histogram is valid
                hist_range = (0, hist_max_radius_display if hist_max_radius_display > 0 else 1.0)
                counts, _ = np.histogram(radii_in_step.dropna(), bins=bins, range=hist_range)
                if counts.size > 0:
                    max_hist_y = max(max_hist_y, counts.max())
    
    if max_hist_y == 0: max_hist_y = 10

    # --- Animation update function ---
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        step = steps[frame]
        df_step = df[df['step'] == step]

        sun = df_step[df_step['particle_id'] == 0]
        jupiter = df_step[df_step['particle_id'] == 1]
        asteroids = df_step[df_step['particle_id'] >= 2]

        ax1.scatter(asteroids['x'], asteroids['y'], s=5, alpha=0.5, label='Asteroids')
        if not sun.empty:
            ax1.scatter(sun['x'], sun['y'], s=100, marker='*', label='Sun', color='gold')
        if not jupiter.empty:
            ax1.scatter(jupiter['x'], jupiter['y'], s=50, marker='o', label='Jupiter', color='orange')
        
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Particle Positions")
        ax1.legend(loc='upper right')
        ax1.axis('equal') # This can cause issues if limits are extreme
        ax1.set_xlim(-max_coord_display, max_coord_display)
        ax1.set_ylim(-max_coord_display, max_coord_display)
        ax1.grid(True, linestyle='--', alpha=0.5)

        asteroids_for_hist = df_step[df_step['particle_id'] >= 2]
        if not asteroids_for_hist.empty:
            radii_for_hist = asteroids_for_hist['radius']
            if not radii_for_hist.empty and radii_for_hist.notnull().any():
                 hist_range = (0, hist_max_radius_display if hist_max_radius_display > 0 else 1.0)
                 ax2.hist(radii_for_hist.dropna(), bins=bins, edgecolor='black', range=hist_range)
        
        ax2.set_xlabel("Radius")
        ax2.set_ylabel("Number of Asteroids")
        ax2.set_title("Radial Distribution")
        ax2.set_xlim(0, hist_max_radius_display)
        ax2.set_ylim(0, max_hist_y * 1.1 if max_hist_y > 0 else 10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        fig.suptitle(f"Asteroid Belt Simulation at Step {step}", fontsize=16)
        progress = (frame + 1) / num_frames * 100
        print(f"\r[Processing] Frame {frame + 1}/{num_frames} ({progress:.1f}%)", end="")

    print("\n[Creating Animation] This may take a few minutes...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False) # Increased interval slightly

    try:
        anim.save(output_mp4, writer='ffmpeg', dpi=150)
        print(f"\n[Success] Animation saved to: {output_mp4}")
    except FileNotFoundError:
        print("\n[Error] ffmpeg not found.")
        print("Please install ffmpeg and ensure it is in your system's PATH.")
    except Exception as e:
        print(f"\n[Error] Failed to save animation: {e}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Animate asteroid belt CSV data.")
    parser.add_argument("--traj_file", type=str, required=True,
                        help="Path to the trajectory CSV file")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for the radial histogram")
    parser.add_argument("--output", type=str, default="asteroid_belt_animation.mp4",
                        help="Output file name for the animation (e.g., animation.mp4)")
    args = parser.parse_args()

    create_animation(args.traj_file, args.bins, args.output)

if __name__ == "__main__":
    main()
