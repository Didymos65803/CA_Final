#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_positions(traj_file, step, output_png):
    """Scatter plot of x vs y for the given simulation step."""
    df = pd.read_csv(traj_file)
    df_step = df[df['step'] == step]
    if df_step.empty:
        print(f"[Warning] No data for step {step} in {traj_file}")
        return

    sun = df_step[df_step['particle_id'] == 0]
    jupiter = df_step[df_step['particle_id'] == 1]
    asteroids = df_step[df_step['particle_id'] >= 2]

    plt.figure(figsize=(6,6))
    plt.scatter(asteroids['x'], asteroids['y'], s=5, alpha=0.5, label='Asteroids')
    if not sun.empty:
        plt.scatter(sun['x'], sun['y'], s=100, marker='*', label='Sun')
    if not jupiter.empty:
        plt.scatter(jupiter['x'], jupiter['y'], s=50, marker='o', label='Jupiter')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-200, 200)  # Adjust as needed based on expected range
    plt.ylim(-200, 200)  # Adjust as needed based on expected range
    plt.title(f"Asteroid Belt at Step {step}")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[Saved] Position plot: {output_png}")
    plt.close()

def plot_radial_histogram(traj_file, step, bins, output_png):
    """Histogram of asteroid radial distances for the given simulation step."""
    df = pd.read_csv(traj_file)
    df_step = df[df['step'] == step]
    if df_step.empty:
        print(f"[Warning] No data for step {step} in {traj_file}")
        return

    asteroids = df_step[df_step['particle_id'] >= 2]
    radii = np.sqrt(asteroids['x']**2 + asteroids['y']**2)

    plt.figure(figsize=(6,4))
    plt.hist(radii, bins=bins, edgecolor='black')
    plt.xlabel("Radius")
    plt.ylabel("Number of Asteroids")
    plt.xlim(0, 200)  # Adjust as needed based on expected radius range
    plt.title(f"Radial Distribution at Step {step}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"[Saved] Radial histogram: {output_png}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Demonstrate asteroid belt CSV data")
    parser.add_argument("--traj_file", type=str, required=True,
                        help="Path to the trajectory CSV file")
    parser.add_argument("--step", type=int, default=None,
                        help="Simulation step to plot (default: final step)")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for the radial histogram")
    args = parser.parse_args()

    df_all = pd.read_csv(args.traj_file, usecols=["step"])
    max_step = df_all["step"].max()
    step = args.step if args.step is not None else max_step

    plot_positions(args.traj_file, step, f"positions_step{step}.png")
    plot_radial_histogram(args.traj_file, step, args.bins, f"radial_hist_step{step}.png")

if __name__ == "__main__":
    main()

