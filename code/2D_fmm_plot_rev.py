import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_all_results(csv_file="timing_results.csv", output_dir="plots_fmm"):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV file is empty. No data to plot.")
        return

    df['NumParticles'] = df['NumParticles'].astype(int)
    df['NumCores'] = df['NumCores'].astype(int)
    df['TimeSeconds'] = df['TimeSeconds'].astype(float)
    df.sort_values(by=['Algorithm', 'NumParticles', 'NumCores'], inplace=True)

    # Filter: remove Direct entries beyond certain size
    MAX_DIRECT_N = 32768
    df = df[~((df['Algorithm'] == 'Direct') & (df['NumParticles'] > MAX_DIRECT_N))]

    algorithms = df['Algorithm'].unique()
    particle_counts = sorted(df['NumParticles'].unique())
    core_counts = sorted(df['NumCores'].unique())

    # Calculate speedup
    df['Speedup'] = 0.0
    for algo in algorithms:
        for n in particle_counts:
            subset = df[(df['Algorithm'] == algo) & (df['NumParticles'] == n)]
            if not subset.empty:
                t1 = subset[subset['NumCores'] == 1]['TimeSeconds']
                if not t1.empty:
                    t1_val = t1.values[0]
                    df.loc[subset.index, 'Speedup'] = t1_val / df.loc[subset.index, 'TimeSeconds']

    plt.style.use('seaborn-v0_8-whitegrid')

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Speedup vs Num Cores (faceted by N) ---
    print("Generating speedup vs cores (faceted)...")
    speedup_df = df[(df['Algorithm'] == 'FMM') & (df['Speedup'] > 0)]
    unique_ns = sorted(speedup_df['NumParticles'].unique())

    if not unique_ns:
        print("No valid data for FMM speedup plot.")
        return

    ncols = 2
    nrows = (len(unique_ns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    cmap = plt.get_cmap("tab10")
    color_cycle = cmap.colors

    for i, n in enumerate(unique_ns):
        ax = axes[i]
        subdf = speedup_df[speedup_df['NumParticles'] == n]
        ax.plot(subdf['NumCores'], subdf['Speedup'], marker='o', label=f'N={n}', color=color_cycle[i % len(color_cycle)])
        ax.plot([1, max(core_counts)], [1, max(core_counts)], linestyle='--', color='gray', label='Ideal')
        ax.set_title(f"Speedup for N={n}")
        ax.set_xticks(core_counts)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend()

    # Clear any extra unused subplots
    for j in range(len(unique_ns), len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Number of Cores")
    fig.supylabel("Speedup (T1 / TN)")
    fig.suptitle("FMM Parallel Speedup vs Cores (Faceted by N)")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "plot_speedup_vs_n_cores_faceted_fmm_only.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    plt.close(fig)

if __name__ == "__main__":
    plot_all_results()

