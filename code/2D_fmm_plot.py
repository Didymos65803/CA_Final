import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_all_results(csv_file="timing_results.csv"):
    """
    Reads timing results from a CSV file and generates multiple performance plots.
    """
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV file is empty. No data to plot.")
        return

    # Ensure correct data types
    df['NumParticles'] = df['NumParticles'].astype(int)
    df['NumCores'] = df['NumCores'].astype(int)
    df['TimeSeconds'] = df['TimeSeconds'].astype(float)

    df.sort_values(by=['Algorithm', 'NumParticles', 'NumCores'], inplace=True)

    algorithms = df['Algorithm'].unique()
    particle_counts = sorted(df['NumParticles'].unique())
    core_counts = sorted(df['NumCores'].unique())

    # --- Calculate Speedup ---
    df['Speedup'] = 0.0
    for particles in particle_counts:
        for algo in algorithms:
            filter_condition = (df['Algorithm'] == algo) & (df['NumParticles'] == particles)
            algo_particle_df = df[filter_condition]
            if not algo_particle_df.empty:
                t_1_core_series = algo_particle_df[algo_particle_df['NumCores'] == 1]['TimeSeconds']
                if not t_1_core_series.empty:
                    t_1_core = t_1_core_series.iloc[0]
                    # df.loc[filter_condition, 'Speedup'] = t_1_core / df.loc[filter_condition, 'TimeSeconds'] # This can be problematic with mixed indices
                    for index, row in algo_particle_df.iterrows():
                        if row['TimeSeconds'] > 1e-9: # Avoid division by zero
                            df.loc[index, 'Speedup'] = t_1_core / row['TimeSeconds']
                        else:
                            df.loc[index, 'Speedup'] = 0 # Or some other indicator like np.nan

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Computation Time vs. Number of Particles ---
    print("Generating Plot 1: Time vs. N_Particles...")
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    line_styles = ['-', '--', ':', '-.']

    for algo_idx, algo in enumerate(algorithms):
        algo_df = df[df['Algorithm'] == algo]
        for core_idx, cores in enumerate(core_counts):
            core_df = algo_df[algo_df['NumCores'] == cores]
            if not core_df.empty:
                ax1.plot(core_df['NumParticles'], core_df['TimeSeconds'],
                         label=f'{algo} ({cores} cores)',
                         marker=markers[algo_idx % len(markers)],
                         linestyle=line_styles[core_idx % len(line_styles)])

    # --- Add theoretical complexity lines ---
    # Use a reference N for normalization, e.g., one of the mid-range particle counts
    # and data from the 1-core run if available.
    theoretical_N_ref = 1024  # You can adjust this
    if particle_counts: # Ensure particle_counts is not empty
        n_range_plot = np.array(sorted(particle_counts)) # Use all N values for the theoretical lines

        # O(N^2) - typically for Direct
        direct_ref_df = df[(df['Algorithm'] == 'Direct') & (df['NumParticles'] == theoretical_N_ref) & (df['NumCores'] == 1)]
        if not direct_ref_df.empty and not pd.isna(direct_ref_df['TimeSeconds'].iloc[0]):
            c_n2 = direct_ref_df['TimeSeconds'].iloc[0] / (theoretical_N_ref**2)
            ax1.plot(n_range_plot, c_n2 * n_range_plot**2, ':', color='gray', label=f'O(N²) (norm @N={theoretical_N_ref})')
        else:
            print(f"ℹ️ Data for Direct method at N={theoretical_N_ref}, 1 core not found. Skipping O(N^2) line.")

        # O(N) - typically for FMM (ideal)
        fmm_ref_df = df[(df['Algorithm'] == 'FMM') & (df['NumParticles'] == theoretical_N_ref) & (df['NumCores'] == 1)]
        if not fmm_ref_df.empty and not pd.isna(fmm_ref_df['TimeSeconds'].iloc[0]):
            c_n = fmm_ref_df['TimeSeconds'].iloc[0] / theoretical_N_ref
            ax1.plot(n_range_plot, c_n * n_range_plot, '-.', color='dimgray', label=f'O(N) (norm @N={theoretical_N_ref})')
        else:
            print(f"ℹ️ Data for FMM method at N={theoretical_N_ref}, 1 core not found. Skipping O(N) line.")

    ax1.set_xlabel("Number of Particles (N)")
    ax1.set_ylabel("Computation Time (seconds)")
    ax1.set_title("Performance: Time vs. Number of Particles")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot_time_vs_n_particles.png")
    print("Saved plot_time_vs_n_particles.png")
    plt.close(fig1)

    # --- Plot 2: Speedup vs. Number of Cores ---
    print("\nGenerating Plot 2: Speedup vs. N_Cores...")
    # Determine number of subplots needed (one per particle count or fewer combined)
    # For simplicity, let's make one plot and use styles to differentiate particle counts if not too many.
    # If many particle counts, faceting is better. Let's try faceting if N_particle_counts > 3.
    
    unique_particle_counts_for_speedup = sorted(df[df['Speedup'] > 0]['NumParticles'].unique())

    if not unique_particle_counts_for_speedup:
        print("No data available for speedup plot (e.g., missing 1-core data).")
    else:
        num_particle_groups = len(unique_particle_counts_for_speedup)
        if num_particle_groups == 0:
             print("Skipping speedup vs cores plot as no particle groups with speedup data.")
        elif num_particle_groups <= 4 : # Single plot if few particle counts
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for algo_idx, algo in enumerate(algorithms):
                for p_idx, particles in enumerate(unique_particle_counts_for_speedup):
                    subset = df[(df['Algorithm'] == algo) &
                                (df['NumParticles'] == particles) &
                                (df['NumCores'] > 0) &
                                (df['Speedup'] > 0)]
                    if not subset.empty:
                        ax2.plot(subset['NumCores'], subset['Speedup'],
                                 marker=markers[algo_idx % len(markers)],
                                 linestyle=line_styles[p_idx % len(line_styles)],
                                 label=f'{algo} (N={particles})')
            if core_counts:
                max_cores_on_plot = max(core_counts) if core_counts else 1
                ax2.plot([1, max_cores_on_plot], [1, max_cores_on_plot], linestyle='--', color='gray', label='Ideal Speedup')

            ax2.set_xlabel("Number of Cores")
            ax2.set_ylabel("Speedup (T_1_core / T_N_cores)")
            ax2.set_title("Parallel Speedup vs. Number of Cores")
            ax2.legend(loc='best', fontsize='small')
            ax2.set_xticks(core_counts)
            ax2.grid(True, which="both", ls="-", alpha=0.5)
            plt.tight_layout()
            plt.savefig("plot_speedup_vs_n_cores.png")
            print("Saved plot_speedup_vs_n_cores.png")
            plt.close(fig2)
        else: # Faceted plot if many particle counts
            num_cols = 2
            num_rows = (num_particle_groups + num_cols - 1) // num_cols
            fig2, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), sharex=True, sharey=True, squeeze=False)
            axes_flat = axes.flatten()

            for i, particles in enumerate(unique_particle_counts_for_speedup):
                ax = axes_flat[i]
                for algo_idx, algo in enumerate(algorithms):
                    subset = df[(df['Algorithm'] == algo) &
                                (df['NumParticles'] == particles) &
                                (df['NumCores'] > 0) &
                                (df['Speedup'] > 0)]
                    if not subset.empty:
                        ax.plot(subset['NumCores'], subset['Speedup'],
                                marker=markers[algo_idx % len(markers)],
                                label=f'{algo}')
                
                if core_counts:
                    max_cores_on_plot = max(core_counts) if core_counts else 1
                    ax.plot([1, max_cores_on_plot], [1, max_cores_on_plot], linestyle='--', color='gray', label='Ideal Speedup')

                ax.set_title(f'N = {particles}')
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, which="both", ls="-", alpha=0.5)
                ax.set_xticks(core_counts)

            # Common labels
            fig2.supxlabel("Number of Cores", y=0.02)
            fig2.supylabel("Speedup (T_1_core / T_N_cores)", x=0.02)
            fig2.suptitle("Parallel Speedup vs. Number of Cores", fontsize=16)
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]) # Adjust layout for suptitle
            plt.savefig("plot_speedup_vs_n_cores_faceted.png")
            print("Saved plot_speedup_vs_n_cores_faceted.png")
            plt.close(fig2)


    # --- Plot 3: Speedup vs. Number of Particles (for fixed core counts) ---
    print("\nGenerating Plot 3: Speedup vs. N_Particles...")
    # Consider only core counts > 1 for this plot
    meaningful_core_counts = [c for c in core_counts if c > 1]

    if not meaningful_core_counts:
        print("No multi-core data available for Speedup vs. N_Particles plot.")
    else:
        num_core_groups = len(meaningful_core_counts)
        if num_core_groups == 0:
            print("Skipping speedup vs N particles plot as no multi-core groups.")
        elif num_core_groups <= 3: # Single plot
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            for algo_idx, algo in enumerate(algorithms):
                for core_idx, cores in enumerate(meaningful_core_counts):
                    subset = df[(df['Algorithm'] == algo) &
                                (df['NumCores'] == cores) &
                                (df['Speedup'] > 0)] # Only plot where speedup is calculated
                    if not subset.empty:
                        ax3.plot(subset['NumParticles'], subset['Speedup'],
                                 label=f'{algo} ({cores} cores)',
                                 marker=markers[algo_idx % len(markers)],
                                 linestyle=line_styles[core_idx % len(line_styles)])
            ax3.set_xlabel("Number of Particles (N)")
            ax3.set_ylabel(f"Speedup (T_1_core / T_N_cores)")
            ax3.set_title("Parallel Speedup vs. Number of Particles")
            ax3.set_xscale('log')
            ax3.legend(loc='best', fontsize='small')
            ax3.grid(True, which="both", ls="-", alpha=0.5)
            plt.tight_layout()
            plt.savefig("plot_speedup_vs_n_particles.png")
            print("Saved plot_speedup_vs_n_particles.png")
            plt.close(fig3)
        else: # Faceted plot
            num_cols = 2
            num_rows = (num_core_groups + num_cols - 1) // num_cols
            fig3, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), sharex=True, sharey=True, squeeze=False)
            axes_flat = axes.flatten()

            for i, cores in enumerate(meaningful_core_counts):
                ax = axes_flat[i]
                for algo_idx, algo in enumerate(algorithms):
                    subset = df[(df['Algorithm'] == algo) &
                                (df['NumCores'] == cores) &
                                (df['Speedup'] > 0)]
                    if not subset.empty:
                        ax.plot(subset['NumParticles'], subset['Speedup'],
                                marker=markers[algo_idx % len(markers)],
                                label=f'{algo}')
                ax.set_title(f'{cores} Cores')
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, which="both", ls="-", alpha=0.5)
                ax.set_xscale('log')

            fig3.supxlabel("Number of Particles (N)", y=0.02)
            fig3.supylabel(f"Speedup (T_1_core / T_N_cores)", x=0.02)
            fig3.suptitle("Parallel Speedup vs. Number of Particles", fontsize=16)
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
            plt.savefig("plot_speedup_vs_n_particles_faceted.png")
            print("Saved plot_speedup_vs_n_particles_faceted.png")
            plt.close(fig3)
        


    
    print("\nAll plotting finished.")
    # plt.show() # Optionally show plots interactively if not running in a script-only environment

if __name__ == "__main__":
    plot_all_results()