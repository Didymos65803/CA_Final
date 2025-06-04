import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_performance(pivot_df):
    """Generates and saves the performance plot (Time vs. N)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    markers = {'Direct': 'x', 'BH': 'o', 'FMM': 's'}
    for method in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[method], 
                marker=markers.get(method, 'v'), 
                linestyle='-', 
                label=f'{method} (Measured)')

    if not pivot_df.empty and 512 in pivot_df.index:
        n_range = pivot_df.index.to_numpy()
        if 'Direct' in pivot_df.columns and not pd.isna(pivot_df.loc[512, 'Direct']):
            c_n2 = pivot_df.loc[512, 'Direct'] / (512**2)
            ax.plot(n_range, c_n2 * n_range**2, ':', color='gray', label='Theoretical O(N²)')
        if 'BH' in pivot_df.columns and not pd.isna(pivot_df.loc[512, 'BH']):
            c_nlogn = pivot_df.loc[512, 'BH'] / (512 * np.log(512))
            ax.plot(n_range, c_nlogn * n_range * np.log(n_range), '--', color='gray', label='Theoretical O(N log N)')
        if 'FMM' in pivot_df.columns and not pd.isna(pivot_df.loc[512, 'FMM']):
            c_n = pivot_df.loc[512, 'FMM'] / 512
            ax.plot(n_range, c_n * n_range, '-.', color='gray', label='Theoretical O(N)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('N-Body Algorithm Performance Comparison')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.savefig('performance_comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✅ Performance plot saved as 'performance_comparison_plot.png'")
    plt.show()


def plot_accuracy(df):
    """Generates and saves the accuracy plot (Error vs. N)."""
    # Filter for methods that have a calculated error (error >= 0)
    error_df = df[(df['Method'] != 'Direct') & (df['Relative_Error'] >= 0)].copy()

    if error_df.empty:
        print("ℹ️ No error data found to plot. Did the C++ benchmark run on N <= 4096?")
        return
        
    error_pivot = error_df.pivot(index='N', columns='Method', values='Relative_Error')

    fig, ax = plt.subplots(figsize=(12, 8))
    markers = {'BH': 'o', 'FMM': 's'}
    for method in error_pivot.columns:
         ax.plot(error_pivot.index, error_pivot[method], 
                marker=markers.get(method, 'v'), 
                linestyle='--', 
                label=f'{method} Error')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel('Relative RMS Error')
    ax.set_title('N-Body Algorithm Accuracy Comparison')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    plt.savefig('accuracy_comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✅ Accuracy plot saved as 'accuracy_comparison_plot.png'")
    plt.show()


def main():
    """Main function to load data and trigger plots."""
    csv_file = 'performance_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: The file '{csv_file}' was not found.")
        print("Please compile and run the C++ benchmark program 'nbody_comparison' first.")
        return

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"❌ Error: The file '{csv_file}' is empty.")
            return
    except Exception as e:
        print(f"❌ Error: Could not read or parse '{csv_file}'. Error: {e}")
        return
        
    print(f"✅ Data loaded successfully from '{csv_file}'.")
    
    # --- Create Performance Plot ---
    time_pivot = df.pivot(index='N', columns='Method', values='Time_sec')
    print("\nPerformance Data (seconds):")
    print(time_pivot)
    plot_performance(time_pivot)

    # --- Create Accuracy Plot ---
    print("\nError Data:")
    print(df[['N', 'Method', 'Relative_Error']])
    plot_accuracy(df)


if __name__ == '__main__':
    main()