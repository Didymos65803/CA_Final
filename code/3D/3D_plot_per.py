import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_performance(df, num_threads_to_plot, theoretical_N_ref=512):
    """Generates and saves the performance plot (Time vs. N) for a specific thread count."""
    
    # Filter data for the specified number of threads
    filtered_df = df[df['Num_Threads'] == num_threads_to_plot]
    if filtered_df.empty:
        print(f"ℹ️ No data found for Num_Threads = {num_threads_to_plot} for performance plot. Skipping.")
        return

    pivot_df = filtered_df.pivot(index='N', columns='Method', values='Time_sec')
    
    if pivot_df.empty:
        print(f"ℹ️ Pivoted data is empty for Num_Threads = {num_threads_to_plot}. Skipping performance plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    markers = {'Direct': 'x', 'BH': 'o', 'FMM': 's'}
    
    for method in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[method], 
                marker=markers.get(method, 'v'), 
                linestyle='-', 
                label=f'{method} (Threads={num_threads_to_plot})')

    n_range_plot = pivot_df.index.to_numpy()
    
    # Check if theoretical_N_ref data exists for normalization
    if theoretical_N_ref in pivot_df.index:
        if 'Direct' in pivot_df.columns and not pd.isna(pivot_df.loc[theoretical_N_ref, 'Direct']):
            c_n2 = pivot_df.loc[theoretical_N_ref, 'Direct'] / (theoretical_N_ref**2)
            ax.plot(n_range_plot, c_n2 * n_range_plot**2, ':', color='gray', label=f'O(N²) (norm @N={theoretical_N_ref})')
        if 'BH' in pivot_df.columns and not pd.isna(pivot_df.loc[theoretical_N_ref, 'BH']):
            c_nlogn = pivot_df.loc[theoretical_N_ref, 'BH'] / (theoretical_N_ref * np.log(max(theoretical_N_ref,2)))
            ax.plot(n_range_plot, c_nlogn * n_range_plot * np.log(np.maximum(n_range_plot, 2)), '--', color='gray', label=f'O(N log N) (norm @N={theoretical_N_ref})')
        if 'FMM' in pivot_df.columns and not pd.isna(pivot_df.loc[theoretical_N_ref, 'FMM']):
            c_n = pivot_df.loc[theoretical_N_ref, 'FMM'] / theoretical_N_ref
            ax.plot(n_range_plot, c_n * n_range_plot, '-.', color='gray', label=f'O(N) (norm @N={theoretical_N_ref})')
    else:
        print(f"ℹ️ N={theoretical_N_ref} not found for Num_Threads={num_threads_to_plot} in pivoted data, theoretical scaling lines might be affected.")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Particles (N)')
    ax.set_ylabel(f'Execution Time (seconds) [Threads={num_threads_to_plot}]')
    ax.set_title(f'N-Body Algorithm Performance (Threads={num_threads_to_plot})')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.savefig(f'performance_comparison_plot_threads_{num_threads_to_plot}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Performance plot for {num_threads_to_plot} threads saved as 'performance_comparison_plot_threads_{num_threads_to_plot}.png'")


def plot_accuracy(df, num_threads_for_error_ref):
    """Generates and saves the accuracy plot (Error vs. N).
       Error should be independent of thread count, so we pick one for reference.
    """
    error_df = df[(df['Method'] != 'Direct') & 
                  (df['Relative_Error'] >= 0) & 
                  (df['Num_Threads'] == num_threads_for_error_ref)].copy()

    if error_df.empty:
        print(f"ℹ️ No valid error data for Num_Threads={num_threads_for_error_ref} to plot accuracy.")
        return
        
    error_pivot = error_df.pivot(index='N', columns='Method', values='Relative_Error')
    if error_pivot.empty:
        print(f"ℹ️ Pivoted error data is empty for Num_Threads={num_threads_for_error_ref}. Skipping accuracy plot.")
        return

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
    ax.set_title(f'N-Body Algorithm Accuracy (Error ref from {num_threads_for_error_ref} threads run)')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    plt.savefig('accuracy_comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✅ Accuracy plot saved as 'accuracy_comparison_plot.png'")


def plot_strong_scaling_speedup(full_df):
    """Plots Speedup vs. Number of Threads for each method and selected N values."""
    methods = sorted(full_df['Method'].unique())
    
    available_N_values = sorted(full_df['N'].unique())
    # Use all available N values for plotting
    N_values_to_plot = available_N_values

    if not N_values_to_plot:
        print("ℹ️ No N values found to plot strong scaling. Skipping.")
        return
    
    for method in methods:
        current_N_values_for_method = [N for N in N_values_to_plot if not (method == "Direct" and N > 32768)] # Limit N for Direct
        if not current_N_values_for_method : 
            print(f"ℹ️ No suitable N values for {method} to plot speedup (Direct method might be limited for large N). Skipping.")
            continue

        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        max_threads_data_for_method = 0

        for N_val in current_N_values_for_method:
            method_N_df = full_df[(full_df['Method'] == method) & (full_df['N'] == N_val)].sort_values(by='Num_Threads')
            
            if method_N_df.empty:
                # print(f"ℹ️ No data for {method} at N={N_val}. Skipping for speedup plot.")
                continue

            time_serial_series = method_N_df[method_N_df['Num_Threads'] == 1]['Time_sec']
            
            if time_serial_series.empty or pd.isna(time_serial_series.iloc[0]):
                # print(f"ℹ️ Serial (1-thread) data not found for {method} at N={N_val}. Cannot compute speedup.")
                continue
            
            time_serial = time_serial_series.iloc[0]
            if time_serial < 1e-9: 
                # print(f"ℹ️ Serial (1-thread) time is ~0 for {method} at N={N_val}. Cannot compute speedup.")
                continue

            method_N_df = method_N_df.copy() 
            method_N_df.loc[:, 'Speedup'] = time_serial / method_N_df['Time_sec']
            
            ax.plot(method_N_df['Num_Threads'], method_N_df['Speedup'], marker='o', linestyle='-', label=f'N={N_val}')
            if not method_N_df['Num_Threads'].empty:
                 max_threads_data_for_method = max(max_threads_data_for_method, method_N_df['Num_Threads'].max())

        if max_threads_data_for_method > 0: # Only plot ideal line if there's actual data
            ax.plot([1, max_threads_data_for_method], [1, max_threads_data_for_method], linestyle=':', color='k', label='Ideal Speedup')
        else: # No data plotted for this method
            plt.close() # Close the empty figure
            print(f"ℹ️ No speedup data plotted for method {method}. Skipping plot generation.")
            continue


        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Speedup (T_serial / T_parallel)')
        ax.set_title(f'Strong Scaling Speedup: {method}')
        
        thread_ticks = sorted(full_df[(full_df['Method'] == method)]['Num_Threads'].unique())
        if thread_ticks and max_threads_data_for_method > 0 : # Ensure ticks are set only if data exists
            ax.set_xticks(thread_ticks)
        
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'strong_scaling_speedup_{method.replace(" ", "_")}.png', dpi=150)
        print(f"✅ Strong scaling speedup plot saved as 'strong_scaling_speedup_{method.replace(' ', '_')}.png'")

def main():
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
        
    print(f"✅ Data loaded successfully from '{csv_file}'. Preview:")
    print(df.head())
    
    # Ensure correct data types
    df['N'] = pd.to_numeric(df['N'])
    df['Num_Threads'] = pd.to_numeric(df['Num_Threads'])
    df['Time_sec'] = pd.to_numeric(df['Time_sec'])
    df['Relative_Error'] = pd.to_numeric(df['Relative_Error'], errors='coerce') # Coerce errors to NaN if unparseable

    ref_threads_for_N_plot = 1
    if not df['Num_Threads'].empty:
        if 1 in df['Num_Threads'].unique():
            ref_threads_for_N_plot = 1 # Prefer 1 thread for baseline N-scaling if available
        else: # If 1-thread data is missing, use max available for N-scaling, but warn for speedup
            ref_threads_for_N_plot = df['Num_Threads'].max() 
            print(f"⚠️ Warning: 1-thread data point not found. Using {ref_threads_for_N_plot} threads for N-scaling plots. Speedup plots might be affected or skipped.")
        if pd.isna(ref_threads_for_N_plot): # Fallback if max() is NaN (empty Num_Threads)
            ref_threads_for_N_plot = 1
    
    # For performance plot, often one wants to see the "best" time, so max threads is also a good choice.
    # Let's use max threads available in the data for the performance vs N plot.
    max_threads_in_data = df['Num_Threads'].max()
    if pd.isna(max_threads_in_data) : max_threads_in_data = 1


    print(f"\nPlotting N-scaling performance using data from {max_threads_in_data} thread(s).")
    try:
        plot_performance(df, num_threads_to_plot=max_threads_in_data)
    except Exception as e:
        print(f"❌ Error creating performance (Time vs N) plot: {e}")

    print(f"\nPlotting accuracy using data from {max_threads_in_data} thread(s) for error reference.")
    try:
        plot_accuracy(df, num_threads_for_error_ref=max_threads_in_data)
    except Exception as e:
        print(f"❌ Error creating accuracy plot: {e}")

    print("\nPlotting strong scaling (Speedup vs Threads).")
    try:
        plot_strong_scaling_speedup(df)
    except Exception as e:
        print(f"❌ Error creating strong scaling speedup plot: {e}")

    if plt.get_fignums():
        print("\nDisplaying plots... Close plot windows to finish.")
        plt.show()
    else:
        print("\nNo plots were generated to display.")

if __name__ == '__main__':
    main()