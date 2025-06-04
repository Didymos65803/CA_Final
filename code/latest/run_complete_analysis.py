#!/usr/bin/env python3
"""run_complete_analysis.py - Python-based analysis runner for cross-platform compatibility"""

import os
import sys
import subprocess
import time
import pathlib

def run_command(cmd, description=""):
    """Run a command and handle errors gracefully."""
    if description:
        print(f"{description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_files():
    """Check if required files exist."""
    required_files = ["fmm_openmp.cpp", "benchmark_improved.py", "setup_improved.py"]
    missing_files = []
    
    for file in required_files:
        if not pathlib.Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files: {', '.join(missing_files)}")
        return False
    
    return True

def clean_build():
    """Clean previous builds."""
    print("Cleaning previous builds...")
    
    dirs_to_remove = ["build", "dist", "__pycache__"]
    files_to_remove = ["*.egg-info", "*.so", "fmm_openmp*.so"]
    
    for dir_name in dirs_to_remove:
        if pathlib.Path(dir_name).exists():
            import shutil
            shutil.rmtree(dir_name)
    
    import glob
    for pattern in files_to_remove:
        for file in glob.glob(pattern):
            pathlib.Path(file).unlink(missing_ok=True)

def build_extension():
    """Build the optimized extension."""
    print("Building optimized FMM extension with OpenMP...")
    
    success = run_command("python setup_improved.py build_ext --inplace", 
                         "Compiling C++ extension")
    
    if not success:
        print("Build failed! Check your compiler and OpenMP installation.")
        return False
    
    # Verify the module can be imported
    try:
        import fmm_openmp
        max_threads = fmm_openmp.get_max_threads()
        print(f"✓ Module imported successfully. OpenMP threads available: {max_threads}")
        return True
    except ImportError as e:
        print(f"ERROR: Cannot import fmm_openmp module: {e}")
        return False

def set_environment():
    """Set optimal OpenMP environment variables."""
    env_vars = {
        'OMP_PROC_BIND': 'true',
        'OMP_PLACES': 'cores', 
        'OMP_DYNAMIC': 'false',
        'OMP_NESTED': 'false'
    }
    
    print("Setting OpenMP environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

def get_system_info():
    """Display system information."""
    print("\nSystem Information:")
    
    # Get CPU count
    cpu_count = os.cpu_count()
    print(f"  CPU cores: {cpu_count}")
    
    # Get Python version
    python_version = sys.version.split()[0]
    print(f"  Python version: {python_version}")
    
    # Check OpenMP status
    try:
        import fmm_openmp
        openmp_status = "Available" if fmm_openmp.get_max_threads() > 1 else "Not available"
        print(f"  OpenMP status: {openmp_status}")
    except ImportError:
        print("  OpenMP status: Module not available")

def run_quick_test():
    """Run quick performance test."""
    print("\n=== Running Quick Performance Test ===")
    success = run_command("python quick_test.py", "Quick performance test")
    return success

def run_basic_analysis():
    """Run basic benchmark analysis."""
    print("\n=== Running Basic Analysis ===")
    
    cmd = """python benchmark_improved.py \
        --sizes 2e3 4e3 8e3 1.6e4 \
        --threads 1 2 4 8 \
        --theta 0.3 0.5 0.7 1.0 \
        --theta_base 0.6"""
    
    success = run_command(cmd, "Basic performance analysis")
    return success

def run_detailed_analysis():
    """Run detailed benchmark analysis."""
    print("\n=== Running Detailed Analysis ===")
    
    cmd = """python benchmark_improved.py \
        --sizes 1e3 2e3 4e3 8e3 1.6e4 3.2e4 \
        --threads 1 2 4 8 16 \
        --theta 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 \
        --theta_base 0.6 \
        --detailed_analysis"""
    
    success = run_command(cmd, "Detailed performance analysis")
    return success

def show_results():
    """Display analysis results."""
    results_dir = pathlib.Path("results_enhanced")
    
    if not results_dir.exists():
        print("WARNING: Results directory not found!")
        return
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {results_dir.absolute()}")
    
    print("\nGenerated files:")
    for file in sorted(results_dir.iterdir()):
        size = file.stat().st_size
        print(f"  {file.name} ({size:,} bytes)")
    
    # Show performance summary if available
    summary_file = results_dir / "performance_summary.txt"
    if summary_file.exists():
        print("\n=== Performance Summary ===")
        with open(summary_file, 'r') as f:
            print(f.read())
    
    # Show quick results from TSV files
    size_sweep_file = results_dir / "detailed_size_sweep.tsv"
    if size_sweep_file.exists():
        print("\n=== Algorithmic Speedups (Direct/FMM) ===")
        with open(size_sweep_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        n, speedup = parts[0], parts[3]
                        print(f"  N={n}: {float(speedup):.1f}× speedup")
    
    thread_scaling_file = results_dir / "detailed_thread_scaling.tsv"
    if thread_scaling_file.exists():
        print("\n=== Thread Scaling Efficiency ===")
        with open(thread_scaling_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        threads, direct_eff, fmm_eff = parts[0], parts[5], parts[6]
                        print(f"  {threads} threads: Direct={float(direct_eff):.2f}, FMM={float(fmm_eff):.2f}")

def main():
    """Main analysis pipeline."""
    print("=== FMM OpenMP Performance Analysis Pipeline ===")
    print("Starting comprehensive performance analysis...")
    
    # Step 1: Check prerequisites
    if not check_files():
        return 1
    
    # Step 2: Clean and build
    clean_build()
    if not build_extension():
        return 1
    
    # Step 3: Setup environment
    set_environment()
    get_system_info()
    
    # Step 4: Run tests based on command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run FMM performance analysis')
    parser.add_argument('--quick', action='store_true', help='Run only quick test')
    parser.add_argument('--detailed', action='store_true', help='Run detailed analysis')
    parser.add_argument('--skip-build', action='store_true', help='Skip compilation step')
    
    args = parser.parse_args()
    
    if args.skip_build:
        print("Skipping build step as requested...")
    
    success = True
    
    if args.quick:
        success = run_quick_test()
    elif args.detailed:
        success = run_basic_analysis() and run_detailed_analysis()
    else:
        # Default: run quick test + basic analysis
        success = run_quick_test() and run_basic_analysis()
    
    # Step 5: Show results
    if success:
        show_results()
        print("\n✓ Analysis completed successfully!")
        print("\nTo view plots, open the PNG files in results_enhanced/ directory")
        print("For detailed data analysis, examine the TSV files")
        return 0
    else:
        print("\n✗ Analysis failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
