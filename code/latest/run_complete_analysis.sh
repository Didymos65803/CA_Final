#!/bin/bash
# run_complete_analysis.sh - Complete FMM performance analysis pipeline
# Make this file executable with: chmod +x run_complete_analysis.sh

set -e  # Exit on any error

echo "=== FMM OpenMP Performance Analysis Pipeline ==="
echo "Starting comprehensive performance analysis..."

# Check if required files exist
required_files=("fmm_openmp.cpp" "benchmark_improved.py" "setup_improved.py")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file $file not found!"
        exit 1
    fi
done

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ *.so fmm_openmp*.so 2>/dev/null || true

# Build optimized extension
echo "Building optimized FMM extension with OpenMP..."
python setup_improved.py build_ext --inplace

# Verify the module can be imported
echo "Verifying module import..."
python -c "import fmm_openmp; print(f'OpenMP threads available: {fmm_openmp.get_max_threads()}')"

# Check system info
echo "System Information:"
echo "  CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
echo "  Python version: $(python --version)"
echo "  OpenMP status: $(python -c 'import fmm_openmp; print("Available" if fmm_openmp.get_max_threads() > 1 else "Not available")')"

# Set optimal OpenMP environment variables
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_NESTED=false

echo "OpenMP Environment:"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo "  OMP_DYNAMIC=$OMP_DYNAMIC"

# Run basic analysis with larger problem sizes for better OpenMP scaling
echo ""
echo "=== Running Basic Analysis ==="
python benchmark_improved.py \
    --sizes 2e3 4e3 8e3 1.6e4 \
    --threads 1 2 4 8 \
    --theta 0.3 0.5 0.7 1.0 \
    --theta_base 0.6

# Run detailed analysis if requested
if [[ "$1" == "--detailed" ]]; then
    echo ""
    echo "=== Running Detailed Analysis ==="
    python benchmark_improved.py \
        --sizes 5e2 1e3 2e3 4e3 8e3 1.6e4 \
        --threads 1 2 4 8 16 \
        --theta 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 \
        --theta_base 0.6 \
        --detailed_analysis
fi

# Generate summary report
echo ""
echo "=== Analysis Complete ==="
echo "Results saved to: results_enhanced/"

if [[ -d "results_enhanced" ]]; then
    echo ""
    echo "Generated files:"
    ls -la results_enhanced/
    
    echo ""
    echo "=== Performance Summary ==="
    if [[ -f "results_enhanced/performance_summary.txt" ]]; then
        cat results_enhanced/performance_summary.txt
    fi
    
    echo ""
    echo "=== Quick Results Check ==="
    if [[ -f "results_enhanced/detailed_size_sweep.tsv" ]]; then
        echo "Algorithmic speedups (Direct/FMM):"
        tail -n +2 results_enhanced/detailed_size_sweep.tsv | awk -F'\t' '{printf "  N=%s: %.1fx speedup\n", $1, $4}'
    fi
    
    if [[ -f "results_enhanced/detailed_thread_scaling.tsv" ]]; then
        echo ""
        echo "Thread scaling efficiency:"
        tail -n +2 results_enhanced/detailed_thread_scaling.tsv | awk -F'\t' '{printf "  %s threads: Direct=%.2f, FMM=%.2f\n", $1, $6, $7}'
    fi
else
    echo "WARNING: Results directory not found!"
fi

echo ""
echo "To view the generated plots, open the PNG files in results_enhanced/ directory"
echo "For detailed analysis, examine the TSV files with your preferred data analysis tool"
