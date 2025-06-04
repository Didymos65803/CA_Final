#!/bin/bash
# build_and_test.sh - Complete build and test script

echo "🚀 Building Final Optimized N-Body Kernel"
echo "========================================"

echo "Step 1: Building kernel..."
python setup_final_optimized.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Step 2: Running quick test..."
    python quick_test.py
    echo ""
    echo "🎉 Complete! Your optimized N-body kernel is ready."
    echo ""
    echo "💡 Usage in your code:"
    echo "import final_optimized_kernel"
    echo "import os"
    echo "os.environ['OMP_NUM_THREADS'] = '2'"
    echo "final_optimized_kernel.benchmark_and_choose(x, y, m, eps2, ax, ay)"
else
    echo "❌ Build failed!"
    echo "Check that you have:"
    echo "  - GCC with OpenMP support"
    echo "  - pybind11 installed"
    echo "  - Python development headers"
fi
