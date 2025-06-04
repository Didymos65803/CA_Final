#!/usr/bin/env python3
"""setup_and_run.py - Simple setup and execution script"""

import os
import sys
import subprocess
import pathlib

def make_executable(filename):
    """Make a file executable on Unix-like systems."""
    if os.name != 'nt':  # Not Windows
        try:
            os.chmod(filename, 0o755)
            print(f"Made {filename} executable")
            return True
        except Exception as e:
            print(f"Could not make {filename} executable: {e}")
            return False
    return True

def main():
    """Setup and run the analysis."""
    print("=== FMM OpenMP Analysis Setup ===")
    
    # Make shell script executable if it exists
    shell_script = "run_complete_analysis.sh"
    if pathlib.Path(shell_script).exists():
        make_executable(shell_script)
    
    # Determine which runner to use
    python_runner = "run_complete_analysis.py"
    
    print("\nAvailable execution methods:")
    print("1. Python runner (cross-platform, recommended)")
    print("2. Shell script (Unix/Linux/macOS only)")
    print("3. Manual step-by-step")
    
    choice = input("\nChoose execution method (1-3) [1]: ").strip()
    
    if choice == "2" and pathlib.Path(shell_script).exists():
        print(f"\nRunning shell script: {shell_script}")
        try:
            subprocess.run([f"./{shell_script}"], check=True)
        except subprocess.CalledProcessError:
            print("Shell script failed. Trying Python runner...")
            choice = "1"
        except FileNotFoundError:
            print("Shell script not executable. Trying Python runner...")
            choice = "1"
    
    if choice == "3":
        print("\nManual execution steps:")
        print("1. Compile: python setup_improved.py build_ext --inplace")
        print("2. Quick test: python quick_test.py")
        print("3. Full analysis: python benchmark_improved.py")
        return
    
    # Default to Python runner
    if pathlib.Path(python_runner).exists():
        print(f"\nRunning Python analysis: {python_runner}")
        
        # Ask for analysis type
        analysis_type = input("Analysis type - (q)uick, (b)asic, (d)etailed [b]: ").strip().lower()
        
        cmd = [sys.executable, python_runner]
        if analysis_type.startswith('q'):
            cmd.append('--quick')
        elif analysis_type.startswith('d'):
            cmd.append('--detailed')
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Python runner failed: {e}")
    else:
        print(f"Python runner {python_runner} not found!")
        print("Running quick test directly...")
        
        # Fallback: just run the quick test
        try:
            # First try to build
            subprocess.run([sys.executable, "setup_improved.py", "build_ext", "--inplace"], check=True)
            print("✓ Build successful")
            
            # Then run quick test
            subprocess.run([sys.executable, "quick_test.py"], check=True)
            print("✓ Quick test completed")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed: {e}")
            print("\nTry manual execution:")
            print("1. python setup_improved.py build_ext --inplace")
            print("2. python quick_test.py")

if __name__ == "__main__":
    main()
