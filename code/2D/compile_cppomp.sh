#!/usr/bin/env bash
# compile_cppomp.sh – build force_kernel for the current Python
python - <<'PY'
import sys, sysconfig, subprocess, shutil, os, textwrap
includes = subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True).strip()
ext = sysconfig.get_config_var('EXT_SUFFIX')
cmd = f"g++ -O3 -std=c++17 -fopenmp -shared -fPIC {includes} bh_omp.cpp -o force_kernel{ext}"
print("[compile]",cmd)
subprocess.check_call(cmd,shell=True)
PY