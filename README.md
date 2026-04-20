# multiCMH

`multiCMH` provides Python bindings to C++ implementations for the multi-scale Cochran-Mantel-Haenszel test.

## Repository structure

- `src/multiCMH/`: Python package
- `src/multiCMH/cpp_modules/`: C++ source files and pybind11 bindings
- `tests/`: tests and examples
- `data/`: simulation files 

## What this package builds

Installing this package compiles a C++ extension module via `setuptools` and `pybind11`.

The extension is built from:

- `src/multiCMH/cpp_modules/cpp_bindings.cpp`
- `src/multiCMH/cpp_modules/Tbl22.cpp`
- `src/multiCMH/cpp_modules/fastcluster_dm.cpp`
- `src/multiCMH/cpp_modules/ward_fastcluster.cpp`
- `src/multiCMH/cpp_modules/median_tree.cpp`
- `src/multiCMH/cpp_modules/CMH.cpp`
- `src/multiCMH/cpp_modules/singleCMH.cpp`
- `src/multiCMH/cpp_modules/multiCMH.cpp`

## Requirements

### Required system tools

- A C++17-compatible compiler
  - Linux: `g++`
  - macOS: `clang++`
  - Windows: MSVC from Visual Studio

### Required libraries

This project depends on:

- `pybind11`
- `Eigen` (header-only)
- Boost headers / Boost Math
- Python packages used by the wrapper:
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## Recommended installation

For Linux, macOS, and most HPC systems, the most reliable approach is to use **micromamba** or **conda** and install all build dependencies inside one environment.

Create and activate an environment:

```bash
micromamba create -n multiCMH -c conda-forge \
  python=3.11 \
  pybind11 \
  numpy scipy pandas matplotlib seaborn scikit-learn \
  eigen boost-cpp \
  pip setuptools
micromamba activate multiCMH