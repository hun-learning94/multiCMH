# Multi-scale Cochran-Mantel-Haenszel Test (multiCMH)

This is a github repository housing a working code of the project ``multiCMH``. 

## Prerequisites
* **C++ Compiler**:
    * **Windows**: MSVC (Microsoft Visual C++) from Visual Studio
    * **Linux/Mac OS**: GCC (G++) 9+ or Clang 9+
* **Python**: 3.13.3+
* **R**: 4.4.1+

## Installation

### C++ Dependencies
The C++ components (codes in `cpp` folder) depend on `boost-math` and `Eigen`.

#### `boost-math`
For Windows users, we recommend using `vcpkg`:
```cmd
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat # compiles the vcpkg executable.
.\vcpkg integrate install # to make Visual Studio find libraries installed by vcpkg.
setx VCPKG_INSTALLATION_ROOT "C:\vcpkg" # add environment variable
.\vcpkg install boost-math
.\vcpkg list # check installed libraries
```
For Linux (Ubuntu/Debian) users,
```bash
sudo apt update
sudo apt install libboost-math-dev
```
We will later write a `cppimport` script at `cppfunctions.cpp` to tell the compiler where to find `boost-math`.

#### `Eigen`
Since `Eigen` is a header-only library, you just need to download, unzip, and place it in a folder where compiler can find. We will place the folder under `cpp/include` directory and write a `cppimport` script at `cppfunctions.cpp` to tell the compiler where to find `Eigen`.
1. Go to [Eigen website](eigen.tuxfamily.org), download the latest stable release `tar.gz`, unzip it.
2. Move `Eigen` folder under `cpp\include`.

or use can type
```cmd
cd C:\vcpkg
.\vcpkg install eigen3
.\vcpkg list # check installed libraries
```

### Python Dependencies
We recommend using virtual environment for Python.
```cmd
python -m venv .venv
.venv\Scripts\activate # On Windows
source .venv/bin/activate # On Linux/macOS
```
To install necessary Python packages,
```cmd
pip install pybind11 cppimport setuptools numpy scipy pandas scikit-learn matplotlib notebook
```

```cmd
micromamba create -n multicmh python=3.10 pybind11 numpy pandas matplotlib seaborn scikit-learn scipy eigen boost pip
```

## Installing other methods
### R packages
At R terminal,
```R
install.packages(c("weightedGCM", "GeneralisedCovarianceMeasure", "CondIndTests", "cdcsis", "bnlearn"))
```
