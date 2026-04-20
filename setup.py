# setup.py
import os
import sys
from setuptools import setup, Extension
import pybind11

# The name of your C++ module, as defined in your pybind11 code
module_name = 'multiCMH.cpp_modules.multiCMH_cpp_module'
source_files = [
    'src/multiCMH/cpp_modules/cpp_bindings.cpp',
    'src/multiCMH/cpp_modules/Tbl22.cpp',
    'src/multiCMH/cpp_modules/fastcluster_dm.cpp',
    'src/multiCMH/cpp_modules/ward_fastcluster.cpp',
    'src/multiCMH/cpp_modules/median_tree.cpp',
    'src/multiCMH/cpp_modules/CMH.cpp',
    'src/multiCMH/cpp_modules/singleCMH.cpp',
    'src/multiCMH/cpp_modules/multiCMH.cpp'
]

# --- Find Headers and Libraries ---
# This part of the script dynamically finds paths from the environment.
# This makes it portable across your HPC and local machine.
if sys.platform == 'win32':
    # On Windows, we'll assume a vcpkg installation in C:/vcpkg
    vcpkg_root = 'C:/vcpkg'
    include_dirs = [
        os.path.join(vcpkg_root, 'installed', 'x64-windows', 'include'),
        os.path.join(vcpkg_root, 'installed', 'x64-windows', 'include', 'eigen3'),
        os.path.join(os.path.dirname(sys.executable), 'include'),
        'src/multiCMH/cpp_modules',
        pybind11.get_include()
    ]
    library_dirs = [
        os.path.join(vcpkg_root, 'installed', 'x64-windows', 'lib'),
        os.path.join(vcpkg_root, 'installed', 'x64-windows', 'lib', 'manual-link'),
    ]
    libraries = []
    extra_compile_args = ['/std:c++17', '/O2'] # /02 maximize optimization for speed
    
else: # Linux or other platforms: assume you have conda and created environment as
    # (Linux) conda create -n <your_environment> python=3.9 pybind11 boost eigen gxx_linux-64 sysroot_linux-64
    # (Mac) conda create -n <your_environment> python=3.9 pybind11 boost eigen clang_osx-64 sysroot_osx-64
    include_dirs = os.environ.get('CPLUS_INCLUDE_PATH', '').split(':')
    include_dirs.extend([
        os.path.join(sys.prefix, 'include'),
        'src/multiCMH/cpp_modules',
        pybind11.get_include()
    ])
    library_dirs = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if not any(library_dirs):
        library_dirs = os.environ.get('DYLD_LIBRARY_PATH', '').split(':') # for mac
    library_dirs.append(os.path.join(sys.prefix, 'lib'))
    # Libraries may be handled by the Conda environment or not needed for header-only parts.
    libraries = [] # The linker will find them if LD_LIBRARY_PATH is set.
    extra_compile_args = ['-std=c++17', '-O2']

# Define the Extension object for setuptools.
extension_module = Extension(
    name=module_name,
    sources=source_files,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args
)

# --- The `setup` function defines your package metadata and build process ---
setup(
    name='multiCMH',
    version='0.1.0',
    description='My multiCMH package with C++ modules',
    author='gk149',
    packages=['multiCMH'],
    package_dir={'': 'src'},
    ext_modules=[extension_module],
    install_requires=[
        'pybind11>=2.6'
    ],
    zip_safe=False
)
