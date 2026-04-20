# setup.py
import os
import sys
from setuptools import setup, Extension, find_packages
import pybind11

module_name = "multiCMH.cpp_modules.multiCMH_cpp_module"

source_files = [
    "src/multiCMH/cpp_modules/cpp_bindings.cpp",
    "src/multiCMH/cpp_modules/Tbl22.cpp",
    "src/multiCMH/cpp_modules/fastcluster_dm.cpp",
    "src/multiCMH/cpp_modules/ward_fastcluster.cpp",
    "src/multiCMH/cpp_modules/median_tree.cpp",
    "src/multiCMH/cpp_modules/CMH.cpp",
    "src/multiCMH/cpp_modules/singleCMH.cpp",
    "src/multiCMH/cpp_modules/multiCMH.cpp",
]


def split_paths(env_var):
    value = os.environ.get(env_var, "")
    return [p for p in value.split(os.pathsep) if p]


def unique_existing_paths(paths):
    seen = set()
    out = []
    for p in paths:
        if not p:
            continue
        p = os.path.abspath(p)
        if p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


if sys.platform.startswith("win"):
    raise RuntimeError(
        "multiCMH setup.py does not support Windows. "
        "Please install on Linux or macOS."
    )

conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)

include_candidates = (
    split_paths("CPLUS_INCLUDE_PATH")
    + [
        pybind11.get_include(),
        os.path.join(conda_prefix, "include"),
        os.path.join(conda_prefix, "include", "eigen3"),
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "include", "eigen3"),
        "src/multiCMH/cpp_modules",
    ]
)

library_candidates = (
    split_paths("LD_LIBRARY_PATH")
    + split_paths("DYLD_LIBRARY_PATH")
    + [
        os.path.join(conda_prefix, "lib"),
        os.path.join(sys.prefix, "lib"),
    ]
)

include_dirs = unique_existing_paths(include_candidates)
library_dirs = unique_existing_paths(library_candidates)

extra_compile_args = ["-O2", "-std=c++17"]

extension_module = Extension(
    name=module_name,
    sources=source_files,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=[],
    extra_compile_args=extra_compile_args,
    language="c++",
)

setup(
    name="multiCMH",
    version="0.1.0",
    description="multiCMH package with pybind11 C++ extensions",
    author="gk149",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[extension_module],
    install_requires=[
        "pybind11>=2.6",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)