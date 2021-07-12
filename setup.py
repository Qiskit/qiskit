# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Qiskit Terra setup file."

import os
import re
import sys
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    import subprocess

    subprocess.call([sys.executable, "-m", "pip", "install", "Cython>=0.27.1"])
    from Cython.Build import cythonize

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# Add Cython extensions here
CYTHON_EXTS = {
    "qiskit/transpiler/passes/routing/cython/stochastic_swap/utils": (
        "qiskit.transpiler.passes.routing.cython.stochastic_swap.utils"
    ),
    "qiskit/transpiler/passes/routing/cython/stochastic_swap/swap_trial": (
        "qiskit.transpiler.passes.routing.cython.stochastic_swap.swap_trial"
    ),
    "qiskit/quantum_info/states/cython/exp_value": "qiskit.quantum_info.states.cython.exp_value",
}

INCLUDE_DIRS = []
# Extra link args
LINK_FLAGS = []
# If on Win and not in MSYS2 (i.e. Visual studio compile)
if sys.platform == "win32" and os.environ.get("MSYSTEM") is None:
    COMPILER_FLAGS = ["/O2"]
# Everything else
else:
    COMPILER_FLAGS = ["-O2", "-funroll-loops", "-std=c++11"]
    if sys.platform == "darwin":
        # These are needed for compiling on OSX 10.14+
        COMPILER_FLAGS.append("-mmacosx-version-min=10.9")
        LINK_FLAGS.append("-mmacosx-version-min=10.9")


EXT_MODULES = []
# Add Cython Extensions
for src, module in CYTHON_EXTS.items():
    ext = Extension(
        module,
        sources=[src + ".pyx"],
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILER_FLAGS,
        extra_link_args=LINK_FLAGS,
        language="c++",
    )
    EXT_MODULES.append(ext)

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )


visualization_extras = [
    "matplotlib>=2.1",
    "ipywidgets>=7.3.0",
    "pydot",
    "pillow>=4.2.1",
    "pylatexenc>=1.4",
    "seaborn>=0.9.0",
    "pygments>=2.4",
]


z3_requirements = [
    "z3-solver>=4.7",
]


bip_requirements = ["cplex", "docplex"]


setup(
    name="qiskit-terra",
    version="0.18.0",
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Qiskit/qiskit-terra",
    author="Qiskit Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    setup_requires=["Cython>=0.27.1"],
    include_package_data=True,
    python_requires=">=3.6",
    extras_require={
        "visualization": visualization_extras,
        "bip-mapper": bip_requirements,
        "crosstalk-pass": z3_requirements,
        "all": visualization_extras + z3_requirements + bip_requirements,
    },
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/qiskit-terra/issues",
        "Documentation": "https://qiskit.org/documentation/",
        "Source Code": "https://github.com/Qiskit/qiskit-terra",
    },
    ext_modules=cythonize(EXT_MODULES),
    zip_safe=False,
)
