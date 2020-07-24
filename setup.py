# -*- coding: utf-8 -*-

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
import sys
from setuptools import setup, find_packages, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    import subprocess
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'Cython>=0.27.1'])
    from Cython.Build import cythonize

REQUIREMENTS = [
    "contextvars>=2.4;python_version<'3.7'",
    "jsonschema>=2.6",
    "networkx>=2.2;python_version>'3.5'",
    # Networkx 2.4 is the final version with python 3.5 support.
    "networkx>=2.2,<2.4;python_version=='3.5'",
    "retworkx>=0.4.0",
    "numpy>=1.17",
    "ply>=3.10",
    "psutil>=5",
    "scipy>=1.4",
    "sympy>=1.3",
    "dill>=0.3",
    "fastjsonschema>=2.10",
    "python-constraint>=1.4",
    "python-dateutil>=2.8.0",
]

# Add Cython extensions here
CYTHON_EXTS = ['utils', 'swap_trial']
CYTHON_MODULE = 'qiskit.transpiler.passes.routing.cython.stochastic_swap'
CYTHON_SOURCE_DIR = 'qiskit/transpiler/passes/routing/cython/stochastic_swap'

INCLUDE_DIRS = []
# Extra link args
LINK_FLAGS = []
# If on Win and not in MSYS2 (i.e. Visual studio compile)
if (sys.platform == 'win32' and os.environ.get('MSYSTEM') is None):
    COMPILER_FLAGS = ['/O2']
# Everything else
else:
    COMPILER_FLAGS = ['-O2', '-funroll-loops', '-std=c++11']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        COMPILER_FLAGS.append('-mmacosx-version-min=10.9')
        LINK_FLAGS.append('-mmacosx-version-min=10.9')


EXT_MODULES = []
# Add Cython Extensions
for ext in CYTHON_EXTS:
    mod = Extension(CYTHON_MODULE + '.' + ext,
                    sources=[CYTHON_SOURCE_DIR + '/' + ext + '.pyx'],
                    include_dirs=INCLUDE_DIRS,
                    extra_compile_args=COMPILER_FLAGS,
                    extra_link_args=LINK_FLAGS,
                    language='c++')
    EXT_MODULES.append(mod)

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name="qiskit-terra",
    version="0.15.0",
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/Qiskit/qiskit-terra",
    author="Qiskit Development Team",
    author_email="qiskit@qiskit.org",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=['test*']),
    install_requires=REQUIREMENTS,
    setup_requires=['Cython>=0.27.1'],
    include_package_data=True,
    python_requires=">=3.5",
    extras_require={
        'visualization': ['matplotlib>=2.1', 'ipywidgets>=7.3.0',
                          'pydot', "pillow>=4.2.1", "pylatexenc>=1.4",
                          "seaborn>=0.9.0", "pygments>=2.4"],
        'full-featured-simulators': ['qiskit-aer>=0.1'],
        'crosstalk-pass': ['z3-solver>=4.7'],
    },
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/qiskit-terra/issues",
        "Documentation": "https://qiskit.org/documentation/",
        "Source Code": "https://github.com/Qiskit/qiskit-terra",
    },
    ext_modules=cythonize(EXT_MODULES),
    zip_safe=False
)
