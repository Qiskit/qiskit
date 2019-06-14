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
import distutils.sysconfig
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

REQUIREMENTS = [
    "jsonschema>=2.6,<2.7",
    "marshmallow>=2.17.0,<3",
    "marshmallow_polyfield>=3.2,<4",
    "networkx>=2.2",
    "numpy>=1.13",
    "pillow>=4.2.1",
    "ply>=3.10",
    "psutil>=5",
    "pylatexenc>=1.4",
    "scipy>=1.0",
    "sympy>=1.3"
]

# Add Cython extensions here
CYTHON_EXTS = ['utils', 'swap_trial']
CYTHON_MODULE = 'qiskit.transpiler.passes.mapping.cython.stochastic_swap'
CYTHON_SOURCE_DIR = 'qiskit/transpiler/passes/mapping/cython/stochastic_swap'

PACKAGE_DATA = {}

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

# Remove -Wstrict-prototypes from cflags
CFG_VARS = distutils.sysconfig.get_config_vars()
if "CFLAGS" in CFG_VARS:
    CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")

EXT_MODULES = []
# Add Cython Extensions
for ext in CYTHON_EXTS:
    mod = Extension(CYTHON_MODULE+'.'+ext,
                    sources=[CYTHON_SOURCE_DIR+'/'+ext+'.pyx'],
                    include_dirs=INCLUDE_DIRS,
                    extra_compile_args=COMPILER_FLAGS,
                    extra_link_args=LINK_FLAGS,
                    language='c++')
    EXT_MODULES.append(mod)


setup(
    name="qiskit-terra",
    version="0.8.2",
    description="Software for developing quantum computing programs",
    long_description="""Terra provides the foundations for Qiskit. It allows the user to write
        quantum circuits easily, and takes care of the constraints of real hardware.""",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=['test*']),
    install_requires=REQUIREMENTS,
    setup_requires=['Cython>=0.27.1'],
    package_data=PACKAGE_DATA,
    include_package_data=True,
    python_requires=">=3.5",
    extras_require={
        'visualization': ['matplotlib>=2.1', 'nxpd>=0.2', 'ipywidgets>=7.3.0',
                          'pydot'],
        'full-featured-simulators': ['qiskit-aer>=0.1']
    },
    ext_modules=cythonize(EXT_MODULES),
    zip_safe=False
)
