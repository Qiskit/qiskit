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

if __name__ == "__main__":
    setup(
        ext_modules=cythonize(EXT_MODULES),
        zip_safe=False
    )

