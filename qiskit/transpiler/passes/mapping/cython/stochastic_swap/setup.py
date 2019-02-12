# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os
import sys
import distutils.sysconfig
from setuptools import setup, Extension
from Cython.Build import cythonize



# Add Cython extensions here
CYTHON_EXTS = ['utils', '_swap_trial']

INCLUDE_DIRS = []
# Extra link args
LINK_FLAGS = []
# If on Win and Python version >= 3.5 and not in MSYS2 (i.e. Visual studio compile)
if (sys.platform == 'win32' and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
        and os.environ.get('MSYSTEM') is None):
    COMPILER_FLAGS = ['/w', '/Ox', '/std:c++11']
# Everything else
else:
    COMPILER_FLAGS = ['-w', '-O3', '-march=native', '-funroll-loops', '-std=c++11']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        COMPILER_FLAGS.append('-mmacosx-version-min=10.9')
        LINK_FLAGS.append('-mmacosx-version-min=10.9')

# Remove -Wstrict-prototypes from cflags
CFG_VARS = distutils.sysconfig.get_config_vars()
if "CFLAGS" in CFG_VARS:
    CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")


EXT_MODULES = []
# Add Cython files from qutip/cy
for ext in CYTHON_EXTS:
    mod = Extension(ext,
                     sources=[ext+'.pyx'],
                     include_dirs=INCLUDE_DIRS,
                     extra_compile_args=COMPILER_FLAGS,
                     extra_link_args=LINK_FLAGS,
                     language='c++')
    EXT_MODULES.append(mod)


setup(name='Qiskit_Cython',
      ext_modules=cythonize(EXT_MODULES)
     )